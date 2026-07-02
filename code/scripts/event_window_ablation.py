#!/usr/bin/env python
"""Experiment 2: event-window causal ablation.

Causal test of the claim that "event-aligned trace components are used by
the planner" (not merely correlated with events).

Protocol per (env, model, ablation_mode):
  1. Build env + load ckpt.
  2. PROBE: run a random-policy rollout and record per-step obs.
  3. Detect "event steps" — steps where ||Δobs||_2 exceeds median + 1·MAD.
     Form event windows by including ±1 step around each event peak.
     Form matched non-event windows from steps where ||Δobs|| < median.
     Form random windows (same total step count) at random positions.
  4. EVAL: run a custom closed-loop loop that mirrors code/eval/closed_loop.py
     but applies the trace ablation ONLY at the post-step history-update
     `predict` call (NOT inside CEM rollouts). The ablation is keyed on the
     env-step counter, so each closed-loop env step is independently either
     ablated or not. Compare env-SR across 4 conditions:
       (a) baseline            — no ablation
       (b) event_window        — zero trace at env steps in event windows
       (c) non_event_window    — zero trace at matched low-Δobs windows
       (d) random_window       — zero trace at random positions
     The KEY claim is supported if (b) drops env-SR more than (c) and (d).

Ablation implementation
----------------------
We monkey-patch `model.predict` so that on each call we consult a global
"ablate_now" flag. When set, we monkey-patch `model.gated_trace.forward`
for the duration of that single call to zero r_t at the env-step index
being predicted.

To do this without modifying closed_loop.py, we replicate the eval logic
inline here (mirroring `code/eval/closed_loop.py::eval_closed_loop`).

Usage:
    python -m code.scripts.event_window_ablation --env ball_in_cup --model stjewm_v2 \
        --out results/aggregate/event_window_ablation/ball_in_cup_stjewm_v2.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Set

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/lx/snn")

from code.eval.closed_loop import (
    make_env,
    ClosedLoopResult,
    _load_eval_dataset,
    encode_obs,
    encode_history,
    _set_env_state,
)
from code.core.cem import CEM
from code.stjewm import STJEWM


# env_name -> (env_kind_in_make_env, data_path, goal_offset, history_size)
ENV_CONFIG = {
    "ball_in_cup": ("ball_in_cup", "/home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz", 25, 1),
    "cartpole_2d": ("cartpole",   "/home/lx/snn/data/dm_control/cartpole_250k.npz", 25, 1),
    "cheetah":     ("cheetah",    "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz", 25, 1),
    "pusht":       ("pusht",      "/home/lx/LeWM/data/pusht_expert_train.h5", 100, 1),
}


# ============================================================
# Ablation hook: a context manager that zeroes the trace at one env step.
# ============================================================
class SingleStepTraceZeros:
    """Context manager: monkey-patch gated_trace.forward for the duration of
    one predict() call to zero r_t at the current env-step index.

    We use a stack-based approach so we can nest (or overlap) hooks safely.
    """

    def __init__(self, model: STJEWM, step_idx: int):
        self.model = model
        self.step_idx = int(step_idx)
        self._orig_gt_forward = None
        self._active = False

    def __enter__(self):
        gt = self.model.gated_trace
        self._orig_gt_forward = gt.forward
        target_step = self.step_idx
        # The closed-loop uses history_size=1, so predict(ctx_emb (B,1,D), ctx_act (B,1,A))
        # produces a trace of shape (B, 1, D). We zero r_t at index 0 of the output
        # when global step == target_step.
        def patched_forward(spike, context):
            B, T, D = spike.shape
            r = torch.zeros(B, D, device=spike.device, dtype=spike.dtype)
            traces = []
            for t in range(T):
                global_idx = target_step  # local to this single predict call
                if global_idx in (target_step,):  # always true in this scope
                    traces.append(torch.zeros_like(r))
                    r = torch.zeros_like(r)
                else:
                    s = spike[:, t]
                    gate_in = torch.cat([r, s, context[:, t]], dim=-1)
                    alpha = torch.sigmoid(gt.gate(gate_in))
                    r = alpha * r + (1.0 - alpha) * s
                    traces.append(r)
            return torch.stack(traces, dim=1)
        gt.forward = patched_forward
        self._active = True
        return self

    def __exit__(self, *args):
        if self._active:
            self.model.gated_trace.forward = self._orig_gt_forward
            self._active = False


# ============================================================
# Probe rollout: collect obs across a random-policy rollout.
# ============================================================
@torch.no_grad()
def probe_rollout(env, n_steps: int = 99, seed: int = 0):
    """Random-policy rollout. Returns (obs_arr, n_resets)."""
    a_low = env.spec.action_low
    a_high = env.spec.action_high
    obs_list = []
    env.reset(seed=seed)
    obs = env.get_state().astype(np.float32)
    obs_list.append(obs)
    t = 0
    n_done = 0
    rng = np.random.default_rng(seed)
    while t < n_steps:
        a = rng.uniform(a_low, a_high).astype(np.float32)
        try:
            _obs, _r, done, _info = env.step(a)
        except Exception:
            done = True
        try:
            obs_next = env.get_state().astype(np.float32)
        except Exception:
            obs_next = obs
        obs = obs_next
        obs_list.append(obs)
        t += 1
        if done and t < n_steps:
            n_done += 1
            env.reset(seed=seed + n_done)
            obs = env.get_state().astype(np.float32)
    obs_arr = np.stack(obs_list, axis=0)
    return obs_arr, n_done


def detect_event_steps(obs_arr: np.ndarray, mad_k: float = 1.0):
    """Return (event_idx, median_d_obs). Event_idx = positions where
    ||Δobs|| > median + mad_k * MAD."""
    d_obs = np.linalg.norm(np.diff(obs_arr, axis=0), axis=1)
    med = float(np.median(d_obs))
    mad = float(np.median(np.abs(d_obs - med))) + 1e-9
    threshold = med + mad_k * mad
    return np.where(d_obs > threshold)[0], med


def build_window_sets(event_idx: np.ndarray, non_event_idx: np.ndarray,
                      half_w: int, max_windows: int) -> dict:
    """Form three window sets in env-step space.

    Returns dict with keys 'event', 'non_event', 'random', each a Set[int].
    Each set has approximately `max_windows * (2*half_w + 1)` elements.
    """
    n_event_peaks = min(len(event_idx), max_windows)
    event_w = set()
    for e in event_idx[:n_event_peaks]:
        for t in range(max(0, int(e) - half_w), int(e) + half_w + 1):
            event_w.add(t)
    n_event_total = len(event_w)

    # Non-event: sample n_event_total distinct positions from non_event_idx.
    non_event_w = set()
    if len(non_event_idx) > 0 and n_event_total > 0:
        rng = np.random.default_rng(1)
        take = min(len(non_event_idx), n_event_total)
        pick_idx = rng.choice(len(non_event_idx), size=take, replace=False)
        for pi in pick_idx:
            non_event_w.add(int(non_event_idx[pi]))

    # Random: sample n_event_total distinct positions uniformly.
    random_w = set()
    if n_event_total > 0:
        # We don't know total_steps here, but each closed-loop runs for ~eval_budget steps.
        # Use a large cap; we'll intersect with the actual trajectory range later.
        rng = np.random.default_rng(2)
        # Pick from a range that's at least eval_budget sized
        ub = max(n_event_total * 5, 200)
        pick = rng.choice(ub, size=n_event_total, replace=False)
        random_w = set(int(t) for t in pick)

    return {"event": event_w, "non_event": non_event_w, "random": random_w}


# ============================================================
# Build a model + load ckpt
# ============================================================
def build_model_and_ckpt(ckpt_path: str, state_dim: int, action_dim: int, device: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {}) or {}
    if ck_args.get("model", "stjewm") == "stjewm":
        n_layers = ck_args.get("n_layers", 4)
        ck_readout_mode = ck_args.get("readout_mode", "hidden_leak")
        model = STJEWM(
            d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
            state_dim=state_dim, cell_n_layers=n_layers, n_d=3,
            trace_beta=0.9, freeze_encoder=True,
            readout_mode=ck_readout_mode,
        )
    else:
        raise ValueError(f"Only STJEWM ckpts supported here; got {ck_args.get('model')}")
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    return model, ck_args


# ============================================================
# Custom eval loop with env-step-localized ablation.
# Mirrors code/eval/closed_loop.py::eval_closed_loop but applies the trace
# ablation hook ONLY at the post-step history-update predict call.
# ============================================================
@torch.no_grad()
def custom_eval_with_ablation(
    model: STJEWM,
    env,
    data_path: str,
    ablation_step_set: Set[int],
    goal_offset: int,
    history_size: int,
    horizon: int,
    eval_budget: int,
    n_episodes: int,
    n_seeds: int,
    device: str,
    cem_samples: int = 300,
    cem_elites: int = 30,
    cem_iters: int = 10,
    success_threshold_cos: float = 0.1,
    split: str = "in_dist",
) -> ClosedLoopResult:
    """Mirror of eval_closed_loop, but with selective trace ablation keyed on
    env-step index. ablation_step_set = env-step indices at which to zero r_t
    during the history-update predict call.
    """
    model.eval()
    action_dim = env.spec.action_dim
    action_low = env.spec.action_low
    action_high = env.spec.action_high

    # Load dataset for sampling init/goal
    ds, _ = _load_eval_dataset(env, data_path, history_size, goal_offset, split=split)

    cem = CEM(
        model, action_dim=action_dim, horizon=horizon,
        n_samples=cem_samples, n_elites=cem_elites, n_iters=cem_iters,
        history_size=history_size, device=device,
    )

    wall_t0 = time.time()
    per_seed_results = []
    per_episode_all = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed * 7919 + 42)
        if ds is not None:
            N = len(ds)
            episode_indices = rng.choice(N, size=min(n_episodes, N), replace=False)
        else:
            episode_indices = list(range(n_episodes))

        seed_episodes = []
        for ep_idx in episode_indices:
            item = ds[int(ep_idx)]
            init_state_np = item["init_state"].numpy() if hasattr(item["init_state"], "numpy") else np.asarray(item["init_state"])
            goal_state_np = item["goal_state"].numpy() if hasattr(item["goal_state"], "numpy") else np.asarray(item["goal_state"])

            try:
                env.reset(seed=int(ep_idx) + seed * 1000)
            except Exception:
                pass
            try:
                _set_env_state(env, init_state_np)
            except Exception:
                pass

            history_states = [init_state_np.copy() for _ in range(history_size)]
            try:
                z_history = encode_history(model, [torch.from_numpy(s).float() for s in history_states], action_dim, device)
            except Exception:
                z_history = encode_obs(model, torch.from_numpy(init_state_np).float(), action_dim, device).unsqueeze(0).repeat(history_size, 1)

            z_goal = encode_obs(model, torch.from_numpy(goal_state_np).float(), action_dim, device)

            actions_taken = 0
            best_actions = None
            t_start = time.time()
            z_init = z_history[-1]
            # env_step_counter tracks the upcoming env step BEFORE the history update.
            # At env step 0, we just stepped into the init state — no history update yet.
            # After taking action 0 and updating history, env_step_counter=1.
            env_step_counter = 0
            done = False
            while actions_taken < eval_budget:
                seq = cem.plan(z_init, z_goal)
                for a_idx in range(min(horizon, eval_budget - actions_taken)):
                    action = seq[a_idx].cpu().numpy().astype(np.float32)
                    action = np.clip(action, action_low, action_high)
                    try:
                        _obs, _r, done, _info = env.step(action)
                    except Exception:
                        done = True
                    actions_taken += 1
                    env_step_counter += 1
                    if done:
                        break
                if done or actions_taken >= eval_budget:
                    break
                # Roll history forward (this is the only place we apply ablation)
                try:
                    a_window = seq[:history_size].unsqueeze(0)
                    if env_step_counter in ablation_step_set:
                        with SingleStepTraceZeros(model, env_step_counter):
                            nxt = model.predict(z_history.unsqueeze(0), a_window)
                    else:
                        nxt = model.predict(z_history.unsqueeze(0), a_window)
                    z_history = torch.cat([z_history[1:], nxt[0:1, -1]], dim=0)
                    z_init = z_history[-1]
                except Exception:
                    break
            plan_time = time.time() - t_start

            try:
                final_state_np = env.get_state()
            except Exception:
                final_state_np = init_state_np

            try:
                z_final = encode_obs(model, torch.from_numpy(final_state_np).float(), action_dim, device)
            except Exception:
                z_final = z_goal

            cos = torch.nn.functional.cosine_similarity(
                z_final.unsqueeze(0), z_goal.unsqueeze(0)
            )
            cos_dist = float((1.0 - cos.item()) / 2.0)
            lewm_success = cos_dist < success_threshold_cos

            env_success, phys_dist = env.check_success(final_state_np, goal_state_np)

            ep_dict = {
                "seed": seed,
                "episode_idx": int(ep_idx),
                "cos_dist": cos_dist,
                "lewm_success": bool(lewm_success),
                "env_success": bool(env_success),
                "phys_dist": float(phys_dist),
                "plan_time_sec": plan_time,
                "env_steps": env_step_counter,
            }
            seed_episodes.append(ep_dict)
            per_episode_all.append(ep_dict)

        if seed_episodes:
            lewm_succ = np.mean([e["lewm_success"] for e in seed_episodes])
            env_succ = np.mean([e["env_success"] for e in seed_episodes])
            mean_cos = np.mean([e["cos_dist"] for e in seed_episodes])
            mean_phys = np.mean([e["phys_dist"] for e in seed_episodes])
            per_seed_results.append({
                "seed": seed,
                "n": len(seed_episodes),
                "success_rate_lewm": float(lewm_succ),
                "success_rate_env": float(env_succ),
                "mean_cos_dist": float(mean_cos),
                "mean_phys_dist": float(mean_phys),
            })

    lewm_arr = np.array([s["success_rate_lewm"] for s in per_seed_results])
    env_arr = np.array([s["success_rate_env"] for s in per_seed_results])
    cos_arr = np.array([s["mean_cos_dist"] for s in per_seed_results])
    phys_arr = np.array([s["mean_phys_dist"] for s in per_seed_results])

    return ClosedLoopResult(
        env_id=env.spec.env_id,
        n_episodes=len(per_episode_all),
        n_seeds=len(per_seed_results),
        cem_samples=cem_samples,
        cem_elites=cem_elites,
        cem_iters=cem_iters,
        horizon=horizon,
        eval_budget=eval_budget,
        goal_offset=goal_offset,
        history_size=history_size,
        success_rate_lewm=float(lewm_arr.mean()) if len(lewm_arr) > 0 else float("nan"),
        success_rate_lewm_std=float(lewm_arr.std()) if len(lewm_arr) > 0 else float("nan"),
        success_rate_env=float(env_arr.mean()) if len(env_arr) > 0 else float("nan"),
        success_rate_env_std=float(env_arr.std()) if len(env_arr) > 0 else float("nan"),
        mean_cos_dist=float(cos_arr.mean()) if len(cos_arr) else float("nan"),
        mean_cos_dist_std=float(cos_arr.std()) if len(cos_arr) else float("nan"),
        mean_phys_dist=float(phys_arr.mean()) if len(phys_arr) else float("nan"),
        mean_phys_dist_std=float(phys_arr.std()) if len(phys_arr) else float("nan"),
        per_seed=per_seed_results,
        per_episode=per_episode_all,
        wall_time_sec=time.time() - wall_t0,
    )


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=list(ENV_CONFIG))
    p.add_argument("--model", default="stjewm_v2",
                   help="Model dir name (under results/<env>/).")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--probe-steps", type=int, default=99,
                   help="Number of probe-rollout steps for event detection.")
    p.add_argument("--mad-k", type=float, default=1.0)
    p.add_argument("--half-w", type=int, default=1,
                   help="Half-width (in steps) of each event window.")
    p.add_argument("--max-windows", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    env_kind, data_path, goal_offset, history_size = ENV_CONFIG[args.env]
    ckpt_path = args.ckpt or f"/home/lx/snn/results/{args.env}/{args.model}/final.pt"
    if not os.path.exists(ckpt_path):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"skipped": True, "reason": f"no ckpt at {ckpt_path}"}, f, indent=2)
        print(f"[event_window_ablation] skip — no ckpt at {ckpt_path}")
        return

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    env = make_env(env_kind, data_path)
    state_dim = env.spec.obs_dim
    action_dim = env.spec.action_dim

    model, ck_args = build_model_and_ckpt(ckpt_path, state_dim, action_dim, device)

    # 1. Probe rollout to detect events
    print(f"[event_window_ablation] probe rollout on {args.env}...", flush=True)
    obs_arr, n_resets = probe_rollout(env, n_steps=args.probe_steps, seed=0)
    (event_idx, med) = detect_event_steps(obs_arr, mad_k=args.mad_k)
    d_obs = np.linalg.norm(np.diff(obs_arr, axis=0), axis=1)
    non_event_idx = np.where(d_obs < med)[0]
    print(f"[event_window_ablation] probe: {len(event_idx)} event steps, "
          f"{len(non_event_idx)} non-event steps, "
          f"median d_obs={med:.4f}, n_resets={n_resets}", flush=True)
    if len(event_idx) == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"skipped": True, "reason": "no event steps detected (env too smooth)"}, f, indent=2)
        print(f"[event_window_ablation] skip — no event steps detected")
        return

    # 2. Build window sets
    window_sets = build_window_sets(
        event_idx, non_event_idx,
        half_w=args.half_w, max_windows=args.max_windows,
    )
    print(f"[event_window_ablation] window sizes: "
          f"event={len(window_sets['event'])}, "
          f"non_event={len(window_sets['non_event'])}, "
          f"random={len(window_sets['random'])}", flush=True)
    if len(window_sets["event"]) == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"skipped": True, "reason": "no event windows built"}, f, indent=2)
        return

    # 3. Run eval under 5 ablation modes (incl. ablate_all sanity check)
    # Use a sentinel ("AB_ALL") to mean "ablate every env step" — confirms the hook works.
    results = {}
    modes = [
        ("baseline", set()),
        ("event_window", window_sets["event"]),
        ("non_event_window", window_sets["non_event"]),
        ("random_window", window_sets["random"]),
        ("ablate_all", set(range(0, 10000))),  # all env steps; sanity check
    ]
    for mode_name, step_set in modes:
        print(f"[event_window_ablation] eval mode={mode_name}, "
              f"steps ablated={len(step_set)}", flush=True)
        t0 = time.time()
        result = custom_eval_with_ablation(
            model, env, data_path,
            ablation_step_set=step_set,
            goal_offset=goal_offset, history_size=history_size,
            horizon=args.horizon, eval_budget=args.eval_budget,
            n_episodes=args.n_episodes, n_seeds=args.n_seeds,
            device=device, split="in_dist",
        )
        elapsed = time.time() - t0
        results[mode_name] = {
            "env": args.env,
            "model": args.model,
            "ablation_mode": mode_name,
            "n_episodes": result.n_episodes,
            "n_seeds": result.n_seeds,
            "horizon": result.horizon,
            "eval_budget": result.eval_budget,
            "success_rate_lewm": result.success_rate_lewm,
            "success_rate_lewm_std": result.success_rate_lewm_std,
            "success_rate_env": result.success_rate_env,
            "success_rate_env_std": result.success_rate_env_std,
            "mean_cos_dist": result.mean_cos_dist,
            "mean_phys_dist": result.mean_phys_dist,
            "wall_time_sec": elapsed,
            "n_ablated_steps": len(step_set),
        }
        print(f"[event_window_ablation] {args.env}/{args.model}/{mode_name}: "
              f"lewm_sr={result.success_rate_lewm:.3f}, "
              f"env_sr={result.success_rate_env:.3f} ({elapsed:.1f}s)", flush=True)
    # 4. Compute drops (env-SR drop in pp, lewm-SR drop in pp)
    base = results["baseline"]
    drops = {}
    for mode_name in ["event_window", "non_event_window", "random_window", "ablate_all"]:
        if mode_name in results:
            drops[mode_name] = {
                "env_sr_drop_pp": (base["success_rate_env"] - results[mode_name]["success_rate_env"]) * 100.0,
                "lewm_sr_drop_pp": (base["success_rate_lewm"] - results[mode_name]["success_rate_lewm"]) * 100.0,
                "cos_dist_increase": results[mode_name]["mean_cos_dist"] - base["mean_cos_dist"],
            }
    drop_random_pp = drops["random_window"]["env_sr_drop_pp"]
    out = {
        "env": args.env,
        "model": args.model,
        "ckpt": ckpt_path,
        "probe": {
            "n_event_steps": int(len(event_idx)),
            "n_non_event_steps": int(len(non_event_idx)),
            "median_d_obs": med,
            "mad_k": args.mad_k,
            "half_w": args.half_w,
            "max_windows": args.max_windows,
            "n_probe_steps": args.probe_steps,
        },
        "windows": {
            "event_n_steps": len(window_sets["event"]),
            "non_event_n_steps": len(window_sets["non_event"]),
            "random_n_steps": len(window_sets["random"]),
            "event_set": sorted(window_sets["event"]),
            "non_event_set": sorted(window_sets["non_event"]),
            "random_set": sorted(window_sets["random"]),
        },
        "results": results,
        "drops": drops,
        "drops_pp": {
            "event": float(drops["event_window"]["env_sr_drop_pp"]),
            "non_event": float(drops["non_event_window"]["env_sr_drop_pp"]),
            "random": float(drops["random_window"]["env_sr_drop_pp"]),
            "ablate_all": float(drops["ablate_all"]["env_sr_drop_pp"]),
        },
        "causal_claim_supported": bool(
            drops["event_window"]["env_sr_drop_pp"] > drops["non_event_window"]["env_sr_drop_pp"]
            and drops["event_window"]["env_sr_drop_pp"] > drops["random_window"]["env_sr_drop_pp"]
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[event_window_ablation] saved -> {args.out}")
    print(f"[event_window_ablation] env-SR drops (pp): "
          f"event={drops['event_window']['env_sr_drop_pp']:.2f}, "
          f"non_event={drops['non_event_window']['env_sr_drop_pp']:.2f}, "
          f"random={drops['random_window']['env_sr_drop_pp']:.2f}, "
          f"ablate_all={drops['ablate_all']['env_sr_drop_pp']:.2f}; "
          f"causal_claim_supported={out['causal_claim_supported']}")


if __name__ == "__main__":
    main()
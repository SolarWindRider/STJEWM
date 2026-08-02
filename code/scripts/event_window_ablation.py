#!/usr/bin/env python
"""B4 event-window causal trace ablation on 5M-aligned checkpoints.

Restores the historical probe -> windows -> five-mode history-path protocol and
adds ``cem_rollout_ablation``, which zeros the STJEWM gated trace for every
model.predict call made by CEM's internal candidate rollouts.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from code.core.envs.dmc_env import DMCPixelEnv
from code.eval.closed_loop import eval_closed_loop, make_env
from code.train.train import build_model

ROOT = Path("/home/lx/snn")
ENV_CONFIG = {
    "cartpole_2d": ("cartpole", ROOT / "data/dm_control/cartpole_250k.npz"),
    "cheetah": ("cheetah", ROOT / "data/dm_control/3d_rollouts_250k/cheetah_250k.npz"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=ENV_CONFIG)
    p.add_argument("--model", default="stjewm_trace_only")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pixel", action="store_true")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-elites", type=int, default=30)
    p.add_argument("--cem-iters", type=int, default=10)
    p.add_argument("--probe-steps", type=int, default=99)
    p.add_argument("--mad-k", type=float, default=1.0)
    p.add_argument("--half-w", type=int, default=2)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def probe_rollout(env, n_steps=99, seed=0):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    obs = np.asarray(env.get_state(), dtype=np.float32)
    observations = [obs.copy()]
    resets = 0
    for _ in range(n_steps):
        action = rng.uniform(env.spec.action_low, env.spec.action_high).astype(np.float32)
        try:
            _, _, done, _ = env.step(action)
            obs = np.asarray(env.get_state(), dtype=np.float32)
        except Exception:
            done = True
        observations.append(obs.copy())
        if done:
            resets += 1
            env.reset(seed=seed + resets)
            obs = np.asarray(env.get_state(), dtype=np.float32)
    return np.stack(observations), resets


def detect_event_steps(observations, mad_k=1.0):
    d_obs = np.linalg.norm(np.diff(observations, axis=0), axis=1)
    median = float(np.median(d_obs))
    mad = float(np.median(np.abs(d_obs - median))) + 1e-9
    events = np.where(d_obs > median + mad_k * mad)[0]
    non_events = np.where(d_obs < median)[0]
    return events, non_events, median


def build_window_sets(event_idx, non_event_idx, half_w=2, max_windows=12):
    event = set()
    for e in event_idx[:max_windows]:
        event.update(range(max(0, int(e) - half_w), int(e) + half_w + 1))
    n = len(event)
    rng = np.random.default_rng(1)
    take = min(n, len(non_event_idx))
    non_event = set(map(int, rng.choice(non_event_idx, size=take, replace=False))) if take else set()
    rng = np.random.default_rng(2)
    upper = max(200, n * 5)
    random = set(map(int, rng.choice(upper, size=n, replace=False))) if n else set()
    return {"event": event, "non_event": non_event, "random": random}


@contextmanager
def zero_stjewm_trace(model):
    """Zero the gated trace for exactly the surrounding predict call."""
    if not hasattr(model, "gated_trace"):
        yield
        return
    gated_trace = model.gated_trace
    original = gated_trace.forward

    def zero_forward(spike, context):
        return torch.zeros_like(spike)

    gated_trace.forward = zero_forward
    try:
        yield
    finally:
        gated_trace.forward = original


def history_hook(step_set):
    def hook(model, ctx_emb, ctx_act, env_step):
        if env_step in step_set:
            with zero_stjewm_trace(model):
                return model.predict(ctx_emb, ctx_act)
        return model.predict(ctx_emb, ctx_act)
    return hook


def cem_hook(model, ctx_emb, ctx_act):
    with zero_stjewm_trace(model):
        return model.predict(ctx_emb, ctx_act)


def build_ckpt_model(ckpt_path, device, pixel=False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {}) or {}
    if pixel:
        model = build_model(
            args.get("model", "stjewm"), args.get("pad_obs_to", 21168),
            args.get("action_dim", 56), args.get("n_layers", 4),
            args.get("readout_mode", "hidden_leak"),
            embed_dim=args.get("embed_dim", 192), image_size=args.get("image_size", 84),
        )
    else:
        from code.stjewm import STJEWM
        model = STJEWM(
            d_hid=192, embed_dim=192, action_dim=args.get("action_dim", 56),
            action_emb_dim=192, state_dim=args.get("pad_obs_to", 128),
            cell_n_layers=args.get("n_layers", 4), n_d=3, trace_beta=0.9,
            freeze_encoder=True, image_size=224,
            readout_mode=args.get("readout_mode", "hidden_leak"),
        )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), args


def pixel_eval(model, env_name, args, history_predict_hook=None, cem_predict_hook=None):
    """Matched live-pixel MPC evaluation with latent and physical metrics."""
    import mujoco
    from code.core.cem import CEM

    env_kind, _ = ENV_CONFIG[env_name]
    image_size = int(getattr(args, "image_size", 84))
    env = DMCPixelEnv(env_kind, image_size=image_size, success_tol=0.1,
                      max_episode_steps=args.eval_budget)
    model_action_dim = model.action_dim
    cem = CEM(model, action_dim=model_action_dim, horizon=args.horizon,
              n_samples=args.cem_samples, n_elites=args.cem_elites,
              n_iters=args.cem_iters, history_size=1, device=args.device,
              predict_hook=cem_predict_hook)

    def encode(pixel):
        x = torch.from_numpy(pixel).float().unsqueeze(0).unsqueeze(0).to(args.device)
        return model._encode_obs(x)[0, 0]

    goal_state = np.zeros(env._nq, dtype=np.float32)
    env.reset(seed=0)
    env._data.qpos[:env._nq] = goal_state
    mujoco.mj_forward(env._model, env._data)
    z_goal = encode(env._render())
    episodes = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        torch.manual_seed(ep)
        np.random.seed(ep)
        obs = env.reset(seed=ep)
        z_history = encode(obs["pixel"]).unsqueeze(0)
        z_init = z_history[-1]
        actions_taken = 0
        done = False
        while actions_taken < args.eval_budget:
            seq = cem.plan(z_init, z_goal)
            for i in range(min(args.horizon, args.eval_budget - actions_taken)):
                action = seq[i].cpu().numpy()[:env.spec.action_dim]
                obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
                actions_taken += 1
                if done:
                    break
            if done or actions_taken >= args.eval_budget:
                break
            a_window = seq[:1].unsqueeze(0)
            if history_predict_hook is None:
                nxt = model.predict(z_history.unsqueeze(0), a_window)
            else:
                nxt = history_predict_hook(model, z_history.unsqueeze(0), a_window, actions_taken)
            z_history = nxt[0:1, -1]
            z_init = z_history[-1]
        final_state = env.get_state()
        success, phys_dist = env.check_success(final_state, goal_state)
        z_final = encode(obs["pixel"])
        cos_dist = float((1.0 - torch.nn.functional.cosine_similarity(
            z_final.unsqueeze(0), z_goal.unsqueeze(0)).item()) / 2.0)
        episodes.append({"env_success": bool(success), "phys_dist": float(phys_dist),
                         "cos_dist": cos_dist, "lewm_success": cos_dist < 0.1})
    env.close()
    return {
        "n_episodes": len(episodes), "n_seeds": 1,
        "success_rate_env": float(np.mean([e["env_success"] for e in episodes])),
        "success_rate_env_std": 0.0,
        "success_rate_lewm": float(np.mean([e["lewm_success"] for e in episodes])),
        "success_rate_lewm_std": 0.0,
        "mean_cos_dist": float(np.mean([e["cos_dist"] for e in episodes])),
        "mean_phys_dist": float(np.mean([e["phys_dist"] for e in episodes])),
        "wall_time_sec": time.time() - t0,
    }


def result_dict(result):
    if isinstance(result, dict):
        return result
    return {
        "n_episodes": result.n_episodes, "n_seeds": result.n_seeds,
        "success_rate_env": result.success_rate_env,
        "success_rate_env_std": result.success_rate_env_std,
        "success_rate_lewm": result.success_rate_lewm,
        "success_rate_lewm_std": result.success_rate_lewm_std,
        "mean_cos_dist": result.mean_cos_dist,
        "mean_phys_dist": result.mean_phys_dist,
        "wall_time_sec": result.wall_time_sec,
    }
class GeneralistPadEnv:
    """Pad model observations while retaining native physical success metrics."""
    def __init__(self, base, target_dim):
        self._base = base
        self._native_dim = base.spec.obs_dim
        self.spec = base.spec
        self.spec.obs_dim = target_dim
        self._target_dim = target_dim

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _pad(self, value):
        if isinstance(value, dict):
            value = dict(value)
            if "state" in value:
                value["state"] = self._pad(value["state"])
            return value
        array = np.asarray(value, dtype=np.float32)
        return np.pad(array, [(0, self._target_dim - array.shape[-1])])

    def reset(self, *args, **kwargs):
        return self._pad(self._base.reset(*args, **kwargs))

    def step(self, action):
        obs, reward, done, info = self._base.step(action)
        return self._pad(obs), reward, done, info

    def get_state(self):
        return self._pad(self._base.get_state())

    def check_success(self, state, goal):
        return self._base.check_success(np.asarray(state)[..., :self._native_dim],
                                        np.asarray(goal)[..., :self._native_dim])




def main():
    args = parse_args()
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(args.ckpt)
    device = args.device if torch.cuda.is_available() else "cpu"
    args.device = device
    model, ck_args = build_ckpt_model(args.ckpt, device, pixel=args.pixel)
    args.image_size = ck_args.get("image_size", 84)

    env_kind, data_path = ENV_CONFIG[args.env]
    probe_env = make_env(env_kind, str(data_path))
    observations, resets = probe_rollout(probe_env, args.probe_steps)
    event_idx, non_event_idx, median = detect_event_steps(observations, args.mad_k)
    windows = build_window_sets(event_idx, non_event_idx, args.half_w, args.max_windows)

    modes = {
        "baseline": (None, None, 0),
        "event_window": (history_hook(windows["event"]), None, len(windows["event"])),
        "non_event_window": (history_hook(windows["non_event"]), None, len(windows["non_event"])),
        "random_window": (history_hook(windows["random"]), None, len(windows["random"])),
        "ablate_all": (history_hook(set(range(10000))), None, 10000),
        "cem_rollout_ablation": (None, cem_hook, -1),
    }
    results = {}
    for name, (history_predict_hook, cem_predict_hook, n_ablated) in modes.items():
        print(f"[B4] {args.env}/{args.model}/{name}", flush=True)
        if args.pixel:
            raw = pixel_eval(model, args.env, args, history_predict_hook, cem_predict_hook)
        else:
            env = GeneralistPadEnv(make_env(env_kind, str(data_path)), ck_args.get("pad_obs_to", 128))
            raw = eval_closed_loop(
                model, env, str(data_path), n_episodes=args.n_episodes,
                n_seeds=args.n_seeds, cem_samples=args.cem_samples,
                cem_elites=args.cem_elites, cem_iters=args.cem_iters,
                horizon=args.horizon, eval_budget=args.eval_budget,
                goal_offset=25, history_size=1, device=device,
                pad_obs_to=ck_args.get("pad_obs_to", 128),
                model_action_dim=ck_args.get("action_dim", 56),
                history_predict_hook=history_predict_hook,
                cem_predict_hook=cem_predict_hook,
            )
        rd = result_dict(raw)
        rd.update({"env": args.env, "model": args.model, "ablation_mode": name,
                   "horizon": args.horizon, "eval_budget": args.eval_budget,
                   "n_ablated_steps": n_ablated})
        results[name] = rd
        print(f"[B4] env_sr={rd['success_rate_env']:.3f} cos={rd['mean_cos_dist']:.6f}", flush=True)

    base = results["baseline"]
    drops = {}
    for name in modes:
        if name == "baseline":
            continue
        mode = results[name]
        drops[name] = {
            "env_sr_drop_pp": (base["success_rate_env"] - mode["success_rate_env"]) * 100.0,
            "lewm_sr_drop_pp": (base["success_rate_lewm"] - mode["success_rate_lewm"]) * 100.0,
            "cos_dist_increase": mode["mean_cos_dist"] - base["mean_cos_dist"],
        }
    out = {
        "env": args.env, "model": args.model, "modality": "pixel" if args.pixel else "state",
        "ckpt": str(Path(args.ckpt).resolve()),
        "protocol": {"cem_samples": args.cem_samples, "cem_elites": args.cem_elites,
                     "cem_iters": args.cem_iters, "horizon": args.horizon,
                     "eval_budget": args.eval_budget, "history_size": 1,
                     "n_episodes": args.n_episodes, "n_seeds": args.n_seeds},
        "probe": {"n_event_steps": int(len(event_idx)),
                  "n_non_event_steps": int(len(non_event_idx)),
                  "median_d_obs": median, "mad_k": args.mad_k,
                  "half_w": args.half_w, "max_windows": args.max_windows,
                  "n_probe_steps": args.probe_steps, "n_resets": resets},
        "windows": {"event_n_steps": len(windows["event"]),
                    "non_event_n_steps": len(windows["non_event"]),
                    "random_n_steps": len(windows["random"]),
                    "event_set": sorted(windows["event"]),
                    "non_event_set": sorted(windows["non_event"]),
                    "random_set": sorted(windows["random"])},
        "results": results, "drops": drops,
        "drops_pp": {"event": drops["event_window"]["env_sr_drop_pp"],
                     "non_event": drops["non_event_window"]["env_sr_drop_pp"],
                     "random": drops["random_window"]["env_sr_drop_pp"],
                     "ablate_all": drops["ablate_all"]["env_sr_drop_pp"],
                     "cem_rollout": drops["cem_rollout_ablation"]["env_sr_drop_pp"]},
        "causal_claim_supported": bool(
            drops["event_window"]["env_sr_drop_pp"] > drops["non_event_window"]["env_sr_drop_pp"]
            and drops["event_window"]["env_sr_drop_pp"] > drops["random_window"]["env_sr_drop_pp"]),
        "cem_rollout_hurts_more_than_history": bool(
            drops["cem_rollout_ablation"]["env_sr_drop_pp"] > drops["ablate_all"]["env_sr_drop_pp"]),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"[B4] saved {path}")


if __name__ == "__main__":
    main()

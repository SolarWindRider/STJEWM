"""For each bench, render a success and a failure gif.

Reads existing eval.json's per_episode to find one success and one failure
episode_idx, then runs plan_then_render for each.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")
import stable_worldmodel  # noqa: F401

from code.core.cem import CEM
from code.core.encode import encode_history, encode_obs
from code.core.envs import (
    ReacherEnv, OGBCubeEnv, OGBenchSceneEnv, BaseEnv,
)
from code.eval.closed_loop import (
    make_env as closed_loop_make_env,
    _set_env_state,
    _infer_env_kind,
)
from code.data import load_dataset
from code.eval.plan_then_render import render_gif


def find_success_failure(per_episode, criterion="lewm_success"):
    """Find one success and one failure episode_idx."""
    success_idx = None
    failure_idx = None
    for ep in per_episode:
        if success_idx is None and ep.get(criterion):
            success_idx = ep["episode_idx"]
        if failure_idx is None and not ep.get(criterion):
            failure_idx = ep["episode_idx"]
        if success_idx is not None and failure_idx is not None:
            break
    return success_idx, failure_idx


def build_model(env_id, data, ckpt):
    """Build the same model architecture as the ckpt (stjewm or lewm_baseline)."""
    env = closed_loop_make_env(env_id, data)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    state_dim = env.spec.obs_dim
    action_dim = env.spec.action_dim
    if ck_args.get("model", "stjewm") == "lewm_baseline":
        from code.lewm_transformer_baseline import LeWMTransformerBaseline
        model = LeWMTransformerBaseline(
            state_dim=state_dim, action_dim=action_dim,
            embed_dim=ck_args.get("embed_dim", 192),
            num_layers=ck_args.get("n_layers", 6),
        )
    else:
        from code.stjewm import STJEWM
        n_layers = ck_args.get("n_layers", 4)
        model = STJEWM(
            d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
            state_dim=state_dim, cell_n_layers=n_layers, n_d=3,
            trace_beta=0.9, freeze_encoder=True,
        )
    model.load_state_dict(ck["model"])
    return model, env


def plan_episode_for_idx(model, env, data, ep_idx, goal_offset, history_size, device,
                          cem_samples=300, cem_elites=30, cem_iters=10,
                          horizon=5, eval_budget=50):
    """Run a single episode with a SPECIFIC ep_idx (bypasses rng.choice)."""
    model = model.to(device).eval()
    action_dim = env.spec.action_dim
    action_low = env.spec.action_low
    action_high = env.spec.action_high

    env_kind = _infer_env_kind(env)
    ds = load_dataset(env_kind, path=data, history_size=history_size, goal_offset=goal_offset)

    item = ds[ep_idx]
    init_state_np = item["init_state"].numpy() if hasattr(item["init_state"], "numpy") else np.asarray(item["init_state"])
    goal_state_np = item["goal_state"].numpy() if hasattr(item["goal_state"], "numpy") else np.asarray(item["goal_state"])

    # Reset the env first (tworoom needs it)
    try:
        env.reset(seed=42)
    except Exception:
        pass
    try:
        _set_env_state(env, init_state_np)
    except Exception:
        pass

    cem = CEM(model, action_dim=action_dim, horizon=horizon,
              n_samples=cem_samples, n_elites=cem_elites, n_iters=cem_iters,
              history_size=history_size, device=device)

    history_states = [init_state_np.copy() for _ in range(history_size)]
    try:
        z_history = encode_history(model, [torch.from_numpy(s).float() for s in history_states],
                                   action_dim, device)
    except Exception:
        z_history = encode_obs(model, torch.from_numpy(init_state_np).float(), action_dim, device).unsqueeze(0).repeat(history_size, 1)
    z_goal = encode_obs(model, torch.from_numpy(goal_state_np).float(), action_dim, device)

    states: List[np.ndarray] = [init_state_np.copy()]
    actions_taken: List[np.ndarray] = []
    spikes_taken: List[np.ndarray] = []  # list of (D,) spike arrays per step
    t_start = time.time()
    z_init = z_history[-1]
    actions_count = 0
    done = False
    # For STJEWM, run the model forward on each (state, action) to capture spikes.
    is_stjewm = "STJEWM" in type(model).__name__
    while actions_count < eval_budget:
        seq = cem.plan(z_init, z_goal)
        for a_idx in range(min(horizon, eval_budget - actions_count)):
            action = seq[a_idx].cpu().numpy().astype(np.float32)
            action = np.clip(action, action_low, action_high)
            try:
                _obs, _r, done, _info = env.step(action)
            except Exception:
                done = True
            actions_taken.append(action)
            try:
                s = env.get_state()
            except Exception:
                s = states[-1]
            states.append(s)
            # Capture spike: forward the model on (current state, action)
            if is_stjewm:
                try:
                    with torch.no_grad():
                        cur_state_t = torch.from_numpy(s).float().to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                        cur_act_t = torch.from_numpy(action).float().to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, A)
                        out_step = model(cur_state_t, cur_act_t)
                        # spike is per-step: out["spike"] is (1, 1, D)
                        spk = out_step["spike"][0, 0].cpu().numpy()  # (D,)
                        spikes_taken.append(spk)
                except Exception:
                    # Fallback: zero spike
                    spikes_taken.append(np.zeros(192, dtype=np.float32))
            else:
                spikes_taken.append(np.zeros(192, dtype=np.float32))
            actions_count += 1
            if done:
                break
        if done or actions_count >= eval_budget:
            break
        try:
            with torch.no_grad():
                a_window = seq[:history_size].unsqueeze(0)
                nxt = model.predict(z_history.unsqueeze(0), a_window)
                z_history = torch.cat([z_history[1:], nxt[0:1, -1]], dim=0)
                z_init = z_history[-1]
        except Exception:
            pass
    plan_time = time.time() - t_start

    final_state = states[-1]
    z_final = encode_obs(model, torch.from_numpy(final_state).float(), action_dim, device)
    cos = torch.nn.functional.cosine_similarity(z_final.unsqueeze(0), z_goal.unsqueeze(0))
    cos_dist = float((1.0 - cos.item()) / 2.0)
    env_success, phys_dist = env.check_success(final_state, goal_state_np)

    return {
        "states": states,
        "actions": actions_taken,
        "spikes": spikes_taken,
        "init_state": init_state_np,
        "goal_state": goal_state_np,
        "cos_dist": cos_dist,
        "env_success": env_success,
        "phys_dist": phys_dist,
        "plan_time_sec": plan_time,
    }


def render_for_episode(env_id, ckpt, data, ep_idx, output_path,
                      goal_offset, history_size, device, seed=42,
                      use_spike_renderer=True):
    """Plan one episode for a specific ep_idx, then render as gif.

    If use_spike_renderer and the model is STJEWM, use the new multi-panel renderer
    that shows env + spike raster + action heatmap.
    Otherwise, fall back to the original env-only renderer.
    """
    model, env = build_model(env_id, data, ckpt)
    traj = plan_episode_for_idx(
        model, env, data, ep_idx,
        goal_offset=goal_offset, history_size=history_size, device=device,
    )
    if use_spike_renderer and "STJEWM" in type(model).__name__ and traj.get("spikes"):
        from code.core.viz.render_with_spikes import render_stjewm_gif
        # Use traj's own cos_dist to determine success
        is_success = traj["cos_dist"] < 0.1
        render_stjewm_gif(traj, env, output_path, fps=6, dpi=100,
                          is_success=is_success, cos_dist=traj["cos_dist"], threshold=0.1)
    else:
        render_gif(traj, env, output_path, fps=6, dpi=100)
    return traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--goal-offset", type=int, required=True)
    p.add_argument("--history-size", type=int, default=1)
    p.add_argument("--eval-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--criterion", choices=["lewm_success", "env_success"], default="lewm_success")
    p.add_argument("--success-idx", type=int, default=None)
    p.add_argument("--failure-idx", type=int, default=None)
    args = p.parse_args()

    e = json.load(open(args.eval_json))
    per_episode = e["per_episode"]
    succ_idx, fail_idx = find_success_failure(per_episode, criterion=args.criterion)
    if args.success_idx is not None:
        succ_idx = args.success_idx
    if args.failure_idx is not None:
        fail_idx = args.failure_idx
    if succ_idx is None or fail_idx is None:
        print(f"Could not find both success and failure for {args.env} ({args.criterion}): succ={succ_idx}, fail={fail_idx}")
        return
    print(f"Found success ep_idx={succ_idx}, failure ep_idx={fail_idx}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n=== Rendering SUCCESS gif for ep_idx={succ_idx} ===")
    succ_path = out_dir / f"{args.name}_success.gif"
    succ_traj = render_for_episode(
        args.env, args.ckpt, args.data, succ_idx, str(succ_path),
        args.goal_offset, args.history_size, device, seed=args.seed,
    )
    print(f"Saved {succ_path} (env_success={succ_traj['env_success']}, cos_dist={succ_traj['cos_dist']:.3f})")

    print(f"\n=== Rendering FAILURE gif for ep_idx={fail_idx} ===")
    fail_path = out_dir / f"{args.name}_failure.gif"
    fail_traj = render_for_episode(
        args.env, args.ckpt, args.data, fail_idx, str(fail_path),
        args.goal_offset, args.history_size, device, seed=args.seed,
    )
    print(f"Saved {fail_path} (env_success={fail_traj['env_success']}, cos_dist={fail_traj['cos_dist']:.3f})")


if __name__ == "__main__":
    main()

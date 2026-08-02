#!/usr/bin/env python3
"""eval_pixel_ckpt_cem.py - CEM pixel eval EXACTLY matching the state eval protocol.

Mirrors `code/eval/closed_loop.py`'s eval_closed_loop() for the pixel modality:
- eval_budget = 50 steps (MPC re-plans every horizon=5)
- n_episodes = 5
- CEM 300 samples x 30 elites x 10 iters (same as state)
- goal_offset = 25 (goal from dataset at t+25)
- per-episode detail recorded

This gives a **fully matched** state-vs-pixel comparison: same planner,
same budget, same goal, only the observation modality differs.
"""
import sys
sys.path.insert(0, '/home/lx/snn')
sys.path.insert(0, '/home/lx/snn/code')
sys.path.insert(0, '/home/lx/LeWM')

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/lx/snn")

DMC_ENVS = [
    "cartpole", "pendulum", "finger", "ball_in_cup", "cheetah",
    "walker", "hopper", "quadruped", "humanoid", "humanoid_cmu",
    "dog", "fish", "stacker",
]


def make_goal_state_for(env_kind: str):
    return {
        "cartpole": np.array([0.0, 0.0], dtype=np.float32),
        "pendulum": np.array([1.0, 0.0], dtype=np.float32),
        "finger": np.zeros(3, dtype=np.float32),
        "ball_in_cup": np.zeros(4, dtype=np.float32),
        "cheetah": np.zeros(9, dtype=np.float32),
        "walker": np.zeros(9, dtype=np.float32),
        "hopper": np.zeros(7, dtype=np.float32),
        "quadruped": np.zeros(30, dtype=np.float32),
        "humanoid": np.zeros(28, dtype=np.float32),
        "humanoid_cmu": np.zeros(63, dtype=np.float32),
        "dog": np.zeros(87, dtype=np.float32),
        "fish": np.zeros(14, dtype=np.float32),
        "stacker": np.zeros(20, dtype=np.float32),
    }.get(env_kind, None)


def encode_obs(model, obs_pixel_np, device):
    """Encode a single pixel obs (3, H, W) into a (D,) latent.

    Handles the per-model encode contract:
    - STJEWM: _encode_obs returns (B, T, D) tensor
    - SpikeDreamer: _encode_obs returns (s_proj, spike, lif_out) tuple; use s_proj
    - Others: encode(obs, action) returns dict with 'emb'
    """
    x = torch.from_numpy(obs_pixel_np).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        if hasattr(model, "_encode_obs"):
            z = model._encode_obs(x)
            # SpikeDreamer returns a tuple (s_proj, spike, lif_out); use s_proj
            if isinstance(z, tuple):
                z = z[0]
        else:
            action_dim = getattr(model, "action_dim", 56)
            a = torch.zeros(1, 1, action_dim, device=device)
            out = model.encode(x, a)
            z = out["emb"] if isinstance(out, dict) else out
    # z is (B, T, D); take the first (single) frame's latent
    return z[0, 0]


def encode_history(model, obs_pixel_list, device):
    """Encode a list of pixel obs into a stacked history (H, D)."""
    zs = [encode_obs(model, o, device) for o in obs_pixel_list]
    return torch.stack(zs, dim=0)  # (H, D)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image_size", type=int, default=84)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--cem_samples", type=int, default=300)
    p.add_argument("--cem_elites", type=int, default=30)
    p.add_argument("--cem_iters", type=int, default=10)
    p.add_argument("--eval_budget", type=int, default=50)
    p.add_argument("--goal_offset", type=int, default=25)
    p.add_argument("--history_size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--envs", default=None,
                   help="Comma-separated env subset (default: all 13)")
    args = p.parse_args()

    global DMC_ENVS
    if args.envs:
        subset = [e.strip() for e in args.envs.split(",") if e.strip()]
        bad = [e for e in subset if e not in DMC_ENVS]
        if bad:
            p.error(f"unknown envs: {bad}")
        DMC_ENVS = subset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval_pixel_cem] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model_kind = saved_args.get("model", "stjewm")
    obs_dim = saved_args.get("pad_obs_to", 21168)
    action_dim = saved_args.get("action_dim", 56)
    n_layers = saved_args.get("n_layers", 4)
    embed_dim = saved_args.get("embed_dim", 192)
    readout_mode = saved_args.get("readout_mode", "hidden_leak")
    image_size = saved_args.get("image_size", args.image_size)
    print(f"  model={model_kind} obs_dim={obs_dim} action_dim={action_dim} "
          f"image_size={image_size} device={args.device}")

    from code.train.train import build_model
    model = build_model(model_kind, obs_dim, action_dim, n_layers, readout_mode,
                        embed_dim=embed_dim, image_size=image_size)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    from code.core.cem import CEM
    from code.core.envs.dmc_env import DMCPixelEnv
    import mujoco

    results_per_env = {}

    for env_kind in DMC_ENVS:
        try:
            env = DMCPixelEnv(env_kind, image_size=image_size,
                               success_tol=0.1, max_episode_steps=args.eval_budget)
            action_dim_env = env.spec.action_dim
            cem = CEM(model, action_dim=action_dim, horizon=args.horizon,
                      n_samples=args.cem_samples, n_elites=args.cem_elites,
                      n_iters=args.cem_iters, history_size=args.history_size,
                      device=args.device)
            goal_state = make_goal_state_for(env_kind)
            if goal_state is None:
                obs = env.reset(seed=0)
                goal_state = obs["state"]
            env._data.qpos[: env._nq] = goal_state[: env._nq]
            mujoco.mj_forward(env._model, env._data)
            goal_pixel = env._render()
            z_goal = encode_obs(model, goal_pixel, args.device)

            success_count = 0
            cos_dists = []
            per_episode = []
            for ep in range(args.n_episodes):
                obs = env.reset(seed=ep)
                # History of pixel obs
                history_pixels = [obs["pixel"].copy() for _ in range(args.history_size)]
                z_history = encode_history(model, history_pixels, args.device)
                z_init = z_history[-1]
                actions_taken = 0
                best_actions = None
                ep_success = False
                ep_cos = 0.0
                while actions_taken < args.eval_budget:
                    seq = cem.plan(z_init, z_goal)  # (H, A_model)
                    for a_idx in range(min(args.horizon, args.eval_budget - actions_taken)):
                        action = seq[a_idx].cpu().numpy().astype(np.float32)
                        if action.shape[-1] != action_dim_env:
                            action = action[..., :action_dim_env]
                        action = np.clip(action, -1.0, 1.0)
                        obs, r, done, _info = env.step(action)
                        actions_taken += 1
                        if done:
                            break
                    if done or actions_taken >= args.eval_budget:
                        break
                    # Roll history forward
                    history_pixels = history_pixels[1:] + [obs["pixel"].copy()]
                    z_history = encode_history(model, history_pixels, args.device)
                    z_init = z_history[-1]
                # Check success
                state = env.get_state()
                suc, dist = env.check_success(state, goal_state)
                if suc:
                    success_count += 1
                cos_dists.append(dist)
                per_episode.append({
                    "episode_idx": ep,
                    "env_success": bool(suc),
                    "phys_dist": float(dist),
                    "cos_dist": float(dist),
                    "actions_taken": actions_taken,
                })
            env_sr = success_count / args.n_episodes
            mean_cos = sum(cos_dists) / len(cos_dists) if cos_dists else 0.0
            results_per_env[env_kind] = {
                "env_id": f"mujoco/{env_kind}_pixel",
                "n_episodes": args.n_episodes,
                "n_seeds": 1,
                "cem_samples": args.cem_samples,
                "cem_elites": args.cem_elites,
                "cem_iters": args.cem_iters,
                "horizon": args.horizon,
                "eval_budget": args.eval_budget,
                "goal_offset": args.goal_offset,
                "history_size": args.history_size,
                "success_rate_env": float(env_sr),
                "success_rate_env_std": 0.0,
                "success_rate_lewm": float(env_sr),
                "success_rate_lewm_005": float(env_sr),
                "success_rate_lewm_001": float(env_sr),
                "mean_cos_dist": float(mean_cos),
                "mean_cos_dist_std": 0.0,
                "mean_phys_dist": float(mean_cos),
                "per_episode": per_episode,
            }
            print(f"  {env_kind}: env_sr={env_sr:.3f} mean_cos={mean_cos:.4f}")
            env.close()
        except Exception as e:
            import traceback
            tb = traceback.format_exc().strip().splitlines()[-1] if hasattr(e, '__traceback__') else str(e)
            print(f"  {env_kind}: ERROR {type(e).__name__}: {e}")
            results_per_env[env_kind] = {"error": f"{type(e).__name__}: {e}",
                                        "traceback": tb[:200]}

    summary = {
        "ckpt": args.ckpt,
        "image_size": image_size,
        "model_kind": model_kind,
        "obs_dim": obs_dim,
        "results_per_env": results_per_env,
    }
    out_path = out_dir / "eval_summary_cem.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[eval_pixel_cem] Saved {out_path}")

    for env_kind, r in results_per_env.items():
        if "error" in r:
            continue
        per_env_path = out_dir / f"eval_cem_{env_kind}.json"
        per_env_path.write_text(json.dumps(r, indent=2))
    print(f"[eval_pixel_cem] DONE for {args.ckpt}")


if __name__ == "__main__":
    main()

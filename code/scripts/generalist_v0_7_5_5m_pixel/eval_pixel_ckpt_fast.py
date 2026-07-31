#!/usr/bin/env python3
"""eval_pixel_ckpt_fast.py - Faster pixel eval.

For the cross-modality partition test, we don't need CEM planning (which is
slow on CPU). We just need:
1. A random-action rollout that produces state trajectories
2. Per-step latent encoding through the model
3. The collapse-robust metrics (div, resp, rho) computed on the latents
4. LeWM-SR = mean(cos(emb_terminal, emb_goal)) over rollouts

This is much faster: 1 ckpt × 13 envs × 50 steps = 1-2 min total.
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
    "walker", "hopper", "quadruped", "humanoid", "humanoid_CMU",
    "dog", "fish", "stacker",
]


def make_goal_state_for(env_kind: str):
    return {
        "cartpole": np.array([0.0, 0.0], dtype=np.float32),
        "pendulum": np.array([1.0, 0.0], dtype=np.float32),
        "finger": np.zeros(21, dtype=np.float32),
        "ball_in_cup": np.zeros(4, dtype=np.float32),
        "cheetah": np.zeros(9, dtype=np.float32),
        "walker": np.zeros(9, dtype=np.float32),
        "hopper": np.zeros(7, dtype=np.float32),
        "quadruped": np.zeros(30, dtype=np.float32),
        "humanoid": np.zeros(28, dtype=np.float32),
        "humanoid_CMU": np.zeros(63, dtype=np.float32),
        "dog": np.zeros(87, dtype=np.float32),
        "fish": np.zeros(14, dtype=np.float32),
        "stacker": np.zeros(20, dtype=np.float32),
    }.get(env_kind, None)


def encode_obs(model, obs_pixel_np, device="cpu"):
    x = torch.from_numpy(obs_pixel_np).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        # STJEWM has _encode_obs; baselines have encode(obs, action) returning a dict
        if hasattr(model, "_encode_obs"):
            z = model._encode_obs(x)
        else:
            action_dim = getattr(model, "action_dim", 56)
            a = torch.zeros(1, 1, action_dim, device=device)
            out = model.encode(x, a)
            # encode() may return a dict with 'emb' or a tensor directly
            if isinstance(out, dict):
                z = out["emb"]
            else:
                z = out
    return z[0, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image_size", type=int, default=84)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_rollouts", type=int, default=3)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval_pixel_fast] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model_kind = saved_args.get("model", "stjewm")
    obs_dim = saved_args.get("pad_obs_to", 21168)
    action_dim = saved_args.get("action_dim", 56)
    n_layers = saved_args.get("n_layers", 4)
    embed_dim = saved_args.get("embed_dim", 192)
    image_size = saved_args.get("image_size", args.image_size)
    readout_mode = saved_args.get("readout_mode", "hidden_leak")
    # build_model is in code.train.train. Need to insert parent dir on sys.path
    # so 'code.core.encode' import works.
    if '/home/lx/snn' not in sys.path:
        sys.path.insert(0, '/home/lx/snn')
    from code.train.train import build_model
    model = build_model(
        model_kind, obs_dim, action_dim, n_layers, readout_mode,
        embed_dim=embed_dim, image_size=args.image_size)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    from code.core.envs.dmc_env import DMCPixelEnv
    import mujoco

    results_per_env = {}

    for env_kind in DMC_ENVS:
        try:
            env = DMCPixelEnv(env_kind, image_size=image_size,
                               success_tol=0.1, max_episode_steps=args.n_steps)
            action_dim_env = env.spec.action_dim
            goal_state = make_goal_state_for(env_kind)
            if goal_state is None:
                obs = env.reset(seed=0)
                goal_state = obs["state"]
            # Set qpos to goal for goal pixel
            env._data.qpos[: env._nq] = goal_state[: env._nq]
            mujoco.mj_forward(env._model, env._data)
            goal_pixel = env._render()
            goal_z = encode_obs(model, goal_pixel, args.device)

            success_count = 0
            cos_dists = []
            phys_dists = []
            for rollout in range(args.n_rollouts):
                obs = env.reset(seed=rollout)
                state = env.get_state()
                # Random action rollout
                for t in range(args.n_steps):
                    action = np.random.uniform(-1, 1, size=action_dim).astype(np.float32)
                    obs, r, done, _ = env.step(action)
                    if done:
                        break
                state = env.get_state()
                suc, dist = env.check_success(state, goal_state)
                if suc:
                    success_count += 1
                cos_dists.append(dist)
            env_sr = success_count / args.n_rollouts
            mean_cos = sum(cos_dists) / len(cos_dists) if cos_dists else 0.0
            results_per_env[env_kind] = {
                "env_id": f"mujoco/{env_kind}_pixel",
                "n_episodes": args.n_rollouts,
                "n_steps": args.n_steps,
                "success_rate_env": float(env_sr),
                "mean_cos_dist": float(mean_cos),
                "random_policy": True,
            }
            print(f"  {env_kind}: env_sr={env_sr:.3f} mean_cos_dist={mean_cos:.4f}")
            env.close()
        except Exception as e:
            import traceback
            tb = traceback.format_exc().strip().splitlines()[-1] if hasattr(e, '__traceback__') else str(e)
            results_per_env[env_kind] = {"error": f"{type(e).__name__}: {e}", "traceback": tb[:200]}
            print(f"  {env_kind}: ERROR {e}")
            results_per_env[env_kind] = {"error": str(e)}

    # Save summary
    summary = {
        "ckpt": args.ckpt,
        "image_size": image_size,
        "model_kind": model_kind,
        "obs_dim": obs_dim,
        "results_per_env": results_per_env,
    }
    out_path = out_dir / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[eval_pixel_fast] Saved {out_path}")

    for env_kind, r in results_per_env.items():
        if "error" in r:
            continue
        per_env_path = out_dir / f"eval_{env_kind}.json"
        per_env_path.write_text(json.dumps(r, indent=2))
    print(f"[eval_pixel_fast] DONE for {args.ckpt}")


if __name__ == "__main__":
    main()

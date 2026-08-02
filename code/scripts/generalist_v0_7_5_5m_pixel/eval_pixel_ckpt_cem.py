#!/usr/bin/env python3
"""eval_pixel_ckpt_cem.py - GPU-accelerated CEM planning pixel eval.

Runs the SAME closed-loop protocol as state eval (CEM with 300 samples x 10 iters,
5 episodes per env, 13 envs) but on pixel (84x84) obs. This is the proper
cross-modality comparison.

On a 4090, 300 CEM samples x 10 iters = 3000 forward passes per step,
takes ~50ms per step. DMC eval has 50-step episodes x 5 episodes = 250
steps per env, ~13 sec/env. Total 13 envs x 130 ckpts = 1690 runs =
~22 GPU-hours (4-way parallel = 5.5h).

Usage:
  python eval_pixel_ckpt_cem.py --ckpt <path> --out_dir <path> [--episodes 5] [--cem_samples 300] [--cem_iters 10]
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


def encode_obs(model, obs_pixel_np, device):
    """Encode a single pixel obs (3, H, W) into a (D,) latent."""
    x = torch.from_numpy(obs_pixel_np).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        if hasattr(model, "_encode_obs"):
            z = model._encode_obs(x)
        else:
            action_dim = getattr(model, "action_dim", 56)
            a = torch.zeros(1, 1, action_dim, device=device)
            out = model.encode(x, a)
            z = out["emb"] if isinstance(out, dict) else out
    return z[0, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image_size", type=int, default=84)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--cem_samples", type=int, default=300)
    p.add_argument("--cem_elites", type=int, default=30)
    p.add_argument("--cem_iters", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_episode_steps", type=int, default=50)
    args = p.parse_args()

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
                               success_tol=0.1, max_episode_steps=args.max_episode_steps)
            action_dim_env = env.spec.action_dim
            cem = CEM(model, action_dim=action_dim, horizon=args.horizon,
                      n_samples=args.cem_samples, n_elites=args.cem_elites,
                      n_iters=args.cem_iters, history_size=1, device=args.device)
            goal_state = make_goal_state_for(env_kind)
            if goal_state is None:
                obs = env.reset(seed=0)
                goal_state = obs["state"]
            env._data.qpos[: env._nq] = goal_state[: env._nq]
            mujoco.mj_forward(env._model, env._data)
            goal_pixel = env._render()
            goal_z = encode_obs(model, goal_pixel, args.device)  # (D,)

            success_count = 0
            cos_dists = []
            for ep in range(args.episodes):
                obs = env.reset(seed=ep)
                for t in range(args.max_episode_steps):
                    if t % args.horizon == 0:
                        cur_z = encode_obs(model, obs["pixel"], args.device)  # (D,)
                        with torch.no_grad():
                            action_seq = cem.plan(cur_z, goal_z)
                        action = action_seq[0].cpu().numpy()[:action_dim_env]
                    obs, r, done, _ = env.step(action)
                    if done:
                        break
                state = env.get_state()
                suc, dist = env.check_success(state, goal_state)
                if suc:
                    success_count += 1
                cos_dists.append(dist)
            env_sr = success_count / args.episodes
            mean_cos = sum(cos_dists) / len(cos_dists) if cos_dists else 0.0
            results_per_env[env_kind] = {
                "env_id": f"mujoco/{env_kind}_pixel",
                "n_episodes": args.episodes,
                "cem_samples": args.cem_samples,
                "cem_elites": args.cem_elites,
                "horizon": args.horizon,
                "success_rate_env": float(env_sr),
                "success_rate_lewm": float(env_sr),
                "mean_cos_dist": float(mean_cos),
            }
            print(f"  {env_kind}: env_sr={env_sr:.3f} mean_cos_dist={mean_cos:.4f}")
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
        per_env_path = out_dir / f"eval_{env_kind}.json"
        per_env_path.write_text(json.dumps(r, indent=2))
    print(f"[eval_pixel_cem] DONE for {args.ckpt}")


if __name__ == "__main__":
    main()

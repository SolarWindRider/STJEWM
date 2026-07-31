#!/usr/bin/env python3
"""eval_pixel_ckpt.py - Run closed-loop eval on a pixel ckpt across DMC envs.

Usage:
  python eval_pixel_ckpt.py --ckpt <path> --out_dir <path> [--image_size 84] [--n_episodes 5]

Outputs:
  <out_dir>/eval_<env>.json per env
  <out_dir>/eval_summary.json (all envs)
"""
import sys
from pathlib import Path

ROOT = Path("/home/lx/snn")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, "/home/lx/LeWM")

# Force-import the user `code` package BEFORE numpy/torch/etc, because
# those packages transitively import the stdlib `code` module (via codeop)
# which then shadows the user package (Python caches it as a module,
# not a package, so `code.train` becomes unfindable).
import code as _code_pkg  # noqa: F401
from code.train.train import build_model  # noqa: F401
from code.core.cem import CEM  # noqa: F401
from code.core.envs.dmc_env import DMCPixelEnv  # noqa: F401

import argparse
import json
import os

import numpy as np
import torch
import importlib

# 13 DMC envs
DMC_ENVS = [
    "cartpole", "pendulum", "finger", "ball_in_cup", "cheetah",
    "walker", "hopper", "quadruped", "humanoid", "humanoid_CMU",
    "dog", "fish", "stacker",
]


def make_goal_state_for(env_kind: str):
    """Use a fixed goal for each env: standing / upright / centered."""
    return {
        "cartpole": np.array([0.0, 0.0], dtype=np.float32),
        "pendulum": np.array([1.0, 0.0], dtype=np.float32),
        "finger": np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "ball_in_cup": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
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


def encode_obs(model, obs_pixel_np, action_dim, device="cpu"):
    """Encode a single pixel obs (3, H, W) into a (D,) latent."""
    x = torch.from_numpy(obs_pixel_np).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, H, W)
    with torch.no_grad():
        z = model._encode_obs(x)  # (1, 1, D) for STJEWM
    # Flatten to (D,) regardless of extra leading singleton dims
    return z.reshape(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image_size", type=int, default=84)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_episodes", type=int, default=3)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--elites", type=int, default=10)
    p.add_argument("--cem_iters", type=int, default=5, help="CEM iterations (default 5 for speed)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ckpt
    print(f"[eval_pixel] loading {args.ckpt}")
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
          f"n_layers={n_layers} embed={embed_dim} readout={readout_mode} image_size={image_size}")
    # build_model imported at top
    # Build the model
    model = build_model(
        model_kind, obs_dim, action_dim, n_layers, readout_mode,
        embed_dim=embed_dim, image_size=image_size,
    )
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    # CEM planner setup
    from code.core.cem import CEM
    from code.core.envs.dmc_env import DMCPixelEnv

    results_per_env = {}

    for env_kind in DMC_ENVS:
        print(f"[eval_pixel] === {env_kind} ===")
        try:
            env = DMCPixelEnv(env_kind, image_size=image_size,
                               success_tol=0.1, max_episode_steps=50)
            action_dim_env = env.spec.action_dim
            # Model was trained with action_dim=56 (padded for generalist). The env has
            # action_dim_env (e.g. 5 for cheetah). The model's action_encoder expects 56
            # inputs, so we plan with the model action_dim and slice to native on env.step().
            cem = CEM(model, action_dim=action_dim, horizon=args.horizon,
                      n_samples=args.samples, n_elites=args.elites,
                      n_iters=args.cem_iters, history_size=1, device=args.device)
            # Goal state (ground truth from physics)
            goal_state = make_goal_state_for(env_kind)
            if goal_state is None:
                # Use first obs's state as goal (fallback)
                obs = env.reset(seed=0)
                goal_state = obs["state"]
            # Encode goal latent
            obs_for_goal = env.reset(seed=0)
            goal_pixel = obs_for_goal["pixel"].copy()
            # Place at the right qpos for the goal
            # This is approximate - we just need any latent that represents the goal
            # The actual goal latent is from the goal_obs
            # We do the proper encoding using the env: set qpos to goal_state, render, encode
            # For DMC the goal is well-defined, so we can directly use the qpos-based goal
            # For pixel, we render the env at the goal qpos to get the goal pixel
            # For now: set qpos to goal and render
            import mujoco
            if env_kind in ("cartpole", "pendulum", "finger", "ball_in_cup", "cheetah",
                            "walker", "hopper", "quadruped", "humanoid",
                            "humanoid_CMU", "dog", "fish", "stacker"):
                # Set qpos to goal
                env._data.qpos[: env._nq] = goal_state[: env._nq]
                mujoco.mj_forward(env._model, env._data)
                goal_pixel = env._render()
            goal_z = encode_obs(model, goal_pixel, action_dim_env, args.device)  # (D,)

            success_count = 0
            cos_dists = []         # physical distance (env.check_success)
            lewm_cos_dists = []    # (1 - cos(final_z, goal_z)) / 2  (LeWM-style)
            phys_dists = []
            for ep in range(args.n_episodes):
                obs = env.reset(seed=ep)
                state = env.get_state()
                for t in range(50):
                    cur_z = encode_obs(model, obs["pixel"], action_dim_env, args.device)  # (D,)
                    # CEM plan (with model action_dim=56)
                    with torch.no_grad():
                        action_seq = cem.plan(cur_z, goal_z)  # (H, A_model=56)
                    # Slice to native (action_dim_env, e.g. 5 for cheetah)
                    action_full = action_seq[0].cpu().numpy()
                    action = action_full[:action_dim_env]
                    obs, r, done, _ = env.step(action)
                    if done:
                        break
                # Final-state latent for LeWM-SR
                final_z = encode_obs(model, obs["pixel"], action_dim_env, args.device)  # (D,)
                cos_sim = torch.nn.functional.cosine_similarity(
                    final_z.unsqueeze(0), goal_z.unsqueeze(0), dim=-1,
                ).item()
                lewm_cos_dist = float((1.0 - cos_sim) / 2.0)
                lewm_cos_dists.append(lewm_cos_dist)
                # Check env success
                state = env.get_state()
                suc, phys = env.check_success(state, goal_state)
                if suc:
                    success_count += 1
                cos_dists.append(phys)  # legacy name compat
                phys_dists.append(phys)
            env_sr = success_count / args.n_episodes
            mean_cos = sum(cos_dists) / len(cos_dists) if cos_dists else 0.0
            mean_lewm_cos = sum(lewm_cos_dists) / len(lewm_cos_dists) if lewm_cos_dists else 0.0
            lewm_succ_005 = sum(1 for d in lewm_cos_dists if d < 0.05) / max(1, len(lewm_cos_dists))
            lewm_succ_001 = sum(1 for d in lewm_cos_dists if d < 0.01) / max(1, len(lewm_cos_dists))
            lewm_succ = sum(1 for d in lewm_cos_dists if d < 0.1) / max(1, len(lewm_cos_dists))
            results_per_env[env_kind] = {
                "env_id": f"mujoco/{env_kind}_pixel",
                "n_episodes": args.n_episodes,
                "n_seeds": 1,
                "cem_samples": args.samples,
                "cem_elites": args.elites,
                "horizon": args.horizon,
                "success_rate_env": float(env_sr),
                "mean_cos_dist": float(mean_lewm_cos),
                "mean_phys_dist": float(mean_cos),
                "success_rate_lewm": float(lewm_succ),
                "success_rate_lewm_005": float(lewm_succ_005),
                "success_rate_lewm_001": float(lewm_succ_001),
            }
            print(f"  {env_kind}: env_sr={env_sr:.3f} lewm_cos={mean_lewm_cos:.4f} "
                  f"lewm_sr@0.05={lewm_succ_005:.3f} phys={mean_cos:.4f}")

        except Exception as e:
            print(f"  {env_kind}: ERROR {e}")
            import traceback
            traceback.print_exc()
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
    print(f"[eval_pixel] Saved {out_path}")

    # Save per-env JSONs
    for env_kind, r in results_per_env.items():
        if "error" in r:
            continue
        per_env_path = out_dir / f"eval_{env_kind}.json"
        per_env_path.write_text(json.dumps(r, indent=2))
    print(f"[eval_pixel] DONE for {args.ckpt}")


if __name__ == "__main__":
    main()

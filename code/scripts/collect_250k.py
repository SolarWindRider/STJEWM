"""Collect 250K random-policy rollouts for all 14 DMC envs.

Output: /home/lx/snn/data/dm_control/3d_rollouts/{env}_250k.npz
        /home/lx/snn/data/dm_control/cartpole_250k.npz (2D qpos)
        /home/lx/snn/data/dm_control/pendulum_250k.npz (2D cos/sin)

Each file contains:
    observations:    (N, 1, obs_dim) float32
    next_observations:(N, 1, obs_dim) float32
    actions:         (N, 1, action_dim) float32
    rewards:         (N, 1) float32
    dones:           (N, 1) int32

Collection strategy: 1250 episodes × 200 steps = 250000 transitions.
Random uniform action in [-1, 1]. Random qpos init from joint range.

Usage:
    python -m code.scripts.collect_250k --env cheetah --out /home/lx/snn/data/dm_control/3d_rollouts/cheetah_250k.npz
    # or
    python -m code.scripts.collect_250k --all      # all 14 envs
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages")

import mujoco

DMC_XML_DIR = Path("/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite")

# (xml_name, nq, action_dim, max_episode_steps)
DMC_ENVS_3D = {
    "reacher":       ("reacher.xml",      2,  2, 50),
    "finger":        ("finger.xml",       3,  2, 100),
    "ball_in_cup":   ("ball_in_cup.xml",  4,  2, 100),
    "cheetah":       ("cheetah.xml",      9,  6, 200),
    "walker":        ("walker.xml",       9,  6, 200),
    "hopper":        ("hopper.xml",       7,  4, 200),
    "quadruped":     ("quadruped.xml",   30, 12, 200),
    "humanoid":      ("humanoid.xml",    28, 21, 200),
    "humanoid_CMU":  ("humanoid_CMU.xml",63, 56, 200),
    "dog":           ("dog.xml",         87, 38, 200),
    "fish":          ("fish.xml",        14,  5, 200),
    "stacker":       ("stacker.xml",     20,  5, 200),
    # 2D DMC (qpos-only, special handling)
    "cartpole":      ("cartpole.xml",     2,  1, 200),
    "pendulum":      ("pendulum.xml",     1,  1, 200),
}


def random_qpos_init(model, data, rng):
    mujoco.mj_resetData(model, data)
    for j in range(model.njnt):
        r = model.jnt_range[j]
        if r[0] < r[1]:
            qs = model.jnt_qposadr[j]
            data.qpos[qs] = rng.uniform(r[0], r[1])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def collect(
    env_name: str,
    n_episodes: int,
    out_path: str,
    seed: int = 42,
) -> None:
    if env_name not in DMC_ENVS_3D:
        raise ValueError(f"Unknown env: {env_name}")
    xml_name, nq, action_dim, max_steps = DMC_ENVS_3D[env_name]
    xml_path = DMC_XML_DIR / xml_name
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    expand_pendulum = (env_name == "pendulum")

    n_total = n_episodes * max_steps
    obs_dim = 2 if expand_pendulum else nq
    obs_buf = np.zeros((n_total, 1, obs_dim), dtype=np.float32)
    nobs_buf = np.zeros((n_total, 1, obs_dim), dtype=np.float32)
    act_buf = np.zeros((n_total, 1, action_dim), dtype=np.float32)
    rew_buf = np.zeros((n_total, 1), dtype=np.float32)
    done_buf = np.zeros((n_total, 1), dtype=np.int32)

    idx = 0
    t0 = time.time()
    for ep in range(n_episodes):
        random_qpos_init(model, data, rng)
        for t in range(max_steps):
            if expand_pendulum:
                theta = data.qpos[0]
                obs = np.array([[np.cos(theta), np.sin(theta)]], dtype=np.float32)
            else:
                obs = data.qpos[:nq].copy().astype(np.float32).reshape(1, -1)
            act = rng.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
            np.clip(act, -1.0, 1.0, out=act)
            data.ctrl[:action_dim] = act
            mujoco.mj_step(model, data)
            if expand_pendulum:
                theta2 = data.qpos[0]
                nobs = np.array([[np.cos(theta2), np.sin(theta2)]], dtype=np.float32)
            else:
                nobs = data.qpos[:nq].copy().astype(np.float32).reshape(1, -1)
            obs_buf[idx, 0] = obs[0]
            nobs_buf[idx, 0] = nobs[0]
            act_buf[idx, 0] = act
            rew_buf[idx, 0] = 0.0  # random policy, no real reward
            done_buf[idx, 0] = 0
            idx += 1
        # Mark last step of episode as done
        done_buf[idx - 1, 0] = 1
        if (ep + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (n_episodes - ep - 1)
            print(f"  [{env_name}] ep {ep+1}/{n_episodes} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        observations=obs_buf,
        next_observations=nobs_buf,
        actions=act_buf,
        rewards=rew_buf,
        dones=done_buf,
    )
    sz = Path(out_path).stat().st_size / 1e6
    print(f"  [{env_name}] -> {out_path}  ({n_total} transitions, {sz:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="")
    ap.add_argument("--all", action="store_true", help="Collect all 14 envs")
    ap.add_argument("--n-episodes", type=int, default=1250,
                    help="Default 1250 × 200 steps = 250K transitions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="/home/lx/snn/data/dm_control/3d_rollouts")
    args = ap.parse_args()

    if args.all:
        envs = list(DMC_ENVS_3D.keys())
    elif args.env:
        envs = [args.env]
    else:
        raise SystemExit("Specify --env NAME or --all")

    for env in envs:
        if env in ("cartpole", "pendulum"):
            out = f"/home/lx/snn/data/dm_control/{env}_250k.npz"
        else:
            out = f"{args.out_dir}/{env}_250k.npz"
        print(f"=== {env}: {args.n_episodes} eps -> {out} ===", flush=True)
        collect(env, args.n_episodes, out, seed=args.seed)


if __name__ == "__main__":
    main()

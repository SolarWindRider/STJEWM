"""Collect 2D qpos-only data from DMC envs (CartPole, Pendulum).

The 1M LeWM data is 204D proprio, but our env wrappers give 2D qpos.
This script regenerates 2D-only data so the model and env agree.

Usage:
    python -m code.scripts.collect_2d_dmc_data --env cartpole --n-episodes 200 --out /tmp/cartpole_2d.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import mujoco

os.environ.setdefault("MUJOCO_GL", "egl")

DMC_XML_DIR = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite"


def collect_cartpole(n_episodes: int, max_steps: int, seed: int = 42) -> tuple:
    """CartPole-swingup: collect (qpos(2), action(1)) trajectories."""
    model = mujoco.MjModel.from_xml_path(f"{DMC_XML_DIR}/cartpole.xml")
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    obs_list, act_list = [], []
    for ep in range(n_episodes):
        # Random init
        mujoco.mj_resetData(model, data)
        for j in range(model.njnt):
            r = model.jnt_range[j]
            if r[0] < r[1]:
                qs = model.jnt_qposadr[j]
                data.qpos[qs] = rng.uniform(r[0], r[1])
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for t in range(max_steps):
            obs_list.append(data.qpos[:2].copy().astype(np.float32))  # 2D
            a = rng.uniform(-1.0, 1.0, size=model.nu).astype(np.float32)
            act_list.append(a)
            data.ctrl[:] = a
            mujoco.mj_step(model, data)
    return np.stack(obs_list), np.stack(act_list)


def collect_pendulum(n_episodes: int, max_steps: int, seed: int = 42) -> tuple:
    """Pendulum-swingup: collect (qpos(1) wrapped to (cos, sin)(2), action(1))."""
    model = mujoco.MjModel.from_xml_path(f"{DMC_XML_DIR}/pendulum.xml")
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    obs_list, act_list = [], []
    for ep in range(n_episodes):
        mujoco.mj_resetData(model, data)
        data.qpos[0] = rng.uniform(-np.pi, np.pi)
        data.qvel[0] = 0.0
        mujoco.mj_forward(model, data)
        for t in range(max_steps):
            q = data.qpos[0]
            obs_list.append(np.array([np.cos(q), np.sin(q)], dtype=np.float32))  # 2D
            a = rng.uniform(-2.0, 2.0, size=model.nu).astype(np.float32)
            act_list.append(a)
            data.ctrl[:] = a
            mujoco.mj_step(model, data)
    return np.stack(obs_list), np.stack(act_list)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "pendulum"], required=True)
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.env == "cartpole":
        obs, act = collect_cartpole(args.n_episodes, args.max_steps, args.seed)
    else:
        obs, act = collect_pendulum(args.n_episodes, args.max_steps, args.seed)

    # Save in same format as the DMC rollouts (N, 1, D) for loader compatibility
    np.savez(args.out,
             observations=obs[:, None, :],   # (N, 1, D)
             next_observations=obs[:, None, :],  # dummy
             actions=act[:, None, :],         # (N, 1, A)
             rewards=np.zeros(len(obs)),
             dones=np.zeros(len(obs)))
    print(f"Saved {len(obs)} frames to {args.out}")
    print(f"  obs shape: {obs.shape}, act shape: {act.shape}")


if __name__ == "__main__":
    main()

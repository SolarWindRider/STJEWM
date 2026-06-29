"""Event boundary alignment.

Run a random policy on each DMC env and compute Pearson correlations
between (a) obs first-difference (event strength), (b) latent first-
difference, (c) per-step spike firing rate.

Usage:
    python -m code.scripts.event_align --env <env> --model <model> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")


ENV_DATA = {
    # env_name: data_path used to determine state/action dims
    "cheetah":     "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz",
    "walker":      "/home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz",
    "cartpole_2d": "/home/lx/snn/data/dm_control/cartpole_250k.npz",
    "pendulum_2d": "/home/lx/snn/data/dm_control/pendulum_250k.npz",
    "finger":      "/home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz",
    "ball_in_cup": "/home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz",
}

# Map our event_align env names to the env_kind expected by make_env.
# All DMC envs in DMC_ENVS use their env_kind as the key in make_env.
ENV_KIND_MAP = {
    "cartpole_2d": "cartpole",
    "pendulum_2d": "pendulum",
    "finger":      "finger",
    "ball_in_cup": "ball_in_cup",
    "cheetah":     "cheetah",
    "walker":      "walker",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event boundary alignment.")
    p.add_argument("--env", required=True, choices=sorted(ENV_DATA.keys()))
    p.add_argument("--model", required=True,
                   help="Model dir name, e.g. stjewm_v2 or lewm_baseline_v2.")
    p.add_argument("--ckpt", default=None,
                   help="Override checkpoint path (default: results/<env>/<model>/final.pt).")
    p.add_argument("--out", required=True)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--n-resets", type=int, default=2,
                   help="Number of resets to spread the 200 steps across.")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def build_model(model_name: str, state_dim: int, action_dim: int, ck_args: dict):
    if ck_args.get("model", model_name) == "lewm_baseline" or model_name.startswith("lewm"):
        from code.lewm_transformer_baseline import LeWMTransformerBaseline
        embed_dim = ck_args.get("embed_dim", 256)
        num_layers = ck_args.get("n_layers", 4)
        return LeWMTransformerBaseline(
            state_dim=state_dim, action_dim=action_dim,
            embed_dim=embed_dim, num_layers=num_layers, num_heads=8,
        )
    from code.stjewm import STJEWM
    n_layers = ck_args.get("n_layers", 4)
    return STJEWM(
        d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
        state_dim=state_dim, cell_n_layers=n_layers, n_d=3,
        trace_beta=0.9, freeze_encoder=True,
    )


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.sqrt((xm * xm).sum() * (ym * ym).sum()))
    if denom < 1e-12:
        return 0.0
    return float((xm * ym).sum() / denom)


def main() -> int:
    args = parse_args()
    env_name = args.env
    model_name = args.model

    # Resolve checkpoint
    ckpt_path = args.ckpt or f"/home/lx/snn/results/{env_name}/{model_name}/final.pt"
    if not os.path.exists(ckpt_path):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"skipped": True, "reason": f"no ckpt at {ckpt_path}"}, f, indent=2)
        print(f"[event_align] skip — no ckpt at {ckpt_path}")
        return 0

    # Load ckpt
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"skipped": True, "reason": f"ckpt load failed: {e}"}, f, indent=2)
        return 0
    ck_args = ck.get("args", {}) or {}

    # Build env (DMCDMCStateEnv) using its native action space.
    try:
        from code.eval.closed_loop import make_env
        env_kind = ENV_KIND_MAP[env_name]
        env = make_env(env_kind, data_path=None)
    except Exception as e:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"skipped": True, "reason": f"env build failed: {e}"}, f, indent=2)
        return 0

    state_dim = env.spec.obs_dim
    action_dim = env.spec.action_dim
    a_low = env.spec.action_low
    a_high = env.spec.action_high

    # Build model
    try:
        model = build_model(model_name, state_dim, action_dim, ck_args)
        model.load_state_dict(ck["model"])
    except Exception as e:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"skipped": True, "reason": f"model build/load failed: {e}"}, f, indent=2)
        return 0
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Run random policy and record obs / latent / spike
    obs_list = []
    lat_list = []
    spike_list = []

    env.reset(seed=0)
    obs = env.get_state()
    obs_list.append(obs.astype(np.float32))

    steps_per_reset = max(1, args.n_steps // args.n_resets)
    n_done = 0
    t = 0
    while t < args.n_steps:
        a = np.random.uniform(a_low, a_high).astype(np.float32)
        out, _, done, _ = env.step(a)
        obs = out.get("state", list(out.values())[0])
        obs = np.asarray(obs, dtype=np.float32)
        # Encode + forward
        s_t = torch.from_numpy(obs).reshape(1, 1, -1).to(args.device)
        a_t = torch.from_numpy(a).reshape(1, 1, -1).to(args.device)
        with torch.no_grad():
            enc = model.encode(s_t, a_t)
            fwd = model.forward(s_t, a_t)
        lat_list.append(enc["emb"][0, 0].cpu().numpy())
        # LeWM baseline has no "spike" key — fall back to None.
        spike = fwd.get("spike", None)
        if spike is not None:
            spike_list.append(float(spike[0, 0].mean().item()))
        else:
            # For LeWM baseline, use the embedding L2 norm as a "rate" proxy.
            spike_list.append(float(np.linalg.norm(enc["emb"][0, 0].cpu().numpy())))
        obs_list.append(obs)
        t += 1
        if done and t < args.n_steps:
            n_done += 1
            env.reset(seed=n_done)

    obs_arr = np.stack(obs_list, axis=0)
    lat_arr = np.stack(lat_list, axis=0)
    rate_arr = np.array(spike_list, dtype=np.float32)

    # First differences (length N-1 each).
    d_obs = np.linalg.norm(np.diff(obs_arr, axis=0), axis=1)
    d_lat = np.linalg.norm(np.diff(lat_arr, axis=0), axis=1)
    # rate has length n_steps; d_obs/d_lat have length n_steps - 1.
    # Align to the shortest of the three.
    L = min(d_obs.shape[0], d_lat.shape[0], rate_arr.shape[0])
    d_obs = d_obs[:L]
    d_lat = d_lat[:L]
    rate_used = rate_arr[:L]

    corr_obs_lat = pearson(d_obs, d_lat)
    corr_obs_rate = pearson(d_obs, rate_used)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "skipped": False,
                "reason": None,
                "env": env_name,
                "model": model_name,
                "corr_obs_latent": float(corr_obs_lat),
                "corr_obs_rate": float(corr_obs_rate),
                "n_steps": int(d_obs.shape[0]),
                "n_resets": int(n_done),
            },
            f, indent=2,
        )
    print(f"[event_align] {env_name}/{model_name}: corr(obs,lat)={corr_obs_lat:.3f}  "
          f"corr(obs,rate)={corr_obs_rate:.3f}  steps={d_obs.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

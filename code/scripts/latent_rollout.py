"""Unified latent-dynamics rollout measurement (E12).

One rollout per (model, env, split) with the SAME protocol as G1
(event_align.py): real dm_control interaction, random uniform policy,
n_steps=200 across n_resets resets, obs padded to 128 / action to 56,
latent = model.encode(obs_t, a_t) after each env step.

Outputs per run:
  <out>.json : div / resp / event-rho computed on the SAME rollout
               (three-axis triplet, single-protocol) + diagnostics
  <out>.npz  : lat_arr (T, D) and obs_arr (T, obs_dim) raw trajectories
               for representative-trajectory figures.

Usage:
    python -m code.scripts.latent_rollout --ckpt <final.pt> --model <dir-name> \
        --env cartpole_2d --out /path/out.json [--pad-obs-to 128] \
        [--action-dim-eval 56] [--n-steps 200] [--device cuda]
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

from code.scripts.event_align import ENV_KIND_MAP, build_model, pearson  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified latent rollout measurement.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", required=True, help="Model dir name (e.g. stjewm_trace_only).")
    p.add_argument("--env", required=True, choices=sorted(ENV_KIND_MAP.keys()))
    p.add_argument("--out", required=True, help="Output json path (.npz written alongside).")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--n-resets", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pad-obs-to", type=int, default=None)
    p.add_argument("--action-dim-eval", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.ckpt):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"skipped": True, "reason": f"no ckpt at {args.ckpt}"},
                  open(args.out, "w"), indent=2)
        print(f"[latent_rollout] skip — no ckpt at {args.ckpt}")
        return 0

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {}) or {}

    from code.eval.closed_loop import make_env
    env = make_env(ENV_KIND_MAP[args.env], data_path=None)

    state_dim = env.spec.obs_dim
    action_dim = env.spec.action_dim
    if args.pad_obs_to is not None:
        state_dim = args.pad_obs_to
    if args.action_dim_eval is not None:
        action_dim = args.action_dim_eval
    a_low = env.spec.action_low
    a_high = env.spec.action_high

    model = build_model(args.model, state_dim, action_dim, ck_args, state_dict=ck["model"])
    model.load_state_dict(ck["model"])
    model = model.to(args.device).eval()
    for p_ in model.parameters():
        p_.requires_grad = False

    obs_list, lat_list = [], []
    env.reset(seed=args.seed)
    obs = env.get_state()
    if args.pad_obs_to is not None and len(obs) < args.pad_obs_to:
        obs = np.concatenate([obs, np.zeros(args.pad_obs_to - len(obs), dtype=np.float32)])

    n_done, t = 0, 0
    while t < args.n_steps:
        a = np.random.uniform(a_low, a_high).astype(np.float32)
        out, _, done, _ = env.step(a)
        obs = out.get("state", list(out.values())[0])
        obs = np.asarray(obs, dtype=np.float32)
        if args.pad_obs_to is not None and len(obs) < args.pad_obs_to:
            obs = np.concatenate([obs, np.zeros(args.pad_obs_to - len(obs), dtype=np.float32)])
        s_t = torch.from_numpy(obs).reshape(1, 1, -1).to(args.device)
        a_padded = np.zeros(action_dim, dtype=np.float32)
        a_padded[: len(a)] = a
        a_t = torch.from_numpy(a_padded).reshape(1, 1, -1).to(args.device)
        with torch.no_grad():
            enc = model.encode(s_t, a_t)
        lat_list.append(enc["emb"][0, 0].float().cpu().numpy())
        obs_list.append(obs)
        t += 1
        if done and t < args.n_steps:
            n_done += 1
            env.reset(seed=args.seed + n_done)

    obs_arr = np.stack(obs_list, axis=0).astype(np.float32)
    lat_arr = np.stack(lat_list, axis=0).astype(np.float32)

    d_obs = np.linalg.norm(np.diff(obs_arr, axis=0), axis=1)
    d_lat = np.linalg.norm(np.diff(lat_arr, axis=0), axis=1)
    resp = float(d_lat.mean() / d_obs.mean()) if d_obs.mean() > 1e-9 else 0.0
    per_dim_std = lat_arr.std(axis=0)
    div = float(per_dim_std.mean())
    rho = pearson(d_obs, d_lat)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "skipped": False,
            "env": args.env,
            "model": args.model,
            "ckpt": args.ckpt,
            "n_steps": int(d_obs.shape[0]),
            "n_resets": int(n_done),
            "divergence": round(div, 6),
            "responsiveness": round(resp, 6),
            "event_rho": round(rho, 6),
            "per_dim_std_max": round(float(per_dim_std.max()), 6),
            "per_dim_std_min": round(float(per_dim_std.min()), 6),
            "mean_norm_latent": round(float(np.linalg.norm(lat_arr, axis=1).mean()), 4),
            "mean_d_obs": round(float(d_obs.mean()), 6),
            "mean_d_lat": round(float(d_lat.mean()), 6),
        }, f, indent=2)
    np.savez(args.out[:-5] + ".npz", lat_arr=lat_arr, obs_arr=obs_arr)
    print(f"[latent_rollout] {args.env}/{args.model}: div={div:.4f} resp={resp:.3f} rho={rho:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

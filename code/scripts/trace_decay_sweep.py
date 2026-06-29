#!/usr/bin/env python
"""Trace decay sweep: fix alpha to constant λ, eval closed-loop.

Proves memory in trace matters: long memory (λ large) > no memory (λ=0).

Usage:
  python -m code.scripts.trace_decay_sweep --env cheetah --decay 0.9
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/home/lx/LeWM")

from code.eval.closed_loop import make_env, eval_closed_loop
from code.stjewm import STJEWM

ENV_CONFIG = {
    "cheetah": ("cheetah", "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz", 25),
    "cartpole_2d": ("cartpole", "/home/lx/snn/data/dm_control/cartpole_250k.npz", 25),
    "pusht": ("pusht", "/home/lx/LeWM/data/pusht_expert_train.h5", 100),
    "tworoom": ("tworoom", "/home/lx/LeWM/data/tworoom_extract/tworoom.h5", 100),
    "walker": ("walker", "/home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz", 25),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=list(ENV_CONFIG))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--decay", type=float, required=True,
                   help="Fixed alpha for trace: r_t = alpha*r_{t-1} + (1-alpha)*s_t")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    return p.parse_args()


@torch.no_grad()
def run_decay_eval(model, env, data_path, args, decay, device="cuda"):
    """Run eval with fixed-decay trace."""
    original_forward = model.gated_trace.forward

    def fixed_decay_forward(spike, context):
        """Fixed-alpha trace: r_t = alpha * r_{t-1} + (1-alpha) * s_t."""
        B, T, D = spike.shape
        r = torch.zeros(B, D, device=spike.device, dtype=spike.dtype)
        traces = []
        alpha = torch.full((B, D), decay, device=spike.device, dtype=spike.dtype)
        for t in range(T):
            s = spike[:, t]
            r = alpha * r + (1.0 - alpha) * s
            traces.append(r)
        return torch.stack(traces, dim=1)

    model.gated_trace.forward = fixed_decay_forward
    try:
        result = eval_closed_loop(
            model, env, data_path,
            n_episodes=args.n_episodes, n_seeds=args.n_seeds,
            cem_samples=300, cem_elites=30, cem_iters=10,
            horizon=args.horizon, eval_budget=args.eval_budget,
            goal_offset=args.goal_offset, history_size=1,
            device=device,
        )
    finally:
        model.gated_trace.forward = original_forward

    return {
        "env": args.env, "decay": decay,
        "success_rate_lewm": result.success_rate_lewm,
        "success_rate_env": result.success_rate_env,
        "mean_cos_dist": result.mean_cos_dist,
        "mean_phys_dist": result.mean_phys_dist,
        "n_episodes": result.n_episodes,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_env_str, data_path_default, goal_offset = ENV_CONFIG[args.env]
    args.goal_offset = goal_offset
    env = make_env(eval_env_str, args.data or data_path_default)
    data_path = args.data or data_path_default
    ckpt_path = args.ckpt or f"/home/lx/snn/results/{args.env}/stjewm_trace_only/final.pt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    state_dim = env.spec.obs_dim
    action_dim = env.spec.action_dim
    n_layers = ck_args.get("n_layers", 4)
    ck_readout_mode = ck_args.get("readout_mode", "trace_only")
    model = STJEWM(
        d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
        state_dim=state_dim, cell_n_layers=n_layers, n_d=3,
        trace_beta=0.9, freeze_encoder=True,
        readout_mode=ck_readout_mode,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    result = run_decay_eval(model, env, data_path, args, args.decay, device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[decay_sweep] {args.env} decay={args.decay:.2f} "
          f"lewm_sr={result['success_rate_lewm']:.3f} cos={result['mean_cos_dist']:.3f}")


if __name__ == "__main__":
    main()

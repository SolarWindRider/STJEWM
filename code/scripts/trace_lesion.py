#!/usr/bin/env python
"""Trace lesion: randomly zero trace dimensions, eval closed-loop.

Proves trace necessity: if lesion -> performance drops, trace is doing the work.

Usage:
  python -m code.scripts.trace_lesion --env cheetah --lesion-ratio 0.5
"""
from __future__ import annotations
import argparse, json, sys, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/home/lx/LeWM")

from code.eval.closed_loop import make_env, eval_closed_loop, ClosedLoopResult, _load_eval_dataset
from code.stjewm import STJEWM


ENV_CONFIG = {
    "cheetah": ("cheetah", "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz", 25, 1),
    "cartpole_2d": ("cartpole", "/home/lx/snn/data/dm_control/cartpole_250k.npz", 25, 1),
    "pusht": ("pusht", "/home/lx/LeWM/data/pusht_expert_train.h5", 100, 1),
    "tworoom": ("tworoom", "/home/lx/LeWM/data/tworoom_extract/tworoom.h5", 100, 1),
    "walker": ("walker", "/home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz", 25, 1),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=list(ENV_CONFIG))
    p.add_argument("--ckpt", default=None, help="default: results/<env>/stjewm_trace_only/final.pt")
    p.add_argument("--data", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--lesion-ratio", type=float, required=True,
                   help="Fraction of trace dims to zero (0.0 = baseline, 1.0 = all zero)")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    return p.parse_args()


@torch.no_grad()
def run_lesion_eval(model, env, data_path, args, lesion_ratio, device="cuda"):
    """Run eval with trace lesion applied."""
    # Monkey-patch _readout to apply lesion
    original_readout = model._readout

    def lesioned_readout(h, spike, trace):
        # lesion: zero random dims of trace
        if lesion_ratio > 0 and trace is not None:
            B, T, D = trace.shape
            mask = (torch.rand(B, 1, D, device=trace.device) > lesion_ratio).float()
            trace = trace * mask
        return original_readout(h, spike, trace)

    model._readout = lesioned_readout

    try:
        result = eval_closed_loop(
            model, env, data_path,
            n_episodes=args.n_episodes, n_seeds=args.n_seeds,
            cem_samples=args.cem_samples if hasattr(args, 'cem_samples') else 300,
            cem_elites=30, cem_iters=10,
            horizon=args.horizon, eval_budget=args.eval_budget,
            goal_offset=args.goal_offset, history_size=args.history_size,
            device=device,
        )
    finally:
        model._readout = original_readout

    return {
        "env": args.env,
        "lesion_ratio": lesion_ratio,
        "success_rate_lewm": result.success_rate_lewm,
        "success_rate_env": result.success_rate_env,
        "mean_cos_dist": result.mean_cos_dist,
        "mean_phys_dist": result.mean_phys_dist,
        "n_episodes": result.n_episodes,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build env and load ckpt
    eval_env_str, data_path_default, goal_offset, history_size = ENV_CONFIG[args.env]
    args.goal_offset = goal_offset
    args.history_size = history_size

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

    result = run_lesion_eval(model, env, data_path, args, args.lesion_ratio, device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[trace_lesion] {args.env} ratio={args.lesion_ratio:.2f} "
          f"lewm_sr={result['success_rate_lewm']:.3f} cos={result['mean_cos_dist']:.3f}")


if __name__ == "__main__":
    main()

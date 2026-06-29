#!/usr/bin/env python
"""Spike timing shuffle: shuffle spike order, keep count fixed. Proves temporal structure matters.

Usage:
  python -m code.scripts.timing_shuffle --env cheetah --shuffle global
"""
from __future__ import annotations
import argparse, json, sys, time, os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/home/lx/LeWM")

from code.eval.closed_loop import make_env, eval_closed_loop
from code.stjewm import STJEWM, GatedSpikeTrace

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
    p.add_argument("--shuffle", choices=["none", "window5", "window10", "global"], default="none")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    return p.parse_args()


def shuffle_spikes(spike, mode, seed=42):
    """Shuffle spike tensor along T dimension. (B, T, D) binary."""
    if mode == "none":
        return spike
    B, T, D = spike.shape
    g = torch.Generator(device=spike.device)
    g.manual_seed(seed)
    if mode == "global":
        # full shuffle across all T for each neuron independently
        idx = torch.argsort(torch.rand(B, T, D, generator=g, device=spike.device), dim=1)
        return spike.gather(1, idx)
    elif mode.startswith("window"):
        w = int(mode.replace("window", ""))
        out = spike.clone()
        for t_start in range(0, T, w):
            t_end = min(t_start + w, T)
            window_len = t_end - t_start
            if window_len <= 1:
                continue
            idx = torch.argsort(torch.rand(B, window_len, D, generator=g, device=spike.device), dim=1)
            out[:, t_start:t_end, :] = spike[:, t_start:t_end, :].gather(1, idx)
        return out
    return spike


@torch.no_grad()
def run_shuffle_eval(model, env, data_path, args, shuffle_mode, device="cuda"):
    """Run eval with shuffled spikes."""
    original_forward = model.forward

    def shuffled_forward(x, a):
        out = original_forward(x, a)
        spike_raw = out["spike"]
        shuffled = shuffle_spikes(spike_raw, shuffle_mode)
        # Recompute trace from shuffled spikes
        act_emb = out["act_emb"]
        h = out["h"]
        context = torch.cat([act_emb, h], dim=-1)
        trace_reshuffled = model.gated_trace(shuffled, context)
        # Override emb using _readout with reshuffled trace
        z_final = model._readout(h, shuffled, trace_reshuffled)
        out["emb"] = z_final
        out["spike"] = shuffled
        out["trace"] = trace_reshuffled
        return out

    model.forward = shuffled_forward
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
        model.forward = original_forward

    return {
        "env": args.env, "shuffle": shuffle_mode,
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
    result = run_shuffle_eval(model, env, data_path, args, args.shuffle, device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[timing_shuffle] {args.env} {args.shuffle} "
          f"lewm_sr={result['success_rate_lewm']:.3f} cos={result['mean_cos_dist']:.3f}")


if __name__ == "__main__":
    main()

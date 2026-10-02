"""Measure trained-checkpoint readouts and observation embeddings separately.

Uses the same seeded, single-frame diagnostic protocol as latent_rollout.py.
Every checkpoint is loaded strictly; failures make the batch fail rather than
silently producing an incomplete summary.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from code.scripts.event_align import build_model
from code.scripts.latent_rollout import PROTOCOL_VERSION, collect_state_rollout


DMC_ENVS = [
    ("cheetah",     "data/dm_control/3d_rollouts_250k/cheetah_250k.npz"),
    ("walker",      "data/dm_control/3d_rollouts_250k/walker_250k.npz"),
    ("humanoid",    "data/dm_control/3d_rollouts_250k/humanoid_250k.npz"),
    ("cartpole_2d", "data/dm_control/cartpole_250k.npz"),
    ("pendulum_2d", "data/dm_control/pendulum_250k.npz"),
    ("finger",      "data/dm_control/3d_rollouts_250k/finger_250k.npz"),
    ("ball_in_cup", "data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz"),
    ("dog",         "data/dm_control/3d_rollouts_250k/dog_250k.npz"),
    ("fish",        "data/dm_control/3d_rollouts_250k/fish_250k.npz"),
    ("stacker",     "data/dm_control/3d_rollouts_250k/stacker_250k.npz"),
    ("quadruped",   "data/dm_control/3d_rollouts_250k/quadruped_250k.npz"),
    ("hopper",      "data/dm_control/3d_rollouts_250k/hopper_250k.npz"),
    ("humanoid_CMU","data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz"),
    ("reacher",     "data/dm_control/3d_rollouts_250k/reacher_250k.npz"),
    ("pusht",       "/home/lx/LeWM/data/pusht_expert_train.h5"),
    ("tworoom",     "/home/lx/LeWM/data/tworoom_extract/tworoom.h5"),
]
CLO_ENV_MAP = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum"}
MODELS = [
    "stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only",
    "stjewm_no_trace", "stjewm_hidden_leak", "stjewm_membrane_readout",
    "alif_timecell_baseline", "gru_baseline", "lewm_baseline_v2",
    "stacked_lif_trace", "stacked_lif_free", "mlp_baseline", "lif_transformer_baseline",
]


def load_model_5m(ckpt_path: str, env, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    state_dim = ck_args.get("pad_obs_to") or env.spec.obs_dim
    action_dim = ck_args.get("action_dim") or env.spec.action_dim
    model_name = Path(ckpt_path).parent.parent.name
    model = build_model(model_name, state_dim, action_dim, ck_args, state_dict=ck["model"])
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval(), ck_args, state_dim, action_dim


def measure_one(ckpt_path, env_kind, data_path, n_steps=200, seed=0, device="cpu"):
    from code.eval.closed_loop import make_env
    env = make_env(CLO_ENV_MAP.get(env_kind, env_kind), data_path)
    try:
        model, _, state_dim, action_dim = load_model_5m(ckpt_path, env, device)
        result, _ = collect_state_rollout(
            model, env, state_dim, action_dim,
            n_steps=n_steps, n_resets=2, seed=seed, device=device,
        )
    finally:
        env.close()
    result.update(
        model=Path(ckpt_path).parent.parent.name,
        split=Path(ckpt_path).parent.parent.parent.name,
        env=env_kind, ckpt=str(ckpt_path), weights_loaded_strict=True,
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("/home/lx/snn/results/5m"))
    p.add_argument("--out", type=Path, default=Path("/home/lx/snn/results/5m_stats"))
    p.add_argument("--splits", nargs="+", default=None,
                   help="Restrict to specific splits (default: all)")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--envs", nargs="+", default=None,
                   help="Restrict to specific envs (default: all DMC)")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=7,
                   help="Number of envs to measure per (split, model)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Discover splits
    splits = args.splits or sorted([p.name for p in args.results.iterdir() if p.is_dir() and p.name != "_logs"])
    envs = args.envs or [e for e, _ in DMC_ENVS][:args.n_envs]

    failures = []
    done = 0
    for split in splits:
        for model in args.models:
            ckpt = args.results / split / model / "seed_0" / "final.pt"
            if not ckpt.exists():
                failures.append(f"Missing checkpoint: {ckpt}")
                continue
            for env in envs:
                # find data path
                data_path = None
                for en, dp in DMC_ENVS:
                    if en == env:
                        data_path = dp
                        break
                if data_path is None:
                    continue
                out_path = args.out / split / model / f"latent_stats_{env}.json"
                if out_path.exists():
                    previous = json.loads(out_path.read_text())
                    if previous.get("protocol_version") == PROTOCOL_VERSION:
                        done += 1
                        continue
                    raise RuntimeError(f"Archive obsolete diagnostics before rerunning: {out_path}")
                t0 = time.time()
                try:
                    r = measure_one(str(ckpt), env, data_path, args.n_steps,
                                    seed=args.seed, device=args.device)
                except Exception as e:
                    failures.append(f"{split}/{model}/{env}: {e}")
                    print(f"  ERR {failures[-1]}", flush=True)
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(r, indent=2, allow_nan=False))
                done += 1
                print(f"  [{done}] {split}/{model}/{env} resp={r['responsiveness']} "
                      f"div={r['divergence']:.6g} ({time.time()-t0:.1f}s)", flush=True)
    print(f"done: {done} stats generated -> {args.out}", flush=True)
    if failures:
        raise RuntimeError("Diagnostic batch failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()

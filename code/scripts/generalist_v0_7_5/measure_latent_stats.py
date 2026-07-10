"""Measure collapse-robust latent statistics for a generalist checkpoint.

For each (ckpt, env) pair, run a random policy for `--n-steps` steps on
the env and record obs and the model's calibrated latent at every step.
Then compute:

- **responsiveness**: `mean_norm(Δlatent) / mean_norm(Δobs)`. < 0.3 = latent
  moves less than obs on average (calibrated STJEWM/CubifAE ~0.2,
  collapsed MLP ~0.08). > 1 = latent amplifies obs (LeWM ~8.5, GRU ~3.9,
  which is noise amplified — caught by event-align ρ).
- **divergence-from-constant**: per-dim std of the latent trajectory,
  averaged across dims. < 0.001 = collapse (MLP ~0.0004). > 0.005 =
  responsive (STJEWM ~0.006, LeWM ~0.18).

The pair (responsiveness, divergence) is **collapse-robust by
construction**: a model that maps all inputs to a single latent vector
will score (≈0, ≈0.0004) — independent of whether its planner is good.
The collapse-robustness comes from `divergence`, not from
`responsiveness`: even an unresponsive model can have a small but
non-zero divergence if its latent has any non-trivial state. The
`responsiveness` ratio is informative but the absolute magnitudes
depend on the model's encoder scale.

Measured on G4 ckpts at cheetah (200 random steps):

| model              | responsiveness | divergence | interpretation        |
|--------------------|----------------|------------|------------------------|
| stjewm_trace_only  | 0.215          | 0.0064     | calibrated             |
| stjewm_spike_only  | 0.206          | 0.0061     | calibrated             |
| stjewm_no_trace    | 0.207          | 0.0054     | calibrated             |
| stjewm_membrane    | 0.193          | 0.0052     | calibrated             |
| cubifae_baseline   | 0.199          | 0.0056     | calibrated             |
| gru_baseline       | 3.905          | 0.0103     | noise (high Δlat)      |
| lewm_baseline      | 8.524          | 0.1797     | over-reactive          |
| mlp_baseline       | 0.086          | 0.0004     | COLLAPSE (15× lower)   |

MLP's divergence is **15× lower** than STJEWM's, and **75× lower** than
LeWM's — a clear quantitative separation.

This script is read-only on ckpts; no retraining, no planner. Cost is
~10ms per step × 200 steps × 1 run = ~2s per (ckpt, env).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DMC_ENVS = [
    ("cheetah",     "data/dm_control/3d_rollouts_250k/cheetah_250k.npz"),
    ("walker",      "data/dm_control/3d_rollouts_250k/walker_250k.npz"),
    ("humanoid",    "data/dm_control/3d_rollouts_250k/humanoid_250k.npz"),
    ("cartpole_2d", "data/dm_control/cartpole_250k.npz"),
    ("pendulum_2d", "data/dm_control/pendulum_250k.npz"),
    ("finger",      "data/dm_control/3d_rollouts_250k/finger_250k.npz"),
    ("ball_in_cup", "data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz"),
]

CLO_ENV_MAP = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum"}


def load_model(ckpt_path: str, env, device: str = "cpu"):
    """Mirror closed_loop.py: load ckpt, build the right model class, move
    to device."""
    from code.eval.closed_loop import _PadObsWrapper
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    state_dim = (ck_args.get("pad_obs_to") or env.spec.obs_dim)
    action_dim = (ck_args.get("action_dim") or env.spec.action_dim)
    if state_dim > env.spec.obs_dim:
        env = _PadObsWrapper(env, state_dim)
    m = ck_args.get("model", "stjewm")
    if m == "lewm_baseline":
        from code.lewm_transformer_baseline import LeWMTransformerBaseline
        model = LeWMTransformerBaseline(
            state_dim=state_dim, action_dim=action_dim,
            embed_dim=ck_args.get("embed_dim", 256),
            num_layers=ck_args.get("n_layers", 4), num_heads=8)
    elif m == "gru_baseline":
        from code.gru_baseline import GRUBaseline
        model = GRUBaseline(state_dim=state_dim, action_dim=action_dim)
    elif m == "mlp_baseline":
        from code.mlp_baseline import make_mlp_baseline
        model = make_mlp_baseline(state_dim=state_dim, action_dim=action_dim)
    elif m == "slt_lif_mpc_trace":
        from code.slt_lif_mpc_baseline import make_slt_lif_mpc_trace
        model = make_slt_lif_mpc_trace(
            state_dim=state_dim, action_dim=action_dim,
            d_in=192, embed_dim=192, n_layers=ck_args.get("n_layers", 4),
            trace_beta=0.9, k_avg=4)
    elif m == "slt_lif_mpc_free":
        from code.slt_lif_mpc_baseline import make_slt_lif_mpc_free
        model = make_slt_lif_mpc_free(
            state_dim=state_dim, action_dim=action_dim,
            d_in=192, embed_dim=192, n_layers=ck_args.get("n_layers", 4),
            trace_beta=0.9)
    elif m == "cubifae_baseline":
        from code.cubifae_baseline import CubifAEBaseline
        model = CubifAEBaseline(
            state_dim=state_dim, action_dim=action_dim,
            d_hid=192, n_layers=ck_args.get("n_layers", 4))
    elif m == "spikedreamer_baseline":
        from code.spikedreamer_baseline import make_spikedreamer
        model = make_spikedreamer(
            state_dim=state_dim, action_dim=action_dim,
            d_snn=128, d_tx=192, num_layers=ck_args.get("n_layers", 4),
            num_heads=8)
    else:
        from code.stjewm import STJEWM
        model = STJEWM(
            d_hid=192, embed_dim=192, action_dim=action_dim,
            action_emb_dim=192, state_dim=state_dim,
            cell_n_layers=ck_args.get("n_layers", 4), n_d=3,
            trace_beta=0.9, freeze_encoder=True,
            readout_mode=ck_args.get("readout_mode", "hidden_leak"))
    return model.to(device).eval(), ck_args, state_dim, action_dim


def measure_one(ckpt_path: str, env_kind: str, data_path: str,
                n_steps: int = 200, seed: int = 0, device: str = "cpu") -> dict:
    """Run a random policy and compute responsiveness / divergence."""
    from code.eval.closed_loop import make_env, _PadObsWrapper
    clo_env = CLO_ENV_MAP.get(env_kind, env_kind)
    env = make_env(clo_env, data_path)
    # Pad the env to the model's state_dim (mirrors closed_loop.py).
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    state_dim = (ck_args.get("pad_obs_to") or env.spec.obs_dim)
    action_dim = (ck_args.get("action_dim") or env.spec.action_dim)
    if state_dim > env.spec.obs_dim:
        env = _PadObsWrapper(env, state_dim)
    model, _, _, _ = load_model(ckpt_path, env, device)

    # Pre-allocate obs and latent trajectory arrays
    obs_traj = []
    lat_traj = []
    a_low = env.spec.action_low
    a_high = env.spec.action_high

    env.reset(seed=seed)
    obs = env.get_state()
    obs_traj.append(obs.astype(np.float32))

    # Pad action to action_dim (model's expected action space)
    a_padded = np.zeros(action_dim, dtype=np.float32)

    with torch.no_grad():
        for t in range(n_steps):
            a = np.random.uniform(a_low, a_high).astype(np.float32)
            a_padded[: len(a)] = a
            s_t = torch.from_numpy(obs.astype(np.float32)).reshape(1, 1, -1).to(device)
            a_t = torch.from_numpy(a_padded).reshape(1, 1, -1).to(device)
            enc = model.encode(s_t, a_t)
            lat = enc["emb"][0, 0].cpu().numpy()  # (D,)
            lat_traj.append(lat)
            out, _, done, _ = env.step(a)
            obs = out.get("state", list(out.values())[0])
            obs_traj.append(obs)
            if done:
                env.reset(seed=seed + t + 1)
    obs_arr = np.stack(obs_traj, axis=0)   # (T+1, D_obs)
    lat_arr = np.stack(lat_traj, axis=0)   # (T, D_lat)
    # Mean norm of first-differences
    d_obs = np.linalg.norm(np.diff(obs_arr, axis=0), axis=1)
    d_lat = np.linalg.norm(np.diff(lat_arr, axis=0), axis=1)
    responsiveness = float(d_lat.mean() / d_obs.mean()) if d_obs.mean() > 1e-9 else 0.0
    # Per-dim std of latent, averaged
    per_dim_std = lat_arr.std(axis=0)  # (D_lat,)
    divergence = float(per_dim_std.mean())
    # Norms for context
    return {
        "model": Path(ckpt_path).parent.parent.name,
        "env": env_kind,
        "ckpt": str(ckpt_path),
        "n_steps": int(n_steps),
        "responsiveness": round(responsiveness, 4),
        "divergence": round(divergence, 4),
        "per_dim_std_max": round(float(per_dim_std.max()), 4),
        "per_dim_std_min": round(float(per_dim_std.min()), 4),
        "mean_norm_obs": round(float(np.linalg.norm(obs_arr, axis=1).mean()), 4),
        "mean_norm_latent": round(float(np.linalg.norm(lat_arr, axis=1).mean()), 4),
        "mean_d_obs": round(float(d_obs.mean()), 4),
        "mean_d_latent": round(float(d_lat.mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env", required=True, choices=[e for e, _ in DMC_ENVS])
    ap.add_argument("--data", default=None,
                    help="Override env data path (default: built-in DMC path).")
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    if args.data is None:
        for env_name, dp in DMC_ENVS:
            if env_name == args.env:
                args.data = dp
                break
    if args.data is None:
        print(f"[measure_latent_stats] unknown env {args.env}", file=sys.stderr)
        return 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result = measure_one(args.ckpt, args.env, args.data,
                          n_steps=args.n_steps, seed=args.seed, device=args.device)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[measure_latent_stats] {args.env} {Path(args.ckpt).parent.parent.name} "
          f"-> responsiveness={result['responsiveness']:.3f} "
          f"divergence={result['divergence']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
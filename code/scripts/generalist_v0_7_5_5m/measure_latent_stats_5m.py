"""Measure collapse-robust latent statistics for 5M-aligned generalist ckpts.

Mirror of code/scripts/generalist_v0_7_5/measure_latent_stats.py but:
  - Reads from results/5m/<split>/<model>/seed_0/final.pt
  - Uses state-dict inference to handle 5M model dims (e.g. cubifae d_hid=186)
  - Iterates over (split, model, env) grid

Usage:
    python -m code.scripts.generalist_v0_7_5_5m.measure_latent_stats_5m
    python -m code.scripts.generalist_v0_7_5_5m.measure_latent_stats_5m --splits oodc_F1 cross_benchmark_F1
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    "cubifae_baseline", "gru_baseline", "lewm_baseline_v2",
    "slt_lif_mpc_trace", "slt_lif_mpc_free", "mlp_baseline", "spikedreamer_baseline",
]


def _infer_dim_from_state_dict(sd, key):
    if sd is None:
        return None
    w = sd.get(key)
    if hasattr(w, "shape") and len(w.shape) >= 1:
        return int(w.shape[0])
    return None


def _state_dict_n_layers(sd, prefix):
    if sd is None:
        return None
    indices = []
    for k in sd:
        if k.startswith(prefix + "."):
            rest = k[len(prefix) + 1:]
            if rest.startswith("weight_ih_l") or rest.startswith("weight_hh_l"):
                try:
                    indices.append(int(rest.split("_l")[-1]))
                except ValueError:
                    pass
    return max(indices) + 1 if indices else None


def load_model_5m(ckpt_path: str, env, device: str = "cpu"):
    from code.eval.closed_loop import _PadObsWrapper
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", {})
    ck_args = ck.get("args", {})
    state_dim = (ck_args.get("pad_obs_to") or env.spec.obs_dim)
    action_dim = (ck_args.get("action_dim") or env.spec.action_dim)
    if state_dim > env.spec.obs_dim:
        env = _PadObsWrapper(env, state_dim)
    m = ck_args.get("model", "stjewm")
    if m == "lewm_baseline":
        from code.lewm_transformer_baseline import LeWMTransformerBaseline
        embed_dim = (_infer_dim_from_state_dict(sd, "state_encoder.proj.0.weight")
                      or ck_args.get("embed_dim", 288))
        return (LeWMTransformerBaseline(
            state_dim=state_dim, action_dim=action_dim,
            embed_dim=embed_dim, num_layers=3, num_heads=8
        ).to(device).eval(), ck_args, state_dim, action_dim)
    if m == "gru_baseline":
        from code.gru_baseline import GRUBaseline
        hidden_dim = (_infer_dim_from_state_dict(sd, "state_proj.0.weight")
                      or ck_args.get("hidden_dim", 560))
        n_layers = _state_dict_n_layers(sd, "gru") or ck_args.get("n_layers", 2)
        return (GRUBaseline(state_dim=state_dim, action_dim=action_dim,
                            hidden_dim=hidden_dim, num_layers=n_layers
                            ).to(device).eval(), ck_args, state_dim, action_dim)
    if m == "mlp_baseline":
        from code.mlp_baseline import MLPBaseline
        hidden_dim = (_infer_dim_from_state_dict(sd, "state_proj.0.weight")
                      or ck_args.get("hidden_dim", 640))
        # num hidden layers = count of net.X.weight of shape [hidden, hidden]
        n_hidden = sum(1 for k, v in sd.items()
                       if k.startswith("net.") and k.endswith(".weight")
                       and hasattr(v, "shape") and len(v.shape) == 2
                       and v.shape[0] == hidden_dim and v.shape[1] == hidden_dim)
        num_layers = n_hidden or 12
        emb_dim = (_infer_dim_from_state_dict(sd, "state_proj.2.weight") or 192)
        return (MLPBaseline(state_dim=state_dim, action_dim=action_dim,
                            hidden_dim=hidden_dim, num_layers=num_layers, emb_dim=emb_dim
                            ).to(device).eval(), ck_args, state_dim, action_dim)
    if m == "slt_lif_mpc_trace":
        from code.slt_lif_mpc_baseline import make_slt_lif_mpc_trace
        d_in = _infer_dim_from_state_dict(sd, "state_projector.proj.0.weight") or 672
        return (make_slt_lif_mpc_trace(state_dim=state_dim, action_dim=action_dim,
                                        d_in=d_in, embed_dim=d_in, n_layers=8,
                                        trace_beta=0.9, k_avg=4
                                        ).to(device).eval(), ck_args, state_dim, action_dim)
    if m == "slt_lif_mpc_free":
        from code.slt_lif_mpc_baseline import make_slt_lif_mpc_free
        d_in = _infer_dim_from_state_dict(sd, "state_projector.proj.0.weight") or 640
        return (make_slt_lif_mpc_free(state_dim=state_dim, action_dim=action_dim,
                                       d_in=d_in, embed_dim=d_in, n_layers=8,
                                       trace_beta=0.9
                                       ).to(device).eval(), ck_args, state_dim, action_dim)
    if m == "cubifae_baseline":
        from code.cubifae_baseline import CubifAEBaseline
        d_hid = _infer_dim_from_state_dict(sd, "state_projector.0.weight") or 186
        return (CubifAEBaseline(state_dim=state_dim, action_dim=action_dim,
                                d_hid=d_hid, n_layers=2
                                ).to(device).eval(), ck_args, state_dim, action_dim)
    if m == "spikedreamer_baseline":
        from code.spikedreamer_baseline import make_spikedreamer
        d_snn = _infer_dim_from_state_dict(sd, "state_proj.proj.0.weight") or 288
        d_tx = d_snn
        if "pos_embed" in sd:
            d_tx = int(sd["pos_embed"].shape[2])
        return (make_spikedreamer(state_dim=state_dim, action_dim=action_dim,
                                 d_snn=d_snn, d_tx=d_tx, num_layers=3, num_heads=8
                                 ).to(device).eval(), ck_args, state_dim, action_dim)
    from code.stjewm import STJEWM
    return (STJEWM(
        d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
        state_dim=state_dim, cell_n_layers=4, n_d=3, trace_beta=0.9,
        freeze_encoder=True, readout_mode=ck_args.get("readout_mode", "hidden_leak")
    ).to(device).eval(), ck_args, state_dim, action_dim)


def measure_one(ckpt_path, env_kind, data_path, n_steps=200, seed=0, device="cpu"):
    from code.eval.closed_loop import make_env, _PadObsWrapper
    clo_env = CLO_ENV_MAP.get(env_kind, env_kind)
    env = make_env(clo_env, data_path)
    # Load ckpt first to know state_dim, then wrap env, then build model
    import torch as _t
    ck = _t.load(ckpt_path, map_location='cpu', weights_only=False)
    ck_args = ck.get('args', {})
    state_dim = ck_args.get('pad_obs_to') or env.spec.obs_dim
    action_dim = ck_args.get('action_dim') or env.spec.action_dim
    if state_dim > env.spec.obs_dim:
        from code.eval.closed_loop import _PadObsWrapper
        env = _PadObsWrapper(env, state_dim)
    model, _, _, _ = load_model_5m(ckpt_path, env, device)

    obs_traj = []
    lat_traj = []
    a_low = env.spec.action_low
    a_high = env.spec.action_high
    env.reset(seed=seed)
    obs = env.get_state()
    obs_traj.append(obs.astype(np.float32))
    a_padded = np.zeros(action_dim, dtype=np.float32)

    with torch.no_grad():
        for t in range(n_steps):
            a = np.random.uniform(a_low, a_high).astype(np.float32)
            a_padded[: len(a)] = a
            s_t = torch.from_numpy(obs.astype(np.float32)).reshape(1, 1, -1).to(device)
            a_t = torch.from_numpy(a_padded).reshape(1, 1, -1).to(device)
            enc = model.encode(s_t, a_t)
            lat = enc["emb"][0, 0].cpu().numpy()
            lat_traj.append(lat)
            out, _, done, _ = env.step(a)
            obs = out.get("state", list(out.values())[0])
            obs_traj.append(obs)
            if done:
                env.reset(seed=seed + t + 1)
    obs_arr = np.stack(obs_traj, axis=0)
    lat_arr = np.stack(lat_traj, axis=0)
    d_obs = np.linalg.norm(np.diff(obs_arr, axis=0), axis=1)
    d_lat = np.linalg.norm(np.diff(lat_arr, axis=0), axis=1)
    responsiveness = float(d_lat.mean() / d_obs.mean()) if d_obs.mean() > 1e-9 else 0.0
    per_dim_std = lat_arr.std(axis=0)
    divergence = float(per_dim_std.mean())
    return {
        "model": Path(ckpt_path).parent.parent.name,
        "split": Path(ckpt_path).parent.parent.parent.name,
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
    p.add_argument("--n-envs", type=int, default=7,
                   help="Number of envs to measure per (split, model)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Discover splits
    splits = args.splits or sorted([p.name for p in args.results.iterdir() if p.is_dir() and p.name != "_logs"])
    envs = args.envs or [e for e, _ in DMC_ENVS][:args.n_envs]

    total = sum(1 for s in splits for m in args.models
                if (args.results / s / m / "seed_0" / "final.pt").exists())
    done = 0
    for split in splits:
        for model in args.models:
            ckpt = args.results / split / model / "seed_0" / "final.pt"
            if not ckpt.exists():
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
                    done += 1
                    continue
                t0 = time.time()
                try:
                    r = measure_one(str(ckpt), env, data_path, args.n_steps, device=args.device)
                except Exception as e:
                    print(f"  ERR {split}/{model}/{env}: {e}")
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(r, indent=2))
                done += 1
                print(f"  [{done}] {split}/{model}/{env} resp={r['responsiveness']:.3f} div={r['divergence']:.3f} ({time.time()-t0:.1f}s)")
    print(f"done: {done} stats generated -> {args.out}")


if __name__ == "__main__":
    main()

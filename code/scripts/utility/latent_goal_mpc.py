"""Latent-goal MPC horizon sweep (v0.7.7 utility experiment 1).

For each (model, env, horizon), run the canonical CEM planner with cosine
distance to a goal latent as the cost. Measures:
  - env_success: fraction of episodes where the env-native check_success passes
  - mean_cos_dist_terminal: terminal cosine distance to goal latent

This is the "can the planner trust latent distance?" experiment. A
collapse/noise/over-reactive latent will hurt as horizon grows; a calibrated
latent should be stable.

Usage:
    python -m code.scripts.utility.latent_goal_mpc \
        --ckpt results/generalist_G16/stjewm_trace_only/seed_0/final.pt \
        --env cheetah --horizons 1,3,5,10,20 \
        --out results/utility/latent_goal_mpc/stjewm_trace_only/cheetah.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")

from code.core.cem import CEM
from code.core.encode import encode_history, encode_obs
from code.core.envs import make_dmc_env
from code.data import load_dataset


DMC_DATA = {
    "cheetah": "data/dm_control/3d_rollouts_250k/cheetah_250k.npz",
    "walker": "data/dm_control/3d_rollouts_250k/walker_250k.npz",
    "finger": "data/dm_control/3d_rollouts_250k/finger_250k.npz",
    "cartpole": "data/dm_control/cartpole_250k.npz",
    "pendulum": "data/dm_control/pendulum_250k.npz",
    "ball_in_cup": "data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz",
    "hopper": "data/dm_control/3d_rollouts_250k/hopper_250k.npz",
    "reacher": "data/dm_control/3d_rollouts_250k/reacher_250k.npz",
}


def build_model_from_ckpt(ck_args: dict, state_dim: int, action_dim: int, device: str):
    # Use padded obs/action dims from the ckpt so cubifae/slt/etc load with the
    # same dims they were trained on (state_projector expects pad_obs_to, not
    # env-native state_dim).
    pad_obs = ck_args.get("pad_obs_to") or 128
    train_state_dim = pad_obs if (state_dim < pad_obs) else state_dim
    train_action_dim = ck_args.get("action_dim") or action_dim
    model_name = ck_args.get("model", "stjewm")
    if model_name == "stjewm":
        from code.stjewm import STJEWM
        return STJEWM(
            d_hid=192, embed_dim=ck_args.get("embed_dim", 192),
            action_dim=train_action_dim, action_emb_dim=192,
            state_dim=train_state_dim,
            cell_n_layers=ck_args.get("n_layers", 2), n_d=3,
            trace_beta=0.9, freeze_encoder=True,
            readout_mode=ck_args.get("readout_mode", "hidden_leak"),
        ).to(device)
    if model_name == "lewm_baseline":
        from code.lewm_transformer_baseline import LeWMTransformerBaseline
        return LeWMTransformerBaseline(
            state_dim=train_state_dim, action_dim=train_action_dim,
            embed_dim=ck_args.get("embed_dim", 256),
            num_layers=ck_args.get("n_layers", 4), num_heads=8,
        ).to(device)
    if model_name == "gru_baseline":
        from code.gru_baseline import GRUBaseline
        return GRUBaseline(state_dim=state_dim, action_dim=action_dim).to(device)
    if model_name == "mlp_baseline":
        from code.mlp_baseline import make_mlp_baseline
        return make_mlp_baseline(state_dim=state_dim, action_dim=action_dim).to(device)
    if model_name == "cubifae_baseline":
        from code.cubifae_baseline import make_cubifae_baseline
        return make_cubifae_baseline(
            state_dim=train_state_dim, action_dim=train_action_dim,
            d_hid=ck_args.get("embed_dim", 192),
            n_layers=ck_args.get("n_layers", 2),
        ).to(device)
    if model_name == "slt_lif_mpc_trace":
        from code.slt_lif_mpc_baseline import make_slt_lif_mpc_trace
        return make_slt_lif_mpc_trace(
            state_dim=train_state_dim, action_dim=train_action_dim,
            d_in=ck_args.get("embed_dim", 192),
            n_layers=ck_args.get("n_layers", 2),
        ).to(device)
    if model_name == "slt_lif_mpc_free":
        from code.slt_lif_mpc_baseline import make_slt_lif_mpc_free
        return make_slt_lif_mpc_free(
            state_dim=train_state_dim, action_dim=train_action_dim,
            d_in=ck_args.get("embed_dim", 192),
            n_layers=ck_args.get("n_layers", 2),
        ).to(device)
    raise ValueError(f"Unknown model_name: {model_name}")


def run_horizon_sweep(
    ckpt_path: str,
    env_kind: str,
    horizons: List[int],
    n_episodes: int = 5,
    cem_samples: int = 100,
    cem_iters: int = 10,
    cem_elites: int = 10,
    goal_offset: int = 25,
    history_size: int = 1,
    device: str = "cpu",
    out_path: str = None,
) -> dict:
    """Sweep CEM horizon and return per-horizon metrics."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    pad_obs_to = ck_args.get("pad_obs_to", 128)
    action_dim = ck_args.get("action_dim", 56)
    state_dim = ck_args.get("state_dim") or pad_obs_to

    model = build_model_from_ckpt(ck_args, state_dim, action_dim, device)
    sd = ck["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        # Some baselines may have non-strict shapes; fall back gracefully.
        result = model.load_state_dict(sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"  [warn] {ckpt_path}: missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    model.eval()

    env = make_dmc_env(env_kind)
    data_path = DMC_DATA.get(env_kind)
    if data_path is None or not os.path.exists(data_path):
        return {"error": f"data not found for env={env_kind}: {data_path}"}
    ds = load_dataset(
        env_kind="dmc", path=data_path,
        history_size=history_size, goal_offset=goal_offset,
        max_windows=min(200, 8000), pad_obs_to=pad_obs_to,
    )

    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=min(n_episodes, len(ds)), replace=False)
    episodes = [ds[int(i)] for i in indices]

    per_horizon = {}
    for H in horizons:
        cem = CEM(
            model, action_dim=action_dim, horizon=H,
            n_samples=cem_samples, n_elites=cem_elites, n_iters=cem_iters,
            history_size=history_size, device=device,
        )
        env_successes = []
        cos_dist_terminals = []
        wall_t0 = time.time()
        for item in episodes:
            init_state_np = item["init_state"].numpy() if hasattr(item["init_state"], "numpy") else np.asarray(item["init_state"])
            goal_state_np = item["goal_state"].numpy() if hasattr(item["goal_state"], "numpy") else np.asarray(item["goal_state"])
            # Pad raw states (e.g. 9-D cheetah) to pad_obs_to so the state-projector branch is taken
            if init_state_np.shape[-1] < pad_obs_to:
                init_state_np = np.concatenate([init_state_np, np.zeros(pad_obs_to - init_state_np.shape[-1], dtype=np.float32)])
            if goal_state_np.shape[-1] < pad_obs_to:
                goal_state_np = np.concatenate([goal_state_np, np.zeros(pad_obs_to - goal_state_np.shape[-1], dtype=np.float32)])

            history_states = [init_state_np.copy() for _ in range(history_size)]
            z_history = encode_history(
                model, [torch.from_numpy(s).float() for s in history_states],
                action_dim, device,
            )
            z_goal = encode_obs(model, torch.from_numpy(goal_state_np).float(), action_dim, device)

            eval_budget = 50
            actions_taken = 0
            done = False
            while actions_taken < eval_budget:
                try:
                    seq = cem.plan(z_init, z_goal)
                except Exception:
                    seq = torch.zeros(H, action_dim)
                for a_idx in range(min(H, eval_budget - actions_taken)):
                    action = seq[a_idx].cpu().numpy().astype(np.float32)
                    if action.shape[-1] > env.spec.action_dim:
                        action = action[..., :env.spec.action_dim]
                    action = np.clip(action, env.spec.action_low, env.spec.action_high)
                    try:
                        _obs, _r, done, _info = env.step(action)
                    except Exception:
                        done = True
                    actions_taken += 1
                    if done:
                        break
                if done or actions_taken >= eval_budget:
                    break
                try:
                    with torch.no_grad():
                        a_window = seq[:history_size].unsqueeze(0)
                        nxt = model.predict(z_history.unsqueeze(0), a_window)
                        z_history = torch.cat([z_history[1:], nxt[0:1, -1]], dim=0)
                        z_init = z_history[-1]
                except Exception:
                    break

            try:
                final_state_np = env.get_state()
            except Exception:
                final_state_np = init_state_np
            # Pad to pad_obs_to so the state_projector branch is taken
            if final_state_np.shape[-1] < pad_obs_to:
                final_state_np = np.concatenate(
                    [final_state_np, np.zeros(pad_obs_to - final_state_np.shape[-1], dtype=np.float32)]
                )
            z_final = encode_obs(model, torch.from_numpy(final_state_np).float(), action_dim, device)
            cos = torch.nn.functional.cosine_similarity(z_final.unsqueeze(0), z_goal.unsqueeze(0))
            cos_dist_terminal = float((1.0 - cos.item()) / 2.0)
            cos_dist_terminals.append(cos_dist_terminal)
            try:
                env_success, _ = env.check_success(final_state_np, goal_state_np)
            except Exception:
                env_success = False
            env_successes.append(1.0 if env_success else 0.0)

        per_horizon[H] = {
            "horizon": H,
            "env_success": float(np.mean(env_successes)),
            "env_success_std": float(np.std(env_successes)),
            "mean_cos_dist_terminal": float(np.mean(cos_dist_terminals)),
            "mean_cos_dist_terminal_std": float(np.std(cos_dist_terminals)),
            "wall_time_sec": time.time() - wall_t0,
            "n_episodes": len(env_successes),
        }

    out = {
        "ckpt": ckpt_path,
        "env": env_kind,
        "n_episodes": n_episodes,
        "cem_samples": cem_samples,
        "cem_elites": cem_elites,
        "cem_iters": cem_iters,
        "horizons": horizons,
        "per_horizon": per_horizon,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  -> {out_path}")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env", required=True, choices=list(DMC_DATA.keys()))
    p.add_argument("--horizons", type=str, default="1,3,5,10,20")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--cem-samples", type=int, default=100)
    p.add_argument("--cem-elites", type=int, default=10)
    p.add_argument("--cem-iters", type=int, default=10)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--history-size", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    run_horizon_sweep(
        ckpt_path=args.ckpt,
        env_kind=args.env,
        horizons=horizons,
        n_episodes=args.n_episodes,
        cem_samples=args.cem_samples,
        cem_elites=args.cem_elites,
        cem_iters=args.cem_iters,
        goal_offset=args.goal_offset,
        history_size=args.history_size,
        device=args.device,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()

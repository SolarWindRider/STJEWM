"""Latent-goal MPC horizon sweep (v0.7.7 utility experiment 1).

For each (model, env, horizon), run the canonical squared-L2 latent CEM planner
with native bounded actions and observed-state replanning. Measures:
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

from code.core.cem import CEM, make_native_action_predict_hook
from code.core.encode import encode_obs
from code.core.envs import make_dmc_env
from code.data import load_dataset
from code.data.loaders import DATA_LOADER_PROTOCOL_VERSION
from code.eval.closed_loop import _set_env_state
from code.scripts.event_align import build_model


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
    """[FIX 2026-09-09] 5M 对齐:委托 trainer build_model(与训练同构),strict 载入由调用方负责。"""
    from code.train.train import build_model
    sd = ck_args.get("pad_obs_to") or state_dim
    ad = ck_args.get("action_dim") or action_dim
    return build_model(
        ck_args.get("model", "stjewm"), obs_dim=sd, action_dim=ad,
        n_layers=ck_args.get("n_layers", 4),
        readout_mode=ck_args.get("readout_mode", "hidden_leak"),
        embed_dim=ck_args.get("embed_dim"), hidden_dim=ck_args.get("hidden_dim"),
        mlp_hidden=ck_args.get("mlp_hidden"), mlp_layers=ck_args.get("mlp_layers"),
        image_size=ck_args.get("image_size", 0),
    ).to(device)
def run_horizon_sweep(
    ckpt_path: str,
    env_kind: str,
    horizons: List[int],
    n_episodes: int = 3,
    cem_samples: int = 100,
    cem_iters: int = 10,
    cem_elites: int = 10,
    goal_offset: int = 25,
    history_size: int = 1,
    device: str = "cpu",
    out_path: str = None,
) -> dict:
    """Evaluate identical offline init/goal pairs at every planning horizon."""
    import mujoco

    if out_path and Path(out_path).exists():
        raise FileExistsError(f"Archive the existing result before rerunning: {out_path}")
    if not horizons or min(horizons) < 1 or n_episodes < 1:
        raise ValueError("Positive horizons and n_episodes are required")
    if not 1 < cem_elites <= cem_samples or cem_iters < 1:
        raise ValueError("CEM requires 1 < n_elites <= n_samples and n_iters >= 1")
    if history_size < 1 or goal_offset < 1:
        raise ValueError("history_size and goal_offset must be positive")
    wall_t0_total = time.time()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    state_dim = next(
        (int(sd[key].shape[1]) for key in (
            "state_encoder.proj.0.weight", "state_projector.proj.0.weight",
            "state_projector.0.weight", "state_proj.0.weight",
        ) if key in sd),
        ck_args.get("pad_obs_to") or ck_args.get("state_dim"),
    )
    action_dim = next(
        (int(sd[key].shape[1]) for key in (
            "action_encoder.proj.0.weight", "action_encoder.0.weight",
            "action_proj.0.weight",
        ) if key in sd),
        ck_args.get("action_dim"),
    )
    if ck_args["model"] == "mlp_baseline":
        action_dim = int(sd["net.0.weight"].shape[1] - sd["state_proj.2.weight"].shape[0])
    if state_dim is None or action_dim is None:
        raise ValueError(f"Cannot determine trained input dimensions for {ckpt_path}")
    model = build_model(
        ck_args["model"], state_dim, action_dim, ck_args, state_dict=sd,
    )
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    data_path = DMC_DATA.get(env_kind)
    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found for env={env_kind}: {data_path}")
    # Keep native states for physics and success; pad only at the model boundary.
    ds = load_dataset(
        env_kind="dmc", path=data_path, history_size=history_size,
        goal_offset=goal_offset, max_windows=200,
    )
    if len(ds) < n_episodes:
        raise ValueError(f"{env_kind} has only {len(ds)} windows for {n_episodes} episodes")
    indices = np.random.default_rng(0).choice(len(ds), size=n_episodes, replace=False)
    episodes = [(int(i), ds[int(i)]) for i in indices]
    env = make_dmc_env(env_kind)
    per_horizon = {}
    eval_budget = 50
    try:
        native_state_dim = env.spec.obs_dim
        native_action_dim = env.spec.action_dim
        if ds.spec.obs_dim != native_state_dim:
            raise ValueError(
                f"{env_kind}: dataset state dimension {ds.spec.obs_dim} "
                f"does not match environment dimension {native_state_dim}"
            )
        if state_dim < native_state_dim or action_dim < native_action_dim:
            raise ValueError(f"Checkpoint inputs are smaller than the native {env_kind} inputs")

        def encode_native_state(state):
            state = np.asarray(state, dtype=np.float32)
            if state.shape != (native_state_dim,) or not np.isfinite(state).all():
                raise ValueError(f"{env_kind}: invalid native state: {state}")
            padded = torch.nn.functional.pad(
                torch.as_tensor(state, device=device), (0, state_dim - native_state_dim),
            )
            return encode_obs(model, padded, action_dim, device)

        predict_native_actions = make_native_action_predict_hook(
            action_dim, env.spec.action_low, env.spec.action_high, device,
        )

        for H in horizons:
            cem = CEM(
                model, action_dim=native_action_dim, horizon=H,
                n_samples=cem_samples, n_elites=cem_elites, n_iters=cem_iters,
                history_size=history_size, device=device,
                predict_hook=predict_native_actions,
            )
            episode_results = []
            wall_t0 = time.time()
            for episode_index, item in episodes:
                init_state = item["init_state"].numpy()
                goal_state = item["goal_state"].numpy()
                # Saved trajectories contain qpos, not qvel. Every comparison
                # starts from the same qpos with zero velocity and a fresh clock.
                torch.manual_seed(episode_index)
                env.reset(seed=episode_index)
                _set_env_state(env, init_state)
                mujoco.mj_forward(env._model, env._data)
                if not np.allclose(env.get_state(), init_state, rtol=0, atol=1e-6):
                    raise RuntimeError(f"{env_kind}: failed to restore episode {episode_index}")
                z_goal = encode_native_state(goal_state)
                actions_taken = 0
                planning_calls = 0
                done = False
                while actions_taken < eval_budget:
                    # Replanning is closed-loop: never substitute a predicted
                    # latent for the observation after the executed action chunk.
                    z_init = encode_native_state(env.get_state())
                    seq = cem.plan(z_init, z_goal)
                    if not torch.isfinite(seq).all():
                        raise FloatingPointError(f"{env_kind}: CEM returned non-finite actions")
                    planning_calls += 1
                    for action in seq[:min(H, eval_budget - actions_taken)].cpu().numpy():
                        action = np.clip(action, env.spec.action_low, env.spec.action_high)
                        _, _, done, _ = env.step(action.astype(np.float32, copy=False))
                        actions_taken += 1
                        if done:
                            break
                    if done:
                        break

                final_state = env.get_state()
                z_final = encode_native_state(final_state)
                cos = torch.nn.functional.cosine_similarity(
                    z_final.unsqueeze(0), z_goal.unsqueeze(0),
                )
                cos_dist_terminal = float((1.0 - cos.item()) / 2.0)
                env_success, phys_dist = env.check_success(final_state, goal_state)
                if not np.isfinite(cos_dist_terminal) or not np.isfinite(phys_dist):
                    raise FloatingPointError(f"{env_kind}: non-finite terminal metrics")
                source_index = (
                    int(ds._starts[episode_index]) if ds._starts is not None else episode_index
                )
                episode_results.append({
                    "episode_index": episode_index,
                    "source_index": source_index,
                    "goal_source_index": source_index + goal_offset,
                    "seed": episode_index,
                    "init_state": init_state.tolist(),
                    "goal_state": goal_state.tolist(),
                    "final_state": final_state.tolist(),
                    "actions_taken": actions_taken,
                    "planning_calls": planning_calls,
                    "env_success": bool(env_success),
                    "phys_dist": float(phys_dist),
                    "cos_dist_terminal": cos_dist_terminal,
                })

            env_successes = [e["env_success"] for e in episode_results]
            cos_dist_terminals = [e["cos_dist_terminal"] for e in episode_results]
            phys_dists = [e["phys_dist"] for e in episode_results]
            per_horizon[H] = {
                "horizon": H,
                "env_success": float(np.mean(env_successes)),
                "env_success_std": float(np.std(env_successes)),
                "mean_cos_dist_terminal": float(np.mean(cos_dist_terminals)),
                "mean_cos_dist_terminal_std": float(np.std(cos_dist_terminals)),
                "mean_phys_dist_terminal": float(np.mean(phys_dists)),
                "mean_phys_dist_terminal_std": float(np.std(phys_dists)),
                "wall_time_sec": time.time() - wall_t0,
                "n_episodes": len(episode_results),
                "per_episode": episode_results,
            }

        out = {
            "protocol_version": 2,
            "ckpt": ckpt_path,
            "data_loader_protocol": DATA_LOADER_PROTOCOL_VERSION,
            "checkpoint_data_protocol_version": ck.get("data_protocol_version"),
            "model": "LeWM" if ck_args["model"] == "lewm_baseline" else ck_args["model"],
            "env": env_kind,
            "n_episodes": n_episodes,
            "cem_samples": cem_samples,
            "cem_elites": cem_elites,
            "cem_iters": cem_iters,
            "cem_cost": "squared_l2",
            "cem_action_dim": native_action_dim,
            "model_action_dim": action_dim,
            "model_state_dim": state_dim,
            "latent_representation": "forward.emb, one observed frame, zero action",
            "goal_offset": goal_offset,
            "history_size": history_size,
            "eval_budget": eval_budget,
            "episode_sampling_seed": 0,
            "episode_indices": indices.tolist(),
            "candidate_windows": len(ds),
            "initial_velocity": "zero (not stored in the offline dataset)",
            "replanning": "observed state after executing up to H actions",
            "physical_metric": "native_state_l2 / sqrt(native_state_dim)",
            "physical_state_dim": native_state_dim,
            "success_tolerance": env._success_tol,
            "horizons": horizons,
            "per_horizon": per_horizon,
            "wall_time_sec_total": time.time() - wall_t0_total,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    finally:
        env.close()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "x") as f:
            json.dump(out, f, indent=2, allow_nan=False)
        print(f"  -> {out_path}")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env", required=True, choices=list(DMC_DATA.keys()))
    p.add_argument("--horizons", type=str, default="1,3,5,10,20")
    p.add_argument("--n-episodes", type=int, default=3)
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

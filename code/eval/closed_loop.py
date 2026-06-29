"""Single canonical closed-loop evaluation.

Unified protocol:
    1. Build a BaseEnv (PushT/TwoRoom/Cube/Reacher/Gym/...)
    2. Build a CEM planner (CEM samples/elites/iters from LeWM paper)

        a. Reset env, get init state
        b. Encode history (history_size frames from dataset, NOT random init)
        c. Sample goal state from dataset at t+goal_offset in same trajectory
        d. Encode goal
        e. CEM plan + receding-horizon execution in the env
        f. At end: report cos_dist (LeWM paper metric) + env-native success
    4. Save JSON with both metrics

Usage:
    python -m code.eval.closed_loop --env pusht --ckpt .../final.pt --data .../pusht.h5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")

from code.core.cem import CEM
from code.core.encode import encode_history, encode_obs, assert_model_compatible, assert_readout_mode
from code.core.envs import (
    PushTEnv, TwoRoomEnv, OGBCubeEnv, OGBenchSceneEnv, ReacherEnv,
    make_gym_env, make_dmc_env, BaseEnv,
)
from code.data import load_dataset


# ============================================================
# Env factory (string -> BaseEnv instance)
# ============================================================
def make_env(env_kind: str, data_path: str = None) -> BaseEnv:
    if env_kind == "pusht":
        return PushTEnv()
    if env_kind == "tworoom" or env_kind == "tworoom_long":
        # tworoom_long uses the same env, but eval_closed_loop forces
        # goal_offset=200 to make the planning horizon much harder.
        return TwoRoomEnv()
    if env_kind == "pusht_ood":
        # Same env as pusht; OOD split is applied in _load_ood_split
        return PushTEnv()
    if env_kind == "cube":
        return OGBCubeEnv()
    if env_kind == "scene":
        return OGBenchSceneEnv()
    if env_kind == "reacher":
        return ReacherEnv()
    if env_kind in (
        "cartpole", "pendulum", "finger", "ball_in_cup", "cheetah",
        "walker", "hopper", "quadruped", "humanoid", "humanoid_cmu",
        "dog", "fish", "stacker", "manipulator",
    ):
        return make_dmc_env(env_kind)
    if env_kind == "cartpole_flicker" or env_kind == "flickering_dmc":
        # FlickeringDMCEnv: obs randomly masked to zero with prob 0.5
        from code.core.envs.dmc_env import FlickeringDMCEnv
        base_kind = "cartpole" if env_kind == "cartpole_flicker" else "cartpole"
        return FlickeringDMCEnv(base_kind, mask_ratio=0.5)
    if env_kind.startswith("vel_hidden_") or env_kind.endswith("_velhidden"):
        # e.g. "vel_hidden_cheetah" or "cheetah_velhidden" -> make_vel_hidden_env("cheetah")
        from code.core.envs.dmc_env import make_vel_hidden_env
        if env_kind.startswith("vel_hidden_"):
            sub = env_kind.replace("vel_hidden_", "")
        else:
            sub = env_kind.replace("_velhidden", "")
        return make_vel_hidden_env(sub)
    if env_kind in ("cartpole_v1", "acrobot", "pendulum_v1", "mountaincar", "mountaincar_cont"):
        eid = {
            "cartpole_v1": "swm/CartPoleControl-v1",
            "acrobot": "swm/AcrobotControl-v1",
            "pendulum_v1": "swm/PendulumControl-v1",
            "mountaincar": "swm/MountainCarControl-v0",
            "mountaincar_cont": "swm/MountainCarContinuousControl-v0",
        }[env_kind]
        return make_gym_env(eid)
    raise ValueError(f"Unknown env_kind: {env_kind}")

# ============================================================
# Result
# ============================================================
@dataclass
class ClosedLoopResult:
    env_id: str
    n_episodes: int
    n_seeds: int
    cem_samples: int
    cem_elites: int
    cem_iters: int
    horizon: int
    eval_budget: int
    goal_offset: int
    history_size: int
    success_rate_lewm: float         # mean(cos_dist < 0.1)
    success_rate_lewm_std: float
    success_rate_env: float          # mean(env-native success)
    success_rate_env_std: float
    mean_cos_dist: float
    mean_cos_dist_std: float
    mean_phys_dist: float
    mean_phys_dist_std: float
    per_seed: List[Dict] = field(default_factory=list)
    per_episode: List[Dict] = field(default_factory=list)
    wall_time_sec: float = 0.0


# ============================================================
# Single canonical eval
# ============================================================
def eval_closed_loop(
    model,
    env: BaseEnv,
    data_path: str,
    n_episodes: int = 50,
    n_seeds: int = 3,
    cem_samples: int = 300,
    cem_elites: int = 30,
    cem_iters: int = 10,
    horizon: int = 5,
    eval_budget: int = 50,
    goal_offset: int = 25,
    history_size: int = 3,
    success_threshold_cos: float = 0.1,
    device: str = "cuda",
    goal_offset_override: Optional[int] = None,
    split: str = "in_dist",
) -> ClosedLoopResult:
    """Run closed-loop CEM planning eval. LeWM-paper protocol + env-native.

    Each episode:
        - Sample (init_state, goal_state) from dataset trajectory (in-distribution)
        - Encode init history (history_size frames ending at init)
        - Encode goal (single frame at t+goal_offset)
        - CEM plan + receding-horizon step
        - Report: cos_dist(z_final, z_goal), env-native success
    """
    model = model.to(device).eval()
    assert_model_compatible(model)
    # Membrane-forbidden protocol: planner/predictor should be in TRACE_ONLY mode.
    # For ablation / comparison purposes, we WARN for other STJEWM modes (hidden_leak,
    # spike_only, no_trace) and raise only for LeWM-style baselines (which do not
    # satisfy any membrane-forbidden contract).
    if hasattr(model, "readout_mode"):
        from code.stjewm import ReadoutMode
        if model.readout_mode != ReadoutMode.TRACE_ONLY:
            import warnings
            warnings.warn(
                f"[closed_loop] model readout_mode={model.readout_mode} is NOT trace_only. "
                f"This is allowed for ablation, but the trace-only protocol expects hidden_leak=0."
            )
    action_dim = env.spec.action_dim
    action_low = env.spec.action_low
    action_high = env.spec.action_high
    # B2: Stress env override — if the env (or caller) requests a
    # different goal_offset (e.g. tworoom_long=200), honor it.
    effective_goal_offset = goal_offset_override if goal_offset_override is not None else goal_offset
    # Load dataset for sampling init/goal
    ds, state_dim = _load_eval_dataset(
        env, data_path, history_size, effective_goal_offset, split=split,
    )
    # Note: we do NOT rebuild the model here — caller is expected to build
    # the model with the correct state_dim (e.g. read from ckpt). state_dim
    # is used only for sanity checks below.
    # Build CEM once
    cem = CEM(
        model, action_dim=action_dim, horizon=horizon,
        n_samples=cem_samples, n_elites=cem_elites, n_iters=cem_iters,
        history_size=history_size, device=device,
    )

    wall_t0 = time.time()
    per_seed_results = []
    per_episode_all = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed * 7919 + 42)
        # Sample init/goal pairs from dataset
        if ds is not None:
            N = len(ds)
            episode_indices = rng.choice(N, size=min(n_episodes, N), replace=False)
        else:
            episode_indices = list(range(n_episodes))

        seed_episodes = []
        for ep_idx in episode_indices:
            item = ds[int(ep_idx)]
            init_state_np = item["init_state"].numpy() if hasattr(item["init_state"], "numpy") else np.asarray(item["init_state"])
            goal_state_np = item["goal_state"].numpy() if hasattr(item["goal_state"], "numpy") else np.asarray(item["goal_state"])

            # Reset the env first (needed for envs that don't support direct state setting,
            # e.g. TwoRoom/Cube — without reset, get_state() returns NaN).
            # Use a per-episode seed derived from the dataset ep_idx for reproducibility.
            try:
                env.reset(seed=int(ep_idx) + seed * 1000)
            except Exception:
                pass
            # Set the env to the init state (best-effort, no-op for envs that don't support it)
            try:
                _set_env_state(env, init_state_np)
            except Exception:
                # Env may not support direct state setting
                pass
            history_states = [init_state_np.copy() for _ in range(history_size)]
            try:
                z_history = encode_history(model, [torch.from_numpy(s).float() for s in history_states], action_dim, device)
            except Exception:
                z_history = encode_obs(model, torch.from_numpy(init_state_np).float(), action_dim, device).unsqueeze(0).repeat(history_size, 1)

            # Encode goal
            z_goal = encode_obs(model, torch.from_numpy(goal_state_np).float(), action_dim, device)

            # CEM plan + step
            actions_taken = 0
            best_actions = None
            t_start = time.time()
            # z_history is (history_size, D); for CEM.plan we need z_init (D,) and z_goal (D,).
            z_init = z_history[-1]  # use the last frame as the "current" latent
            while actions_taken < eval_budget:
                seq = cem.plan(z_init, z_goal)  # (H, A)
                for a_idx in range(min(horizon, eval_budget - actions_taken)):
                    action = seq[a_idx].cpu().numpy().astype(np.float32)
                    action = np.clip(action, action_low, action_high)
                    try:
                        _obs, _r, done, _info = env.step(action)
                    except Exception:
                        done = True
                    actions_taken += 1
                    if done:
                        break
                if done or actions_taken >= eval_budget:
                    break
                # Roll history forward
                try:
                    with torch.no_grad():
                        a_window = seq[:history_size].unsqueeze(0)
                        nxt = model.predict(z_history.unsqueeze(0), a_window)
                        z_history = torch.cat([z_history[1:], nxt[0:1, -1]], dim=0)
                        z_init = z_history[-1]
                except Exception:
                    break
            plan_time = time.time() - t_start

            # Compute final state and metrics
            try:
                final_state_np = env.get_state()
            except Exception:
                final_state_np = init_state_np

            try:
                z_final = encode_obs(model, torch.from_numpy(final_state_np).float(), action_dim, device)
            except Exception:
                z_final = z_goal  # fallback
            cos = torch.nn.functional.cosine_similarity(
                z_final.unsqueeze(0), z_goal.unsqueeze(0)
            )
            cos_dist = float((1.0 - cos.item()) / 2.0)
            lewm_success = cos_dist < success_threshold_cos

            env_success, phys_dist = env.check_success(final_state_np, goal_state_np)

            ep_dict = {
                "seed": seed,
                "episode_idx": int(ep_idx),
                "cos_dist": cos_dist,
                "lewm_success": bool(lewm_success),
                "env_success": bool(env_success),
                "phys_dist": float(phys_dist),
                "plan_time_sec": plan_time,
            }
            seed_episodes.append(ep_dict)
            per_episode_all.append(ep_dict)

        if seed_episodes:
            lewm_succ = np.mean([e["lewm_success"] for e in seed_episodes])
            env_succ = np.mean([e["env_success"] for e in seed_episodes])
            mean_cos = np.mean([e["cos_dist"] for e in seed_episodes])
            mean_phys = np.mean([e["phys_dist"] for e in seed_episodes])
            per_seed_results.append({
                "seed": seed,
                "n": len(seed_episodes),
                "success_rate_lewm": float(lewm_succ),
                "success_rate_env": float(env_succ),
                "mean_cos_dist": float(mean_cos),
                "mean_phys_dist": float(mean_phys),
            })

    lewm_arr = np.array([s["success_rate_lewm"] for s in per_seed_results])
    env_arr = np.array([s["success_rate_env"] for s in per_seed_results])
    cos_arr = np.array([s["mean_cos_dist"] for s in per_seed_results])
    phys_arr = np.array([s["mean_phys_dist"] for s in per_seed_results])

    return ClosedLoopResult(
        env_id=env.spec.env_id,
        n_episodes=len(per_episode_all),
        n_seeds=len(per_seed_results),
        cem_samples=cem_samples,
        cem_elites=cem_elites,
        cem_iters=cem_iters,
        horizon=horizon,
        eval_budget=eval_budget,
        goal_offset=goal_offset,
        history_size=history_size,
        success_rate_lewm=float(lewm_arr.mean()) if len(lewm_arr) > 0 else float("nan"),
        success_rate_lewm_std=float(lewm_arr.std()) if len(lewm_arr) > 0 else float("nan"),
        success_rate_env=float(env_arr.mean()) if len(env_arr) > 0 else float("nan"),
        success_rate_env_std=float(env_arr.std()) if len(env_arr) > 0 else float("nan"),
        mean_cos_dist=float(cos_arr.mean()) if len(cos_arr) > 0 else float("nan"),
        mean_cos_dist_std=float(cos_arr.std()) if len(cos_arr) > 0 else float("nan"),
        mean_phys_dist=float(phys_arr.mean()) if len(phys_arr) > 0 else float("nan"),
        mean_phys_dist_std=float(phys_arr.std()) if len(phys_arr) > 0 else float("nan"),
        per_seed=per_seed_results,
        per_episode=per_episode_all,
        wall_time_sec=time.time() - wall_t0,
    )


# ============================================================
# Helpers
# ============================================================
def _load_eval_dataset(env, data_path, history_size, goal_offset, split: str = "in_dist"):
    """Load the eval dataset for sampling init/goal pairs.

    Returns (ds, state_dim). ds may be None if no dataset is available
    (eval will use random init from env instead).
    """
    use_env_based = (data_path is None or data_path == "(none)" or data_path == "")
    if use_env_based:
        env_kind_for_load = _infer_env_kind(env)
        if env_kind_for_load in ("ogb_cube", "ogb_scene"):
            ds = load_dataset(env_kind_for_load + "_env", n_episodes=20, max_steps_per_ep=50,
                              history_size=history_size, goal_offset=goal_offset, seed=42)
            return ds, ds.spec.obs_dim
        elif env_kind_for_load.startswith("mujoco/"):
            # DMC / Reacher: can't collect, need offline data
            return None, env.spec.obs_dim
        return None, env.spec.obs_dim
    if data_path.endswith(".npz") or data_path.endswith(".h5"):
        ds = load_dataset(_infer_env_kind(env), path=data_path,
                          history_size=history_size, goal_offset=goal_offset)
        # B4: OOD split for held-out goal states
        if split == "unseen_goal" and hasattr(ds, "spec"):
            ds = _make_unseen_goal_subset(ds)
        return ds, ds.spec.obs_dim
    # gym_live: collect from env
    ds = load_dataset("gym_live", path=data_path, history_size=history_size,
                      goal_offset=goal_offset, n_episodes=20, seed=42)
    return ds, ds.spec.obs_dim


def _make_unseen_goal_subset(ds) -> "torch.utils.data.Subset":
    """Return a Subset of `ds` that only contains the held-out 20% of windows.

    The split is by dataset index (deterministic). The eval will use these
    windows as (init, goal) pairs — the goal states will be from a region
    of the trajectory distribution not seen during training, so the
    model has to generalize.
    """
    n = len(ds)
    if n <= 10:
        return ds  # too small to split
    cut = int(n * 0.8)
    indices = list(range(cut, n))
    return torch.utils.data.Subset(ds, indices)


def _infer_env_kind(env: BaseEnv) -> str:
    eid = env.spec.env_id
    if "PushT" in eid: return "pusht"
    if "TwoRoom" in eid: return "tworoom"
    if "OGBCube" in eid: return "ogb_cube"
    if "OGBScene" in eid: return "ogb_scene"
    if "reacher" in eid.lower(): return "reacher_4d"  # 4D state with target
    if "mujoco/" in eid:
        return "mujoco_3d"
    return "mujoco_3d"

def _set_env_state(env: BaseEnv, state: np.ndarray) -> None:
    """Best-effort: set the env to a specific state.

    For DMCStateEnv: set qpos[:nq] from state (only the qpos part — no target).
    For ReacherEnv: qpos[:2] + target[:2].
    """
    import mujoco
    if isinstance(env, ReacherEnv):
        env._data.qpos[:2] = state[:2]
        env._data.qvel[:] = 0.0
        env._model.geom_pos[env._target_id, :2] = state[2:4]
        mujoco.mj_forward(env._model, env._data)
        return
    # DMC state env: set qpos directly
    if hasattr(env, "_model") and hasattr(env, "_data") and hasattr(env, "_nq"):
        if env._expand_pendulum:
            # state is (cos, sin); convert back to single angle
            angle = float(np.arctan2(state[1], state[0]))
            env._data.qpos[0] = angle
        else:
            env._data.qpos[: env._nq] = state[: env._nq]
        env._data.qvel[:] = 0.0
    return
# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="pusht|tworoom|cube|reacher|cartpole|...")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-elites", type=int, default=30)
    p.add_argument("--cem-iters", type=int, default=10)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--goal-offset", type=int, default=25,
                   help="Default goal_offset (overridden by stress env like tworoom_long)")
    p.add_argument("--split", choices=["in_dist", "unseen_goal"], default="in_dist",
                   help="Dataset split: 'in_dist' (default) or 'unseen_goal' (B4: held-out)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[closed_loop/{args.env}] ckpt={args.ckpt}, data={args.data}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build env
    env = make_env(args.env, args.data)

    # Build model
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    state_dim = env.spec.obs_dim
    action_dim = env.spec.action_dim
    if ck_args.get("model", "stjewm") == "lewm_baseline":
        from code.lewm_transformer_baseline import LeWMTransformerBaseline
        embed_dim = ck_args.get("embed_dim", 256)  # default matches the 5.07M LeWM-style we trained
        model = LeWMTransformerBaseline(state_dim=state_dim, action_dim=action_dim, embed_dim=embed_dim,
                                         num_layers=ck_args.get("n_layers", 4))
    else:
        from code.stjewm import STJEWM
        n_layers = ck_args.get("n_layers", 4)
        # ReadoutMode: read from ckpt args (added by Workstream A)
        ck_readout_mode = ck_args.get("readout_mode", "hidden_leak")
        model = STJEWM(
            d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
            state_dim=state_dim, cell_n_layers=n_layers, n_d=3,
            trace_beta=0.9, freeze_encoder=True,
            readout_mode=ck_readout_mode,
        )
    model.load_state_dict(ck["model"])

    # B2: stress env goal_offset override (tworoom_long -> 200)
    goal_offset_override = None
    if args.env == "tworoom_long":
        goal_offset_override = 200

    # Run eval
    result = eval_closed_loop(
        model, env, args.data,
        n_episodes=args.n_episodes,
        n_seeds=args.n_seeds,
        goal_offset=args.goal_offset, history_size=args.history_size,
        horizon=args.horizon, eval_budget=args.eval_budget,
        device=device,
        goal_offset_override=goal_offset_override,
        split=args.split,
    )
    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\n=== FINAL (closed_loop/{args.env}) ===")
    print(f"  LeWM SR: {result.success_rate_lewm:.3f} ± {result.success_rate_lewm_std:.3f}")
    print(f"  Env-native SR: {result.success_rate_env:.3f} ± {result.success_rate_env_std:.3f}")
    print(f"  Mean cos_dist: {result.mean_cos_dist:.4f}")
    print(f"  Mean phys_dist: {result.mean_phys_dist:.4f}")
    print(f"  Saved to {args.out}")


if __name__ == "__main__":
    main()

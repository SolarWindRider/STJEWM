"""Single canonical closed-loop evaluation.

Unified protocol:
    1. Build a BaseEnv (PushT/TwoRoom/Cube/Reacher/Gym/...)
    2. Build a CEM planner (CEM samples/elites/iters from LeWM paper)

        a. Reset env, get init state
        b. Restore the sampled initial state and encode observed history
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
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")

from code.core.cem import CEM, make_native_action_predict_hook
from code.core.encode import encode_history, encode_obs
from code.core.envs import (
    PushTEnv, TwoRoomEnv, OGBCubeEnv, OGBenchSceneEnv, ReacherEnv,
    make_gym_env, make_dmc_env, BaseEnv, DMCStateEnv, FlickeringDMCEnv,
)
from code.data import load_dataset
from code.data.loaders import DATA_LOADER_PROTOCOL_VERSION


# ============================================================
# Env factory (string -> BaseEnv instance)
# ============================================================
def make_env(env_kind: str, data_path: str = None, *, flicker_mask_ratio: float = None,
             delay_length: int = None, cue_visibility: int = None) -> BaseEnv:
    if env_kind == "pusht":
        return PushTEnv()
    if env_kind == "tworoom" or env_kind == "tworoom_long":
        # tworoom_long uses the same env; eval_closed_loop honors --goal-offset
        # for the long variant so the caller can sweep difficulty.
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
        # FlickeringDMCEnv: obs randomly masked to zero with prob mask_ratio.
        # mask_ratio is a per-eval knob so the caller can sweep difficulty.
        from code.core.envs.dmc_env import FlickeringDMCEnv
        base_kind = "cartpole"
        ratio = 0.5 if flicker_mask_ratio is None else float(flicker_mask_ratio)
        return FlickeringDMCEnv(base_kind, mask_ratio=ratio)
    if env_kind.endswith("_qpos_masked"):
        from code.core.envs.dmc_env import make_qpos_mask_env
        return make_qpos_mask_env(env_kind.removesuffix("_qpos_masked"))
    if env_kind in ("cartpole_v1", "acrobot", "pendulum_v1", "mountaincar", "mountaincar_cont"):
        eid = {
            "cartpole_v1": "swm/CartPoleControl-v1",
            "acrobot": "swm/AcrobotControl-v1",
            "pendulum_v1": "swm/PendulumControl-v1",
            "mountaincar": "swm/MountainCarControl-v0",
            "mountaincar_cont": "swm/MountainCarContinuousControl-v0",
        }[env_kind]
        return make_gym_env(eid)
    if env_kind == "delayed_t_maze":
        from code.core.envs.delayed_t_maze import make_delayed_t_maze
        return make_delayed_t_maze(
            delay_length=delay_length or 50,
            cue_visibility=cue_visibility or 3,
        )
    if env_kind == "event_window":
        from code.core.envs.event_window import make_event_window
        return make_event_window()
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
    success_rate_lewm: float         # mean(cos_dist < 0.1) — v0.7.13: legacy, mostly noisy
    success_rate_lewm_std: float
    success_rate_lewm_005: float = 0.0  # mean(cos_dist < 0.05) — calibrated threshold
    success_rate_lewm_001: float = 0.0  # mean(cos_dist < 0.01) — strict threshold
    success_rate_env: float = 0.0          # mean(env-native success)
    success_rate_env_std: float = 0.0
    mean_cos_dist: float = 0.0
    mean_cos_dist_std: float = 0.0
    mean_phys_dist: float = 0.0
    mean_phys_dist_std: float = 0.0
    mean_reward: float = 0.0
    mean_reward_std: float = 0.0
    per_seed: List[Dict] = field(default_factory=list)
    per_episode: List[Dict] = field(default_factory=list)
    wall_time_sec: float = 0.0
    protocol: Dict[str, Any] = field(default_factory=dict)


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
    pad_obs_to: Optional[int] = None,
    model_action_dim: Optional[int] = None,
    step_predict_hook=None,
    cem_predict_hook=None,
    seed_start: int = 0,
) -> ClosedLoopResult:
    """Plan from observed states; evaluate physical goals in native units."""
    if min(n_episodes, n_seeds, horizon, eval_budget, history_size) < 1:
        raise ValueError("Episode, seed, horizon, budget and history counts must be positive")
    if seed_start < 0:
        raise ValueError("seed_start must be nonnegative")
    if step_predict_hook is not None and cem_predict_hook is not None:
        raise ValueError("Use either a timed predictor intervention or a CEM-wide intervention")
    physical_env = env
    qpos_mask_indices = set()
    while True:
        qpos_mask_indices.update(getattr(physical_env, "observation_mask_indices", ()))
        if not hasattr(physical_env, "_base"):
            break
        physical_env = physical_env._base
    qpos_mask_indices = sorted(qpos_mask_indices)
    native_state_dim = physical_env.spec.obs_dim
    action_dim = physical_env.spec.action_dim
    action_low = physical_env.spec.action_low
    action_high = physical_env.spec.action_high
    if model_action_dim is None:
        model_action_dim = getattr(model, "action_dim", action_dim)
    effective_goal_offset = goal_offset_override if goal_offset_override is not None else goal_offset
    ds, state_dim = _load_eval_dataset(
        physical_env, data_path, history_size, effective_goal_offset,
        split=split, pad_obs_to=pad_obs_to,
    )
    if ds is None or len(ds) < n_episodes:
        raise ValueError(f"Need {n_episodes} valid offline init/goal windows for {env.spec.env_id}")
    state_dim = pad_obs_to or state_dim
    if state_dim < native_state_dim:
        raise ValueError(f"Model state dimension {state_dim} < native dimension {native_state_dim}")
    env_kind = _infer_env_kind(physical_env)
    state_scale = {"pusht": 500.0, "tworoom": 250.0}.get(env_kind, 1.0)
    def physical_state():
        if isinstance(physical_env, DMCStateEnv):
            return DMCStateEnv.get_state(physical_env)
        return physical_env.get_state()

    def apply_observation_mask(state: np.ndarray) -> np.ndarray:
        if not qpos_mask_indices:
            return np.asarray(state, dtype=np.float32)
        masked = np.asarray(state, dtype=np.float32).copy()
        masked[qpos_mask_indices] = 0.0
        return masked

    def model_state(raw_state):
        native = apply_observation_mask(
            np.asarray(raw_state, dtype=np.float32).reshape(-1)[:native_state_dim],
        )
        if native.shape != (native_state_dim,) or not np.isfinite(native).all():
            raise ValueError(f"Invalid native state for {env.spec.env_id}: {native}")
        scaled = native / state_scale
        return np.pad(scaled, (0, state_dim - native_state_dim))

    def observation_state(obs):
        if isinstance(obs, dict):
            for key in ("state", "observation"):
                if key in obs:
                    return obs[key]
            return env.get_state()
        return obs

    current_env_step = 0

    def predict_for_step(model, ctx_emb, ctx_act):
        if step_predict_hook is not None:
            return step_predict_hook(model, ctx_emb, ctx_act, current_env_step)
        if cem_predict_hook is not None:
            return cem_predict_hook(model, ctx_emb, ctx_act)
        return model.predict(ctx_emb, ctx_act)

    cem = CEM(
        model, action_dim=action_dim, horizon=horizon,
        n_samples=cem_samples, n_elites=cem_elites, n_iters=cem_iters,
        history_size=history_size, device=device,
        predict_hook=make_native_action_predict_hook(
            model_action_dim, action_low, action_high, device, predict_for_step,
        ),
    )
    model.eval()
    wall_t0 = time.time()
    per_seed_results = []
    per_episode_all = []

    for seed in range(seed_start, seed_start + n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed * 7919 + 42)
        episode_indices = rng.choice(len(ds), size=n_episodes, replace=False)
        seed_episodes = []
        for ep_idx in episode_indices:
            item = ds[int(ep_idx)]
            init_state = np.asarray(item["init_state"], dtype=np.float32)[:native_state_dim] * state_scale
            goal_state = np.asarray(item["goal_state"], dtype=np.float32)[:native_state_dim] * state_scale
            episode_seed = int(ep_idx) + seed * 1000
            env.reset(seed=episode_seed)
            _set_env_state(env, init_state, sim_state=item.get("sim_state"))
            restored = physical_state()
            if isinstance(physical_env, PushTEnv):
                physical_env._env.unwrapped._set_goal_state(goal_state)
            if not np.allclose(restored, init_state, rtol=1e-5, atol=1e-4):
                raise RuntimeError(f"Failed to restore {env.spec.env_id} episode {int(ep_idx)}")
            history_states = [model_state(env.get_state()) for _ in range(history_size)]
            z_goal = encode_obs(model, torch.from_numpy(model_state(goal_state)), model_action_dim, device)
            actions_taken = 0
            planning_calls = 0
            episode_reward = 0.0
            t_start = time.time()
            done = False
            final_observed_state = env.get_state()
            while actions_taken < eval_budget:
                current_env_step = actions_taken
                z_history = encode_history(
                    model, [torch.from_numpy(s) for s in history_states], model_action_dim, device,
                )
                seq = cem.plan(z_history, z_goal)
                if not torch.isfinite(seq).all():
                    raise FloatingPointError(f"CEM returned non-finite controls for {env.spec.env_id}")
                planning_calls += 1
                for candidate in seq[:min(horizon, eval_budget - actions_taken)].cpu().numpy():
                    action = np.clip(candidate, action_low, action_high).astype(np.float32, copy=False)
                    obs, reward, done, info = env.step(action)
                    actions_taken += 1
                    episode_reward += float(reward)
                    final_observed_state = observation_state(obs)
                    history_states = history_states[1:] + [model_state(final_observed_state)]
                    if done:
                        break
                if done:
                    break
            plan_time = time.time() - t_start
            final_state = physical_state()
            z_final = encode_obs(
                model, torch.from_numpy(model_state(final_observed_state)), model_action_dim, device,
            )
            cos = torch.nn.functional.cosine_similarity(z_final.unsqueeze(0), z_goal.unsqueeze(0))
            cos_dist = float((1.0 - cos.item()) / 2.0)
            if isinstance(physical_env, ReacherEnv):
                # This benchmark asks for the sampled future configuration,
                # not equality of the unchanged/synthetic target coordinates.
                from code.core.envs.dmc_env import DMC_ENVS
                phys_dist = float(np.linalg.norm(final_state[:2] - goal_state[:2]) / np.sqrt(2))
                env_success = phys_dist < DMC_ENVS["reacher"][4]
            else:
                env_success, phys_dist = physical_env.check_success(final_state, goal_state)
            if not np.isfinite([cos_dist, phys_dist, episode_reward]).all():
                raise FloatingPointError(f"Non-finite terminal metrics for {env.spec.env_id}")
            ep_dict = {
                "seed": seed,
                "episode_idx": int(ep_idx),
                "initialization_seed": episode_seed,
                "init_state": init_state.tolist(),
                "goal_state": goal_state.tolist(),
                "final_state": np.asarray(final_state).tolist(),
                "actions_taken": actions_taken,
                "planning_calls": planning_calls,
                "cos_dist": cos_dist,
                "lewm_success": cos_dist < success_threshold_cos,
                "lewm_success_at_005": cos_dist < 0.05,
                "lewm_success_at_001": cos_dist < 0.01,
                "env_success": bool(env_success),
                "phys_dist": float(phys_dist),
                "plan_time_sec": plan_time,
                "episode_reward": episode_reward,
            }
            if "sim_state" in item:
                ep_dict["initial_sim_state"] = item["sim_state"]
                ep_dict["source_index"] = item["source_index"]
                ep_dict["source_episode"] = item["source_episode"]
            seed_episodes.append(ep_dict)
            per_episode_all.append(ep_dict)

        per_seed_results.append({
            "seed": seed,
            "n": len(seed_episodes),
            "success_rate_lewm": float(np.mean([e["lewm_success"] for e in seed_episodes])),
            "success_rate_lewm_005": float(np.mean([e["lewm_success_at_005"] for e in seed_episodes])),
            "success_rate_lewm_001": float(np.mean([e["lewm_success_at_001"] for e in seed_episodes])),
            "success_rate_env": float(np.mean([e["env_success"] for e in seed_episodes])),
            "mean_cos_dist": float(np.mean([e["cos_dist"] for e in seed_episodes])),
            "mean_phys_dist": float(np.mean([e["phys_dist"] for e in seed_episodes])),
            "mean_reward": float(np.mean([e["episode_reward"] for e in seed_episodes])),
        })

    def mean_metric(key):
        return float(np.mean([s[key] for s in per_seed_results]))

    def std_metric(key):
        return float(np.std([s[key] for s in per_seed_results]))

    physical_metric = {
        "units": "native (no model padding or normalization)",
        "state_dim": native_state_dim,
        "definition": "environment-native success",
        "tolerance": getattr(physical_env, "_success_tol", None),
    }
    if isinstance(physical_env, ReacherEnv):
        physical_metric.update(definition="qpos goal RMS < tolerance", state_dim=2, tolerance=0.05)
    elif hasattr(physical_env, "_nq"):
        physical_metric["definition"] = "qpos goal RMS < tolerance"
        if getattr(physical_env, "_expand_pendulum", False):
            physical_metric["definition"] = "goal angle distance < tolerance (radians)"
    elif isinstance(physical_env, PushTEnv):
        physical_metric.update(
            definition="agent+block position L2 < 20 px and wrapped block angle error < pi/9 radians",
            position_tolerance=20.0, angle_tolerance_radians=float(np.pi / 9),
            position_comparison_dim=4, angle_index=4,
            reported_distance="full native 7D state L2 (upstream eval_state)",
        )
    elif isinstance(physical_env, TwoRoomEnv):
        physical_metric.update(
            definition="agent goal position L2 < 16 px", tolerance=16.0,
            comparison_dim=2, reported_distance="agent position L2",
        )
    elif getattr(physical_env, "ENV_KIND", None) == "delayed_t_maze":
        physical_metric.update(
            definition="a terminal decision was made and its side equals the recorded true cue",
            reported_distance="binary task failure (0 success, 1 failure)",
            comparison_dim=1,
        )
    protocol = {
        "version": 2, "cem_cost": "squared_l2",
        "data_loader_protocol": getattr(
            getattr(ds, "dataset", ds), "data_protocol_version", DATA_LOADER_PROTOCOL_VERSION,
        ),
        "latent_representation": "forward.emb, independent observed frames, zero action",
        "cem_action_dim": action_dim, "model_action_dim": model_action_dim,
        "model_state_dim": state_dim, "observation_scale": state_scale,
        "declared_env_id": env.spec.env_id,
        "physical_env_id": physical_env.spec.env_id,
        "initialization": (
            "recorded simulator phase, true cue and prior decision state"
            if getattr(physical_env, "ENV_KIND", None) == "delayed_t_maze"
            else "offline native state; unrecorded velocities zero"
        ),
        "goal_definition": getattr(
            getattr(ds, "dataset", ds), "goal_definition", "same-episode state at t+goal_offset",
        ),
        "replanning": "observed history after executing up to H actions",
        "physical_success": physical_metric,
        "timed_intervention": step_predict_hook is not None,
    }
    if qpos_mask_indices:
        protocol.update({
            "observation_mask": {
                "indices": qpos_mask_indices, "quantity": "qpos", "goal_masked": True,
            },
            "condition": "qpos_indices_" + "_".join(map(str, qpos_mask_indices)) + "_masked",
            "mask_ratio": 1.0,
            "physical_success_uses": "unmasked native state and goal",
        })
    if isinstance(physical_env, FlickeringDMCEnv):
        protocol["observation_dropout"] = {
            "probability": physical_env.mask_ratio,
            "sampling": "one cached mask per reset/step; restoration preserves reset mask",
            "goal_masked": False,
            "physical_success_uses": "unmasked native state and goal",
        }
    return ClosedLoopResult(
        env_id=env.spec.env_id, n_episodes=len(per_episode_all), n_seeds=n_seeds,
        cem_samples=cem_samples, cem_elites=cem_elites, cem_iters=cem_iters,
        horizon=horizon, eval_budget=eval_budget, goal_offset=effective_goal_offset,
        history_size=history_size,
        success_rate_lewm=mean_metric("success_rate_lewm"),
        success_rate_lewm_std=std_metric("success_rate_lewm"),
        success_rate_lewm_005=mean_metric("success_rate_lewm_005"),
        success_rate_lewm_001=mean_metric("success_rate_lewm_001"),
        success_rate_env=mean_metric("success_rate_env"),
        success_rate_env_std=std_metric("success_rate_env"),
        mean_cos_dist=mean_metric("mean_cos_dist"),
        mean_cos_dist_std=std_metric("mean_cos_dist"),
        mean_phys_dist=mean_metric("mean_phys_dist"),
        mean_phys_dist_std=std_metric("mean_phys_dist"),
        mean_reward=mean_metric("mean_reward"), mean_reward_std=std_metric("mean_reward"),
        per_seed=per_seed_results, per_episode=per_episode_all,
        wall_time_sec=time.time() - wall_t0,
        protocol=protocol,
    )


# ============================================================
# Helpers
# ============================================================
def _load_eval_dataset(env, data_path, history_size, goal_offset, split: str = "in_dist",
                       pad_obs_to: Optional[int] = None):
    """Load the eval dataset for sampling init/goal pairs.

    Missing offline state/goal data is an explicit error; it is never replaced
    with unrelated random initial states.
    """
    use_env_based = (data_path is None or data_path == "(none)" or data_path == "")
    if use_env_based:
        env_kind_for_load = _infer_env_kind(env)
        if env_kind_for_load in ("ogb_cube", "ogb_scene"):
            ds = load_dataset(env_kind_for_load + "_env", n_episodes=20, max_steps_per_ep=50,
                              history_size=history_size, goal_offset=goal_offset, seed=42)
            return ds, ds.spec.obs_dim
        raise ValueError(f"Offline init/goal data is required for {env.spec.env_id}")
    if data_path.endswith(".npz") or data_path.endswith(".h5"):
        env_id = env.spec.env_id if hasattr(env, "spec") else None
        if _infer_env_kind(env) == "delayed_t_maze":
            from code.data.delayed_t_maze import load_delayed_t_maze
            ds = load_delayed_t_maze(
                data_path, history_size=history_size, goal_offset=goal_offset,
                pad_obs_to=pad_obs_to, env_id=env_id, include_sim_state=True,
            )
        else:
            ds = load_dataset(_infer_env_kind(env), path=data_path,
                              history_size=history_size, goal_offset=goal_offset,
                              pad_obs_to=pad_obs_to, env_id=env_id)
        spec_dim = ds.spec.obs_dim
        # B4: OOD split for held-out goal states
        if split == "unseen_goal" and hasattr(ds, "spec"):
            ds = _make_unseen_goal_subset(ds)
        return ds, spec_dim
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
    if "delayed" in eid.lower() or "t_maze" in eid.lower() or "t-maze" in eid.lower():
        return "delayed_t_maze"
    if "PushT" in eid: return "pusht"
    if "TwoRoom" in eid: return "tworoom"
    if "OGBCube" in eid: return "ogb_cube"
    if "OGBScene" in eid: return "ogb_scene"
    if "reacher" in eid.lower(): return "reacher_4d"  # 4D state with target
    if "mujoco/" in eid:
        return "dmc"
    return "mujoco_3d"

def _set_env_state(env: BaseEnv, state: np.ndarray, *, sim_state: dict | None = None) -> None:
    """Restore the recorded native state without advancing simulation time."""
    import mujoco

    while hasattr(env, "_base"):
        env = env._base
    state = np.asarray(state, dtype=np.float32)[:env.spec.obs_dim]
    if state.shape != (env.spec.obs_dim,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid initial state for {env.spec.env_id}: {state}")
    if getattr(env, "ENV_KIND", None) == "delayed_t_maze":
        if sim_state is None:
            raise ValueError("Maze restoration requires recorded phase and hidden-cue metadata")
        for key in ("delay_length", "cue_visibility", "distractor"):
            if sim_state[key] != getattr(env.cfg, key):
                raise ValueError(f"Maze {key} differs from the recorded simulator configuration")
        env._step_count = int(sim_state["step_count"])
        env._agent_y = float(state[1])
        env._cue_side = int(sim_state["cue_side"])
        env._terminal_choice = int(sim_state["terminal_choice"])
        env._reward_given = bool(sim_state["reward_given"])
        env._accumulated_reward = float(sim_state["accumulated_reward"])
        env._last_obs_dict = env._obs_dict()
        return
    if isinstance(env, ReacherEnv):
        env._data.qpos[:2] = state[:2]
        env._data.qvel[:] = 0.0
        env._model.geom_pos[env._target_id, :2] = state[2:4]
        mujoco.mj_forward(env._model, env._data)
        return
    if hasattr(env, "_model") and hasattr(env, "_data") and hasattr(env, "_nq"):
        if env._expand_pendulum:
            env._data.qpos[0] = float(np.arctan2(state[1], state[0]))
        else:
            env._data.qpos[:env._nq] = state[:env._nq]
        env._data.qvel[:] = 0.0
        mujoco.mj_forward(env._model, env._data)
        if isinstance(env, FlickeringDMCEnv):
            env.refresh_observation()
        return
    if isinstance(env, PushTEnv):
        native = env._env.unwrapped
        # Upstream _set_state() advances physics by dt, so use the same
        # documented body assignments and reindex collision shapes directly.
        native.agent.position = tuple(state[:2])
        native.agent.velocity = tuple(state[5:7])
        # Match upstream ordering: rotating the asymmetric body's local center
        # of gravity changes its reported position, so set world position last.
        native.block.angle = float(state[4])
        native.block.position = tuple(state[2:4])
        native.block.velocity = (0.0, 0.0)
        native.block.angular_velocity = 0.0
        native.agent.force = (0.0, 0.0)
        native.block.force = (0.0, 0.0)
        native.block.torque = 0.0
        native.space.reindex_shapes_for_body(native.agent)
        native.space.reindex_shapes_for_body(native.block)
        restored = native._get_obs()
        env._current_obs = {
            "state": restored,
            "proprio": np.concatenate([restored[:2], restored[-2:]]),
        }
        return
    if isinstance(env, TwoRoomEnv):
        native = env._env.unwrapped
        # The recorded data fixes door geometry; widths/wall thickness are not
        # stored. Refuse an incompatible world rather than inventing geometry.
        if not np.allclose(np.asarray(native._get_obs())[4:], state[4:], atol=1e-4, rtol=0):
            raise ValueError("TwoRoom door geometry differs from the recorded initial state")
        native._set_state(state[:2])
        native._set_goal_state(state[2:4])
        env._current_obs = native._get_obs()
        return
    raise NotImplementedError(
        f"Exact state restoration is unsupported for {env.spec.env_id}; "
        "a simulator-state snapshot is required"
    )


class _PadObsWrapper:
    """Wraps a BaseEnv so every obs returned is zero-padded to `target_dim` along the last axis.

    Used at eval time to load a generalist checkpoint (trained on padded obs) against
    any env with a smaller native obs_dim. The wrapper also updates `env.spec.obs_dim`
    to the padded dim so downstream code (state_dim computation, LeWM-SR checks) sees
    the right shape.

    Methods overridden: reset, step, get_state, check_success.
    The native spec/action_dim/action_low/action_high are preserved so CEM and
    env-native success detection still work.
    """
    def __init__(self, base: BaseEnv, target_dim: int):
        self._base = base
        self._native_dim = base.spec.obs_dim
        self.spec = replace(base.spec, obs_dim=target_dim)
        self._target = target_dim
        self._step_count = 0

    def reset(self, seed=None, **kw):
        obs = self._base.reset(seed=seed, **kw)
        return self._pad(obs)

    def step(self, action):
        obs, r, d, info = self._base.step(action)
        return self._pad(obs), r, d, info

    def get_state(self):
        return self._pad_arr(self._base.get_state())

    def check_success(self, s, g):
        return self._base.check_success(
            np.asarray(s)[..., :self._native_dim], np.asarray(g)[..., :self._native_dim],
        )

    def _pad(self, obs):
        if isinstance(obs, dict) and "state" in obs:
            obs = dict(obs); obs["state"] = self._pad_arr(obs["state"]); return obs
        return self._pad_arr(obs)
    def _pad_arr(self, arr):
        a = np.array(arr, dtype=np.float32, copy=True)
        if a.ndim == 0:
            return a
        last = a.shape[-1]
        if last < self._target:
            pad = np.zeros(a.shape[:-1] + (self._target - last,), dtype=np.float32)
            a = np.concatenate([a, pad], axis=-1)
        return a


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="pusht|tworoom|cube|reacher|cartpole|...")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-elites", type=int, default=30)
    p.add_argument("--cem-iters", type=int, default=10)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--goal-offset", type=int, default=25,
                   help="Default goal_offset. For tworoom_long, you can override "
                        "this per-call (the legacy hard-coded 200 override was removed).")
    p.add_argument("--split", choices=["in_dist", "unseen_goal"], default="in_dist",
                   help="Dataset split: 'in_dist' (default) or 'unseen_goal' (B4: held-out)")
    p.add_argument("--flicker-mask-ratio", type=float, default=None,
                   help="FlickeringDMCEnv mask probability (cartpole_flicker). "
                        "Default 0.5 if not provided.")
    p.add_argument("--qpos-mask-obs-ratio", type=float, default=None,
                   help="If >0, additionally zero this fraction of NON-masked obs dims "
                        "(cheetah_qpos_masked hard). The base condition masks recorded "
                        "qpos channels, not velocities.")
    p.add_argument("--pad-obs-eval", type=int, default=None,
                   help="If set, pad env obs to this dim at eval time. Required for "
                        "loading a generalist checkpoint trained with --pad-obs-to.")
    p.add_argument("--action-dim-eval", type=int, default=None,
                   help="Override the model's padded action dimension. CEM always "
                        "optimizes native-dimensional controls.")
    p.add_argument("--delay-length", type=int, default=None,
                   help="Override delayed_t_maze corridor length (default 50).")
    p.add_argument("--cue-visibility", type=int, default=None,
                   help="Override delayed_t_maze cue visibility (default 3).")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[closed_loop/{args.env}] ckpt={args.ckpt}, data={args.data}", flush=True)
    device = args.device
    if Path(args.out).exists():
        raise FileExistsError(f"Archive the existing result before rerunning: {args.out}")

    # Build env (supports per-eval flicker mask_ratio)
    env = make_env(args.env, args.data, flicker_mask_ratio=args.flicker_mask_ratio,
                     delay_length=args.delay_length, cue_visibility=args.cue_visibility)

    # Resolve architecture from trained tensors, never by trying random sizes.
    from code.scripts.event_align import build_model
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    state_dim = next(
        (int(sd[key].shape[1]) for key in (
            "state_encoder.proj.0.weight", "state_projector.proj.0.weight",
            "state_projector.0.weight", "state_proj.0.weight",
        ) if key in sd),
        ck_args.get("pad_obs_to") or args.pad_obs_eval or env.spec.obs_dim,
    )
    action_dim = next(
        (int(sd[key].shape[1]) for key in (
            "action_encoder.proj.0.weight", "action_encoder.0.weight",
            "action_proj.0.weight",
        ) if key in sd),
        ck_args.get("action_dim") or args.action_dim_eval or env.spec.action_dim,
    )
    if ck_args["model"] == "mlp_baseline":
        action_dim = int(sd["net.0.weight"].shape[1] - sd["state_proj.2.weight"].shape[0])
    model = build_model(ck_args["model"], state_dim, action_dim, ck_args, state_dict=sd)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print(f"[closed_loop] trained weights loaded strictly on {device}", flush=True)


    # Extra corruption uses the same observation/goal boundary as the base mask.
    if args.qpos_mask_obs_ratio is not None:
        from code.core.envs.dmc_env import MaskedQposEnv
        if not 0.0 <= args.qpos_mask_obs_ratio <= 1.0:
            raise ValueError("qpos-mask-obs-ratio must be between zero and one")
        if not isinstance(env, MaskedQposEnv):
            raise ValueError("qpos-mask-obs-ratio requires a *_qpos_masked environment")
        non_masked_idx = [
            i for i in range(env.spec.obs_dim) if i not in env.observation_mask_indices
        ]
        n_extra = int(round(args.qpos_mask_obs_ratio * len(non_masked_idx)))
        if n_extra:
            rng = np.random.default_rng(12345)
            drop_idx = sorted(rng.choice(non_masked_idx, size=n_extra, replace=False).tolist())
            env = MaskedQposEnv(env, drop_idx)
        print(f"[closed_loop] extra-drop {n_extra}/{len(non_masked_idx)} qpos dims "
              f"(ratio={args.qpos_mask_obs_ratio})", flush=True)

    # The previous hard-coded tworoom_long -> goal_offset=200 override was removed.
    # tworoom_long now honors --goal-offset so the sweep can vary difficulty.
    goal_offset_override = None
    # Run eval
    result = eval_closed_loop(
        model, env, args.data,
        n_episodes=args.n_episodes,
        cem_samples=args.cem_samples, cem_elites=args.cem_elites, cem_iters=args.cem_iters,
        n_seeds=args.n_seeds,
        goal_offset=args.goal_offset, history_size=args.history_size,
        horizon=args.horizon, eval_budget=args.eval_budget,
        device=device,
        goal_offset_override=goal_offset_override,
        split=args.split,
        pad_obs_to=state_dim,
        model_action_dim=action_dim,
    )
    # Save
    result.protocol["checkpoint"] = str(Path(args.ckpt).resolve())
    result.protocol["checkpoint_data_protocol_version"] = ck.get("data_protocol_version")
    for field in ("training_protocol_version", "step"):
        if field in ck:
            result.protocol["checkpoint_" + field] = ck[field]
    if "training_provenance" in ck:
        result.protocol["checkpoint_training_provenance"] = ck["training_provenance"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x") as f:
        json.dump(asdict(result), f, indent=2, allow_nan=False)
    print(f"\n=== FINAL (closed_loop/{args.env}) ===")
    print(f"  LeWM SR: {result.success_rate_lewm:.3f} ± {result.success_rate_lewm_std:.3f}")
    print(f"  Env-native SR: {result.success_rate_env:.3f} ± {result.success_rate_env_std:.3f}")
    print(f"  Mean cos_dist: {result.mean_cos_dist:.4f}")
    print(f"  Mean phys_dist: {result.mean_phys_dist:.4f}")
    print(f"  Saved to {args.out}")


if __name__ == "__main__":
    main()

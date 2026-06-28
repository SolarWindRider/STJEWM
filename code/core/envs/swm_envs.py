"""Stable-worldmodel env wrappers: PushT, TwoRoom, OGBCube.

These use the official stable_worldmodel gymnasium interface. They wrap the
gym env + provide env-native success checks.

State spec (per env):
    PushT-v1:    obs = Dict(proprio(4), state(7));  state input = 7D state
    TwoRoom-v1:  obs = Box(10,);                    state input = 10D
    OGBCube-v0:  obs = Box(28,);                    state input = 28D
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Ensure mujoco uses EGL (headless GPU) so gym envs work in our env
os.environ.setdefault("MUJOCO_GL", "egl")

from .base import BaseEnv, EnvSpec


# ============================================================
# Helper: per-env state extraction
# ============================================================
def _state_from_pusht_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """PushT: use the 7D 'state' (agent pos+vel + block pos+vel+angle+angvel)."""
    return np.asarray(obs["state"], dtype=np.float32).flatten()


def _state_from_tworoom_obs(obs) -> np.ndarray:
    """TwoRoom: obs is a 10D Box."""
    return np.asarray(obs, dtype=np.float32).flatten()


def _state_from_cube_obs(obs) -> np.ndarray:
    """OGBench Cube: obs is a 28D Box (proprio + cube pos)."""
    return np.asarray(obs, dtype=np.float32).flatten()


# ============================================================
# PushT wrapper
# ============================================================
class PushTEnv(BaseEnv):
    """Wrapper around swm/PushT-v1.

    Env-native success (stable_worldmodel PushT task):
        block within tolerance of target — implemented via env.unwrapped._is_success()
        (stable_worldmodel defines block-in-target-pose success).
    """

    def __init__(self):
        super().__init__()
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        self._env = gym.make("swm/PushT-v1")
        action_low = self._env.action_space.low.astype(np.float32)
        action_high = self._env.action_space.high.astype(np.float32)
        self.spec = EnvSpec(
            env_id="swm/PushT-v1",
            obs_dim=7,
            action_dim=2,
            action_low=action_low,
            action_high=action_high,
            obs_keys=("state",),
            max_episode_steps=200,
        )
        self._current_obs: Optional[Dict[str, np.ndarray]] = None

    def reset(self, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        out, _ = self._env.reset(seed=seed)
        self._current_obs = out
        self._step_count = 0
        return out

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        out, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        self._current_obs = out
        self._step_count += 1
        return out, float(reward), bool(done), info

    def get_state(self) -> np.ndarray:
        return _state_from_pusht_obs(self._current_obs)

    def check_success(self, state: np.ndarray, goal_state: np.ndarray) -> Tuple[bool, float]:
        """Block-in-target-pose check on the 7D state vector.

        state layout (swm/PushT-v1, from lejepa swm_pushT):
            [agent_x, agent_y, agent_vx, agent_vy, block_x, block_y, block_angle]
        goal_state has the same layout. We measure block (x, y, angle) distance
        to the goal. The "tolerance" comes from stable_worldmodel's PushT task
        definition (default: 0.07 m for x/y, 1.0 rad for angle).
        """
        # Block pose is in indices 4, 5, 6
        diff = state[4:7] - goal_state[4:7]
        # Position (x, y) tolerance 0.07 m, angle tolerance 1.0 rad
        pos_tol = 0.07
        ang_tol = 1.0
        dist = float(np.linalg.norm(diff[:2]))
        ang_dist = float(abs(diff[2]))
        # Normalized: how many tolerances exceeded
        if dist < pos_tol and ang_dist < ang_tol:
            return True, 0.0
        # Use the max of (dist/pos_tol, ang_dist/ang_tol) as the distance
        return False, max(dist / pos_tol, ang_dist / ang_tol)

    def close(self):
        self._env.close()


# ============================================================
# TwoRoom wrapper
# ============================================================
class TwoRoomEnv(BaseEnv):
    """Wrapper around swm/TwoRoom-v1.

    Env-native success: agent within tolerance of target.
    """

    def __init__(self):
        super().__init__()
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        self._env = gym.make("swm/TwoRoom-v1")
        action_low = self._env.action_space.low.astype(np.float32)
        action_high = self._env.action_space.high.astype(np.float32)
        self.spec = EnvSpec(
            env_id="swm/TwoRoom-v1",
            obs_dim=10,
            action_dim=2,
            action_low=action_low,
            action_high=action_high,
            obs_keys=("observation",),
            max_episode_steps=200,
        )
        self._current_obs: Optional[np.ndarray] = None

    def reset(self, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        out, _ = self._env.reset(seed=seed)
        self._current_obs = out
        self._step_count = 0
        return {"observation": out}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        out, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        self._current_obs = out
        self._step_count += 1
        return {"observation": out}, float(reward), bool(done), info

    def get_state(self) -> np.ndarray:
        return _state_from_tworoom_obs(self._current_obs)

    def check_success(self, state: np.ndarray, goal_state: np.ndarray) -> Tuple[bool, float]:
        """TwoRoom: agent position within tolerance of target.

        state layout (swm/TwoRoom-v1, 10D):
            [agent_x, agent_y, target_x, target_y, ...internal state...]
        The first 4 dims are the agent + target positions. We use those.
        """
        diff = state[:4] - goal_state[:4]
        # 2D Euclidean distance between agent and target
        dist = float(np.linalg.norm(diff[:2]))
        # Tolerance: 1.0 unit (TwoRoom room is ~5x5 units, door at x=0)
        return dist < 1.0, dist

    def close(self):
        self._env.close()


# ============================================================
# OGBCube wrapper
# ============================================================
class OGBCubeEnv(BaseEnv):
    """Wrapper around swm/OGBCube-v0.

    Env-native success: cube within tolerance of target.
    """

    def __init__(self):
        super().__init__()
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        self._env = gym.make("swm/OGBCube-v0")
        action_low = self._env.action_space.low.astype(np.float32)
        action_high = self._env.action_space.high.astype(np.float32)
        self.spec = EnvSpec(
            env_id="swm/OGBCube-v0",
            obs_dim=28,
            action_dim=5,
            action_low=action_low,
            action_high=action_high,
            obs_keys=("observation",),
            max_episode_steps=200,
        )
        self._current_obs: Optional[np.ndarray] = None

    def reset(self, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        out, _ = self._env.reset(seed=seed)
        self._current_obs = out
        self._step_count = 0
        return {"observation": out}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        out, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        self._current_obs = out
        self._step_count += 1
        return {"observation": out}, float(reward), bool(done), info

    def get_state(self) -> np.ndarray:
        return _state_from_cube_obs(self._current_obs)

    def check_success(self, state: np.ndarray, goal_state: np.ndarray) -> Tuple[bool, float]:
        """OGBCube: cube position within tolerance of target.

        The 28D obs layout (OGBench cube) is:
            [robot_arm_proprio(7), cube_pos(3), cube_quat(4), target_pos(3), target_quat(4), ...]
        The exact layout is environment-specific. We default to using the last 6
        dims (target pos + quat) and matching cube to target by position.
        For a 3D cube placement, cube_pos is typically at indices [7:10] and
        target_pos at [-6:-3] (last 6 dims are target pos+quat).
        """
        # Heuristic: cube at indices [7:10], target at [-6:-3]
        cube_pos = state[7:10] if state.shape[0] >= 10 else state[:3]
        target_pos = goal_state[-6:-3] if goal_state.shape[0] >= 6 else goal_state[:3]
        dist = float(np.linalg.norm(cube_pos - target_pos))
        # OGBench cube tolerance: 0.1 m (10 cm)
        return dist < 0.1, dist

    def close(self):
        self._env.close()


# ============================================================
# Factory: build env by name
# ============================================================
def make_swm_env(env_id: str) -> BaseEnv:
    if env_id == "swm/PushT-v1":
        return PushTEnv()
    if env_id == "swm/TwoRoom-v1":
        return TwoRoomEnv()
    if env_id == "swm/OGBCube-v0":
        return OGBCubeEnv()
    raise ValueError(f"Unknown swm env: {env_id}. Supported: PushT-v1, TwoRoom-v1, OGBCube-v0")

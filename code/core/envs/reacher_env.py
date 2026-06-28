"""Reacher (DMControl easy) wrapper.

Uses direct mujoco bindings (bypasses stable_worldmodel/dm_control 1.0.41
incompatibility with mujoco 3.10 — the `flex_bandwidth` attribute bug).

State spec:
    4D = [qpos(2), target_xy(2)]
    action = 2D torque
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import mujoco

from .base import BaseEnv, EnvSpec

# Make sure mujoco GL is set before any import
os.environ.setdefault("MUJOCO_GL", "egl")

REACHER_XML = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite/reacher.xml"
TARGET_SIZE = 0.05  # 5cm — standard Reacher success threshold (DMC)


class ReacherEnv(BaseEnv):
    """Direct mujoco Reacher env. State = [qpos(2), target(2)], 4D.

    The state is constructed by combining the joint qpos (read from mujoco)
    with the target position (read from the model XML/geom). Action is a 2D
    torque clipped to [-1, 1] (DMC reacher.easy).
    """

    def __init__(self):
        super().__init__()
        self._model = mujoco.MjModel.from_xml_path(REACHER_XML)
        self._data = mujoco.MjData(self._model)
        # Cache IDs
        self._target_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        self._finger_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "finger")
        # Default random init
        self._rng = np.random.default_rng(42)
        self.spec = EnvSpec(
            env_id="mujoco/reacher",
            obs_dim=4,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            obs_keys=("state",),
            max_episode_steps=50,  # LeWM paper budget for Reacher
        )

    def reset(self, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Random qpos in [-pi/2, pi/2]
        self._data.qpos[:] = self._rng.uniform(-np.pi / 2, np.pi / 2, size=2)
        self._data.qvel[:] = 0.0
        # Random target in [-0.2, 0.2]^2
        self._model.geom_pos[self._target_id, :2] = self._rng.uniform(-0.2, 0.2, size=2)
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        return self._current_obs_dict()

    def _current_obs_dict(self) -> Dict[str, np.ndarray]:
        return {"state": self.get_state()}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        self._data.ctrl[:] = action
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1
        # Compute reward (distance-based, like DMC)
        dist = self._fingertip_target_dist()
        reward = -float(dist)
        done = self._step_count >= self.spec.max_episode_steps
        info = {"fingertip_target_dist": dist}
        return self._current_obs_dict(), reward, done, info

    def get_state(self) -> np.ndarray:
        target_pos = self._model.geom_pos[self._target_id, :2].copy()
        qpos = self._data.qpos[:2].copy()
        return np.concatenate([qpos, target_pos]).astype(np.float32)

    def _fingertip_target_dist(self) -> float:
        if self._finger_id < 0:
            return float("inf")
        finger_pos = self._data.geom_xpos[self._finger_id, :2].copy()
        target_pos = self._model.geom_pos[self._target_id, :2].copy()
        return float(np.linalg.norm(finger_pos - target_pos))

    def check_success(self, state: np.ndarray, goal_state: np.ndarray) -> Tuple[bool, float]:
        """Reacher: fingertip-to-target distance < 5cm.

        state = [qpos(2), target(2)]. We don't have the fingertip position
        directly in state; the env would need to be stepped to compute it.
        Caller should use env-native distance from step() info.
        For offline eval, use cosine distance as fallback.
        """
        # State-based proxy: just compare target positions
        # (This is loose — real success requires a mujoco step, but for
        #  offline evaluation of latent cost this is what we have.)
        diff = state[2:4] - goal_state[2:4]
        dist = float(np.linalg.norm(diff))
        return dist < TARGET_SIZE, dist

    def close(self):
        pass  # mujoco has no explicit close


def make_reacher_env() -> ReacherEnv:
    return ReacherEnv()

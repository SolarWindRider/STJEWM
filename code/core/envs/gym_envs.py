"""Gymnasium classic-control env wrappers.

Uses the stable_worldmodel gym wrapper (swm/X-v1/v0). State = raw obs (1-6D).

Supported:
    swm/CartPoleControl-v1     (4D obs, Discrete(2) action)
    swm/AcrobotControl-v1       (6D obs, Discrete(3) action)
    swm/PendulumControl-v1      (3D obs, Box(-2, 2, 1D) action)
    swm/MountainCarControl-v0   (2D obs, Discrete(3) action)
    swm/MountainCarContinuousControl-v0 (2D obs, Box(-1, 1, 1D) action)

Success criteria (env-native, per Gymnasium-v1/v0 specs):
    CartPole-v1:    reward >= 475 (last 100-eps avg)
    Acrobot-v1:      reward <= -100 (close to upright)
    Pendulum-v1:     reward > -1.0 (close to upright)
    MountainCar-v0:  reward >= -110
    MountainCarCont: reward >= 90
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from .base import BaseEnv, EnvSpec


class GymControlEnv(BaseEnv):
    """Generic wrapper for stable_worldmodel gym classic-control envs.

    All these envs have a simple obs box + env-native reward signal. We
    report env-native success based on reward thresholds per env.
    """

    def __init__(self, env_id: str, success_reward_threshold: float, lower_is_better: bool = False):
        super().__init__()
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        self._env = gym.make(env_id)
        self._env_id = env_id
        self._success_reward_threshold = success_reward_threshold
        self._lower_is_better = lower_is_better

        # Compute action spec — handle both Discrete and Box action spaces
        if hasattr(self._env.action_space, "n"):
            # Discrete — we use one-hot encoding as the "action" (world model input)
            self._action_dim = int(self._env.action_space.n)
            self._action_low = np.zeros(self._action_dim, dtype=np.float32)
            self._action_high = np.ones(self._action_dim, dtype=np.float32)
            self._is_discrete = True
        else:
            self._action_dim = int(self._env.action_space.shape[0])
            self._action_low = self._env.action_space.low.astype(np.float32)
            self._action_high = self._env.action_space.high.astype(np.float32)
            self._is_discrete = False

        self._obs_dim = int(np.prod(self._env.observation_space.shape))
        self.spec = EnvSpec(
            env_id=env_id,
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            action_low=self._action_low,
            action_high=self._action_high,
            obs_keys=("observation",),
            max_episode_steps=int(getattr(self._env.spec, "max_episode_steps", 500)),
        )
        self._current_obs: Optional[np.ndarray] = None
        self._last_reward: float = 0.0

    def reset(self, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        out, _ = self._env.reset(seed=seed)
        self._current_obs = out
        self._last_reward = 0.0  # Reset so success is computed from this episode's rewards only
        self._step_count = 0
        return {"observation": np.asarray(out, dtype=np.float32)}
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        # Convert one-hot -> discrete int if needed
        if self._is_discrete:
            act = int(np.argmax(action))
        else:
            act = np.clip(np.asarray(action, dtype=np.float32), self._action_low, self._action_high)
        out, reward, terminated, truncated, info = self._env.step(act)
        done = terminated or truncated
        self._current_obs = out
        self._last_reward = float(reward)
        self._step_count += 1
        return {"observation": np.asarray(out, dtype=np.float32)}, float(reward), bool(done), info

    def get_state(self) -> np.ndarray:
        return np.asarray(self._current_obs, dtype=np.float32).flatten()

    def check_success(self, state: np.ndarray, goal_state: np.ndarray) -> Tuple[bool, float]:
        """Env-native success: based on reward threshold.

        For goal-conditioned Gym classic-control, the "goal" is typically
        "upright pendulum" (Pendulum/CartPole/Acrobot) or "reached flag"
        (MountainCar). We approximate by reward threshold.
        """
        if self._lower_is_better:
            success = self._last_reward <= self._success_reward_threshold
        else:
            success = self._last_reward >= self._success_reward_threshold
        return success, float(self._last_reward)

    def close(self):
        self._env.close()


# ============================================================
# Factory
# ============================================================
def make_gym_env(env_id: str) -> GymControlEnv:
    if env_id == "swm/CartPoleControl-v1":
        return GymControlEnv(env_id, success_reward_threshold=475.0)
    if env_id == "swm/AcrobotControl-v1":
        return GymControlEnv(env_id, success_reward_threshold=-100.0, lower_is_better=True)
    if env_id == "swm/PendulumControl-v1":
        return GymControlEnv(env_id, success_reward_threshold=-1.0, lower_is_better=True)
    if env_id == "swm/MountainCarControl-v0":
        return GymControlEnv(env_id, success_reward_threshold=-110.0)
    if env_id == "swm/MountainCarContinuousControl-v0":
        return GymControlEnv(env_id, success_reward_threshold=90.0)
    raise ValueError(f"Unknown gym env: {env_id}")

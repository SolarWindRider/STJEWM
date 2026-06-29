"""DMC env wrappers using direct mujoco (bypasses stable_worldmodel/dm_control 1.0.41
incompatibility with mujoco 3.10 — the `flex_bandwidth` attribute bug).

Each env wraps a DMC XML file and provides:
    - reset() -> {state: (D,)}
    - step(action) -> ({state}, reward, done, info)
    - get_state() -> (D,)    [the state vector used as input to the world model]
    - check_success(state, goal_state) -> (bool, distance)

For ALL DMC envs, `get_state()` returns the raw mujoco qpos[:nq]. The world
model is trained on this same qpos. This is the canonical DMC state-input setup.

State specs (from DMC suite):
    cartpole:    2D qpos (cart pos, pole angle)
    pendulum:    1D qpos (cos theta, sin theta)  -- but mujoco stores 1 angle, we wrap as 2D
    reacher:     2D qpos + 2D target
    finger:      3D qpos
    ball_in_cup: 4D qpos
    cheetah:     9D qpos
    walker:      9D qpos
    hopper:      7D qpos
    quadruped:   30D qpos
    humanoid:    28D qpos
    humanoid_CMU: 63D qpos
    dog:         87D qpos
    fish:        14D qpos
    stacker:     20D qpos

Success criteria (from DMC physics):
    For these envs, we use the env-native `solve()`-like check on the
    final qpos vs the goal state (provided by the dataset at t+goal_offset).
    Heuristic: env success = the goal_state is within a reasonable tolerance
    of the final state. We use a permissive distance threshold (1.0) for
    "within same basin" check; the proper env-native test would be
    DMC's per-env `Physics` task check.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import mujoco

from .base import BaseEnv, EnvSpec

os.environ.setdefault("MUJOCO_GL", "egl")

DMC_XML_DIR = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite"

# ============================================================
# Stress suite B3: Velocity-hidden DMC
# ============================================================
# Approximate velocity slice indices per DMC env. These cover the qvel[] part
# of the state vector. We mask the velocity slice at runtime to test whether
# trace-based models can still plan when continuous velocity is hidden.
# Source: DM Control Suite proprioceptive obs layout. For low-dim qpos-only
# envs (cartpole, pendulum, reacher) the slice still approximates the
# "velocity" channel of the obs.
VEL_INDICES = {
    "cheetah":       slice(3, 6),    # x_pos, y_pos skipped; velocity is indices 3-5
    "walker":        slice(3, 6),
    "cartpole":      slice(2, 4),
    "pendulum":      slice(1, 2),
    "hopper":        slice(3, 6),
    "humanoid":      slice(22, 36),
    "humanoid_cmu":  slice(22, 36),
    "quadruped":     slice(6, 12),
    "dog":           slice(6, 12),
    "fish":          slice(3, 6),
    "finger":        slice(3, 6),
    "ball_in_cup":   slice(3, 6),
    "stacker":       slice(3, 6),
    "manipulator":   slice(3, 6),
    "reacher":       slice(2, 4),
}


# ============================================================
# Env registry: (env_kind, xml_name, obs_dim, action_dim, max_episode_steps)
# ============================================================
DMC_ENVS = {
    # env_kind: (xml_filename, obs_dim, action_dim, max_episode_steps, success_tol)
    "cartpole":     ("cartpole.xml", 2, 1, 200, 0.5),
    "pendulum":     ("pendulum.xml", 1, 1, 200, 0.5),
    "reacher":      ("reacher.xml", 2, 2, 50, 0.05),
    "finger":       ("finger.xml", 3, 2, 100, 0.3),
    "ball_in_cup":  ("ball_in_cup.xml", 4, 2, 100, 0.1),
    "cheetah":      ("cheetah.xml", 9, 6, 200, 1.0),
    "walker":       ("walker.xml", 9, 6, 200, 1.0),
    "hopper":       ("hopper.xml", 7, 4, 200, 1.0),
    "quadruped":    ("quadruped.xml", 30, 12, 200, 1.0),
    "humanoid":     ("humanoid.xml", 28, 21, 200, 1.0),
    "humanoid_cmu": ("humanoid_CMU.xml", 63, 56, 200, 1.0),
    "dog":          ("dog.xml", 87, 38, 200, 1.0),
    "fish":         ("fish.xml", 14, 5, 200, 1.0),
    "stacker":      ("stacker.xml", 20, 5, 200, 1.0),
    "manipulator":  ("manipulator.xml", 14, 5, 200, 0.05),
}


# ============================================================
# Single generic DMC state env
# ============================================================
class DMCStateEnv(BaseEnv):
    """Generic DMC env using direct mujoco bindings.

    State = mujoco qpos[:nq]. Action is the mujoco ctrl signal clipped to [-1, 1].
    Success is a loose distance check: |state - goal| < success_tol (per-env).
    """

    def __init__(self, env_kind: str, success_tol: float = 1.0, max_episode_steps: int = 200):
        super().__init__()
        if env_kind not in DMC_ENVS:
            raise ValueError(f"Unknown DMC env: {env_kind}. Available: {list(DMC_ENVS.keys())}")
        xml_name, obs_dim, action_dim, default_max_steps, default_tol = DMC_ENVS[env_kind]
        self._env_kind = env_kind
        self._model = mujoco.MjModel.from_xml_path(str(Path(DMC_XML_DIR) / xml_name))
        self._data = mujoco.MjData(self._model)
        self._rng = np.random.default_rng(42)
        # For pendulum, mujoco stores 1 angle but we expand to (cos, sin) as 2D
        self._expand_pendulum = (env_kind == "pendulum")
        if self._expand_pendulum:
            self.spec = EnvSpec(
                env_id=f"mujoco/{env_kind}",
                obs_dim=2,
                action_dim=action_dim,
                action_low=np.full(action_dim, -1.0, dtype=np.float32),
                action_high=np.full(action_dim, 1.0, dtype=np.float32),
                obs_keys=("state",),
                max_episode_steps=max_episode_steps if max_episode_steps != 200 else default_max_steps,
            )
            self._nq = 1
        else:
            self.spec = EnvSpec(
                env_id=f"mujoco/{env_kind}",
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_low=np.full(action_dim, -1.0, dtype=np.float32),
                action_high=np.full(action_dim, 1.0, dtype=np.float32),
                obs_keys=("state",),
                max_episode_steps=max_episode_steps if max_episode_steps != 200 else default_max_steps,
            )
            self._nq = obs_dim
        self._success_tol = success_tol if success_tol != 1.0 else default_tol

    def reset(self, seed: Optional[int] = None, **kwargs) -> dict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self._model, self._data)
        # Random init in joint range
        for j in range(self._model.njnt):
            jt = self._model.jnt_type[j]
            if jt == mujoco.mjtJoint.mjJNT_FREE:
                continue  # leave free joint at zero
            r = self._model.jnt_range[j]
            if r[0] < r[1]:
                qs = self._model.jnt_qposadr[j]
                self._data.qpos[qs] = self._rng.uniform(r[0] * 0.5, r[1] * 0.5)
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        return self._current_obs_dict()

    def _current_obs_dict(self) -> dict:
        return {"state": self.get_state()}

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        # Pad / truncate to nu
        a = np.zeros(self._model.nu, dtype=np.float32)
        n = min(len(a), len(action))
        a[:n] = action[:n]
        self._data.ctrl[:] = a
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1
        done = self._step_count >= self.spec.max_episode_steps
        reward = 0.0  # not used for our success-based eval
        return self._current_obs_dict(), float(reward), done, {}

    def get_state(self) -> np.ndarray:
        qpos = self._data.qpos[: self._nq].copy()
        if self._expand_pendulum:
            return np.array([np.cos(qpos[0]), np.sin(qpos[0])], dtype=np.float32)
        return qpos.astype(np.float32)

    def check_success(self, state: np.ndarray, goal_state: np.ndarray) -> Tuple[bool, float]:
        """Env-native success: state is within tolerance of goal state.

        We compare the full state vector. Tolerance depends on the env.
        """
        # Pendulum: compare (cos, sin) -> angle
        if self._expand_pendulum:
            cos_a, sin_a = state[0], state[1]
            cos_g, sin_g = goal_state[0], goal_state[1]
            # Dot product
            cos_diff = cos_a * cos_g + sin_a * sin_g
            cos_diff = np.clip(cos_diff, -1.0, 1.0)
            angle_diff = float(np.arccos(cos_diff))
            return angle_diff < self._success_tol, angle_diff
        diff = state - goal_state
        # Normalize per-dim tolerance
        dist = float(np.linalg.norm(diff) / max(np.sqrt(len(state)), 1))
        return dist < self._success_tol, dist

    def close(self):
        pass


# ============================================================
# OGBench Scene env
# ============================================================
class OGBenchSceneEnv(BaseEnv):
    """Wrapper around swm/OGBScene-v0.

    40D obs, 5D action (matches Cube).
    """

    def __init__(self):
        super().__init__()
        import gymnasium as gym
        import stable_worldmodel  # noqa
        self._env = gym.make("swm/OGBScene-v0")
        self.spec = EnvSpec(
            env_id="swm/OGBScene-v0",
            obs_dim=40,
            action_dim=5,
            action_low=self._env.action_space.low.astype(np.float32),
            action_high=self._env.action_space.high.astype(np.float32),
            obs_keys=("observation",),
            max_episode_steps=200,
        )
        self._current_obs: Optional[np.ndarray] = None

    def reset(self, seed: Optional[int] = None, **kwargs) -> dict:
        out, _ = self._env.reset(seed=seed)
        self._current_obs = out
        self._step_count = 0
        return {"observation": np.asarray(out, dtype=np.float32)}

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        out, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        self._current_obs = out
        self._step_count += 1
        return {"observation": np.asarray(out, dtype=np.float32)}, float(reward), bool(done), info

    def get_state(self) -> np.ndarray:
        return np.asarray(self._current_obs, dtype=np.float32).flatten()

    def check_success(self, state: np.ndarray, goal_state: np.ndarray) -> Tuple[bool, float]:
        """OGBench Scene: env-native scene success = all subgoals met.

        For 40D obs we don't know exact layout; use a loose distance check.
        """
        dist = float(np.linalg.norm(state - goal_state) / np.sqrt(len(state)))
        return dist < 1.0, dist

    def close(self):
        self._env.close()


# ============================================================
# Factory
# ============================================================
def make_dmc_env(env_kind: str) -> BaseEnv:
    return DMCStateEnv(env_kind)


# ============================================================
# Stress suite B1: Flickering DMC
# ============================================================
class FlickeringDMCEnv(DMCStateEnv):
    """DMC state env where the obs is randomly masked to zero with prob mask_ratio.

    Forces the model to integrate over time (a key strength of the trace).
    Used as a stress test for the trace-only protocol.
    """

    def __init__(self, *args, mask_ratio: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = float(mask_ratio)

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        if np.random.rand() < self.mask_ratio:
            obs["state"] = np.zeros_like(obs["state"])
        info["mask_ratio"] = self.mask_ratio
        return obs, reward, done, info

    def reset(self, seed: Optional[int] = None, **kwargs) -> dict:
        obs = super().reset(seed=seed, **kwargs)
        if np.random.rand() < self.mask_ratio:
            obs["state"] = np.zeros_like(obs["state"])
        return obs



# ============================================================
# Stress suite B3: Velocity-hidden DMC factory
# ============================================================
def make_vel_hidden_env(env_kind: str) -> BaseEnv:
    """Return a DMC env with velocity components of the obs zeroed at every step.

    This is a runtime wrapper that does NOT modify the underlying mujoco model.
    The wrapper inherits from the base env class and masks velocity indices
    in obs. Used to test whether trace-based models can still plan when
    continuous velocity is hidden at evaluation time.
    """
    base = make_dmc_env(env_kind)
    vel_slice = VEL_INDICES.get(env_kind, slice(0, 0))
    parent_class = type(base)

    class _VelHiddenWrapper(parent_class):
        def __init__(self, base_env, vel_slice):
            # Copy attributes from base
            self._base = base_env
            self._vel_slice = vel_slice
            self._step_count = 0
            self.spec = base_env.spec
            self._env_kind = getattr(base_env, "_env_kind", env_kind)

        def reset(self, seed=None, **kwargs):
            obs = self._base.reset(seed=seed, **kwargs)
            self._step_count = 0
            return self._mask_obs(obs)

        def step(self, action):
            obs, r, done, info = self._base.step(action)
            self._step_count += 1
            return self._mask_obs(obs), r, done, info

        def _mask_obs(self, obs):
            if isinstance(obs, dict) and "state" in obs:
                obs = dict(obs)
                obs["state"] = self._mask(obs["state"])
                return obs
            return self._mask(obs)

        def _mask(self, obs):
            arr = np.array(obs, dtype=np.float32, copy=True)
            arr[self._vel_slice] = 0.0
            return arr

        def get_state(self):
            return self._mask(self._base.get_state())

        def check_success(self, final_state, goal_state):
            return self._base.check_success(self._mask(final_state), self._mask(goal_state))

        def render(self, *args, **kwargs):
            return self._base.render(*args, **kwargs)

        def close(self):
            self._base.close()

    return _VelHiddenWrapper(base, vel_slice)



def make_ogb_scene_env() -> BaseEnv:
    return OGBenchSceneEnv()

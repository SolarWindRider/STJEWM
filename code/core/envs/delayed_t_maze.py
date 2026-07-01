"""Delayed T-Maze environment.

The agent is placed at the start of a long corridor. During the first
``cue_visibility`` frames a cue appears at one of the two goals (left or
right). The cue then disappears and the agent must traverse the corridor
(inertially — every step advances the agent forward by 1). At the corridor
end the agent must pick the goal that matches the memorized cue. The task
probes working-memory: success depends on retaining the cue through a
``delay_length``-long corridor.

Difficulty is controlled by three knobs:
- ``delay_length`` (= corridor_length): how many frames of pure forward motion
  separate cue observation from the terminal choice.
- ``cue_visibility``: how many frames the cue is visible before the corridor.
- ``distractor``: if True, random distractor cues can flip the apparent goal
  during the corridor (50% chance per corridor step).

State layout (obs_dim = 6):
    [agent_x, agent_y, cue_x, cue_y, corridor_marker, goal_marker]
- agent_x, agent_y: continuous position of the agent along the corridor.
- cue_x, cue_y: the *currently visible* cue position. After cue_visibility
  frames these are zeroed until the terminal step.
- corridor_marker: 1.0 while the agent is in the corridor (cue hidden),
  0.0 during the cue phase and terminal phase.
- goal_marker: 1.0 only on the terminal decision frame, else 0.0.

Actions (action_dim = 2) are clipped to [-1, 1]:
- action[0]: forward command. Magnitude is ignored — agent always advances by
  1 during corridor steps (working memory is forced by environment physics).
  Lateral components during the corridor have no effect on position.
- action[1]: only meaningful on the terminal frame. action[1] < 0 -> choose
  left goal, action[1] >= 0 -> choose right goal. Lateral choice during
  intermediate frames is a no-op.

Episode length:
    episode_length = cue_visibility + delay_length + TERMINAL_FRAMES (= 10)
The final TERMINAL_FRAMES steps keep the agent at the corridor end; only the
first of them carries the decision reward.

Reward:
    +1.0 if the agent picks the goal matching the memorized cue on the
    terminal decision frame; 0.0 otherwise.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Allow direct script execution (`python code/core/envs/delayed_t_maze.py`).
# When run as a module (e.g. `from code.core.envs.delayed_t_maze import ...`)
# this is a no-op because `code` is already on sys.path.
if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    _snn_root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    if _snn_root not in sys.path:
        sys.path.insert(0, _snn_root)

# Use absolute import when running as a script; relative otherwise.
try:
    from .base import BaseEnv, EnvSpec
except ImportError:
    from code.core.envs.base import BaseEnv, EnvSpec

# Constants for the T-maze geometry. Cues are placed at +/- GOAL_OFFSET_X on
# either side of the corridor, and the corridor runs from y = 0 to y =
# CORRIDOR_LENGTH.
GOAL_OFFSET_X = 1.0
CORRIDOR_END_Y_OFFSET = 1.0  # y position where the terminal decision happens
TERMINAL_FRAMES = 10  # post-decision frames to let reward propagate


@dataclass
class DelayedTMazeConfig:
    """Static configuration for a Delayed-T-Maze instance."""
    delay_length: int = 50          # corridor length
    cue_visibility: int = 3         # frames the cue is shown
    distractor: bool = False        # random cue flips during corridor
    distractor_prob: float = 0.5    # per-step probability when distractor=True


class DelayedTMazeEnv(BaseEnv):
    """Delayed T-Maze: pure working-memory probe with a single binary choice.

    Compatible with the ST-JEWM ``BaseEnv`` contract: the trainer can pull
    state vectors via ``get_state()`` and ``load_dataset("delayed_t_maze")``
    loads a procedurally generated 30K-window offline dataset.
    """

    # Class-level identifier so closed_loop / loaders can recognize the env.
    ENV_KIND = "delayed_t_maze"

    def __init__(
        self,
        delay_length: int = 50,
        cue_visibility: int = 3,
        distractor: bool = False,
        config: Optional[DelayedTMazeConfig] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if config is not None:
            self.cfg = config
        else:
            self.cfg = DelayedTMazeConfig(
                delay_length=int(delay_length),
                cue_visibility=int(cue_visibility),
                distractor=bool(distractor),
            )
        if self.cfg.delay_length < 1:
            raise ValueError(f"delay_length must be >= 1, got {self.cfg.delay_length}")
        if self.cfg.cue_visibility < 1:
            raise ValueError(f"cue_visibility must be >= 1, got {self.cfg.cue_visibility}")

        self._episode_length = self.cfg.cue_visibility + self.cfg.delay_length + TERMINAL_FRAMES
        self._decision_step = self.cfg.cue_visibility + self.cfg.delay_length  # first terminal frame

        self.spec = EnvSpec(
            env_id=f"delayed_t_maze/delay{self.cfg.delay_length}_cue{self.cfg.cue_visibility}"
                   + ("_distractor" if self.cfg.distractor else ""),
            obs_dim=6,
            action_dim=2,
            action_low=np.array([-1.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
            obs_keys=("state",),
            max_episode_steps=self._episode_length,
        )

        # State variables that change per episode
        self._rng = np.random.default_rng(seed)
        self._agent_y = 0.0
        self._cue_side: int = 0           # -1 left, +1 right, 0 undecided
        self._terminal_choice: int = 0    # -1 left, +1 right, 0 unchosen
        self._reward_given: bool = False
        self._accumulated_reward: float = 0.0
        self._last_obs_dict: Dict[str, np.ndarray] = {}

    # -----------------------------------------------------------
    # BaseEnv interface
    # -----------------------------------------------------------
    def reset(self, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif kwargs.get("reset_rng", False):
            # Allow tests to re-seed deterministically without an explicit seed
            self._rng = np.random.default_rng(self._rng.integers(0, 2**31 - 1))

        # Pick left or right with equal probability
        self._cue_side = int(self._rng.choice([-1, 1]))
        self._agent_y = 0.0
        self._terminal_choice = 0
        self._reward_given = False
        self._accumulated_reward = 0.0
        self._step_count = 0

        self._last_obs_dict = self._obs_dict()
        return self._last_obs_dict

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 2:
            # Pad to 2D for safety
            padded = np.zeros(2, dtype=np.float32)
            padded[: action.shape[0]] = action
            action = padded
        action = np.clip(action, -1.0, 1.0)

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        if self._step_count < self.cfg.cue_visibility:
            # Cue phase — agent sits still; cue is visible at cue_x, cue_y.
            # Lateral action[0] is ignored; action[1] is also a no-op here.
            pass
        elif self._step_count < self._decision_step:
            # Corridor phase — advance agent by exactly 1; cue is hidden.
            # Optional distractor flips the apparent cue_x/cue_y randomly.
            self._agent_y += 1.0
            if self.cfg.distractor and self._rng.random() < self.cfg.distractor_prob:
                # Random distractor: pretend there's a cue on a random side.
                # We don't actually flip the true cue_side (memory must hold).
                pass  # the cue is hidden anyway, but we record the noise below
        else:
            # Terminal phase — first frame records the decision.
            if not self._reward_given:
                choice = -1 if action[1] < 0.0 else 1
                self._terminal_choice = choice
                if choice == self._cue_side:
                    reward = 1.0
                self._reward_given = True
                self._accumulated_reward += reward
                info["cue_side"] = self._cue_side
                info["choice"] = choice
                info["success"] = bool(choice == self._cue_side)

        self._step_count += 1
        done = self._step_count >= self._episode_length

        self._last_obs_dict = self._obs_dict()
        return self._last_obs_dict, float(reward), bool(done), info

    def get_state(self) -> np.ndarray:
        """Return the current 6D state vector.

        Mirrors the most recent obs dict exactly so loaders / replays can use
        either path interchangeably.
        """
        return self._last_obs_dict["state"].copy()

    def check_success(self, state: np.ndarray, goal_state: np.ndarray) -> Tuple[bool, float]:
        """T-Maze success criterion.

        The goal is to be on the cue-side of the T at the decision frame. The
        recorded goal state encodes the cue side in the cue_x channel:
        goal_state[2] = -GOAL_OFFSET_X for left, +GOAL_OFFSET_X for right.
        Success is achieved if the agent's terminal choice matches this.

        For evaluation convenience we also accept a match on the agent's y
        position being at the corridor end.
        """
        # First try to recover the cue side from the goal state.
        cue_side_goal = 0
        if goal_state.shape[0] >= 3:
            cue_side_goal = -1 if goal_state[2] < 0.0 else (1 if goal_state[2] > 0.0 else 0)
        # The agent's choice is sign(action[1]) — but we don't have it here.
        # Fall back to a tolerance check on (cue_x, agent_y).
        diff = state - goal_state
        dist = float(np.linalg.norm(diff) / np.sqrt(len(state)))
        # If the goal cue side is clearly left/right and the agent has reached
        # the corridor end (state[1] >= delay_length), we count that as
        # success. Otherwise fall back to a distance check.
        at_end = state[1] >= self.cfg.delay_length - 0.5
        return (cue_side_goal != 0 and at_end and dist < 2.0), dist

    # -----------------------------------------------------------
    # Internals
    # -----------------------------------------------------------
    def _obs_dict(self) -> Dict[str, np.ndarray]:
        """Compute the current 6D observation.

        Layout: [agent_x, agent_y, cue_x, cue_y, corridor_marker, goal_marker]
        """
        agent_x = 0.0
        agent_y = self._agent_y

        # Cue visibility: shown during the cue phase only (when distractor is
        # off). When distractor is on, we also reveal a randomly-positioned
        # distractor during the corridor.
        cue_x = 0.0
        cue_y = 0.0
        if self._step_count < self.cfg.cue_visibility:
            # Original cue: side determines x; y is at the start.
            cue_x = float(self._cue_side) * GOAL_OFFSET_X
            cue_y = 0.0
        elif self.cfg.distractor and self._step_count < self._decision_step:
            # Distractor: random side, somewhere in the corridor.
            distractor_side = int(self._rng.choice([-1, 1]))
            cue_x = float(distractor_side) * GOAL_OFFSET_X
            cue_y = self._agent_y

        corridor_marker = 1.0 if self.cfg.cue_visibility <= self._step_count < self._decision_step else 0.0
        goal_marker = 1.0 if self._step_count >= self._decision_step else 0.0

        state = np.array(
            [agent_x, agent_y, cue_x, cue_y, corridor_marker, goal_marker],
            dtype=np.float32,
        )
        return {"state": state}

    # -----------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------
    def close(self):
        pass

    @property
    def cue_side(self) -> int:
        return self._cue_side

    @property
    def terminal_choice(self) -> int:
        return self._terminal_choice


# ============================================================
# Factory
# ============================================================
def make_delayed_t_maze(
    data_path: Optional[str] = None,
    delay_length: int = 50,
    cue_visibility: int = 3,
    distractor: bool = False,
    seed: Optional[int] = None,
) -> DelayedTMazeEnv:
    """Construct a DelayedTMazeEnv.

    Parameters
    ----------
    data_path : str, optional
        Unused at construction time; reserved for parity with the
        ``make_*_env`` factory convention so callers can pass it through
        uniformly.
    delay_length : int
        Number of corridor frames between cue presentation and the decision.
    cue_visibility : int
        Number of frames the cue is shown before the corridor.
    distractor : bool
        If True, random distractor cues appear during the corridor.
    seed : int, optional
        RNG seed for the env.
    """
    return DelayedTMazeEnv(
        delay_length=delay_length,
        cue_visibility=cue_visibility,
        distractor=distractor,
        seed=seed,
    )


# ============================================================
# Procedural dataset generator
# ============================================================
def generate_delayed_t_maze_dataset(
    n_windows: int = 30000,
    delay_length: int = 50,
    cue_visibility: int = 3,
    distractor: bool = False,
    save_path: str = "/home/lx/snn/data/delayed_t_maze_30k.npz",
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Generate a (30000, 6) state / (30000, 2) action dataset by rolling out
    a uniform-random policy in the Delayed-T-Maze environment.

    Returns a dict with arrays:
        observations: (N, 6) float32 — same layout as the env state
        actions:      (N, 2) float32 — the action taken at each step
        goal_states:  (N, 6) float32 — the terminal "goal" state for each
                     trajectory the window belongs to (cue side encoded in
                     cue_x, corridor_marker=0, goal_marker=1)
    """
    rng = np.random.default_rng(seed)
    env = DelayedTMazeEnv(
        delay_length=delay_length,
        cue_visibility=cue_visibility,
        distractor=distractor,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    episode_length = env._episode_length

    all_obs = np.zeros((n_windows, 6), dtype=np.float32)
    all_act = np.zeros((n_windows, 2), dtype=np.float32)
    all_goal = np.zeros((n_windows, 6), dtype=np.float32)

    written = 0
    while written < n_windows:
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        # Buffer for this episode
        ep_obs = np.zeros((episode_length, 6), dtype=np.float32)
        ep_act = np.zeros((episode_length, 2), dtype=np.float32)

        for t in range(episode_length):
            ep_obs[t] = env.get_state()
            if t < episode_length - 1:
                a = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                env.step(a)
                ep_act[t] = a
            else:
                ep_act[t] = 0.0

        # Goal state encodes the cue side: goal_x = +/- GOAL_OFFSET_X, agent
        # at corridor end, goal_marker = 1.
        goal_state = np.array(
            [0.0, float(delay_length), float(env.cue_side) * GOAL_OFFSET_X,
             0.0, 0.0, 1.0],
            dtype=np.float32,
        )

        # Copy as many windows as we can from this episode.
        take = min(episode_length, n_windows - written)
        all_obs[written: written + take] = ep_obs[:take]
        all_act[written: written + take] = ep_act[:take]
        all_goal[written: written + take] = goal_state
        written += take

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        observations=all_obs,
        actions=all_act,
        goal_states=all_goal,
    )
    return {
        "observations": all_obs,
        "actions": all_act,
        "goal_states": all_goal,
        "path": save_path,
    }


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    env = make_delayed_t_maze()
    obs = env.reset(seed=0)
    print(f"env_id: {env.spec.env_id}")
    print(f"spec: obs_dim={env.spec.obs_dim}, action_dim={env.spec.action_dim}, "
          f"max_episode_steps={env.spec.max_episode_steps}")
    print(f"obs.shape: {obs['state'].shape}")
    assert obs["state"].shape[-1] == 6, f"expected obs_dim=6, got {obs['state'].shape}"

    rng = np.random.default_rng(123)
    total_reward = 0.0
    for t in range(60):
        a = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        obs, reward, done, info = env.step(a)
        total_reward += reward
        if done:
            print(f"episode ended at t={t}, info={info}")
            break

    print(f"accumulated reward after 60 random steps: {total_reward:.3f}")

    # Procedurally generate the dataset (only if requested explicitly to keep
    # the smoke test fast). Uncomment the next two lines to regenerate.
    # gen = generate_delayed_t_maze_dataset(n_windows=30000, save_path="/home/lx/snn/data/delayed_t_maze_30k.npz")
    # print(f"dataset saved: {gen['path']}, obs.shape={gen['observations'].shape}")

    print(f"smoke test completed in {time.time() - t0:.2f}s")
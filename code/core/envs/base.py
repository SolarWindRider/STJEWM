"""Base Env abstraction for ST-JEWM.

Every concrete env wrapper (PushT, TwoRoom, Reacher, Cube, Gym, etc.) must
expose these 4 methods. This is the contract that:

    code/eval/closed_loop.py
    code/eval/lewm_protocol.py
    code/eval/plan_then_render.py
    code/scripts/render_*.py

all depend on.

Design rule: state extraction is the loader's job, not the env's job. The env
just provides raw obs (Dict, Box, etc.) via get_state() and an env-native
success check.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class EnvSpec:
    """Static metadata about an env. Computed once at construction time."""
    env_id: str
    obs_dim: int              # dimensionality of the raw obs (flattened)
    action_dim: int           # dimensionality of the action
    action_low: np.ndarray     # (action_dim,)
    action_high: np.ndarray    # (action_dim,)
    obs_keys: tuple            # e.g. ("state",) or ("proprio", "state")
    max_episode_steps: int     # env-defined timeout


class BaseEnv(ABC):
    """Abstract base for all ST-JEWM env wrappers."""

    def __init__(self):
        self.spec: EnvSpec = None  # set by subclass
        self._step_count: int = 0

    @abstractmethod
    def reset(self, seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Reset the env. Returns initial obs (dict of np.ndarray)."""
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Take a step. Returns (obs, reward, done, info)."""
        ...

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the state vector (1D) used as input to the world model.

        The loader decides which obs keys to use; the env provides them via
        reset()/step() and the subclass assembles the state vector.
        """
        ...

    @abstractmethod
    def check_success(self, state: np.ndarray, goal_state: np.ndarray) -> Tuple[bool, float]:
        """Env-native success test.

        Returns:
            (success: bool, distance: float)
            - success: True iff `state` is within env-defined tolerance of `goal_state`
            - distance: the env-native distance metric (lower = closer to goal)
        """
        ...

    @property
    def step_count(self) -> int:
        return self._step_count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env_id={self.spec.env_id if self.spec else 'unset'})"

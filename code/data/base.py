"""Base window dataset for sliding-window training.

All env-specific loaders (PushT, TwoRoom, Reacher, Cube, etc.) return a
`WindowDataset` instance. The trainer uses only the `__getitem__` and `__len__`
methods, so loader implementations can vary internally as long as they return
the right shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowSpec:
    """Metadata for a windowed dataset."""
    obs_dim: int  # post-pad if pad_obs_to is set, else raw
    action_dim: int
    history_size: int = 3
    goal_offset: int = 25
    max_windows: Optional[int] = None  # cap on number of windows (for fast smoke tests)
    # Generalist-training extensions (preserved when absent)
    obs_dim_original: int = 0  # pre-pad raw env dim (0 means: same as obs_dim)
    action_dim_original: int = 0  # pre-pad raw env dim (0 means: same as action_dim)
    pad_obs_to: Optional[int] = None  # if set, obs is zero-padded to this dim at __init__
    env_id: Optional[str] = None  # which env produced this dataset (for multi-env mixing)


class WindowDataset(Dataset):
    """Generic sliding-window dataset over (obs, action) trajectories.

    Each item is a dict with:
        - "state":      (history_size + goal_offset, obs_dim) float
        - "action":     (history_size + goal_offset - 1, action_dim) float,
                        with a zero-pad at the end (matching the goal step)
        - "goal_state": (obs_dim,) float — the goal state at t + goal_offset
        - "init_state": (obs_dim,) float — the state at the start of the window
    """

    def __init__(
        self,
        obs: np.ndarray,           # (N, obs_dim) flat
        actions: np.ndarray,        # (N, action_dim) flat
        spec: WindowSpec,
        max_windows: Optional[int] = None,
    ):
        # Capture pre-pad dims before any mutation
        if not spec.obs_dim_original:
            spec.obs_dim_original = obs.shape[1]
        if not spec.action_dim_original:
            spec.action_dim_original = actions.shape[1]

        # Pad obs to spec.pad_obs_to (or override pad from arg) if larger than current
        target = spec.pad_obs_to
        if target is not None and obs.shape[1] < target:
            pad_width = target - obs.shape[1]
            obs = np.concatenate(
                [obs.astype(np.float32),
                 np.zeros((obs.shape[0], pad_width), dtype=np.float32)],
                axis=1,
            )
            spec.obs_dim = target  # post-pad dim; trainer reads this

        self.obs = obs.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.spec = spec
        N = len(obs)
        # Window = history_size + goal_offset, so we can have one full window
        # per starting step from t=0 to t=N-window-1.
        window = spec.history_size + spec.goal_offset + 1
        self._max_starts = N - window
        if self._max_starts <= 0:
            raise ValueError(
                f"Dataset too small: N={N}, need > {window}. "
                f"history={spec.history_size}, goal_offset={spec.goal_offset}"
            )
        cap = max_windows or spec.max_windows
        if cap is not None and cap < self._max_starts:
            self._max_starts = cap


    def __len__(self) -> int:
        return self._max_starts

    def __getitem__(self, idx: int) -> dict:
        spec = self.spec
        window = spec.history_size + spec.goal_offset + 1
        s = idx
        e = s + window
        state_window = self.obs[s:e]                            # (window, obs_dim)
        action_window = self.actions[s:e - 1]                    # (window-1, action_dim)
        # Pad action window with one zero row so it has the same time-dim as state
        if action_window.shape[0] < window:
            pad = np.zeros((window - action_window.shape[0], self.actions.shape[1]), dtype=np.float32)
            action_window = np.concatenate([action_window, pad], axis=0)
        return {
            "state": torch.from_numpy(state_window).float(),
            "action": torch.from_numpy(action_window).float(),
            "init_state": torch.from_numpy(self.obs[s]).float(),
            "goal_state": torch.from_numpy(self.obs[s + spec.goal_offset]).float(),
        }

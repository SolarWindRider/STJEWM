"""Multi-env dataset factory for generalist training.

A generalist ST-JEWM model takes one ckpt and serves many envs. To make
that work the trainer needs:

1. A unified obs_dim across all envs — handled by `WindowDataset(pad_obs_to=...)`
   (each per-env loader pads its native obs to a common dim at load time).
2. A unified action_dim — handled here, by post-padding per-env action windows
   in `_ActionPaddedDataset` so the model sees the same `(W, ACTION_DIM)`
   regardless of which env produced the row.

Usage:

    from code.data.multi_env import load_multi_env_dataset
    ds = load_multi_env_dataset(
        env_specs=[{"env_kind": "dmc", "path": "...", "history_size": 1,
                    "goal_offset": 25, "max_windows": 10000, "env_id": "cartpole_2d"},
                   ...],
        pad_obs_to=128, action_dim_target=56,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .loaders import load_dataset


class _ActionPaddedDataset(Dataset):
    """Wraps a WindowDataset and zero-pads action dim AND time length uniformly.

    Two normalizations, all done here so the trainer sees a single uniform
    (B, T_max, D_padded) layout regardless of which env produced the row:

    1. Action dim: pad to action_dim_target (last axis).
    2. Window length: pad to time_target along axis 0.
       (different envs use different history_size + goal_offset, so a single
        batch would otherwise fail to stack.)
    """

    def __init__(self, base: Dataset, action_dim_target: int,
                 time_target: int, obs_dim_target: Optional[int] = None):
        self._base = base
        self._action_dim_target = action_dim_target
        self._time_target = time_target
        self._obs_dim_target = obs_dim_target
        # Spec inherits everything from the base, with action_dim set to target
        self.spec = base.spec
        self.spec.action_dim = action_dim_target
        if not self.spec.action_dim_original:
            self.spec.action_dim_original = getattr(base, "actions", np.zeros((0, 1))).shape[1] if hasattr(base, "actions") else action_dim_target

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._base[idx]
        # ---- state window: pad time axis ----
        state = item["state"]  # (T, D_obs)
        T, D = state.shape
        if T < self._time_target:
            pad = torch.zeros((self._time_target - T, D), dtype=state.dtype)
            state = torch.cat([state, pad], dim=0)
        # ---- action window: pad action dim AND time axis ----
        action = item["action"]  # (T, action_dim_native)
        W, native = action.shape
        if W < self._time_target:
            t_pad = torch.zeros((self._time_target - W, native), dtype=action.dtype)
            action = torch.cat([action, t_pad], dim=0)
            W = self._time_target
        if native < self._action_dim_target:
            pad = torch.zeros((W, self._action_dim_target - native), dtype=action.dtype)
            action = torch.cat([action, pad], dim=-1)
        elif native > self._action_dim_target:
            raise ValueError(
                f"native action_dim {native} > target {self._action_dim_target}; "
                "raise action_dim_target in the spec"
            )
        # ---- init_state / goal_state: pad last dim if needed ----
        init = item["init_state"]
        goal = item["goal_state"]
        if self._obs_dim_target is not None:
            for tag, vec in (("init", init), ("goal", goal)):
                d = vec.shape[-1]
                if d < self._obs_dim_target:
                    pad = torch.zeros(self._obs_dim_target - d, dtype=vec.dtype)
                    vec = torch.cat([vec, pad], dim=-1)
                    if tag == "init":
                        init = vec
                    else:
                        goal = vec
        return {**item, "state": state, "action": action,
                "init_state": init, "goal_state": goal}


def load_multi_env_dataset(
    env_specs: List[Dict[str, Any]],
    pad_obs_to: Optional[int] = None,
    action_dim_target: Optional[int] = None,
    seed: int = 3072,
) -> Dataset:
    """Load a list of per-env datasets and return one ConcatDataset over them.

    Each entry in `env_specs` is a dict:
        {"env_kind": "dmc", "path": "/abs/path.npz", "history_size": 1,
         "goal_offset": 25, "max_windows": 10000, "env_id": "cartpole_2d"}

    Optional loader-specific kwargs can be added (e.g. `state_dim` for reacher).
    The whole dict is forwarded to `load_dataset` as kwargs (after popping the
    bookkeeping keys), so per-loader requirements are honored.

    Returns `ConcatDataset` over `_ActionPaddedDataset` wrappers. Obs windows
    are padded to `pad_obs_to` inside `WindowDataset`; action windows are
    padded to `action_dim_target` here.

    If `action_dim_target` is None, the factory uses
    `max(spec.action_dim_original for spec in env_specs)` — i.e. no padding
    of the largest env, just smaller ones.
    """
    children: List[Dataset] = []
    for entry in env_specs:
        env_id = entry.get("env_id", entry.get("env_kind"))
        kwargs = {k: v for k, v in entry.items() if k not in ("env_id",)}
        kwargs.setdefault("pad_obs_to", pad_obs_to)
        kwargs["env_id"] = env_id
        ds = load_dataset(**kwargs)
        children.append(ds)

    # Auto-pick action_dim_target if not given
    if action_dim_target is None:
        action_dim_target = max(
            (getattr(c, "spec", None) and c.spec.action_dim_original) or 0 for c in children
        )
        action_dim_target = max(action_dim_target, 1)

    # Compute the time-target as max(window length) across children. Each child's
    # window length is history_size + goal_offset + 1; we sample one row to read it.
    time_targets = []
    for c in children:
        s = c[0]
        time_targets.append(s["state"].shape[0])
    time_target = max(time_targets)

    wrapped = [_ActionPaddedDataset(
        c, action_dim_target=action_dim_target,
        time_target=time_target, obs_dim_target=pad_obs_to,
    ) for c in children]
    return ConcatDataset(wrapped)


def load_multi_env_dataset_from_json(
    json_path: str,
    pad_obs_to: Optional[int] = None,
    action_dim_target: Optional[int] = None,
    seed: int = 3072,
) -> Dataset:
    """Convenience: read env_specs from a JSON file and call load_multi_env_dataset.

    Supports two formats:
    (A) A flat list of env-spec dicts (e.g. configs/generalist_16env.json)
    (B) A dict with metadata keys + a "specs" key holding the list
        (e.g. configs/oodc_5m/oodc_F1.json, with _split_name + specs)
    """
    raw = json.loads(Path(json_path).read_text())
    if isinstance(raw, dict):
        specs = raw.get("specs", [])
    else:
        specs = raw
    return load_multi_env_dataset(
        env_specs=specs,
        pad_obs_to=pad_obs_to,
        action_dim_target=action_dim_target,
        seed=seed,
    )

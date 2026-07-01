"""Data loaders, one per env.

Each loader reads the raw dataset file (.h5 / .npz / per-episode dir), extracts
a state vector (per `BENCHMARKS.md` conventions), and returns a `WindowDataset`.

Usage:
    from code.data import load_dataset
    ds = load_dataset("pusht", "/path/to/pusht_expert_train.h5")
    state, action = ds[0]["state"], ds[0]["action"]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .base import WindowDataset, WindowSpec


# ============================================================
# Helper: episode-aware indexing for h5py with ep_idx / step_idx
# ============================================================
def h5_episode_index(h5_path: str):
    """Read (ep_ids, ep_offsets, ep_lens) from a LeWM-style h5 file."""
    with h5py.File(h5_path, "r", swmr=True) as f:
        ep_key = "episode_idx" if "episode_idx" in f else "ep_idx"
        ep_idx = f[ep_key][:]
        steps = f["step_idx"][:]
        if "ep_offset" in f:
            offsets = f["ep_offset"][:]
            lens = f["ep_len"][:]
            n = len(ep_idx)
        else:
            order = np.argsort(ep_idx, kind="stable")
            sorted_ep = ep_idx[order]
            boundaries = np.where(np.diff(sorted_ep) != 0)[0] + 1
            ep_ids_unique = np.split(sorted_ep, boundaries)
            offsets = np.array([grp[0] for grp in ep_ids_unique])
            lens = np.array([len(g) for g in ep_ids_unique])
            n = len(ep_idx)
        return ep_idx, steps, offsets, lens, n


def h5_load_field(h5_path: str, field: str) -> np.ndarray:
    """Read a field from an h5 file."""
    with h5py.File(h5_path, "r", swmr=True) as f:
        return f[field][:]


# ============================================================
# PushT loader
# ============================================================
def load_pusht(
    h5_path: str = "/home/lx/LeWM/data/pusht_expert_train.h5",
    history_size: int = 3,
    goal_offset: int = 25,
    max_windows: Optional[int] = None,
) -> WindowDataset:
    """PushT: use the 7D `state` field (agent pos+vel + block pos+vel+angle+angvel)."""
    state = h5_load_field(h5_path, "state").astype(np.float32)  # (N, 7)
    actions = h5_load_field(h5_path, "action").astype(np.float32)  # (N, 2)
    spec = WindowSpec(
        obs_dim=7,
        action_dim=2,
        history_size=history_size,
        goal_offset=goal_offset,
    )
    return WindowDataset(state, actions, spec, max_windows=max_windows)


# ============================================================
# TwoRoom loader
# ============================================================
def load_tworoom(
    h5_path: str = "/home/lx/LeWM/data/tworoom_extract/tworoom.h5",
    history_size: int = 1,        # LeWM paper uses history=1 for TwoRoom
    goal_offset: int = 100,        # LeWM App. F.1: goal=100 steps ahead for TwoRoom
    max_windows: Optional[int] = None,
) -> WindowDataset:
    """TwoRoom: use the 10D `observation` field."""
    state = h5_load_field(h5_path, "observation").astype(np.float32)
    actions = h5_load_field(h5_path, "action").astype(np.float32)
    spec = WindowSpec(
        obs_dim=10,
        action_dim=2,
        history_size=history_size,
        goal_offset=goal_offset,
    )
    return WindowDataset(state, actions, spec, max_windows=max_windows)


# ============================================================
# Reacher loader (supports BOTH 4D and 204D formats)
# ============================================================
def load_reacher(
    npz_path: str,
    history_size: int = 1,        # LeWM paper uses history=1 for Reacher
    goal_offset: int = 25,         # LeWM App. F.1: goal=25 steps ahead
    state_dim: int = 4,            # which sub-dim of obs to use
    max_windows: Optional[int] = None,
) -> WindowDataset:
    """Reacher: load dm_control 1M reacher_easy.npz.

    Two data formats supported:
        - state_dim=4  : from our 4D simplified data ([qpos(2), target(2)])
        - state_dim=204: from LeWM's 1M data (full DMC proprio)
        - state_dim=5  : [qpos(2), target(2)] (synthesized from 204D by extracting
                          [41:43] for qpos, [142:144] for target) — see notes below.
    """
    d = np.load(npz_path)
    obs_raw = d["observations"][:, 0, :].astype(np.float32)  # (N, obs_raw_dim)
    actions = d["actions"][:, 0, :].astype(np.float32)  # (N, 2)

    if state_dim == 2:
        # 2D: synthesize [qpos(2), target(2)] — add zero target for 4D model compat.
        assert obs_raw.shape[1] == 2, f"Reacher 2D data must be 2D, got {obs_raw.shape[1]}"
        n = obs_raw.shape[0]
        target = np.zeros((n, 2), dtype=np.float32)
        state = np.concatenate([obs_raw[:, :2], target], axis=1)
        spec_obs_dim = 4
    elif state_dim == 4:
        if obs_raw.shape[1] == 4:
            state = obs_raw[:, :4]
        else:
            assert obs_raw.shape[1] >= 2, f"Reacher 4D state but data is {obs_raw.shape[1]}"
            n = obs_raw.shape[0]
            target = np.zeros((n, 2), dtype=np.float32)
            state = np.concatenate([obs_raw[:, :2], target], axis=1)
        spec_obs_dim = 4
    elif state_dim == 5:
        # in some frames), target(2) at [142:144]. Use first 3 of [41:43] is
        # not safe; better to use obs_raw shape to detect.
        # NOTE: For 204D LeWM data, frame 0 is all zeros; only [41,42,142,143] are
        # non-zero in the very first frames. We use those + zero qvel as the 5D state.
        assert obs_raw.shape[1] == 204
        qpos = obs_raw[:, 41:43]
        qvel = obs_raw[:, 14:16]
        target = obs_raw[:, 142:144]
        state = np.concatenate([qpos, qvel, target], axis=1)
        spec_obs_dim = 5
    elif state_dim == 204:
        # Use full 204D proprio
        state = obs_raw
        spec_obs_dim = 204
    else:
        raise ValueError(f"Unsupported state_dim={state_dim}; choose 2, 4, 5, or 204")
    spec = WindowSpec(
        obs_dim=spec_obs_dim,
        action_dim=2,
        history_size=history_size,
        goal_offset=goal_offset,
    )
    return WindowDataset(state, actions, spec, max_windows=max_windows)


# ============================================================
# OGBCube loader
# ============================================================
def load_ogb_metadata(
    data_root: str,
    history_size: int = 3,
    goal_offset: int = 25,
    split: str = "train",
    max_episodes: Optional[int] = None,
    state_field: str = "qpos",     # "qpos" (35D) or "qvel" (32D) or both concatenated
    max_frames_per_ep: int = 200, # OGBench standard: 200 per ep
    max_windows: Optional[int] = None,
) -> WindowDataset:
    """OGBench loader: read per-frame `metadata*.pt` files (PyTorch state dicts).

    Each frame's metadata contains:
        qpos:  (35D) joint positions
        qvel:  (32D) joint velocities
        actions: (5D)

    OGBench data is stored as <root>/<split>/ep<NNNN>/metadata<NNNN>.pt
    """
    split_dir = Path(data_root) / split
    ep_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if max_episodes is not None:
        ep_dirs = ep_dirs[:max_episodes]

    import torch
    obs_list, act_list = [], []
    for ep_dir in ep_dirs:
        md_files = sorted(ep_dir.glob("metadata*.pt"))
        if not md_files:
            continue
        md_files = md_files[:max_frames_per_ep + 1]  # +1 for terminal
        frames_qpos, frames_act = [], []
        for fn in md_files:
            d = torch.load(fn, map_location="cpu", weights_only=False)
            if state_field == "qpos":
                frames_qpos.append(d["qpos"].numpy().astype(np.float32))
            elif state_field == "qvel":
                frames_qpos.append(d["qvel"].numpy().astype(np.float32))
            else:  # "qpos+qvel"
                qp = d["qpos"].numpy().astype(np.float32)
                qv = d["qvel"].numpy().astype(np.float32)
                frames_qpos.append(np.concatenate([qp, qv]))
            frames_act.append(d["actions"].numpy().astype(np.float32))
        if len(frames_qpos) < history_size + goal_offset + 2:
            continue
        obs_list.append(np.stack(frames_qpos, axis=0))
        act_list.append(np.stack(frames_act, axis=0))

    if not obs_list:
        raise FileNotFoundError(f"No valid metadata files under {split_dir}")

    state = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    spec = WindowSpec(
        obs_dim=state.shape[1],
        action_dim=actions.shape[1],
        history_size=history_size,
        goal_offset=goal_offset,
    )
    return WindowDataset(state, actions, spec, max_windows=max_windows)


# Back-compat alias (old name was load_ogb_cube, but it expected .npz which we don't have)
def load_ogb_cube(
    data_root: str = "/home/lx/LeWM/data/ogbench/ogbench_ds/cube",
    history_size: int = 3,
    goal_offset: int = 25,
    split: str = "train",
    max_episodes: Optional[int] = None,
    max_windows: Optional[int] = None,
) -> WindowDataset:
    return load_ogb_metadata(data_root, history_size, goal_offset, split, max_episodes,
                             state_field="qpos", max_frames_per_ep=200,
                             max_windows=max_windows)
# ============================================================
# DMC loader (12 envs, all use npz format with observations[N, 1, D])
# ============================================================
def load_dmc(
    npz_path: str,
    history_size: int = 1,
    goal_offset: int = 25,
    max_windows: Optional[int] = None,
) -> WindowDataset:
    """Generic DMC npz loader. Used for cartpole, pendulum, finger, cheetah, etc."""
    d = np.load(npz_path)
    obs_raw = d["observations"][:, 0, :].astype(np.float32)
    actions = d["actions"][:, 0, :].astype(np.float32)
    spec = WindowSpec(
        obs_dim=obs_raw.shape[1],
        action_dim=actions.shape[1],
        history_size=history_size,
        goal_offset=goal_offset,
    )
    return WindowDataset(obs_raw, actions, spec, max_windows=max_windows)


# ============================================================
# Delayed T-Maze loader (6D obs, 2D action; procedurally generated npz)
# ============================================================
def load_delayed_t_maze(
    npz_path: str = "/home/lx/snn/data/delayed_t_maze_30k.npz",
    history_size: int = 1,
    goal_offset: int = 25,
    max_windows: Optional[int] = None,
) -> WindowDataset:
    """Delayed T-Maze loader.

    State vector layout (6D):
        [agent_x, agent_y, cue_x, cue_y, corridor_marker, goal_marker]
    Actions (2D):
        [forward_command, lateral_choice]
        The forward component is always saturated during the corridor; the
        lateral component only matters on the terminal decision frame.
    """
    d = np.load(npz_path)
    observations = d["observations"].astype(np.float32)  # (N, 6)
    actions = d["actions"].astype(np.float32)            # (N, 2)
    spec = WindowSpec(
        obs_dim=6,
        action_dim=2,
        history_size=history_size,
        goal_offset=goal_offset,
    )
    return WindowDataset(observations, actions, spec, max_windows=max_windows)

# ============================================================
# Gym loader (live environment, no offline data file)
# ============================================================
class GymLiveDataset(Dataset):
    """For gym classic-control envs: collect data on-the-fly via random policy.

    Use only for sanity testing; for the actual NMI submission we use stable_worldmodel
    pre-collected expert data.
    """

    def __init__(
        self,
        env_id: str,
        n_episodes: int = 50,
        max_episode_steps: int = 500,
        history_size: int = 1,
        goal_offset: int = 25,
        seed: int = 42,
    ):
        import gymnasium as gym
        self.spec = WindowSpec(
            obs_dim=-1,  # determined dynamically
            action_dim=-1,
            history_size=history_size,
            goal_offset=goal_offset,
        )
        rng = np.random.default_rng(seed)
        env = gym.make(env_id)
        obs_dim = int(np.prod(env.observation_space.shape))
        if hasattr(env.action_space, "n"):
            action_dim = int(env.action_space.n)
        else:
            action_dim = int(env.action_space.shape[0])
        self.spec.obs_dim = obs_dim
        self.spec.action_dim = action_dim

        all_obs, all_act = [], []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            for t in range(max_episode_steps):
                if hasattr(env.action_space, "n"):
                    a = rng.integers(0, action_dim)
                    a_onehot = np.zeros(action_dim, dtype=np.float32)
                    a_onehot[a] = 1.0
                    all_act.append(a_onehot)
                else:
                    a = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                    all_act.append(a)
                all_obs.append(np.asarray(obs, dtype=np.float32).flatten())
                obs, _, term, trunc, _ = env.step(a if not hasattr(env.action_space, "n") else int(a))
                if term or trunc:
                    break
        env.close()

        self.obs = np.stack(all_obs, axis=0)
        self.actions = np.stack(all_act, axis=0)
        N = len(self.obs)
        window = history_size + goal_offset + 1
        self._max_starts = N - window

    def __len__(self):
        return self._max_starts

    def __getitem__(self, idx):
        spec = self.spec
        window = spec.history_size + spec.goal_offset + 1
        s = idx
        e = s + window
        state_window = self.obs[s:e]
        action_window = self.actions[s:e - 1]
        if action_window.shape[0] < window:
            pad = np.zeros((window - action_window.shape[0], self.actions.shape[1]), dtype=np.float32)
            action_window = np.concatenate([action_window, pad], axis=0)
        return {
            "state": torch.from_numpy(state_window).float(),
            "action": torch.from_numpy(action_window).float(),
            "init_state": torch.from_numpy(self.obs[s]).float(),
            "goal_state": torch.from_numpy(self.obs[s + spec.goal_offset]).float(),
        }


def load_gym_live(
    env_id: str,
    n_episodes: int = 50,
    **kwargs,
) -> GymLiveDataset:
    """Wrapper for `GymLiveDataset`. Use only for smoke tests."""
    return GymLiveDataset(env_id, n_episodes=n_episodes, **kwargs)


# ============================================================
# 3D mujoco rollouts loader (manipulator, finger, cheetah, walker, etc.)
# ============================================================
def load_mujoco_3d(
    npz_path: str,
    history_size: int = 3,
    goal_offset: int = 25,
    max_windows: Optional[int] = None,
) -> WindowDataset:
    """Load our own 3D rollouts (from stage38_gen_3d_rollouts.py output).

    Format: npz with 'observations' (N, 1, D), 'actions' (N, 1, A), etc.
    """
    d = np.load(npz_path)
    obs_raw = d["observations"][:, 0, :].astype(np.float32)
    actions = d["actions"][:, 0, :].astype(np.float32)
    spec = WindowSpec(
        obs_dim=obs_raw.shape[1],
        action_dim=actions.shape[1],
        history_size=history_size,
        goal_offset=goal_offset,
    )
    return WindowDataset(obs_raw, actions, spec, max_windows=max_windows)


# ============================================================
# Factory
# ============================================================
def load_dataset(
    env_kind: str,
    path: str = None,
    **kwargs,
) -> Dataset:
    """Top-level factory: load_dataset("pusht") / load_dataset("reacher_4d", path=...) / etc.

    Supported env_kind values:
        pusht           LeWM PushT h5 (state 7D)
        tworoom         LeWM TwoRoom h5 (state 10D)
        reacher_4d      our 4D reacher (4D state)
        reacher_lewm    LeWM 1M reacher (5D state, synthesized from 204D proprio)
        reacher_full    LeWM 1M reacher (full 204D proprio)
        ogb_cube        OGBench Cube (env-collected, 28D obs + target from reset)
        ogb_cube_env   OGBench Cube via env, goal from info["target"]
        ogb_cube_meta   OGBench Cube via metadata*.pt files (35D qpos)
        ogb_scene       OGBench Scene per-episode metadata*.pt
        dmc             generic DMC npz (cartpole/pendulum/etc)
        mujoco_3d       our 3D rollouts npz
        delayed_t_maze   synthetic Delayed-T-Maze npz (state 6D, 2D action)
        gym_live        gym env_id (collects random data on the fly)
    """
    if env_kind == "pusht":
        return load_pusht(path or "/home/lx/LeWM/data/pusht_expert_train.h5", **kwargs)
    if env_kind == "tworoom":
        return load_tworoom(path or "/home/lx/LeWM/data/tworoom_extract/tworoom.h5", **kwargs)
    if env_kind == "reacher_4d":
        assert path is not None
        return load_reacher(path, state_dim=4, **kwargs)
    if env_kind == "reacher_lewm":
        assert path is not None
        return load_reacher(path, state_dim=5, **kwargs)
    if env_kind == "reacher_full":
        assert path is not None
        return load_reacher(path, state_dim=204, **kwargs)
    if env_kind == "ogb_cube":
        return load_ogb_cube(path or "/home/lx/LeWM/data/ogbench/ogbench_ds/cube", **kwargs)
    if env_kind == "ogb_cube_env":
        return load_ogb_cube_env(**kwargs)
    if env_kind == "ogb_scene":
        return load_ogb_metadata(path or "/home/lx/LeWM/data/ogbench/ogbench_ds/scene", **kwargs)
    if env_kind == "ogb_scene_env":
        return load_ogb_scene_env(**kwargs)
    if env_kind == "dmc":
        assert path is not None
        return load_dmc(path, **kwargs)
    if env_kind == "mujoco_3d":
        assert path is not None
        return load_mujoco_3d(path, **kwargs)
    if env_kind == "delayed_t_maze":
        return load_delayed_t_maze(path or "/home/lx/snn/data/delayed_t_maze_30k.npz", **kwargs)
    if env_kind == "gym_live":
        assert path is not None  # env_id
        return load_gym_live(path, **kwargs)
    raise ValueError(f"Unknown env_kind: {env_kind}")


# ============================================================
# OGBench loader (env-based): reset env to get init obs + target obs
# ============================================================
def load_ogb_env_based(
    env_id: str = "swm/OGBCube-v0",
    n_episodes: int = 100,
    max_steps_per_ep: int = 200,
    seed: int = 42,
) -> "WindowDataset":
    """OGBench loader using the actual env: collect (obs, action) pairs.

    Each episode: env.reset() gives init_obs (28D) + info["target"] (28D goal).
    We run random actions for `max_steps_per_ep` steps, collecting (obs, action).
    """
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium as gym
    import stable_worldmodel  # noqa

    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    all_obs, all_act, all_init, all_goal = [], [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        init_obs = obs.astype(np.float32)
        target_obs = info["target"].astype(np.float32)
        for t in range(max_steps_per_ep):
            a = env.action_space.sample().astype(np.float32)
            all_obs.append(obs.astype(np.float32))
            all_act.append(a)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc: break
        all_init.append(init_obs)
        all_goal.append(target_obs)
    env.close()

    state = np.stack(all_obs, axis=0)
    actions = np.stack(all_act, axis=0)
    spec = WindowSpec(
        obs_dim=state.shape[1],
        action_dim=actions.shape[1],
        history_size=3,
        goal_offset=25,
    )
    # Wrap to add init/goal fields
    class _OGBWindowDataset(WindowDataset):
        def __init__(self, states, acts, init_states, goal_states, spec):
            self.obs = states
            self.actions = acts
            self.init_states = init_states
            self.goal_states = goal_states
            self.spec = spec
            self._ep_boundaries = [0]
            cur = 0
            for s in init_states:
                cur += max_steps_per_ep
                self._ep_boundaries.append(cur)
            self._max_starts = max(0, len(states) - spec.history_size - spec.goal_offset - 1)
        def __len__(self):
            return self._max_starts
        def __getitem__(self, idx):
            spec = self.spec
            window = spec.history_size + spec.goal_offset + 1
            s = idx
            e = s + window
            state_window = self.obs[s:e]
            action_window = self.actions[s:e - 1]
            if action_window.shape[0] < window:
                pad = np.zeros((window - action_window.shape[0], self.actions.shape[1]), dtype=np.float32)
                action_window = np.concatenate([action_window, pad], axis=0)
            # Find which episode this index belongs to
            ep_idx = 0
            for i in range(len(self._ep_boundaries) - 1):
                if self._ep_boundaries[i] <= s < self._ep_boundaries[i + 1]:
                    ep_idx = i
                    break
            return {
                "state": torch.from_numpy(state_window).float(),
                "action": torch.from_numpy(action_window).float(),
                "init_state": torch.from_numpy(self.init_states[ep_idx]).float(),
                "goal_state": torch.from_numpy(self.goal_states[ep_idx]).float(),
            }
    return _OGBWindowDataset(state, actions, all_init, all_goal, spec)


def load_ogb_cube_env(
    n_episodes: int = 100,
    max_steps_per_ep: int = 200,
    seed: int = 42,
    **kwargs,
) -> "WindowDataset":
    return load_ogb_env_based("swm/OGBCube-v0", n_episodes, max_steps_per_ep, seed)


def load_ogb_scene_env(
    n_episodes: int = 100,
    max_steps_per_ep: int = 200,
    seed: int = 42,
    **kwargs,
) -> "WindowDataset":
    return load_ogb_env_based("swm/OGBScene-v0", n_episodes, max_steps_per_ep, seed)

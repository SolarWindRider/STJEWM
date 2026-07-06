"""Linear probe on frozen encoder outputs.

Train a single Linear layer (frozen encoder + single Linear head, 5 epochs
of Adam) to predict physical / event variables from a model's latent embedding.

Usage:
    python -m code.scripts.probe --env <env> --model <model \\
            --probe-target <target> --out <json>

Targets (window-level, one (z, y) per window; MSE loss, R^2 metric):
    position         predict state[:nq]  (configurable per-env slice, see ENV_PROBE)
    velocity         predict state[nq:nq+nv]   (skipped for pixel-input)
    contact          predict a binary contact flag (window-level, from state[1]-state[0])
    future_k         predict state[t + k=10]
    goal_direction   predict (goal - state) / ||...||

Event-type targets (per-step, one (z_t, y_t) per timestep in each window;
BCE loss + accuracy metric for binary, MSE + R^2 for continuous):
    event_contact            ||state[t+1]-state[t]||_inf > per-window state-std
    event_persistent         ||state[t+1]-state[t]||_2 > median(diffs) + 1*MAD(diffs)
    event_high_motion        ||state[t+1]-state[t]||_2 in top quartile of window diffs
    event_low_motion         ||state[t+1]-state[t]||_2 in bottom quartile
    event_future_k5          state[t+5] is an event_contact step (look-ahead)
    event_future_k10         state[t+10] is an event_contact step
    event_vel_above_median   ||state[t]-state[t-1]||_2 above window median diff
    event_room_entered       x crosses room-divider wall (tworoom)
    event_block_near_target  ||block-target|| < 0.3 (pusht, normalized)
    event_cue_state          corridor_marker (idx 4) = 1 (delayed_t_maze)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/lx/snn")


# ============================================================
# Env registry (mirrors code/scripts/train_all.sh ENVS)
# ============================================================
# (env_kind, data_path, history_size, goal_offset)
ENV_REGISTRY: dict[str, tuple[str, str, int, int]] = {
    "pusht":           ("pusht",       "/home/lx/LeWM/data/pusht_expert_train.h5",                              1, 100),
    "tworoom":         ("tworoom",     "/home/lx/LeWM/data/tworoom_extract/tworoom.h5",                          1, 100),
    "reacher":         ("reacher_4d",  "/home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz",         1,  25),
    "delayed_t_maze":  ("delayed_t_maze", "/home/lx/snn/data/delayed_t_maze_30k.npz",                          1,  25),
    "cartpole_2d":     ("dmc",         "/home/lx/snn/data/dm_control/cartpole_250k.npz",                        1,  25),
    "pendulum_2d":     ("dmc",         "/home/lx/snn/data/dm_control/pendulum_250k.npz",                        1,  25),
    "finger":          ("dmc",         "/home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz",         1,  25),
    "ball_in_cup":     ("dmc",         "/home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz",    1,  25),
    "cheetah":         ("dmc",         "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz",        1,  25),
    "walker":          ("dmc",         "/home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz",         1,  25),
    "hopper":          ("dmc",         "/home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz",         1,  25),
    "quadruped":       ("dmc",         "/home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz",      1,  25),
    "humanoid":        ("dmc",         "/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz",       1,  25),
    "humanoid_CMU":    ("dmc",         "/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz",   1,  25),
    "dog":             ("dmc",         "/home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz",            1,  25),
    "fish":            ("dmc",         "/home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz",           1,  25),
    "stacker":         ("dmc",         "/home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz",        1,  25),
}


# Per-env probe-target slices (start, end) for the *position* dim.
#   For DMC envs, "position" = qpos, "velocity" = qvel.
#   We only have flat state vectors from the .npz; we approximate the
#   split by the conventional nq values from dm_control.suite.
ENV_PROBE: dict[str, dict] = {
    "pusht":        {"pos": (0, 5),  "vel": (5, 7),  "obs_dim": 7},   # [block(3)+target(2)] + [vel(2)]
    "tworoom":      {"pos": (0, 4),  "vel": (4, 6),  "obs_dim": 10},  # agent(2) + goal(2) + vel(2)+...
    # For DMC envs the .npz state is qpos-only (or sin/cos-encoded), NOT qvel.
    # "velocity" target is the discrete 1-step Δstate over a subset of dims
    # (a velocity proxy that works regardless of whether qvel is stored).
    "reacher":      {"pos": (0, 2),  "vel": None,    "obs_dim": 4},   # 4-D state, no qvel stored
    "cartpole_2d":  {"pos": (0, 2),  "vel": None,    "obs_dim": 2},   # 2-D state
    "pendulum_2d":  {"pos": (0, 2),  "vel": None,    "obs_dim": 2},   # 2-D state
    "finger":       {"pos": (0, 3),  "vel": None,    "obs_dim": 3},   # 3-D state
    "ball_in_cup":  {"pos": (0, 4),  "vel": None,    "obs_dim": 4},   # 4-D state
    "cheetah":      {"pos": (0, 9),  "vel": (0, 3),  "obs_dim": 9},   # 9-D state, Δ over first 3
    "walker":       {"pos": (0, 9),  "vel": (0, 3),  "obs_dim": 9},
    "hopper":       {"pos": (0, 7),  "vel": (0, 3),  "obs_dim": 7},
    "quadruped":    {"pos": (0, 12), "vel": (0, 6),  "obs_dim": 30},  # 30-D state, Δ over first 6
    "humanoid":     {"pos": (0, 14), "vel": (0, 6),  "obs_dim": 28},  # 28-D, Δ over first 6
    "humanoid_CMU": {"pos": (0, 27), "vel": (0, 6),  "obs_dim": 63},  # 63-D, Δ over first 6
    "dog":          {"pos": (0, 19), "vel": (0, 6),  "obs_dim": 87},  # 87-D, Δ over first 6
    "fish":         {"pos": (0, 7),  "vel": (0, 3),  "obs_dim": 14},  # 14-D, Δ over first 3
    "stacker":      {"pos": (0, 10), "vel": (0, 3),  "obs_dim": 20},  # 20-D, Δ over first 3
    "delayed_t_maze": {"pos": (0, 2), "vel": None,   "obs_dim": 6},   # agent(2) + cue(2) + corridor/goal
}


# ============================================================
# Event-type probe registry
# ============================================================
# Per-env list of event-type targets. Each is binary unless noted.
#
# Conventions:
#   * event_contact            : ||state[t+1] - state[t]||_inf > window-state-std
#   * event_persistent         : ||state[t+1] - state[t]||_2 > median(diffs) + 1*MAD(diffs)
#   * event_high_motion        : ||state[t+1] - state[t]||_2 in top quartile of window diffs
#   * event_low_motion         : ||state[t+1] - state[t]||_2 in bottom quartile
#   * event_future_k5/k10      : is state[t+k] an event_contact step?
#   * event_vel_above_median   : ||state[t] - state[t-1]||_2 above window median
#   * event_room_entered       : x crosses room-divider wall (tworoom)
#   * event_block_near_target  : ||block-target|| < 0.3 normalized (pusht)
#   * event_cue_state          : corridor_marker (idx 4) = 1 (delayed_t_maze)
EVENT_PROBES_PER_ENV: dict[str, list[str]] = {
    # DMC contact-rich envs
    "ball_in_cup":     ["event_contact", "event_high_motion", "event_future_k5"],
    "cartpole_2d":     ["event_contact", "event_high_motion", "event_future_k5"],
    "cheetah":         ["event_high_motion", "event_low_motion", "event_future_k10"],
    "finger":          ["event_contact", "event_high_motion", "event_future_k5"],
    "pendulum_2d":     ["event_high_motion", "event_future_k10", "event_persistent"],
    "walker":          ["event_contact", "event_high_motion", "event_future_k10"],
    "hopper":          ["event_contact", "event_high_motion", "event_future_k10"],
    "quadruped":       ["event_high_motion", "event_low_motion", "event_future_k10"],
    "humanoid":        ["event_high_motion", "event_low_motion", "event_future_k10"],
    "humanoid_CMU":    ["event_high_motion", "event_low_motion", "event_future_k10"],
    "dog":             ["event_high_motion", "event_future_k10", "event_persistent"],
    "fish":            ["event_high_motion", "event_low_motion", "event_future_k10"],
    "stacker":         ["event_contact", "event_high_motion", "event_future_k10"],
    "reacher":         ["event_high_motion", "event_low_motion", "event_future_k5"],
    # Non-DMC event envs
    "pusht":           ["event_contact", "event_block_near_target", "event_future_k10"],
    "tworoom":         ["event_room_entered", "event_high_motion", "event_future_k5"],
    "delayed_t_maze":  ["event_cue_state", "event_future_k5", "event_high_motion"],
}


# Targets that produce a binary {0,1} label (BCE loss + accuracy metric).
EVENT_BINARY_TARGETS: set[str] = {
    "event_contact", "event_persistent", "event_high_motion", "event_low_motion",
    "event_future_k5", "event_future_k10", "event_vel_above_median",
    "event_room_entered", "event_block_near_target", "event_cue_state",
}


# All targets that the probe can produce.
ALL_PROBE_TARGETS: list[str] = [
    # window-level (legacy)
    "position", "velocity", "contact", "future_k", "goal_direction",
    # per-step event-type
    "event_contact", "event_persistent", "event_high_motion", "event_low_motion",
    "event_future_k5", "event_future_k10", "event_vel_above_median",
    "event_room_entered", "event_block_near_target", "event_cue_state",
]
# ============================================================
# Helpers
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probe on frozen encoder outputs.")
    p.add_argument("--env", required=True, choices=sorted(ENV_REGISTRY.keys()))
    p.add_argument("--model", required=True, help="Model dir name, e.g. stjewm_v2.")
    p.add_argument(
        "--probe-target",
        required=True,
        choices=ALL_PROBE_TARGETS,
    )
    p.add_argument("--ckpt", default=None,
                   help="Override checkpoint path (default: results/<env>/<model>/final.pt).")
    p.add_argument("--future-k", type=int, default=10,
                   help="When --probe-target=future_k, predict state at t+k.")
    p.add_argument("--out", required=True, help="Path to write JSON result.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-windows", type=int, default=5000,
                   help="Cap dataset size to keep probe fast.")
    p.add_argument("--device", default="cpu",
                   help="cpu or cuda (probe is tiny, cpu is fine).")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--pad-obs-to", type=int, default=None,
                   help="Override state_dim with this padded dim (for generalist ckpts).")
    p.add_argument("--action-dim-eval", type=int, default=None,
                   help="Override action_dim with this value (for generalist ckpts).")
    return p.parse_args()


def build_model(model_name: str, state_dim: int, action_dim: int, ck_args: dict,
                ck_state_dict: dict | None = None):
    """Build the model exactly as code/train/train.py and code/eval/closed_loop.py do.

    For GRU / MLP, the n_layers stored in ck_args can be wrong (training
    bookkeeping drift). We prefer the actual layer count from the state_dict
    when available.
    """
    def _state_dict_n_layers(prefix: str) -> int | None:
        if ck_state_dict is None:
            return None
        indices = []
        for k in ck_state_dict:
            if k.startswith(prefix + "."):
                rest = k[len(prefix) + 1:]
                if rest.startswith("weight_ih_l") or rest.startswith("weight_hh_l"):
                    try:
                        indices.append(int(rest.split("_l")[-1]))
                    except ValueError:
                        pass
        return max(indices) + 1 if indices else None

    if model_name.startswith("lewm"):
        embed_dim = ck_args.get("embed_dim", 256)
        num_layers = ck_args.get("n_layers", 4)
        return LeWMTransformerBaseline(
            state_dim=state_dim, action_dim=action_dim,
            embed_dim=embed_dim, num_layers=num_layers, num_heads=8,
        )
    if model_name.startswith("gru"):
        from code.gru_baseline import GRUBaseline
        hidden_dim = ck_args.get("hidden_dim", 576)
        num_layers = _state_dict_n_layers("gru") or ck_args.get("n_layers", 3)
        return GRUBaseline(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, num_layers=num_layers,
        )
    if model_name.startswith("mlp"):
        from code.mlp_baseline import MLPBaseline
        hidden_dim = ck_args.get("hidden_dim", 576)
        num_layers = _state_dict_n_layers("mlp_cells") or _state_dict_n_layers("mlp") or ck_args.get("n_layers", 4)
        emb_dim = ck_args.get("emb_dim", 192)
        return MLPBaseline(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, num_layers=num_layers, emb_dim=emb_dim,
        )
    if model_name.startswith("slt_lif_mpc_trace"):
        from code.slt_lif_mpc_baseline import make_slt_lif_mpc_trace
        n_layers = ck_args.get("n_layers", 4)
        return make_slt_lif_mpc_trace(
            state_dim=state_dim, action_dim=action_dim,
            d_in=192, embed_dim=192, n_layers=n_layers, trace_beta=0.9, k_avg=4,
        )
    if model_name.startswith("slt_lif_mpc_free"):
        from code.slt_lif_mpc_baseline import make_slt_lif_mpc_free
        n_layers = ck_args.get("n_layers", 4)
        return make_slt_lif_mpc_free(
            state_dim=state_dim, action_dim=action_dim,
            d_in=192, embed_dim=192, n_layers=n_layers, trace_beta=0.9,
        )
    if model_name.startswith("spikedreamer"):
        from code.spikedreamer_baseline import make_spikedreamer
        n_layers = ck_args.get("n_layers", 4)
        return make_spikedreamer(
            state_dim=state_dim, action_dim=action_dim,
            d_snn=128, d_tx=192, num_layers=n_layers, num_heads=8,
        )
    if model_name.startswith("cubifae"):
        from code.cubifae_baseline import CubifAEBaseline
        n_layers = ck_args.get("n_layers", 4)
        return CubifAEBaseline(
            state_dim=state_dim, action_dim=action_dim,
            d_hid=192, n_layers=n_layers,
        )
    from code.stjewm import STJEWM
    n_layers = ck_args.get("n_layers", 4)
    return STJEWM(
        d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
        state_dim=state_dim, cell_n_layers=n_layers, n_d=3,
        trace_beta=0.9, freeze_encoder=True,
    )


def collect_latents_and_targets(
    model, dataset, action_dim: int, probe_dim: int,
    target_kind: str, env: str, device: str, k: int = 10,
    max_windows: int = 5000,
):
    """Walk the dataset, run model.encode() per window, return (Z, Y) tensors.

    target_kind: position | velocity | future_k | goal_direction | contact
    """
    n = min(len(dataset), max_windows)
    Zs, Ys = [], []
    pos_slice = ENV_PROBE[env]["pos"]
    vel_slice = ENV_PROBE[env].get("vel")

    # Walk the dataset in batches of BATCH windows; each window is a (T, obs_dim)
    # trajectory. We mean-pool the per-timestep emb across T for the probe input.
    BATCH = 64
    for batch_start in range(0, n, BATCH):
        batch_end = min(batch_start + BATCH, n)
        # Truncate the encode input to the first `history_size + 1` frames; that's
        # the only part the encoder actually needs. The full window is still used
        # for target extraction (init_state / goal_state / future_k).
        h_trim = dataset.spec.history_size + 1 if hasattr(dataset, "spec") else 2
        s_list, a_list, target_list = [], [], []
        for i in range(batch_start, batch_end):
            item = dataset[i]
            s_full = item["state"]            # (T, obs_dim)
            a_full = item["action"]           # (T, action_dim)
            s_list.append(s_full[:h_trim])    # trimmed for encode
            a_list.append(a_full[:h_trim])
            init_state = item["init_state"]
            goal_state = item["goal_state"]
            if target_kind == "position":
                tgt = init_state[pos_slice[0]: pos_slice[1]]
            elif target_kind == "velocity":
                # Predict Δstate over the vel slice: state[t+1] - state[t]
                # (1-step discrete velocity proxy; works whether or not the env
                #  has a true qvel channel — we always have the full state vector.)
                if vel_slice is None:
                    return None, None, f"no velocity slice for env={env}"
                v0 = s_full[0, vel_slice[0]: vel_slice[1]]
                v1 = s_full[1, vel_slice[0]: vel_slice[1]]
                tgt = v1 - v0
            elif target_kind == "future_k":
                t_idx = min(k, s_full.shape[0] - 1)
                tgt = s_full[t_idx, pos_slice[0]: pos_slice[1]]
            elif target_kind == "goal_direction":
                diff = (goal_state - init_state)[pos_slice[0]: pos_slice[1]]
                norm = diff.norm() + 1e-8
                tgt = diff / norm
            elif target_kind == "contact":
                # Predict whether ||state[t+1] - state[t]||_∞ > std(state)
                # (a contact-like event: state changes more than its typical
                #  within-trajectory variation.) Binary {0,1} target.
                diff = (s_full[1] - s_full[0]).abs().max()
                state_std = s_full.std(dim=0).mean()
                tgt = torch.tensor([float(diff > state_std)])
            else:
                return None, None, f"unknown target {target_kind}"
            target_list.append(tgt)
        # Pad T to a common length
        T_max = max(s.shape[0] for s in s_list)
        obs_dim = s_list[0].shape[-1]
        action_dim_ = a_list[0].shape[-1]
        s_pad = torch.zeros(len(s_list), T_max, obs_dim, dtype=torch.float32)
        a_pad = torch.zeros(len(a_list), T_max, action_dim_, dtype=torch.float32)
        for j, (s, a) in enumerate(zip(s_list, a_list)):
            s_pad[j, : s.shape[0]] = s
            a_pad[j, : a.shape[0]] = a
        s_dev = s_pad.to(device)
        a_dev = a_pad.to(device)
        with torch.no_grad():
            enc = model.encode(s_dev, a_dev)
        z = enc["emb"]                               # (B, T_max, D)
        z_pooled = z.mean(dim=1)                     # (B, D)
        for j in range(z_pooled.shape[0]):
            Zs.append(z_pooled[j].cpu())
        for tgt in target_list:
            Ys.append(tgt.cpu())
    if not Zs:
        return None, None, "empty dataset"
    Z = torch.stack(Zs, dim=0)        # (N, D)
    Y = torch.stack(Ys, dim=0)        # (N, probe_dim)
    return Z, Y, None

# ============================================================
# Per-step event-type target extractors
# ============================================================
def _per_step_diffs(state_window: torch.Tensor) -> torch.Tensor:
    """Per-step L2 diff ||state[t+1] - state[t]||_2 over the window.

    Returns a (T-1,) tensor. state_window shape: (T, obs_dim).
    """
    diff = state_window[1:] - state_window[:-1]
    return diff.norm(dim=-1)


def _per_step_event_target(state_window: torch.Tensor, target_kind: str, env: str,
                            goal_state: torch.Tensor) -> torch.Tensor | None:
    """Compute a per-step (T,) binary target tensor for one window.

    Returns None if the target is not applicable to this env.
    """
    if target_kind == "event_contact":
        # Per-step: ||state[t+1]-state[t]||_inf above per-window p90 of diffs.
        # (state_std-based threshold is too strict for the smooth DMC rollouts
        # we use; a per-window percentile keeps ~10% positive in every window.)
        diffs = _per_step_diffs(state_window)  # (T-1,)
        if diffs.numel() < 4:
            return None
        thr = torch.quantile(diffs, 0.90)
        return (diffs >= thr).float()
    if target_kind == "event_persistent":
        diffs = _per_step_diffs(state_window)
        if diffs.numel() < 2:
            return None
        med = diffs.median()
        mad = (diffs - med).abs().median()
        thr = med + 1.0 * mad
        return (diffs > thr).float()
    if target_kind == "event_high_motion":
        diffs = _per_step_diffs(state_window)
        if diffs.numel() < 4:
            return None
        q75 = torch.quantile(diffs, 0.75)
        return (diffs >= q75).float()
    if target_kind == "event_low_motion":
        diffs = _per_step_diffs(state_window)
        if diffs.numel() < 4:
            return None
        q25 = torch.quantile(diffs, 0.25)
        return (diffs <= q25).float()
    if target_kind == "event_vel_above_median":
        diffs = _per_step_diffs(state_window)
        if diffs.numel() == 0:
            return None
        med = diffs.median()
        return (diffs > med).float()
    if target_kind in ("event_future_k5", "event_future_k10"):
        k = 5 if target_kind == "event_future_k5" else 10
        # At step t, predict whether state[t+k] is an event_contact step.
        diffs = _per_step_diffs(state_window)
        if diffs.numel() < k + 4:
            return None
        # event-contact label at each step (T-1,) — p90 threshold for fairness
        # with the rest of the event targets.
        thr = torch.quantile(diffs, 0.90)
        evt = (diffs >= thr).float()             # (T-1,)
        # For step t in [0, T-1-k], target = evt[t+k] (if t+k < T-1)
        # Length of output = T-1-k (the last k steps have no t+k label).
        out_len = evt.numel() - k
        if out_len <= 0:
            return None
        out = evt[k: k + out_len]
        return out
    if target_kind == "event_room_entered":
        # tworoom: agent x is dim 0. Wall at ~112. Transition = x crosses the wall.
        if env != "tworoom":
            return None
        x = state_window[:, 0]
        prev_room = (x[:-1] < 112).float()
        next_room = (x[1:] < 112).float()
        crossed = (prev_room != next_room).float()
        return crossed
    if target_kind == "event_block_near_target":
        # pusht: state = [block(3)+target(2)+vel(2)]. block-target dist < 0.3.
        if env != "pusht":
            return None
        block = state_window[:, 0:3]      # x,y,angle
        target = state_window[:, 3:5]     # x,y
        # PushT state is in pixel coordinates (block x,y in [0,224]); the
        # block is "near the target" when within 200 px (~25% of typical
        # block-to-target distance; base rate ~11%).
        diff = block[:, :2] - target
        dist = diff.norm(dim=-1)
        near = (dist < 200.0).float()
        return near[:-1]
    if target_kind == "event_cue_state":
        # delayed_t_maze: corridor_marker is at index 4. 1 = corridor visible.
        if env != "delayed_t_maze":
            return None
        corridor = state_window[:, 4]
        cue = (corridor > 0.5).float()
        return cue[:-1]
    return None

def collect_event_targets(
    model, dataset, target_kind: str, env: str, device: str,
    max_windows: int = 5000,
    pad_obs_to: int | None = None,
    action_dim_eval: int | None = None,
):
    """Walk the dataset, run model.encode() per window, return per-step (Z, Y).

    For each window we feed the FULL trajectory to model.encode() (padded to
    the longest window in the batch) and get (B, T, D) embeddings. We then
    compute a per-step binary target of length T-1 (inter-step diff axis)
    and align them with z[:, :T-1, :].

    Returns (Z, Y, binary_flag, err) where:
        Z:            (sum_windows x (T-1), D)
        Y:            (sum_windows x (T-1), 1)
        binary_flag:  True if target is binary (use BCE + accuracy)
    """
    n = min(len(dataset), max_windows)
    Zs, Ys = [], []
    BATCH = 32  # smaller because we expand to per-step T
    for batch_start in range(0, n, BATCH):
        batch_end = min(batch_start + BATCH, n)
        s_list, a_list, target_list = [], [], []
        T_real_list = []
        for i in range(batch_start, batch_end):
            item = dataset[i]
            s_full = item["state"]            # (T_full, obs_dim)
            a_full = item["action"]           # (T_full, action_dim)
            goal_state = item["goal_state"]
            T_real_list.append(s_full.shape[0])
            s_list.append(s_full)
            a_list.append(a_full)
            tgt = _per_step_event_target(s_full, target_kind, env, goal_state)
            if tgt is None:
                # Append a placeholder so encode() still gets the right shapes.
                tgt = torch.zeros(s_full.shape[0] - 1, dtype=torch.float32)
            target_list.append(tgt)

        # Pad to T_max (the longest window in this batch).
        T_max = max(T_real_list)
        obs_dim = s_list[0].shape[-1]
        action_dim_ = a_list[0].shape[-1]
        # When state_dim was overridden (generalist ckpt with --pad-obs-to),
        # pad obs to state_dim; same for action_dim.
        obs_dim_eff = pad_obs_to if pad_obs_to is not None and obs_dim < pad_obs_to else obs_dim
        action_dim_eff = action_dim_eval if action_dim_eval is not None and action_dim_ < action_dim_eval else action_dim_
        s_pad = torch.zeros(len(s_list), T_max, obs_dim_eff, dtype=torch.float32)
        a_pad = torch.zeros(len(s_list), T_max, action_dim_eff, dtype=torch.float32)
        for j, (s, a) in enumerate(zip(s_list, a_list)):
            s_pad[j, : s.shape[0], : s.shape[-1]] = s
            a_pad[j, : a.shape[0], : a.shape[-1]] = a
        s_dev = s_pad.to(device)
        a_dev = a_pad.to(device)
        with torch.no_grad():
            out = model.forward(s_dev, a_dev)
        # We probe the GATED SPIKE TRACE (pre-projection), not the post-readout
        # latent. The trace is the model-visible state for trace-only STJEWM and
        # the membrane-forbidden protocol is most clearly tested on it. For
        # baselines (LeWM, GRU, MLP) forward() does not return a 'trace' key;
        # fall back to 'emb' in that case.
        if isinstance(out, dict) and "trace" in out:
            z = out["trace"]                            # (B, T_max, D)
        else:
            z = out["emb"] if isinstance(out, dict) else out
        # Per-step z for the real window length. Targets are aligned to T-1
        # steps (the inter-step diff axis). We index z[:, :T-1, :].
        for j, T_real in enumerate(T_real_list):
            z_win = z[j, : T_real - 1, :]            # (T_real-1, D)
            tgt_win = target_list[j]                  # (T_real-1,) or subset
            n_match = min(z_win.shape[0], tgt_win.shape[0])
            if n_match <= 0:
                continue
            Zs.append(z_win[:n_match].cpu())
            Ys.append(tgt_win[:n_match].cpu())
    if not Zs:
        return None, None, True, "empty dataset"
    Z = torch.cat(Zs, dim=0)         # (N_total_steps, D)
    Y = torch.cat(Ys, dim=0)         # (N_total_steps,)
    Y = Y.unsqueeze(-1)              # (N, 1)
    binary = target_kind in EVENT_BINARY_TARGETS
    return Z, Y, binary, None


def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Per-output R² averaged over output dims."""
    yp = y_pred.numpy()
    yt = y_true.numpy()
    if yt.shape[1] == 1:
        ss_res = float(((yt - yp) ** 2).sum())
        ss_tot = float(((yt - yt.mean()) ** 2).sum()) + 1e-9
        return 1.0 - ss_res / ss_tot
    # Per-dim R² averaged
    r2s = []
    for d in range(yt.shape[1]):
        ss_res = float(((yt[:, d] - yp[:, d]) ** 2).sum())
        ss_tot = float(((yt[:, d] - yt[:, d].mean()) ** 2).sum()) + 1e-9
        r2s.append(1.0 - ss_res / ss_tot)
    return float(np.mean(r2s))


def save_skip(out_path: str, reason: str, n_train: int = 0, n_val: int = 0) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"skipped": True, "reason": reason, "r2": 0.0, "n_train": n_train, "n_val": n_val},
            f, indent=2,
        )


# ============================================================
# Main
# ============================================================
def main() -> int:
    args = parse_args()
    env = args.env
    model_name = args.model
    target = args.probe_target

    ckpt_path = args.ckpt or f"/home/lx/snn/results/{env}/{model_name}/final.pt"
    if not os.path.exists(ckpt_path):
        save_skip(args.out, f"checkpoint missing: {ckpt_path}")
        print(f"[probe] skip — no ckpt at {ckpt_path}")
        return 0

    # Load ckpt args
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        save_skip(args.out, f"ckpt load failed: {e}")
        return 0
    ck_args = ck.get("args", {}) or {}

    # Build dataset
    env_kind, data_path, history_size, goal_offset = ENV_REGISTRY[env]
    if not os.path.exists(data_path):
        save_skip(args.out, f"data missing: {data_path}")
        return 0
    try:
        from code.data import load_dataset
        ds = load_dataset(env_kind, path=data_path, history_size=history_size,
                          goal_offset=goal_offset, max_windows=args.max_windows)
    except Exception as e:
        save_skip(args.out, f"dataset load failed: {e}")
        return 0
    if len(ds) == 0:
        save_skip(args.out, "empty dataset")
        return 0

    # Determine state_dim / action_dim. When --pad-obs-to/--action-dim-eval
    # are set (generalist ckpt), override the per-env native dims.
    sample = ds[0]
    state_dim = args.pad_obs_to if args.pad_obs_to is not None else sample["state"].shape[-1]
    action_dim = args.action_dim_eval if args.action_dim_eval is not None else sample["action"].shape[-1]

    # Build model + load weights
    try:
        model = build_model(model_name, state_dim, action_dim, ck_args,
                            ck_state_dict=ck.get("model"))
        model.load_state_dict(ck["model"])
    except Exception as e:
        save_skip(args.out, f"model build/load failed: {e}")
        return 0
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Determine probe target dim + dispatch event-type vs window-level
    is_event = target.startswith("event_")
    binary = target in EVENT_BINARY_TARGETS
    if is_event:
        probe_dim = 1   # per-step binary or scalar
        Z, Y, binary, err = collect_event_targets(
            model, ds, target, env, args.device,
            max_windows=args.max_windows,
            pad_obs_to=args.pad_obs_to,
            action_dim_eval=args.action_dim_eval,
        )
    else:
        probe_dim = ENV_PROBE[env]["pos"][1] - ENV_PROBE[env]["pos"][0]
        if target == "velocity":
            if ENV_PROBE[env].get("vel") is None:
                save_skip(args.out, f"no velocity slice for env={env}")
                return 0
            probe_dim = ENV_PROBE[env]["vel"][1] - ENV_PROBE[env]["vel"][0]
        elif target == "contact":
            probe_dim = 1
        elif target == "future_k":
            probe_dim = ENV_PROBE[env]["pos"][1] - ENV_PROBE[env]["pos"][0]
        elif target == "goal_direction":
            probe_dim = ENV_PROBE[env]["pos"][1] - ENV_PROBE[env]["pos"][0]
        Z, Y, err = collect_latents_and_targets(
            model, ds, action_dim, probe_dim, target, env, args.device,
            k=args.future_k, max_windows=args.max_windows,
        )
    if Z is None:
        save_skip(args.out, err or "no data")
        return 0

    # Z is already (N, D) from the batched collect
    n_total = Z.shape[0]
    if n_total < 16:
        save_skip(args.out, f"too few samples: {n_total}", n_train=0, n_val=0)
        return 0

    # Train/val split (last val_frac is val)
    n_val = max(1, int(args.val_frac * n_total))
    n_train = n_total - n_val
    Z_train, Z_val = Z[:n_train], Z[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]

    # Move to device
    Z_train = Z_train.to(args.device)
    Z_val = Z_val.to(args.device)
    Y_train = Y_train.to(args.device)
    Y_val = Y_val.to(args.device)

    # Linear probe
    embed_dim = Z_train.shape[-1]
    head = nn.Linear(embed_dim, probe_dim).to(args.device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    if binary:
        # Per-step classification. The probe output is a single logit; we
        # apply BCEWithLogitsLoss on (pred, target) and report balanced acc.
        # We weight the positive class inversely to its base rate to prevent
        # the probe from collapsing to all-zeros for rare-event targets
        # (e.g., room_entered at 4% base rate).
        y_for_weight = Y_train.float()
        base = float(y_for_weight.mean().clamp(min=1e-3))
        pos_weight = torch.tensor([max((1 - base) / base, 1.0)],
                                  device=args.device).clamp(max=50.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # Standardize targets and use MSE -> R^2 on raw scale.
        y_mean = Y_train.mean(dim=0, keepdim=True)
        y_std = Y_train.std(dim=0, keepdim=True) + 1e-6
        Y_train_n = (Y_train - y_mean) / y_std
        Y_val_n = (Y_val - y_mean) / y_std
        loss_fn = nn.MSELoss()

    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n_train)
        for s in range(0, n_train, args.batch):
            idx = perm[s: s + args.batch]
            pred = head(Z_train[idx])
            if binary:
                loss = loss_fn(pred, Y_train[idx])
            else:
                loss = loss_fn(pred, Y_train_n[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    dt = time.time() - t0

    # Eval
    with torch.no_grad():
        pred_val = head(Z_val)
    if binary:
        # For imbalanced binary targets (e.g. room_entered ~6%, block_near_target
        # ~10%) bal_acc on argmax can be misleading. We report AUROC as the
        # headline number (calibration-free, threshold-free, handles imbalance
        # naturally), keep bal_acc + raw_acc for context, and additionally report
        # a threshold-free AUPRC which is more sensitive to the rare-class signal.
        prob = torch.sigmoid(pred_val).cpu().numpy().reshape(-1)
        y = Y_val.float().cpu().numpy().reshape(-1)
        # Per-class recall at argmax (kept for sanity)
        pred_class = (prob > 0.5).astype(np.float32)
        tp = float(((pred_class == 1) & (y == 1)).sum())
        fn = float(((pred_class == 0) & (y == 1)).sum())
        tn = float(((pred_class == 0) & (y == 0)).sum())
        fp = float(((pred_class == 1) & (y == 0)).sum())
        pos_recall = tp / max(tp + fn, 1.0)
        neg_recall = tn / max(tn + fp, 1.0)
        raw_acc = float((pred_class == y).mean())
        bal_acc = 0.5 * (pos_recall + neg_recall)
        # AUROC + AUPRC
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auroc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else 0.5
            auprc = float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else float(y.mean())
        except Exception:
            auroc = 0.5
            auprc = float(y.mean())
        r2 = auroc
        metric_name = "auroc"
        extra = {"raw_acc": raw_acc, "pos_recall": pos_recall,
                 "neg_recall": neg_recall, "base_rate": float(y.mean()),
                 "bal_acc": bal_acc, "auprc": auprc}
    else:
        pred_val_dn = pred_val * y_std + y_mean
        r2 = r2_score(pred_val_dn.cpu(), Y_val.cpu())
        metric_name = "r2"
        extra = {}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "skipped": False,
        "reason": None,
        "r2": float(r2),
        "metric": metric_name,
        "binary": bool(binary),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "probe_target": target,
        "env": env,
        "model": model_name,
        "probe_dim": int(probe_dim),
        "wall_time_sec": round(dt, 2),
    }
    payload.update(extra)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    if binary:
        print(f"[probe] {env}/{model_name}/{target}: AUROC={r2:.4f} bal_acc={bal_acc:.4f} raw_acc={raw_acc:.4f} base={float(y.mean()):.3f}  ({dt:.1f}s)")
    else:
        print(f"[probe] {env}/{model_name}/{target}: R^2={r2:.4f}  (n_train={n_train}, n_val={n_val}, {dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

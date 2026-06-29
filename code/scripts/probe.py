"""Linear probe on frozen encoder outputs.

Train a single Linear layer (frozen encoder + single Linear head, 5 epochs
of Adam/MSE) to predict physical variables from a model's latent embedding.

Usage:
    python -m code.scripts.probe --env <env> --model <model> \\
            --probe-target <target> --out <json>

Targets:
    position         predict state[:nq]  (configurable per-env slice, see ENV_PROBE)
    velocity         predict state[nq:nq+nv]   (skipped for pixel-input)
    contact          predict a binary contact flag
    future_k         predict state[t + k=10]
    goal_direction   predict (goal - state) / ||...||
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
    "reacher":      {"pos": (0, 2),  "vel": None,    "obs_dim": 4},   # qpos(2), no qvel stored
    "cartpole_2d":  {"pos": (0, 2),  "vel": None,    "obs_dim": 2},   # [cart, pole_angle]
    "pendulum_2d":  {"pos": (0, 2),  "vel": None,    "obs_dim": 2},   # [cos, sin]
    "finger":       {"pos": (0, 3),  "vel": None,    "obs_dim": 3},   # [finger_pos]
    "ball_in_cup":  {"pos": (0, 4),  "vel": None,    "obs_dim": 4},   # [ball_pos]
    "cheetah":      {"pos": (0, 9),  "vel": None,    "obs_dim": 9},   # qpos only in 9D
    "walker":       {"pos": (0, 9),  "vel": None,    "obs_dim": 9},
    "hopper":       {"pos": (0, 7),  "vel": None,    "obs_dim": 7},
    "quadruped":    {"pos": (0, 12), "vel": None,    "obs_dim": 30},  # first half
    "humanoid":     {"pos": (0, 14), "vel": None,    "obs_dim": 28},  # first half
    "humanoid_CMU": {"pos": (0, 27), "vel": None,    "obs_dim": 63},  # first half
    "dog":          {"pos": (0, 19), "vel": None,    "obs_dim": 87},  # first half
    "fish":         {"pos": (0, 7),  "vel": None,    "obs_dim": 14},  # first half
    "stacker":      {"pos": (0, 10), "vel": None,    "obs_dim": 20},  # first half
}


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
        choices=["position", "velocity", "contact", "future_k", "goal_direction"],
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
    return p.parse_args()


def build_model(model_name: str, state_dim: int, action_dim: int, ck_args: dict):
    """Build the model exactly as code/train/train.py and code/eval/closed_loop.py do."""
    if ck_args.get("model", model_name) == "lewm_baseline" or model_name.startswith("lewm"):
        from code.lewm_transformer_baseline import LeWMTransformerBaseline
        embed_dim = ck_args.get("embed_dim", 256)
        num_layers = ck_args.get("n_layers", 4)
        return LeWMTransformerBaseline(
            state_dim=state_dim, action_dim=action_dim,
            embed_dim=embed_dim, num_layers=num_layers, num_heads=8,
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
                if vel_slice is None:
                    return None, None, "no velocity slice"
                tgt = init_state[vel_slice[0]: vel_slice[1]]
            elif target_kind == "future_k":
                t_idx = min(k, s_full.shape[0] - 1)
                tgt = s_full[t_idx, pos_slice[0]: pos_slice[1]]
            elif target_kind == "goal_direction":
                diff = (goal_state - init_state)[pos_slice[0]: pos_slice[1]]
                norm = diff.norm() + 1e-8
                tgt = diff / norm
            elif target_kind == "contact":
                state_std = s_full.std(dim=0).mean()
                state_mean = s_full.mean()
                tgt = torch.tensor([float((init_state - state_mean).abs().max() > state_std)])
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

    # Determine state_dim / action_dim from first sample
    sample = ds[0]
    state_dim = sample["state"].shape[-1]
    action_dim = sample["action"].shape[-1]

    # Build model + load weights
    try:
        model = build_model(model_name, state_dim, action_dim, ck_args)
        model.load_state_dict(ck["model"])
    except Exception as e:
        save_skip(args.out, f"model build/load failed: {e}")
        return 0
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Determine probe target dim
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

    # Collect latents and targets
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

    # Standardize targets
    y_mean = Y_train.mean(dim=0, keepdim=True)
    y_std = Y_train.std(dim=0, keepdim=True) + 1e-6
    Y_train_n = (Y_train - y_mean) / y_std
    Y_val_n = (Y_val - y_mean) / y_std

    # Linear probe
    embed_dim = Z_train.shape[-1]
    head = nn.Linear(embed_dim, probe_dim).to(args.device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    for ep in range(args.epochs):
        # Mini-batch over train
        perm = torch.randperm(n_train)
        for s in range(0, n_train, args.batch):
            idx = perm[s: s + args.batch]
            pred = head(Z_train[idx])
            loss = loss_fn(pred, Y_train_n[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    dt = time.time() - t0

    # Eval (denormalize predictions for R² on raw scale)
    with torch.no_grad():
        pred_val = head(Z_val) * y_std + y_mean
    r2 = r2_score(pred_val.cpu(), Y_val.cpu())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "skipped": False,
                "reason": None,
                "r2": float(r2),
                "n_train": int(n_train),
                "n_val": int(n_val),
                "probe_target": target,
                "env": env,
                "model": model_name,
                "probe_dim": int(probe_dim),
                "wall_time_sec": round(dt, 2),
            },
            f, indent=2,
        )
    print(f"[probe] {env}/{model_name}/{target}: R²={r2:.4f}  (n_train={n_train}, n_val={n_val}, {dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

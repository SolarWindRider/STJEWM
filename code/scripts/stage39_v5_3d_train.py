"""v4.4 trainer: v4.3 architecture + goal-conditioned + trained on mujoco rollouts.

Diff from v4.3:
  - Training data: mujoco rollouts (random policies, qpos != 0)
  - Same architecture, same loss
  - Trained to learn (qpos, action, target) -> next_qpos
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/lx/LeWM")
sys.path.insert(0, "/home/lx/snn/code")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="reacher_mujoco")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--lambda-sigreg", type=float, default=0.09)
    p.add_argument("--lambda-goal", type=float, default=0.5)
    p.add_argument("--goal-offset", type=int, default=5)
    return p.parse_args()


class MuJoCoWindowDataset(Dataset):
    """Sliding windows from mujoco rollouts. state = (qpos[2], target[2])."""
    def __init__(self, npz_path, history_size=3, goal_offset=5):
        data = np.load(npz_path)
        self.obs = data["observations"][:, 0, :].astype(np.float32)  # (N, 4)
        self.actions = data["actions"][:, 0, :].astype(np.float32)  # (N, 2)
        self.next_obs = data["next_observations"][:, 0, :].astype(np.float32)  # (N, 4)
        # For windows: history (3) + goal at t+H (5) + buffer
        window_size = history_size + goal_offset + 1
        N = len(self.obs) - window_size
        self.windows = [(i, window_size) for i in range(N)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        s, w = self.windows[i]
        state = self.obs[s:s+w]
        action = self.actions[s:s+w-1]
        zero_act = np.zeros((1, action.shape[1]), dtype=action.dtype)
        action = np.concatenate([action, zero_act], axis=0)
        return {
            "state": torch.from_numpy(state).float(),
            "action": torch.from_numpy(action).float(),
        }


def build_v4_dmc(obs_dim, action_dim, n_layers=4, n_neurons=192):
    from lewm_stjewm_v4 import STJEWMv4
    return STJEWMv4(
        d_hid=n_neurons, embed_dim=n_neurons,
        action_dim=action_dim, action_emb_dim=n_neurons,
        state_dim=obs_dim, cell_n_layers=n_layers, n_d=3,
        trace_beta=0.9, freeze_encoder=True,
    )


def train(model, loader, args, device, save_dir, n_windows_per_epoch):
    """v4.4 = v4.3 + goal + mujoco rollouts."""
    from src.sigreg import SIGReg
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[v4.4] params total={n_params/1e6:.2f}M trainable={n_train/1e6:.2f}M, "
        f"{n_windows_per_epoch} windows/epoch, batch={args.batch}, epochs={args.epochs}, "
        f"lr={args.lr}, lambda_sigreg={args.lambda_sigreg}, lambda_goal={args.lambda_goal}, "
        f"goal_offset={args.goal_offset}",
        flush=True,
    )
    t0 = time.time()
    step = 0
    H = 3
    T_pred = 3
    losses_log = []
    for epoch in range(args.epochs):
        for batch in loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(state, action)
                emb = out["emb"]
                emb_pre = out["emb_pre_cell"]
                ctx_emb = emb[:, :H]
                ctx_act = action[:, :H]
                pred_emb = model.predict(ctx_emb, ctx_act)
                tgt_emb = emb[:, H:H + T_pred]
                pred_loss = F.mse_loss(pred_emb, tgt_emb)
                sigreg_loss = sigreg(emb_pre.transpose(0, 1))
                # Goal loss: state at t+H should match goal (last 2 dims of state)
                # state[t] = [qpos_t, target_t]
                # We want model.predict(history) ≈ encode(state[t+H], action[t+H])
                with torch.no_grad():
                    goal_state = state[:, H + args.goal_offset:H + args.goal_offset + 1]  # (B, 1, D)
                    zero_act_g = torch.zeros(goal_state.shape[0], 1, action.shape[-1], device=device)
                    out_goal = model(goal_state, zero_act_g)
                    goal_emb_target = out_goal["emb_pre_cell"][:, 0]  # (B, 192)
                goal_pred = model.predict(ctx_emb, ctx_act)[:, -1]  # (B, 192)
                goal_loss = F.mse_loss(goal_pred, goal_emb_target)
                loss = pred_loss + args.lambda_sigreg * sigreg_loss + args.lambda_goal * goal_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                speed = step / elapsed if elapsed > 0 else 0
                eta = (args.epochs * n_windows_per_epoch / args.batch - step) / speed if speed > 0 else 0
                sparsity = 1.0 - out["spike"].float().mean().item()
                print(
                    f"[v4.4] ep {epoch+1}/{args.epochs} step {step} "
                    f"pred={pred_loss.item():.4f} sigreg={sigreg_loss.item():.4f} "
                    f"goal={goal_loss.item():.4f} total={loss.item():.4f} "
                    f"speed={speed:.2f}/s ETA={eta/3600:.1f}h sparsity={sparsity:.3f}",
                    flush=True,
                )
                losses_log.append({
                    "step": step,
                    "pred": float(pred_loss.item()),
                    "sigreg": float(sigreg_loss.item()),
                    "goal": float(goal_loss.item()),
                    "total": float(loss.item()),
                })
            if args.save_every > 0 and step % args.save_every == 0:
                ck_path = save_dir / f"step{step}.pt"
                torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, ck_path)
                print(f"[v4.4] saved {ck_path}", flush=True)
    final_path = save_dir / "final.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, final_path)
    print(f"[v4.4] final saved {final_path}", flush=True)
    log_path = save_dir / "loss_log.json"
    with open(log_path, "w") as f:
        json.dump({"step": step, "losses": losses_log}, f)
    print(f"[v4.4] loss log saved {log_path}")


def main():
    args = parse_args()
    print(f"[v4.4] cmd: {vars(args)}", flush=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = MuJoCoWindowDataset(
        args.data, history_size=3, goal_offset=args.goal_offset,
    )
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    data = np.load(args.data)
    obs_dim = data["observations"].shape[-1]
    action_dim = data["actions"].shape[-1]

    model = build_v4_dmc(obs_dim, action_dim, n_layers=args.n_layers)
    model = model.to(DEVICE)

    save_dir = Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)
    train(model, loader, args, DEVICE, save_dir, n_windows_per_epoch=len(ds))


if __name__ == "__main__":
    main()
"""Single canonical trainer for STJEWM (pure SNN) and LeWM-style baseline.

Replaces the old stage34/stage39/stage47 trainers (now deleted).

Single trainer, model architecture arg controls whether we get the SNN or the
Transformer baseline. Hyperparams via args. Loss is the same LeWM-derived
two-term objective: pred_loss + lambda_sigreg * sigreg_loss + lambda_goal * goal_loss.

Usage:
    python -m code.train.train --model stjewm --data ... --out ...
    python -m code.train.train --model lewm_baseline --data ... --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Register stable_worldmodel envs before any loader / eval can use them
import stable_worldmodel  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, "/home/lx/snn")

from code.core.encode import assert_model_compatible
from code.data import load_dataset
from code.sigreg import SIGReg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["stjewm", "lewm_baseline"], required=True,
                   help="Which model architecture to train")
    p.add_argument("--env-kind", required=True,
                   help="Loader kind: pusht, tworoom, reacher_4d, reacher_lewm, "
                        "reacher_full, ogb_cube, dmc, mujoco_3d, gym_live")
    p.add_argument("--data", default=None,
                   help="Path to data file (or env_id for gym_live; not required for env-based loaders like ogb_cube_env)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--n-layers", type=int, default=4,
                   help="Number of SNN/Transformer layers (default 4 for STJEWM, 6 for LeWM-style baseline)")
    p.add_argument("--lambda-sigreg", type=float, default=0.09)
    p.add_argument("--lambda-goal", type=float, default=0.5)
    p.add_argument("--goal-offset", type=int, default=5)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--t-pred", type=int, default=3,
                   help="Number of next-step predictions per loss term. Independent of goal_offset.")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Cap dataset size (for fast smoke tests)")
    p.add_argument("--n-episodes", type=int, default=50,
                   help="Number of episodes to collect (for env-based data, e.g. ogb_cube_env)")
    p.add_argument("--max-steps-per-ep", type=int, default=200,
                   help="Max steps per collected episode (for env-based data)")
    p.add_argument("--readout-mode", type=str, default="hidden_leak",
                   choices=["trace_only", "hidden_leak", "membrane_readout",
                            "spike_only", "rate_only", "no_trace"],
                   help="STJEWM readout mode (membrane-forbidden protocol)")
    return p.parse_args()
# Model builders
# ============================================================
def build_model(model_kind: str, obs_dim: int, action_dim: int, n_layers: int,
                readout_mode: str = "hidden_leak"):
    if model_kind == "stjewm":
        from code.stjewm import STJEWM
        return STJEWM(
            d_hid=192, embed_dim=192,
            action_dim=action_dim, action_emb_dim=192,
            state_dim=obs_dim,
            cell_n_layers=n_layers, n_d=3,
            trace_beta=0.9, freeze_encoder=True,
            readout_mode=readout_mode,
        )
    if model_kind == "lewm_baseline":
        from code.lewm_transformer_baseline import LeWMTransformerBaseline
        # 4-layer 256-hidden matches STJEWM (5.03M vs 5.07M = 0.7% delta).
        # n_layers is configurable via --n-layers (default 4).
        return LeWMTransformerBaseline(
            state_dim=obs_dim, action_dim=action_dim,
            embed_dim=256, num_layers=n_layers, num_heads=8,
        )
    raise ValueError(f"Unknown model: {model_kind}")


# ============================================================
# Training loop (single canonical)
# ============================================================
def train(
    model,
    loader: DataLoader,
    args,
    device: str,
    save_dir: Path,
    n_windows_per_epoch: int,
):
    """Canonical training loop.
    Loss (LeWM App. A + goal-conditioned term):
        L_total = L_pred + lambda_sigreg * L_sigreg + lambda_goal * L_goal
    """
    from code.sigreg import SIGReg
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = type(model).__name__
    print(
        f"[train/{args.model}] {model_name} params total={n_params/1e6:.2f}M "
        f"trainable={n_train/1e6:.2f}M, {n_windows_per_epoch} windows/epoch, "
        f"batch={args.batch}, epochs={args.epochs}, lr={args.lr}, "
        f"lambda_sigreg={args.lambda_sigreg}, lambda_goal={args.lambda_goal}, "
        f"goal_offset={args.goal_offset}, history={args.history_size}",
        flush=True,
    )
    t0 = time.time()
    step = 0
    H = args.history_size
    T_pred = min(H, args.t_pred, args.goal_offset)  # never exceed history, t_pred, or goal_offset
    losses_log = []
    for epoch in range(args.epochs):
        for batch in loader:
            state = batch["state"].to(device)            # (B, W, D)
            action = batch["action"].to(device)          # (B, W, A)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(state, action)
                emb = out["emb"]
                emb_pre = out["emb_pre_cell"]
                # Context: first H steps; predict next T_pred
                ctx_emb = emb[:, :H]
                ctx_act = action[:, :H]
                pred_emb = model.predict(ctx_emb, ctx_act)
                tgt_emb = emb[:, H:H + T_pred]
                pred_loss = F.mse_loss(pred_emb, tgt_emb)
                # SIGReg on the pre-cell embedding (LeWM App. A)
                sigreg_loss = sigreg(emb_pre.transpose(0, 1))
                # Goal loss: predict the goal state embedding from the current history.
                # (LeWM App. F.1)
                #
                # Critical fix: the original code used `model.predict(ctx, ctx_act)[:, -1]`
                # which is a 1-step prediction (H+1 ahead), not a goal_offset-step prediction.
                # The goal loss should compare the model's predicted latent for the goal state
                # to the actual goal state latent.
                #
                # The model's `emb[:, t]` is its predicted next-state latent for step t+1.
                # So if we forward on a window of (H + goal_offset) states, the model's
                # predicted latent for the goal state (at position H+goal_offset) is
                # `emb[:, H+goal_offset-1]`. The target is the model's own latent
                # for the goal state, which is `out_goal["emb"][:, 0]` (forwarding on the
                # goal state alone gives its predicted next-state latent — but for the
                # goal state, this is the closest "self-distillation" target).
                # Both are in the same "post-stack predicted latent" space (matches
                # the pred_loss formulation above).
                #
                # Implementation: single forward on (H + goal_offset) states. No rollout
                # needed because the model is already doing next-step prediction at each
                # position — the last position is the goal_offset-th step prediction.
                with torch.no_grad():
                    goal_state = state[:, H + args.goal_offset:H + args.goal_offset + 1]
                    zero_act_g = torch.zeros(
                        goal_state.shape[0], 1, action.shape[-1],
                        device=device, dtype=action.dtype,
                    )
                    out_goal = model(goal_state, zero_act_g)
                    # Use the post-stack predicted latent of the goal state as target.
                    # Note: this is what the model itself produces as the next-state
                    # prediction for the goal state. This is the standard JEPA-style
                    # "self-distillation" target.
                    goal_emb_target = out_goal["emb"][:, 0]  # (B, D)
                # Prediction: forward on (H + goal_offset) states. The model's predicted
                # next-state latent at the last position is its goal prediction.
                full_state = state[:, :H + args.goal_offset]
                full_action = action[:, :H + args.goal_offset]
                out_full = model(full_state, full_action)
                full_emb = out_full["emb"]  # (B, H+goal_offset, D)
                goal_pred = full_emb[:, -1]  # (B, D) — prediction for the goal state
                goal_loss = F.mse_loss(goal_pred, goal_emb_target)
                loss = pred_loss + args.lambda_sigreg * sigreg_loss + args.lambda_goal * goal_loss
                sparsity = 1.0 - out["spike"].float().mean().item() if "spike" in out else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                speed = step / elapsed if elapsed > 0 else 0
                eta = (args.epochs * n_windows_per_epoch / args.batch - step) / speed if speed > 0 else 0
                sparsity = 1.0 - out["spike"].float().mean().item() if "spike" in out else None
                sparsity_str = f" sparsity={sparsity:.3f}" if sparsity is not None else ""
                print(
                    f"[train/{args.model}] ep {epoch+1}/{args.epochs} step {step} "
                    f"pred={pred_loss.item():.4f} sigreg={sigreg_loss.item():.4f} "
                    f"goal={goal_loss.item():.4f} total={loss.item():.4f} "
                    f"speed={speed:.2f}/s ETA={eta/3600:.1f}h{sparsity_str}",
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
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "step": step,
                }, ck_path)
                print(f"[train/{args.model}] saved {ck_path}", flush=True)
    final_path = save_dir / "final.pt"
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "step": step,
    }, final_path)
    print(f"[train/{args.model}] final saved {final_path}", flush=True)
    log_path = save_dir / "loss_log.json"
    with open(log_path, "w") as f:
        json.dump({"step": step, "losses": losses_log}, f)
    print(f"[train/{args.model}] loss log saved {log_path}")


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    print(f"[train/{args.model}] cmd: {vars(args)}", flush=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset via unified loader
    if args.env_kind == "gym_live":
        ds = load_dataset("gym_live", path=args.data, history_size=args.history_size,
                          goal_offset=args.goal_offset, n_episodes=50, seed=args.seed)
    elif args.env_kind in ("ogb_cube_env", "ogb_scene_env"):
        # env-based OGBench loader: collect from env, no path needed
        ds = load_dataset(args.env_kind, n_episodes=args.n_episodes,
                          max_steps_per_ep=args.max_steps_per_ep,
                          history_size=args.history_size,
                          goal_offset=args.goal_offset,
                          seed=args.seed)
    else:
        ds = load_dataset(args.env_kind, path=args.data, history_size=args.history_size,
                          goal_offset=args.goal_offset, max_windows=args.max_windows)

    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    # Determine obs_dim / action_dim from first batch
    sample = ds[0]
    obs_dim = sample["state"].shape[-1]
    action_dim = sample["action"].shape[-1]

    # Build model — both architectures use 4 layers for ~5M-param match.
    # STJEWM = 4-layer SNN stack. LeWM-style = 4-layer Transformer (4-layer
    # 256-hidden is the closest match to STJEWM 5.03M: LeWM 5.07M = 0.7% delta).
    n_layers = args.n_layers
    # Save the actual n_layers used (not the user-provided default)
    args.n_layers = n_layers
    # Save embed_dim for eval (LeWM-style uses 256, STJEWM uses 192)
    if args.model == "lewm_baseline":
        args.embed_dim = 256
    else:
        args.embed_dim = 192
    model = build_model(args.model, obs_dim, action_dim, n_layers, args.readout_mode).to(device)
    assert_model_compatible(model)

    save_dir = Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)
    train(model, loader, args, device, save_dir, n_windows_per_epoch=len(ds))


if __name__ == "__main__":
    main()

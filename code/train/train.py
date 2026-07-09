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
from code.native_losses import (
    NATIVE_LOSS_DISPATCH,
    stjewm_loss,
    cubifae_loss,
    spikedreamer_loss,
    slt_lif_mpc_loss,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["stjewm", "lewm_baseline", "gru_baseline", "mlp_baseline",
                                       "slt_lif_mpc_trace", "slt_lif_mpc_free",
                                       "spikedreamer_baseline",
                                       "cubifae_baseline"], required=True,
                   help="Which model architecture to train")
    p.add_argument("--env-kind", required=False, default=None,
                   help="Loader kind: pusht, tworoom, reacher_4d, reacher_lewm, "
                        "reacher_full, ogb_cube, dmc, mujoco_3d, gym_live. "
                        "Required unless --multi-env-spec is set.")
    p.add_argument("--multi-env-spec", default=None,
                   help="Path to a JSON file with shape "
                        "[{env_kind, path, history_size, goal_offset, max_windows, env_id}, ...]. "
                        "Mutually exclusive with --env-kind for the dataset layer; "
                        "when set, --pad-obs-to and --action-dim should be set too.")
    p.add_argument("--pad-obs-to", type=int, default=None,
                   help="If set, pad each loaded obs to this dim at the data layer. "
                        "Required for generalist training.")
    p.add_argument("--action-dim", type=int, default=None,
                   help="Override the per-env action_dim used to construct the model. "
                        "Required for generalist training; sets a fixed action_dim across envs.")
    p.add_argument("--embed-dim", type=int, default=None,
                   help="Override the embed_dim used by build_model (e.g. 192 for generalist LeWM). "
                        "If None, build_model picks per-model defaults.")
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
                            "spike_only", "rate_only", "no_trace", "raw_spike"],
                   help="STJEWM readout mode (membrane-forbidden protocol)")
    args = p.parse_args()
    # Validate that exactly one of --env-kind / --multi-env-spec is set
    if args.env_kind is None and args.multi_env_spec is None:
        p.error("One of --env-kind or --multi-env-spec is required.")
    if args.env_kind is not None and args.multi_env_spec is not None:
        p.error("--env-kind and --multi-env-spec are mutually exclusive.")
    return args

# Model builders
# ============================================================
def build_model(model_kind: str, obs_dim: int, action_dim: int, n_layers: int,
                readout_mode: str = "hidden_leak", embed_dim: Optional[int] = None):
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
            embed_dim=(embed_dim or 256), num_layers=n_layers, num_heads=8,
        )
    if model_kind == "gru_baseline":
        from code.gru_baseline import GRUBaseline
        return GRUBaseline(state_dim=obs_dim, action_dim=action_dim)
    if model_kind == "mlp_baseline":
        from code.mlp_baseline import make_mlp_baseline
        return make_mlp_baseline(state_dim=obs_dim, action_dim=action_dim)
    if model_kind == "slt_lif_mpc_trace":
        from code.slt_lif_mpc_baseline import make_slt_lif_mpc_trace
        return make_slt_lif_mpc_trace(
            state_dim=obs_dim, action_dim=action_dim,
            d_in=192, embed_dim=192, n_layers=n_layers, trace_beta=0.9, k_avg=4,
        )
    if model_kind == "slt_lif_mpc_free":
        from code.slt_lif_mpc_baseline import make_slt_lif_mpc_free
        return make_slt_lif_mpc_free(
            state_dim=obs_dim, action_dim=action_dim,
            d_in=192, embed_dim=192, n_layers=n_layers, trace_beta=0.9,
        )
    if model_kind == "spikedreamer_baseline":
        from code.spikedreamer_baseline import make_spikedreamer
        return make_spikedreamer(
            state_dim=obs_dim, action_dim=action_dim,
            d_snn=128, d_tx=192, num_layers=n_layers, num_heads=8,
        )
    if model_kind == "cubifae_baseline":
        from code.cubifae_baseline import CubifAEBaseline
        return CubifAEBaseline(
            state_dim=obs_dim, action_dim=action_dim,
            d_hid=192, n_layers=n_layers,
        )

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
    Loss dispatch (v0.7+): each SNN baseline uses its native loss via
    code.native_losses.NATIVE_LOSS_DISPATCH.
        - stjewm / lewm_baseline / gru / mlp: 3-term pred + sigreg + goal
        - cubifae_baseline: 2-term pred + L1 spike sparsity
        - spikedreamer_baseline: 4-term pred + KL + recon + sparse
        - slt_lif_mpc_{trace,free}: 3-term pred + sparse + action (action=0 in CEM-eval)
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

                # ============== Native-loss dispatch ==============
                # v0.7: Each SNN baseline has its own native loss; the
                # trainer dispatches on args.model and uses the appropriate
                # function from code/native_losses. STJEWM and the
                # Transformer/RNN/MLP baselines still use the 3-term loss
                # (pred + sigreg + goal). CubifAE, SpikeDreamer, SLT-LIF-MPC
                # use their paper-native losses.
                model_kind = args.model
                loss_fn = NATIVE_LOSS_DISPATCH.get(model_kind, stjewm_loss)

                if loss_fn is stjewm_loss:
                    # 3-term JEPA loss: pred + lambda_sigreg*sigreg + lambda_goal*goal
                    sigreg_loss = sigreg(emb_pre.transpose(0, 1))
                    with torch.no_grad():
                        goal_state = state[:, H + args.goal_offset:H + args.goal_offset + 1]
                        zero_act_g = torch.zeros(
                            goal_state.shape[0], 1, action.shape[-1],
                            device=device, dtype=action.dtype,
                        )
                        out_goal = model(goal_state, zero_act_g)
                        goal_emb_target = out_goal["emb"][:, 0]
                    full_state = state[:, :H + args.goal_offset]
                    full_action = action[:, :H + args.goal_offset]
                    out_full = model(full_state, full_action)
                    goal_pred = out_full["emb"][:, -1]
                    loss, parts = loss_fn(
                        pred_emb, tgt_emb, emb_pre, sigreg,
                        goal_pred, goal_emb_target,
                        args.lambda_sigreg, args.lambda_goal,
                    )
                elif loss_fn is cubifae_loss:
                    # 2-term: pred + L1 spike sparsity. No sigreg / no goal.
                    loss, parts = loss_fn(
                        pred_emb, tgt_emb,
                        spike_layers=out.get("spike_layers", []),
                        goal_pred=None, goal_emb=None,
                        lambda_pred=1.0, lambda_sparse=1e-3, lambda_goal=0.0,
                    )
                elif loss_fn is spikedreamer_loss:
                    # 4-term: pred + KL + recon + sparse. State-based obs
                    # means no recon. LIF encoder is deterministic (no VAE),
                    # so mu=logvar=None and lambda_kl * 0 is fine.
                    spike_count = out.get("spike_count", out.get("spike"))
                    loss, parts = loss_fn(
                        pred_emb, tgt_emb,
                        obs_recon=None, obs_target=None,
                        mu=out.get("mu"), logvar=out.get("logvar"),
                        spike_count=spike_count,
                        lambda_recon=0.0, lambda_kl=1e-3,
                        lambda_pred=1.0, lambda_sparse=1e-3,
                    )
                elif loss_fn is slt_lif_mpc_loss:
                    # 3-term: pred + sparse + action. No action supervision
                    # in CEM-eval, so action term is 0. Reduces to pred + sparse.
                    loss, parts = loss_fn(
                        pred_emb, tgt_emb,
                        spike_count=out.get("spike"),
                        action_pred=None, action_target=None,
                        lambda_pred=1.0, lambda_sparse=1e-4, lambda_action=0.0,
                    )
                else:
                    # Fallback: plain MSE.
                    loss = pred_loss
                    parts = {"pred": pred_loss.item(), "total": pred_loss.item()}

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
                # Build a flat string from whatever loss terms are in `parts`.
                # STJEWM/JePA: pred/sigreg/goal. CubifAE: pred/sparse.
                # SpikeDreamer: pred/kl/recon/sparse. SLT-LIF-MPC: pred/sparse/action.
                def _fmt(p: dict) -> str:
                    keys = ("pred", "sigreg", "goal", "sparse", "kl", "recon", "action")
                    return " ".join(
                        f"{k}={p[k]:.4f}" for k in keys if k in p and p[k] is not None
                    )
                print(
                    f"[train/{args.model}] ep {epoch+1}/{args.epochs} step {step} "
                    f"{_fmt(parts)} total={loss.item():.4f} "
                    f"speed={speed:.2f}/s ETA={eta/3600:.1f}h{sparsity_str}",
                    flush=True,
                )
                log_entry = {"step": step, "total": float(loss.item())}
                log_entry.update({k: float(v) for k, v in parts.items()})
                losses_log.append(log_entry)
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

    # Load dataset via unified loader (specialist OR generalist)
    if args.multi_env_spec is not None:
        # Generalist path: union of per-env datasets, padded to common obs_dim and action_dim
        from code.data import load_multi_env_dataset_from_json
        if args.pad_obs_to is None:
            raise ValueError("--pad-obs-to is required when --multi-env-spec is set")
        if args.action_dim is None:
            raise ValueError("--action-dim is required when --multi-env-spec is set")
        ds = load_multi_env_dataset_from_json(
            args.multi_env_spec,
            pad_obs_to=args.pad_obs_to,
            action_dim_target=args.action_dim,
            seed=args.seed,
        )
    elif args.env_kind == "gym_live":
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
    sample_action_dim = sample["action"].shape[-1]
    # In generalist mode, --action-dim overrides whatever the data layer produced
    action_dim = args.action_dim if args.action_dim is not None else sample_action_dim
    if args.action_dim is not None and args.action_dim != sample_action_dim:
        raise RuntimeError(
            f"--action-dim={args.action_dim} but dataset action_dim={sample_action_dim}; "
            "the multi-env factory should have produced this shape. Check load_multi_env_dataset."
        )

    # Build model — both architectures use 4 layers for ~5M-param match.
    # STJEWM = 4-layer SNN stack. LeWM-style = 4-layer Transformer (4-layer
    # 256-hidden is the closest match to STJEWM 5.03M: LeWM 5.07M = 0.7% delta).
    n_layers = args.n_layers
    # Save the actual n_layers used (not the user-provided default)
    args.n_layers = n_layers
    # Save embed_dim for eval (LeWM-style uses 256, STJEWM uses 192; generalist overrides via --embed-dim)
    if args.embed_dim is not None:
        args.embed_dim = args.embed_dim
    elif args.model == "lewm_baseline":
        args.embed_dim = 256
    else:
        args.embed_dim = 192
    model = build_model(
        args.model, obs_dim, action_dim, n_layers, args.readout_mode,
        embed_dim=args.embed_dim,
    ).to(device)
    assert_model_compatible(model)

    save_dir = Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)
    train(model, loader, args, device, save_dir, n_windows_per_epoch=len(ds))


if __name__ == "__main__":
    main()

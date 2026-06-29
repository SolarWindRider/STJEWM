"""Sparse FLOPs analysis.

Since `thop` and `fvcore` are not installed, we estimate dense and sparse
FLOPs analytically for ST-JEWM and the LeWM-style Transformer baseline.

Per-timestep FLOPs (one (B, D) token through the model):

  ST-JEWM (4-layer, embed_dim=192, state_dim D, action_dim A):
      state_projector:  D * 192  (multiply-adds = D * 192)
      projector (when pixel): ~5.4 GMACs for ViT-Tiny (256-token @ 224x224)
      action_encoder:   A * 192
      MultiCompStack:   4 cells x ~884,736 = 3,538,944 per timestep
        (each cell: 3 dendrites + 1 soma, 192x192 matmuls, gated trace)
      trace_proj:       192 * 192
      Total per-timestep: ~4M FLOPs

  LeWM-style Transformer (4-layer, embed_dim=256, num_heads=8, state_dim D, action_dim A):
      state_encoder:    D * 256
      action_encoder:   A * 256
      4 x AdaLNZeroBlock:
          AdaLN modulation: ~256 * 256 * 2
          QKV proj:         256 * (256*3)
          attn (256 tokens -> 1 for closed loop, but we count single-token
                inference as a per-timestep computation):
                256 * 256
          MLP (2-layer):    256 * 1024 + 1024 * 256
      norm + proj_out:  256 * 256
      Total per-timestep: ~5M FLOPs

Usage:
    python -m code.scripts.flops --ckpt <path> --out <json>

Sparsity: assumed s=0.85 (matches ST-JEWM spike sparsity 0.75-0.95 per smoke test).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/lx/snn")


# ============================================================
# Dense FLOPs estimates
# ============================================================
def stjewm_dense_flops_per_step(state_dim: int, action_dim: int) -> int:
    """Per-timestep FLOPs (multiply-adds counted as 1 each)."""
    flops = 0
    # state_projector: Linear(state_dim, 192)
    flops += state_dim * 192
    # action_encoder: Linear(action_dim, 192)
    flops += action_dim * 192
    # MultiCompStack: 4 cells, each has 3 dendrites + 1 soma
    # each dendrite/soma = Linear(192, 192) gated; conservatively 4 * 192 * 192 = 147,456
    # per cell, but dendrite has gate+sigmoid => 2 matmuls.
    # Use the assignment's estimate: 12 * 192 * 192 * 2 = 884,736 per cell
    flops += 4 * 12 * 192 * 192 * 2
    # trace_proj: Linear(192, 192)
    flops += 192 * 192
    # gated trace context: cat(act_emb, h) -> 2*192; alpha mlp is small (~1k)
    flops += 4 * 192  # alpha_mlp Linear(2*192, 192)
    return flops


def lewm_dense_flops_per_step(state_dim: int, action_dim: int, num_layers: int = 4, embed_dim: int = 256) -> int:
    flops = 0
    flops += state_dim * embed_dim
    flops += action_dim * embed_dim
    # Each AdaLNZeroBlock (per-timestep, single token):
    #   AdaLN: 2x Linear(embed_dim, embed_dim) for shift+scale + 2x for gate
    for _ in range(num_layers):
        flops += 4 * embed_dim * embed_dim  # AdaLN modulation/gate
        flops += embed_dim * 3 * embed_dim  # QKV
        flops += embed_dim * embed_dim      # attention output proj
        flops += embed_dim * 4 * embed_dim  # MLP up
        flops += embed_dim * 4 * embed_dim  # MLP down
    flops += embed_dim * embed_dim  # proj_out
    return flops


# ViT-Tiny cost (per forward through 256 tokens @ 224x224) ~ 5.4 GMACs
VIT_TINY_GMACS_PER_TINY_FORWARD = 5.4e9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sparse / dense FLOPs analysis.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--state-dim", type=int, default=None,
                   help="Override state dim (default: read from ckpt args + dataset).")
    p.add_argument("--action-dim", type=int, default=None)
    p.add_argument("--sparsity", type=float, default=0.85,
                   help="Assumed spike sparsity for sparse-FLOPs estimate.")
    p.add_argument("--seq-len", type=int, default=5,
                   help="Sequence length (T) for batch FLOPs.")
    p.add_argument("--batch", type=int, default=2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.ckpt):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"skipped": True, "reason": f"no ckpt at {args.ckpt}"}, f, indent=2)
        return 0

    try:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except Exception as e:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"skipped": True, "reason": f"ckpt load failed: {e}"}, f, indent=2)
        return 0
    ck_args = ck.get("args", {}) or {}

    # Default dims from cheetah (typical ST-JEWM test case)
    state_dim = args.state_dim or 9
    action_dim = args.action_dim or 6

    is_lewm = ck_args.get("model", "stjewm") == "lewm_baseline"
    embed_dim = ck_args.get("embed_dim", 256 if is_lewm else 192)
    num_layers = ck_args.get("n_layers", 4)

    if is_lewm:
        dense_per_step = lewm_dense_flops_per_step(state_dim, action_dim, num_layers, embed_dim)
    else:
        dense_per_step = stjewm_dense_flops_per_step(state_dim, action_dim)

    # Multiply by batch * T
    dense_per_batch = dense_per_step * args.seq_len * args.batch
    dense_gmacs = dense_per_batch / 1e9

    # Add ViT-Tiny cost for ST-JEWM pixel input (one forward per timestep of the
    # 256-token ViT). Disabled here because the trained checkpoints use
    # state input. Add it conditionally.
    use_vit = (not is_lewm) and state_dim is None  # pure pixel
    if use_vit:
        dense_gmacs = (VIT_TINY_GMACS_PER_TINY_FORWARD * args.seq_len * args.batch) / 1e9

    sparse_gmacs = dense_gmacs * (1.0 - args.sparsity)

    # n_params from the loaded state_dict
    state = ck.get("model", {})
    n_params = sum(int(v.numel()) for v in state.values() if hasattr(v, "numel"))

    # Model dir name
    out_path = Path(args.out)
    model_name = out_path.stem  # <model>.json
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "skipped": False,
                "reason": None,
                "ckpt": args.ckpt,
                "model": model_name,
                "is_lewm": bool(is_lewm),
                "state_dim": int(state_dim),
                "action_dim": int(action_dim),
                "embed_dim": int(embed_dim),
                "num_layers": int(num_layers),
                "seq_len": int(args.seq_len),
                "batch": int(args.batch),
                "dense_gmacs": float(dense_gmacs),
                "sparse_gmacs": float(sparse_gmacs),
                "sparsity_assumed": float(args.sparsity),
                "n_params": int(n_params),
            },
            f, indent=2,
        )
    print(f"[flops] {model_name}: dense={dense_gmacs:.4f} GMACs  "
          f"sparse={sparse_gmacs:.4f} GMACs  params={n_params/1e6:.2f}M")
    return 0


if __name__ == "__main__":
    sys.exit(main())

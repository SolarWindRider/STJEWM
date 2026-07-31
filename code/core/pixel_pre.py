"""Frozen ViT-Tiny pixel preprocessor — shared by all baselines.

When `obs_dim >= 100` (i.e. obs is a flattened 3*image_size*image_size pixel
tensor rather than a low-dim qpos), every baseline uses this frozen ViT-Tiny
preprocessor instead of its state_dim-proportional Linear. The preprocessor
is **always frozen** (5.5M params, no gradients), so the trainable parameter
budget is the same as the state version of each baseline (within ±3.2%).

Usage:

    from code.core.pixel_pre import FrozenPixelPreprocessor
    pre = FrozenPixelPreprocessor(image_size=84, embed_dim=192)
    x_pixel = torch.rand(B, T, 3, 84, 84)
    emb = pre(x_pixel)  # (B, T, 192)
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_encoder():
    """Lazy import: ViT-Tiny lives in LeWM/src/encoder.py."""
    if "/home/lx/LeWM" not in sys.path:
        sys.path.insert(0, "/home/lx/LeWM")
    from src.encoder import Encoder
    return Encoder


class FrozenPixelPreprocessor(nn.Module):
    """Frozen ViT-Tiny (5.5M) + 2-layer MLP projector → (B, T, embed_dim).

    Always frozen. Total params ~5.6M. Used by all baselines when
    obs is pixel.
    """

    def __init__(self, image_size: int = 84, embed_dim: int = 192,
                 patch_size: int = 14):
        super().__init__()
        Encoder = _get_encoder()
        self.encoder = Encoder(image_size=image_size, patch_size=patch_size)
        self.image_size = image_size
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.hidden_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Freeze encoder ALWAYS — the per-model "5M-aligned trainable"
        # budget does NOT include this 5.5M frozen.
        for p in self.encoder.parameters():
            p.requires_grad = False

    def trainable_params(self):
        return [p for p in self.proj.parameters() if p.requires_grad]

    def frozen_params(self):
        return [p for p in self.encoder.parameters()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) → (B, T, embed_dim)."""
        B, T = x.shape[:2]
        flat = x.reshape(B * T, *x.shape[2:])  # (B*T, 3, H, W)
        feat = self.encoder(flat)  # (B*T, hidden_dim)
        emb = self.proj(feat).reshape(B, T, -1)
        return emb

    @staticmethod
    def is_pixel_obs_dim(obs_dim: int) -> bool:
        """Heuristic: pixel obs is large (≥ 3*32*32 = 3072) and not a
        small state vector.
        """
        return obs_dim >= 100

    @staticmethod
    def image_size_from_obs_dim(obs_dim: int) -> int:
        """If obs_dim looks pixel-like, infer image_size; else 0."""
        if obs_dim % 3 != 0:
            return 0
        per_channel = obs_dim // 3
        # Find integer sqrt
        s = int(per_channel ** 0.5)
        if s * s == per_channel:
            return s
        return 0

"""LeWM-style Transformer baseline for state-based 3D arm world model.

Faithful port of LeWM's JEPA architecture but for state input:
- 6-layer Transformer (vs 4-layer SNN stack in ST-JEWM)
- AdaLN-zero conditioning (LeWM's signature)
- ViT-like state encoder (Linear + LayerNorm)
- ~3.5M trainable params (similar to LeWM 3.49M for reacher)

Used as the baseline for comparison against ST-JEWM on the SAME 3D arm bench.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

sys.path.insert(0, "/home/lx/LeWM")


class StateEncoder(nn.Module):
    """Encode state (any dim) to embed_dim tokens."""
    def __init__(self, state_dim, embed_dim=192):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        # x: (B, T, state_dim) -> (B, T, embed_dim)
        return self.proj(x)


class ActionEncoder(nn.Module):
    """Encode action to embed_dim tokens."""
    def __init__(self, action_dim, embed_dim=192):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(action_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, a):
        return self.proj(a)


class AdaLNZeroBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning (LeWM style)."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # AdaLN modulation
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # Zero-init the AdaLN output (LeWM AdaLN-zero trick)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond):
        # x: (B, T, D), cond: (B, T, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa * attn_out
        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(h)
        return x


class LeWMTransformerBaseline(nn.Module):
    """LeWM-style Transformer world model for state input."""
    def __init__(self, state_dim, action_dim, embed_dim=192, num_layers=6, num_heads=8, history_size=3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.history_size = history_size
        self.num_layers = num_layers

        # Encoders
        self.state_encoder = StateEncoder(state_dim, embed_dim)
        self.action_encoder = ActionEncoder(action_dim, embed_dim)
        # LeWM-style baseline uses 6 layers by default (LeWM paper)
        # pos_embed size must be >= max(goal_offset + history + 1) across all envs.
        # LeWM paper uses goal_offset=100 (TwoRoom), so we need 1+1+100+1=102.
        # Use 256 to be safe.
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks (LeWM style: AdaLN-zero)
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)

        # Output projection
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def encode(self, state, action):
        """Alias of forward() to satisfy the model API contract in code/core/encode.py.

        Both STJEWM and LeWMTransformerBaseline expose:
        """
        return self.forward(state, action)

    def forward(self, state, action):
        """
        state: (B, T, state_dim)
        action: (B, T, action_dim)
        returns: dict with 'emb' = (B, T, embed_dim)
        """
        B, T, _ = state.shape
        s_emb = self.state_encoder(state)  # (B, T, D)
        a_emb = self.action_encoder(action)  # (B, T, D)
        x = s_emb + a_emb + self.pos_embed[:, :T]
        # Condition is the state embedding
        cond = s_emb + self.pos_embed[:, :T]
        for block in self.blocks:
            x = block(x, cond)
        x = self.norm_out(x)
        return {"emb": self.proj_out(x), "emb_pre_cell": s_emb}

    def predict(self, ctx_emb, ctx_act):
        """Per-step prediction."""
        # ctx_emb: (B, H, D), ctx_act: (B, H, A) or (B, H, D)
        if ctx_act.shape[-1] == self.action_dim:
            a_emb = self.action_encoder(ctx_act)
        else:
            a_emb = ctx_act
        # Add pos embed
        B, H, _ = ctx_emb.shape
        x = ctx_emb + a_emb + self.pos_embed[:, :H]
        cond = ctx_emb + self.pos_embed[:, :H]
        for block in self.blocks:
            x = block(x, cond)
        x = self.norm_out(x)
        return self.proj_out(x)


if __name__ == "__main__":
    # Test
    model = LeWMTransformerBaseline(state_dim=17, action_dim=5)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LeWM-style Transformer baseline: {n_params/1e6:.2f}M trainable params")
    state = torch.randn(2, 5, 17)
    action = torch.randn(2, 5, 5)
    out = model(state, action)
    print(f"Output shape: {out['emb'].shape}")

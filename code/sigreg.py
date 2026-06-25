"""SIGReg: Sketch Isotropic Gaussian Regularizer (Epps-Pulley CF distance).

Ported from LeWM (Maes et al., 2026). Projects token embeddings onto random
1-D directions, then measures discrepancy of empirical characteristic function
vs N(0,1) on a knot grid t in [0, 3]. Returned scalar ~ 0 when the marginals
match an isotropic standard Gaussian.

Public surface:
    SIGReg(knots=17, num_proj=1024) -> nn.Module
        .forward(proj) where proj has shape (T, B, D)
"""
import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU friendly Epps-Pulley)."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be >= 2")
        self.knots = knots
        self.num_proj = num_proj
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)  # N(0,1) characteristic fn
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("wphi", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.dim() != 3:
            raise ValueError(f"SIGReg expects (T, B, D); got {proj.shape}")
        T, B, D = proj.shape
        # Random 1-D projections (detached: not part of the differentiable graph).
        A = torch.randn(D, self.num_proj, device=proj.device, dtype=proj.dtype).detach()
        A = A / A.norm(p=2, dim=0, keepdim=False).clamp_min(1e-12)
        # proj @ A : (T, B, num_proj); multiply by t grid
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (T, B, num_proj, knots)
        cos_mean = x_t.cos().mean(dim=-3)        # (B, num_proj, knots)
        sin_mean = x_t.sin().mean(dim=-3)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = (err * self.wphi).sum(dim=-1) * T  # (B, num_proj)
        return statistic.mean()

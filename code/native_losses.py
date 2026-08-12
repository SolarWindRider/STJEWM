"""Native losses for SNN world-model baselines.

The ST-JEWM trainer (code/train/train.py) uses a 3-term loss:
    L = L_pred + lambda_sigreg * L_sigreg + lambda_goal * L_goal
which is appropriate for ST-JEWM's JEPA-style trace but is WRONG for
most other SNN world models. Each baseline here ships its own native
loss; the trainer dispatches on args.model_kind to call the right one.

Reference: docs/SNN_BASELINE_NATIVE_LOSSES.md
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def stjewm_loss(
    pred_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    emb_pre: torch.Tensor,
    sigreg_fn,
    goal_pred: torch.Tensor,
    goal_emb: torch.Tensor,
    lambda_sigreg: float = 0.09,
    lambda_goal: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """ST-JEWM's native loss (3 terms).

    L = ||pred - sg(tgt)||^2 + lambda_sigreg * SIGReg(emb_pre) + lambda_goal * ||goal_pred - sg(goal_emb)||^2
    """
    pred_loss = F.mse_loss(pred_emb, tgt_emb.detach())
    sigreg_loss = sigreg_fn(emb_pre.transpose(0, 1))  # (B,T,D) -> (B, knots)
    goal_loss = F.mse_loss(goal_pred, goal_emb.detach())
    total = pred_loss + lambda_sigreg * sigreg_loss + lambda_goal * goal_loss
    return total, {
        "pred": pred_loss.item(),
        "sigreg": sigreg_loss.item() if torch.is_tensor(sigreg_loss) else sigreg_loss,
        "goal": goal_loss.item(),
        "total": total.item(),
    }


def cubifae_loss(
    pred_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    spike_layers: list[torch.Tensor],
    goal_pred: torch.Tensor | None = None,
    goal_emb: torch.Tensor | None = None,
    lambda_pred: float = 1.0,
    lambda_sparse: float = 1e-3,
    lambda_goal: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """ALIF-timecell (in-house; code id cubifae_baseline) native loss.

    Two terms:
      L_pred  = ||pred - sg(tgt)||^2              # predictive loss on the time-cell readout
      L_sparse = sum over layers of mean(|spikes|) # L1 population sparsity

    For our 1-epoch budget we use mean(|spikes|) as population sparsity.

    The paper does NOT use a SIGReg or a goal term. The "anchor" readout
    (time-cell 1D-conv over membrane) is the predictive latent z_t.
    """
    pred_loss = F.mse_loss(pred_emb, tgt_emb.detach())
    if spike_layers:
        sparse_terms = [sl.abs().mean() for sl in spike_layers if sl is not None]
        sparse_loss = sum(sparse_terms) / max(len(sparse_terms), 1)
    else:
        sparse_loss = torch.tensor(0.0, device=pred_emb.device)
    total = lambda_pred * pred_loss + lambda_sparse * sparse_loss
    if goal_pred is not None and goal_emb is not None and lambda_goal > 0:
        total = total + lambda_goal * F.mse_loss(goal_pred, goal_emb.detach())
    return total, {
        "pred": pred_loss.item(),
        "sparse": sparse_loss.item(),
        "total": total.item(),
    }


def spikedreamer_loss(
    pred_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    obs_recon: torch.Tensor | None = None,
    obs_target: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    logvar: torch.Tensor | None = None,
    spike_count: torch.Tensor | None = None,
    lambda_recon: float = 1.0,
    lambda_kl: float = 1e-3,
    lambda_pred: float = 1.0,
    lambda_sparse: float = 1e-3,
) -> tuple[torch.Tensor, dict]:
    """LIF-Transformer (in-house; code id spikedreamer_baseline) native loss.

    A hybrid: LIF encoder + Transformer decoder. Total loss is:
      L_recon   = ||obs_recon - sg(obs_target)||^2   (only if pixel obs)
      L_KL      = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))  (VAE-style)
      L_pred    = ||pred - sg(tgt)||^2              (Transformer world-model loss)
      L_sparse  = mean(spike_count)                 (LIF sparsity prior)

    For state-based control (our setup), L_recon is not applicable;
    we pass obs_recon = None. The paper's λ_kl is typically 1e-3 to
    1e-4 (β-VAE annealed).
    """
    pred_loss = F.mse_loss(pred_emb, tgt_emb.detach())
    if mu is not None and logvar is not None:
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        kl = torch.tensor(0.0, device=pred_emb.device)
    if obs_recon is not None and obs_target is not None:
        recon = F.mse_loss(obs_recon, obs_target.detach())
    else:
        recon = torch.tensor(0.0, device=pred_emb.device)
    if spike_count is not None:
        sparse = spike_count.abs().mean()
    else:
        sparse = torch.tensor(0.0, device=pred_emb.device)
    total = (
        lambda_recon * recon
        + lambda_kl * kl
        + lambda_pred * pred_loss
        + lambda_sparse * sparse
    )
    return total, {
        "pred": pred_loss.item(),
        "kl": kl.item(),
        "recon": recon.item(),
        "sparse": sparse.item(),
        "total": total.item(),
    }


def slt_lif_mpc_loss(
    pred_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    spike_count: torch.Tensor | None = None,
    action_pred: torch.Tensor | None = None,
    action_target: torch.Tensor | None = None,
    lambda_pred: float = 1.0,
    lambda_sparse: float = 1e-4,
    lambda_action: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Stacked-LIF (in-house; code id slt_lif_mpc_*) native loss for a closed-loop LIF controller.

    Three terms, matching the canonical closed-loop SNN-MPC recipe:

      L_pred   = ||pred - sg(tgt)||^2              # next-state prediction
      L_sparse = mean(|spikes|)                   # LIF firing-rate prior
      L_action = ||action_pred - action_target||^2  # policy/control loss
                                                   (only if action_pred given)

    Stacked-LIF is a controller, not a pure world model — it
    explicitly trains the SNN to predict the next action (control
    output). For our CEM-eval setup we have no action supervision,
    so lambda_action defaults to 0 and the loss reduces to pred + sparse.
    """
    pred_loss = F.mse_loss(pred_emb, tgt_emb.detach())
    if spike_count is not None:
        sparse = spike_count.abs().mean()
    else:
        sparse = torch.tensor(0.0, device=pred_emb.device)
    if action_pred is not None and action_target is not None and lambda_action > 0:
        action_loss = F.mse_loss(action_pred, action_target.detach())
    else:
        action_loss = torch.tensor(0.0, device=pred_emb.device)
    total = lambda_pred * pred_loss + lambda_sparse * sparse + lambda_action * action_loss
    return total, {
        "pred": pred_loss.item(),
        "sparse": sparse.item(),
        "action": action_loss.item(),
        "total": total.item(),
    }


# Dispatch table used by code/train/train.py
NATIVE_LOSS_DISPATCH = {
    "stjewm_trace_only":     stjewm_loss,
    "stjewm_hidden_leak":    stjewm_loss,
    "stjewm_spike_only":     stjewm_loss,
    "stjewm_no_trace":       stjewm_loss,
    "stjewm_membrane_readout":stjewm_loss,
    "stjewm_rate_only":      stjewm_loss,
    "lewm_baseline_v2":      stjewm_loss,           # LeWM shares JEPA-style pred
    "lewm_baseline_no_goal": stjewm_loss,
    "lewm_baseline_trace_only": stjewm_loss,
    "gru_baseline":          stjewm_loss,           # GRU is JEPA-style in our wrapper
    "mlp_baseline":          stjewm_loss,           # MLP is JEPA-style in our wrapper
    "cubifae_baseline":      cubifae_loss,
    "spikedreamer_baseline": spikedreamer_loss,
    "slt_lif_mpc_trace":     slt_lif_mpc_loss,
    "slt_lif_mpc_free":      slt_lif_mpc_loss,
}

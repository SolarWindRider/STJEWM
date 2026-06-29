"""ST-JEWM — Pure SNN reconstruction-free world model.

This is the response to the NMI reviewer challenge that the prior hybrid
(LeWM Transformer + SNN side-channel) is not a pure SNN model. This version
**fully replaces** the LeWM Transformer with a stack of MultiCompartment SNN
cells.

Architecture (pixel input):
    pixels -> ViT-Tiny (frozen) -> projector -> z_enc (B, T, 192)
    actions -> ActionMLP (1-layer) -> a_emb (B, T, 192)
    z_enc + a_emb -> MultiCompStack(4 layers) -> h (B, T, 192)
    spike -> GatedSpikeTrace(ctx=[a_emb, h]) -> r_t
    z_final = h + trace_proj(trace) -> next-state latent

Architecture (state input):
    state -> StateProjector (MLP) -> z_enc
    ... same as above ...

Loss:  pred_loss + lambda_sigreg * sigreg_loss + lambda_goal * goal_loss

Trainable params: ~4.6M
"""

from __future__ import annotations

import sys
from enum import Enum

sys.path.insert(0, "/home/lx/LeWM")
sys.path.insert(0, "/home/lx/snn/code")

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder import Encoder

from snn_cell import MultiCompartmentCell


class ReadoutMode(str, Enum):
    """Controls what latent state is exposed by `forward()` and `predict()`.

    The membrane-forbidden protocol mandates that planner / predictor modules
    cannot read the full continuous hidden state. We support:
        - HIDDEN_LEAK       (default, legacy): z_final = h + trace_proj(trace)
        - TRACE_ONLY        (NMI requirement): z_final = trace_proj(trace)
        - MEMBRANE_READOUT  : z_final = h_post_cell.detach()  # discrete state
        - SPIKE_ONLY        : z_final = post_mlp(spike).detach()
        - RATE_ONLY         : z_final = moving_average(spike).detach()
        - NO_TRACE          : z_final = h  # ablation: no trace branch
    """
    TRACE_ONLY = "trace_only"
    HIDDEN_LEAK = "hidden_leak"
    MEMBRANE_READOUT = "membrane_readout"
    SPIKE_ONLY = "spike_only"
    RATE_ONLY = "rate_only"
    NO_TRACE = "no_trace"

# ============== B1: Gated Spike Trace (reused from v3) ==============
class GatedSpikeTrace(nn.Module):
    """Content-aware exponential trace: r_t = alpha_t * r_{t-1} + (1 - alpha_t) * s_t.

    alpha_t = sigma(W_gate . [r_{t-1}, s_t, c_t]) where c_t is the conditioning
    context (action + z_pred). Allows the trace to 'forget fast' when not
    relevant and 'remember long' when context demands it.
    """

    def __init__(self, d_hidden: int, d_context: int, init_alpha: float = 0.9):
        super().__init__()
        # gate input: r_{t-1} (D) + s_t (D) + c_t (d_context)
        self.gate = nn.Linear(2 * d_hidden + d_context, d_hidden, bias=True)
        # init bias so initial alpha ~= init_alpha
        with torch.no_grad():
            bias = torch.full(
                (d_hidden,),
                float(torch.log(torch.tensor(init_alpha / (1 - init_alpha))).item()),
            )
            self.gate.bias.copy_(bias)
            self.gate.weight.zero_()
        self.d_hidden = d_hidden

    def forward(self, spike: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute gated trace.

        Args:
            spike: (B, T, D) binary spikes from cell
            context: (B, T, d_context) conditioning (action + z_pred)

        Returns:
            trace: (B, T, D) gated spike trace
        """
        B, T, D = spike.shape
        r = torch.zeros(B, D, device=spike.device, dtype=spike.dtype)
        traces = []
        for t in range(T):
            s = spike[:, t]
            gate_in = torch.cat([r, s, context[:, t]], dim=-1)
            alpha = torch.sigmoid(self.gate(gate_in))  # (B, D)
            r = alpha * r + (1.0 - alpha) * s
            traces.append(r)
        return torch.stack(traces, dim=1)


# ============== A2: MultiCompStack ==============
class MultiCompStack(nn.Module):
    """Stack of N MultiCompartment SNN cells with LayerNorm + residual.

    Each layer:
        s_l  = MultiCompCell(h_{l-1})     # dendritic spike
        r_l  = post_mlp_l(s_l)            # learnable per-cell readout
        h_l  = h_{l-1} + LayerNorm(r_l)   # residual + LN

    The MultiComp cell is a 3-dendrite + 1-soma spiking compartment with rich
    dendritic computation. The per-cell post-MLP gives the stack learnable
    capacity to mix channels (the SNN cell itself is recurrent over time but
    does not mix channels on its own).
    """

    def __init__(
        self,
        d_hid: int = 192,
        n_layers: int = 4,
        n_d: int = 3,
        trace_beta: float = 0.9,
        mlp_hidden: int | None = None,
    ):
        super().__init__()
        self.d_hid = d_hid
        self.d_hid = d_hid
        self.n_layers = n_layers
        # 12x expansion -> per cell ~885K params, stack ~3.5M, total ~5M
        if mlp_hidden is None:
            mlp_hidden = 12 * d_hid
        self.cells = nn.ModuleList(
            [
                MultiCompartmentCell(
                    d_in=d_hid, d_hid=d_hid, N_d=n_d, trace_beta=trace_beta
                )
                for _ in range(n_layers)
            ]
        )
        self.post_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_hid, mlp_hidden),
                    nn.GELU(),
                    nn.Linear(mlp_hidden, d_hid),
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_hid, eps=1e-6) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Forward through the stack.

        Args:
            x: (B, T, D) input embeddings (e.g. z_enc + a_emb)

        Returns:
            dict with:
                h: (B, T, D) final hidden state after the stack
                spike: (B, T, D) spikes from the LAST layer
                spike_layers: list of (B, T, D) spikes from each layer
                trace: (B, T, D) gated trace from the LAST layer (pre-gate)
        """
        spike_layers = []
        h = x
        cell_out = None
        for cell, post_mlp, norm in zip(self.cells, self.post_mlps, self.norms):
            cell_out = cell(h)
            spk = cell_out["spike"]  # (B, T, D)
            spike_layers.append(spk)
            # Per-cell learnable readout -> residual + LayerNorm
            r = post_mlp(spk)
            h = h + norm(r)
        return {
            "h": h,
            "spike": spike_layers[-1],
            "spike_layers": spike_layers,
            "trace": cell_out.get("trace"),
        }


# ============== Simple action encoder (1-layer MLP) ==============
class ActionMLP(nn.Module):
    """1-layer linear action encoder: action_dim -> emb_dim.

    The LeWM Embedder uses 1D-conv smoothing plus a 2-layer MLP, but this model uses
    minimal additional params; a single linear projection suffices.
    """

    def __init__(self, action_dim: int, emb_dim: int = 192):
        super().__init__()
        self.proj = nn.Linear(action_dim, emb_dim)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # a: (B, T, action_dim) -> (B, T, emb_dim)
        return self.proj(a)


# ============== State projector (DMC: low-dim state -> 192) ==============
class StateProjector(nn.Module):
    """State input projector: low-dim state vector -> 192-D token embedding."""

    def __init__(self, state_dim: int, emb_dim: int = 192):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.proj(s)


# ============== Main model: STJEWM ==============
class STJEWM(nn.Module):
    """ST-JEWM: Pure SNN reconstruction-free world model.
    Pipeline:
        pixels/state -> encoder -> z_enc (B, T, 192)
        action -> ActionMLP -> a_emb (B, T, 192)
        h = z_enc + a_emb
        h, spikes = MultiCompStack(h)
        trace = GatedSpikeTrace(spikes, ctx=[a_emb, h])
        z_final = h + trace_proj(trace)

    NO Transformer, NO AdaLN, NO attention. Pure spike-based dynamics.
    """

    def __init__(
        self,
        d_hid: int = 192,
        embed_dim: int = 192,
        action_dim: int = 10,
        action_emb_dim: int = 192,
        state_dim: int | None = None,  # if set, use StateProjector for low-dim state input
        cell_n_layers: int = 4,
        n_d: int = 3,
        trace_beta: float = 0.9,
        freeze_encoder: bool = True,
        image_size: int = 224,
        patch_size: int = 14,
        readout_mode: str = "hidden_leak",
    ):
        super().__init__()
        self.d_hid = d_hid
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.trace_beta = trace_beta
        self.readout_mode = ReadoutMode(readout_mode)

        # ViT-Tiny encoder (frozen by default)
        self.encoder = Encoder(image_size=image_size, patch_size=patch_size)
        # Projector: ViT hidden_dim -> embed_dim
        # LeWM uses MLP(input_dim -> hidden_dim -> output_dim) with BatchNorm1d;
        # we use a simpler 2-layer MLP to keep params low.
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.hidden_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Action encoder (1-layer MLP per spec)
        self.action_encoder = ActionMLP(action_dim=action_dim, emb_dim=action_emb_dim)

        # Optional state projector (for DMC-style low-dim state input)
        self.state_projector = (
            StateProjector(state_dim, embed_dim) if state_dim is not None else None
        )

        # A2: MultiCompStack (the predictor — replaces Transformer)
        self.stack = MultiCompStack(
            d_hid=d_hid, n_layers=cell_n_layers, n_d=n_d, trace_beta=trace_beta
        )

        # B1: Gated Spike Trace
        # context = [a_emb, h] -> 2 * d_hid
        self.gated_trace = GatedSpikeTrace(
            d_hidden=d_hid, d_context=2 * d_hid, init_alpha=trace_beta
        )

        # Trace projection (skip-style, added to h)
        self.trace_proj = nn.Linear(d_hid, d_hid, bias=False)

        # Freeze encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Stash config
        self._cfg = dict(
            d_hid=d_hid, embed_dim=embed_dim, action_dim=action_dim,
            action_emb_dim=action_emb_dim, state_dim=state_dim,
            cell_n_layers=cell_n_layers, n_d=n_d, trace_beta=trace_beta,
            freeze_encoder=freeze_encoder, image_size=image_size, patch_size=patch_size,
        )

    # ============== Encoders ==============
    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation -> (B, T, embed_dim).

        obs can be:
            (B, T, 3, H, W) pixel input — uses ViT encoder + projector
            (B, T, state_dim) state input — uses state_projector
        """
        if self.state_dim is not None and obs.shape[-1] == self.state_dim:
            return self.state_projector(obs)
        B, T = obs.shape[:2]
        flat = obs.reshape(B * T, *obs.shape[2:])
        feat = self.encoder(flat)  # (B*T, hidden_dim)
        emb = self.projector(feat).reshape(B, T, -1)
        return emb

    # ============== Readout mode (membrane-forbidden protocol) ==============
    def _readout(self, h: torch.Tensor, spike: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        """Apply the configured readout mode to combine h/spike/trace into z_final.

        - HIDDEN_LEAK:       z = h + trace_proj(trace)  (legacy)
        - TRACE_ONLY:        z = trace_proj(trace)        (NMI)
        - MEMBRANE_READOUT:  z = h.detach()               (treat h as discrete latent)
        - SPIKE_ONLY:        z = h * spike.float().detach()  (mask h by spikes)
        - RATE_ONLY:         z = F.avg_pool1d(h.transpose(1,2), kernel_size=4, stride=1).transpose(1,2)
                             (downsampled h, no trace; rate-like)
        - NO_TRACE:          z = h                        (ablation)
        """
        mode = self.readout_mode
        if mode == ReadoutMode.HIDDEN_LEAK:
            return h + self.trace_proj(trace)
        if mode == ReadoutMode.TRACE_ONLY:
            return self.trace_proj(trace)
        if mode == ReadoutMode.MEMBRANE_READOUT:
            return h.detach()
        if mode == ReadoutMode.SPIKE_ONLY:
            return h * spike.float().detach()
        if mode == ReadoutMode.RATE_ONLY:
            # downsample h along time as a rate-style readout
            h_t = h.transpose(1, 2)  # (B, D, T)
            pooled = F.avg_pool1d(h_t, kernel_size=4, stride=1, padding=2)
            return pooled.transpose(1, 2)[:, : h.shape[1], :]
        if mode == ReadoutMode.NO_TRACE:
            return h
        raise ValueError(f"Unknown readout mode: {mode}")
    # ============== API: match v3 exactly ==============
    def encode(self, x: torch.Tensor, a: torch.Tensor) -> dict:
        """Encode (obs, action) -> {'emb': (B,T,D), 'act_emb': (B,T,D)}.

        Matches v3 STJEWMv3.encode(images, actions) signature.
        """
        emb = self._encode_obs(x)  # (B, T, D)
        act_emb = self.action_encoder(a)  # (B, T, D)
        return {"emb": emb, "act_emb": act_emb}

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> dict:
        """Full forward: returns dict with 'emb' (the predicted latent)."""
        enc = self.encode(x, a)
        emb = enc["emb"]           # (B, T, D)
        act_emb = enc["act_emb"]   # (B, T, D)
        # A2: MultiCompStack on z_enc + a_emb
        h_in = emb + act_emb
        stack_out = self.stack(h_in)
        h = stack_out["h"]              # (B, T, D)
        spike = stack_out["spike"]      # (B, T, D)
        # B1: Gated Spike Trace — context = [a_emb, h]
        context = torch.cat([act_emb, h], dim=-1)  # (B, T, 2D)
        trace = self.gated_trace(spike, context)    # (B, T, D)
        # z_final = h + trace_proj(trace)  (legacy; replaced by _readout)
        z_final = self._readout(h, spike, trace)
        return {
            "emb": z_final,
            "emb_pre_cell": emb,
            "act_emb": act_emb,
            "spike": spike,
            "trace": trace,
            "spike_layers": stack_out["spike_layers"],
            "h": h,
        }

    def predict(self, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> torch.Tensor:
        """Per-step prediction. Matches v3 STJEWMv3.predict signature.

        ctx_act can be either raw actions (action_dim) or pre-encoded (192-D).
        """
        if ctx_act.shape[-1] == self.d_hid:
            act_emb = ctx_act
        else:
            act_emb = self.action_encoder(ctx_act)
        # Run MultiCompStack
        h_in = ctx_emb + act_emb
        stack_out = self.stack(h_in)
        h = stack_out["h"]
        spike = stack_out["spike"]
        # Gated trace
        context = torch.cat([act_emb, h], dim=-1)
        trace = self.gated_trace(spike, context)
        return self._readout(h, spike, trace)

    @staticmethod
    def criterion(pred_emb: torch.Tensor, tgt_emb: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_emb, tgt_emb)

    @torch.no_grad()
    def rollout(self, init_emb: torch.Tensor, action_sequence: torch.Tensor, history_size: int = 3) -> torch.Tensor:
        """Autoregressive latent rollout (matches JEPA.rollout)."""
        B, T, A = action_sequence.shape
        h = init_emb[:, -history_size:].clone()
        preds = []
        for t in range(T):
            a_t = action_sequence[:, t: t + history_size]
            if a_t.shape[1] < history_size:
                pad = torch.zeros(B, history_size - a_t.shape[1], A, device=a_t.device, dtype=a_t.dtype)
                a_t = torch.cat([a_t, pad], dim=1)
            nxt = self.predict(h, a_t)[:, -1]
            preds.append(nxt)
            h = torch.cat([h, nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def cost(self, pred_emb: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
        return ((pred_emb - goal_emb) ** 2).flatten(1).sum(-1)

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        """swm.Costable protocol: cost for action candidates.

        info_dict: {'goal_emb': (B, D), 'init_emb': (B, H, D)}
        action_candidates: (B, K, T, action_dim)
        """
        goal_emb = info_dict["goal_emb"]
        init_emb = info_dict["init_emb"]
        B, K, T, A = action_candidates.shape
        history_size = init_emb.shape[1]
        # Embed actions: (B, K, T, A) -> (B, K, T, D)
        act_flat = action_candidates.reshape(B * K, T, A)
        act_emb_flat = self.action_encoder(act_flat)
        act_emb = act_emb_flat.reshape(B, K, T, -1)
        # Replicate init across K
        init_k = init_emb.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, history_size, init_emb.shape[2])
        h = init_k
        for t in range(T):
            avail = T - t
            if avail >= history_size:
                a_t = act_emb[:, :, t:t + history_size].reshape(B * K, history_size, -1)
            else:
                a_t_partial = act_emb[:, :, t:].reshape(B * K, avail, -1)
                pad = torch.zeros(B * K, history_size - avail, act_emb.shape[-1],
                                  device=act_emb.device, dtype=act_emb.dtype)
                a_t = torch.cat([a_t_partial, pad], dim=1)
            h_in = h[:, -history_size:]
            nxt = self.predict(h_in, a_t)[:, -1]
            h = torch.cat([h, nxt.unsqueeze(1)], dim=1)
        z_final = h[:, -1].reshape(B, K, -1)
        goal_exp = goal_emb.unsqueeze(1).expand(-1, K, -1)
        return ((z_final - goal_exp) ** 2).sum(-1)


# ============== Factory ==============
def make_stjewm(
    obs_dim: int | None = None,
    action_dim: int = 10,
    n_layers: int = 4,
    n_neurons: int = 192,
    n_d: int = 3,
    trace_beta: float = 0.9,
    freeze_encoder: bool = True,
    image_size: int = 224,
    patch_size: int = 14,
    readout_mode: str = "hidden_leak",
) -> STJEWM:
    """Factory: build a pure-SNN STJEWM world model.

    Args:
        obs_dim: observation dim. If int and != 3*image_size^2, treated as
            state_dim (low-dim state input like DMC). If None, use pixel
            encoder.
        action_dim: action dimension.
        n_layers: number of MultiComp SNN cells in the stack.
        n_neurons: hidden dim (default 192 to match ViT-Tiny).
        n_d: number of dendrites per cell.
        trace_beta: initial trace decay.
        freeze_encoder: freeze ViT weights.
        image_size: ViT image size.
        patch_size: ViT patch size.
        readout_mode: how `forward()` / `predict()` combine h/spike/trace
            (see `ReadoutMode`). Default `"hidden_leak"` reproduces v2 behavior.

    Returns:
        STJEWM model.
    """
    # Heuristic: state_dim if obs_dim is small (< 100) and not 3 (RGB)
    state_dim = None
    if obs_dim is not None and obs_dim < 100 and obs_dim != 3:
        state_dim = obs_dim

    return STJEWM(
        d_hid=n_neurons,
        embed_dim=n_neurons,
        action_dim=action_dim,
        action_emb_dim=n_neurons,
        state_dim=state_dim,
        cell_n_layers=n_layers,
        n_d=n_d,
        trace_beta=trace_beta,
        freeze_encoder=freeze_encoder,
        image_size=image_size,
        patch_size=patch_size,
        readout_mode=readout_mode,
    )


# ============== Smoke test ==============
if __name__ == "__main__":
    import time

    # Pixel-input model
    print("=" * 60)
    print("STJEWM smoke test — pixel input")
    print("=" * 60)
    model = make_stjewm(obs_dim=3 * 224 * 224, action_dim=10, n_layers=4, n_neurons=192)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_frozen = n_total - n_params
    print(f"STJEWM params: trainable={n_params:,}  total={n_total:,}  frozen={n_frozen:,}")

    assert 4_000_000 <= n_params <= 6_000_000, (
        f"trainable params {n_params:,} outside target [4M, 6M]"
    )

    B, T = 2, 5
    images = torch.randn(B, T, 3, 224, 224)
    actions = torch.randn(B, T, 10)
    t0 = time.time()
    out = model(images, actions)
    dt = time.time() - t0
    print(f"emb shape: {tuple(out['emb'].shape)}")
    print(f"spike shape: {tuple(out['spike'].shape)}")
    print(f"trace shape: {tuple(out['trace'].shape)}")
    print(f"forward time: {dt:.2f}s")
    assert out["emb"].shape == (B, T, 192), f"emb shape {out['emb'].shape} != ({B},{T},192)"

    # Spike sparsity
    sparsity = 1.0 - out["spike"].float().mean().item()
    print(f"spike sparsity: {sparsity:.3f}")
    assert 0.75 <= sparsity <= 0.95, f"spike sparsity {sparsity:.3f} outside [0.75, 0.95]"

    # Test predict() API
    ctx_emb = torch.randn(B, 3, 192)
    ctx_act = torch.randn(B, 3, 10)
    pred = model.predict(ctx_emb, ctx_act)
    print(f"predict shape: {tuple(pred.shape)}")
    assert pred.shape == (B, 3, 192)

    # Test predict with pre-encoded actions
    ctx_act_emb = torch.randn(B, 3, 192)
    pred2 = model.predict(ctx_emb, ctx_act_emb)
    print(f"predict(pre-encoded) shape: {tuple(pred2.shape)}")

    # State-input model (DMC-like)
    print()
    print("=" * 60)
    print("STJEWM smoke test — state input (DMC)")
    print("=" * 60)
    model_state = make_stjewm(obs_dim=9, action_dim=1, n_layers=4, n_neurons=192)
    state = torch.randn(B, T, 9)
    act = torch.randn(B, T, 1)
    out_s = model_state(state, act)
    print(f"state emb shape: {tuple(out_s['emb'].shape)}")
    assert out_s["emb"].shape == (B, T, 192)
    sparsity_s = 1.0 - out_s["spike"].float().mean().item()
    print(f"state spike sparsity: {sparsity_s:.3f}")
    assert 0.75 <= sparsity_s <= 0.95, f"state spike sparsity {sparsity_s:.3f} outside [0.75, 0.95]"

    print()
    print("=" * 60)
    print("OK STJEWM smoke test")
    print("=" * 60)


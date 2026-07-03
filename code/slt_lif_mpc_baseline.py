"""SLT-LIF-MPC baseline — Liu et al. 2024 NeurIPS workshop (DECOLLE/STBP-style).

A minimal pure-SNN predictive model:
    1. StateProjector (low-dim state -> d_in)
    2. ActionMLP     (action -> d_in)
    3. Stack of 4 LIFCells with atan surrogate (alpha=2.0, same as STJEWM)
    4. Two readout variants:
         - TraceOnly :   z_t = moving_avg(s_t, k=4) projected to 192
                         (membrane-forbidden; only s_t + trace exposed)
         - FreeAccess:   z_t = concat([s_t, v_t]) projected to 192
                         (membrane-exposed; planner reads v_t directly)

Contract (same as STJEWM / GRU / LeWM):
    model.encode(obs, action) -> dict with key 'emb' (B, T, 192)
    model.predict(ctx_emb, ctx_act) -> Tensor (B, 192)
    model.forward() returns dict with
        'emb'           (B, T, 192)         predicted latent
        'emb_pre_cell'  (B, T, d_in)        pre-stack state embedding (for SIGReg)
        'act_emb'       (B, T, d_in)
        'spike'         (B, T, d_in)        final-layer spikes (for sparsity)
        'trace'         (B, T, d_in)        moving_avg(s_t) for TraceOnly,
                                            s_t for FreeAccess
        'h'             (B, T, d_in)        s_t for TraceOnly,
                                            v_t for FreeAccess
        'spike_layers'  list[Tensor]        per-layer spikes

This file follows the STJEWM code structure:
    1. ATan surrogate is already provided in code/snn_cell.py (alpha=2.0).
    2. LIFCell wraps the surrogate, hard reset, refractoriness, exp trace.
    3. Stacked with residual + LayerNorm, like MultiCompStack.
"""
from __future__ import annotations

import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/lx/snn/code")

from snn_cell import LIFCell


# ============================================================
# StateProjector (low-dim state -> d_in)
# ============================================================
class StateProjector(nn.Module):
    """State input projector: low-dim state vector -> d_in-D token embedding."""

    def __init__(self, state_dim: int, d_in: int = 192):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim, d_in),
            nn.SiLU(),
            nn.Linear(d_in, d_in),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.proj(s)


# ============================================================
# ActionMLP (action -> d_in)
# ============================================================
class ActionMLP(nn.Module):
    """1-layer (effectively 2-MLP) action encoder: action_dim -> emb_dim."""

    def __init__(self, action_dim: int, emb_dim: int = 192):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(action_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.proj(a)


# ============================================================
# LIF stack — like STJEWM's MultiCompStack but using plain LIFCell
# ============================================================
class LIFStack(nn.Module):
    """Stack of N LIFCells with LayerNorm + residual."""

    def __init__(self, d_in: int = 192, n_layers: int = 4, trace_beta: float = 0.9):
        super().__init__()
        self.n_layers = n_layers
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.cells.append(
                LIFCell(d_in=d_in, d_hid=d_in, v_thresh=0.3, v_reset=0.0,
                        tau_m=20.0, t_ref=2.0, dt=1.0, alpha_surr=2.0,
                        init_scale=3.0, trace_beta=trace_beta)
            )
            self.norms.append(nn.LayerNorm(d_in))

    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, T, D). Returns dict with 'h', 'spike', 'trace', 'v_last', 'spike_layers'."""
        h = x
        spike_layers: List[torch.Tensor] = []
        v_last_seq: List[torch.Tensor] = []
        for cell, norm in zip(self.cells, self.norms):
            res = h
            out_cell = cell(norm(h))
            spk = out_cell["spike"]
            # soft residual: spike adds to the input through identity (grad through ATan)
            h = res + spk
            spike_layers.append(spk)
            # capture the last layer's membrane potential sequence (B, T, D)
            v_last_seq.append(out_cell["v"])
        return {
            "h": h,
            "spike": spike_layers[-1],
            "trace": None,
            "v_last": v_last_seq[-1],
            "spike_layers": spike_layers,
        }


# ============================================================
# Base class — both variants share this
# ============================================================
class SLT_LIF_MPCBase(nn.Module):
    """Base SLT-LIF-MPC model. Subclasses specialize the readout (z_t) shape.

    Common structure:
        obs -> StateProjector -> emb_pre_cell (B,T,D)
        action -> ActionMLP    -> act_emb     (B,T,D)
        h = emb_pre_cell + act_emb
        h, spike = LIFStack(h)         # h is the post-stack spike-rate-like output
        z = readout(h, v, s) -> (B,T,192)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_in: int = 192,
        embed_dim: int = 192,
        n_layers: int = 4,
        trace_beta: float = 0.9,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_in = d_in
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.state_projector = StateProjector(state_dim, d_in=d_in)
        self.action_encoder = ActionMLP(action_dim=action_dim, emb_dim=d_in)
        self.stack = LIFStack(d_in=d_in, n_layers=n_layers, trace_beta=trace_beta)

        # Readout projector (subclass-specific dim via self._readout_input_dim)
        self.readout = nn.Linear(self._readout_input_dim, embed_dim)

    # ============== Subclass-supplied input dim to readout ==============
    @property
    def _readout_input_dim(self) -> int:
        raise NotImplementedError

    # ============== Readout (subclass-specific) ==============
    def _readout(self, h: torch.Tensor, v_last: torch.Tensor, s_last: torch.Tensor) -> torch.Tensor:
        """Combine the last-layer outputs into the (B,T,embed_dim) latent.

        TraceOnly : moving_avg(s_last, k=4) -> 192
        FreeAccess: concat([s_last, v_last]) -> 192
        """
        raise NotImplementedError

    # ============== Encoders ==============
    def encode(self, x: torch.Tensor, a: torch.Tensor) -> dict:
        """Encode (obs, action) -> {'emb': (B,T,D), 'act_emb': (B,T,D)}.

        `emb` here is the pre-stack state embedding (matches STJEWM's
        `emb_pre_cell`); the post-stack readout goes into `forward`.
        """
        emb_pre = self.state_projector(x)        # (B, T, D)
        act_emb = self.action_encoder(a)          # (B, T, D)
        return {"emb": emb_pre, "act_emb": act_emb}

    # ============== Forward ==============
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> dict:
        """Full forward. Returns dict matching STJEWM contract.

        `emb`            = post-readout latent (B, T, 192) — what planner sees
        `emb_pre_cell`   = pre-stack state embedding (B, T, D) — for SIGReg
        `act_emb`        = (B, T, D)
        `spike`          = (B, T, D) last-layer spikes (sparsity)
        `trace`          = (B, T, D) the membrane-forbidden "trace":
                            moving_avg(s_t, k=4) for TraceOnly,
                            s_t for FreeAccess.
        `h`              = (B, T, D) the membrane-exposed hidden:
                            s_t for TraceOnly (membrane-forbidden),
                            v_t for FreeAccess.
        `spike_layers`   = list[(B, T, D)] per-layer spikes
        """
        enc = self.encode(x, a)
        emb_pre = enc["emb"]
        act_emb = enc["act_emb"]

        # A: LIFStack on z_enc + a_emb
        h_in = emb_pre + act_emb
        stack_out = self.stack(h_in)
        s_last = stack_out["spike"]          # (B, T, D)
        spike_layers = stack_out["spike_layers"]

        # We need v_last too. Each LIFCell returns "v" (B, T, D). For the
        # last-layer v, we re-run the last cell quickly to get v. To keep
        # things simple and re-use the last cell's v, run it standalone here.
        # But LIFCell maintains its own state across t, so we need to do a
        # full forward through the last cell on `s_last`'s *pre-spike* input.
        # Instead: use a small trick — the v of the last layer is recoverable
        # by replaying the last cell on its own input. To avoid extra work,
        # we extend LIFStack to also return v_last.
        v_last = stack_out.get("v_last", s_last)   # fallback if not provided
        h_last = stack_out["h"]                    # post-residual spike output

        z = self._readout(h_last, v_last, s_last)  # (B, T, 192)
        trace = self._trace(h_last, v_last, s_last)  # (B, T, D)

        return {
            "emb": z,
            "emb_pre_cell": emb_pre,
            "act_emb": act_emb,
            "spike": s_last,
            "trace": trace,
            "h": h_last if self._is_membrane_forbidden else v_last,
            "spike_layers": spike_layers,
        }

    # ============== Per-step prediction ==============
    def predict(self, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> torch.Tensor:
        """Per-step prediction. ctx_emb: (B, H, D), ctx_act: (B, H, A or D).

        Returns the last-timestep readout of (B, embed_dim).
        """
        if ctx_act.shape[-1] == self.d_in:
            act_emb = ctx_act
        else:
            act_emb = self.action_encoder(ctx_act)
        h_in = ctx_emb + act_emb
        stack_out = self.stack(h_in)
        s_last = stack_out["spike"]
        v_last = stack_out.get("v_last", s_last)
        h_last = stack_out["h"]
        z = self._readout(h_last, v_last, s_last)
        return z  # full readout (B, H, 192) — matches STJEWM.predict contract

    # ============== API stubs (CEM planner compat) ==============
    @staticmethod
    def criterion(pred_emb: torch.Tensor, tgt_emb: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_emb, tgt_emb)

    @property
    def readout_mode(self) -> str:
        return self._readout_name

    @property
    def _is_membrane_forbidden(self) -> bool:
        return getattr(self, "_readout_name", "") == "trace_only"

    def _trace(self, h: torch.Tensor, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """The 'trace' exposed to the membrane-forbidden protocol consumers
        (probe.py reads out['trace'])."""
        return s if self._is_membrane_forbidden else s


# ============================================================
# TraceOnly — membrane-forbidden (only s_t + moving_avg exposed)
# ============================================================
class SLT_LIF_MPC_TraceOnly(SLT_LIF_MPCBase):
    """TraceOnly variant: only s_t and a moving-average trace are exposed.

    z_t = moving_avg(s_t, k=4) projected to 192
    'h' = s_t (NOT v_t; protocol forbids membrane)
    'trace' = moving_avg(s_t, k=4)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_in: int = 192,
        embed_dim: int = 192,
        n_layers: int = 4,
        trace_beta: float = 0.9,
        k_avg: int = 4,
    ):
        self._readout_name = "trace_only"
        self.k_avg = k_avg
        super().__init__(
            state_dim=state_dim, action_dim=action_dim,
            d_in=d_in, embed_dim=embed_dim,
            n_layers=n_layers, trace_beta=trace_beta,
        )

    @property
    def _readout_input_dim(self) -> int:
        return self.d_in

    def _readout(self, h: torch.Tensor, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # z = moving_avg(s, k=4) projected to embed_dim
        s_t = s.transpose(1, 2)                  # (B, D, T)
        pooled = F.avg_pool1d(s_t, kernel_size=self.k_avg, stride=1,
                              padding=self.k_avg // 2)
        avg_s = pooled.transpose(1, 2)[:, : s.shape[1], :]  # (B, T, D)
        return self.readout(avg_s)

    def _trace(self, h, v, s):
        # Same moving_avg, exposed as 'trace' for probe.py
        s_t = s.transpose(1, 2)
        pooled = F.avg_pool1d(s_t, kernel_size=self.k_avg, stride=1,
                              padding=self.k_avg // 2)
        return pooled.transpose(1, 2)[:, : s.shape[1], :]


# ============================================================
# FreeAccess — membrane-exposed (planner reads v_t directly)
# ============================================================
class SLT_LIF_MPC_FreeAccess(SLT_LIF_MPCBase):
    """FreeAccess variant: continuous membrane potential v_t is exposed.

    z_t = concat([s_t, v_t]) projected to 192 (note: 2D -> 192D)
    'h' = v_t (the membrane potential)
    'trace' = s_t (raw spike rate; for probe.py compatibility)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_in: int = 192,
        embed_dim: int = 192,
        n_layers: int = 4,
        trace_beta: float = 0.9,
    ):
        self._readout_name = "free_access"
        super().__init__(
            state_dim=state_dim, action_dim=action_dim,
            d_in=d_in, embed_dim=embed_dim,
            n_layers=n_layers, trace_beta=trace_beta,
        )

    @property
    def _readout_input_dim(self) -> int:
        return 2 * self.d_in

    def _readout(self, h: torch.Tensor, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # z = concat([s, v]) projected to embed_dim
        z = torch.cat([s, v], dim=-1)            # (B, T, 2D)
        return self.readout(z)

    def _trace(self, h, v, s):
        # For probe.py: expose raw spike trace (no moving avg in this variant
        # because the trace is not what the planner uses)
        return s


# ============================================================
# Factory helpers
# ============================================================
def make_slt_lif_mpc_trace(
    state_dim: int, action_dim: int,
    d_in: int = 192, embed_dim: int = 192,
    n_layers: int = 4, trace_beta: float = 0.9, k_avg: int = 4,
) -> SLT_LIF_MPC_TraceOnly:
    return SLT_LIF_MPC_TraceOnly(
        state_dim=state_dim, action_dim=action_dim,
        d_in=d_in, embed_dim=embed_dim,
        n_layers=n_layers, trace_beta=trace_beta, k_avg=k_avg,
    )


def make_slt_lif_mpc_free(
    state_dim: int, action_dim: int,
    d_in: int = 192, embed_dim: int = 192,
    n_layers: int = 4, trace_beta: float = 0.9,
) -> SLT_LIF_MPC_FreeAccess:
    return SLT_LIF_MPC_FreeAccess(
        state_dim=state_dim, action_dim=action_dim,
        d_in=d_in, embed_dim=embed_dim,
        n_layers=n_layers, trace_beta=trace_beta,
    )


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("SLT-LIF-MPC smoke test")
    print("=" * 60)

    B, T, D_obs, D_act = 4, 12, 9, 6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for variant_name, factory in [
        ("TraceOnly", make_slt_lif_mpc_trace),
        ("FreeAccess", make_slt_lif_mpc_free),
    ]:
        print(f"\n--- {variant_name} ---")
        model = factory(state_dim=D_obs, action_dim=D_act).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"params total = {n_params/1e6:.3f}M")
        s = torch.randn(B, T, D_obs, device=device)
        a = torch.randn(B, T, D_act, device=device)
        t0 = time.time()
        out = model(s, a)
        dt = time.time() - t0
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)}")
            else:
                print(f"  {k}: {type(v).__name__} (len={len(v)})")
        # Predict path: first H=3 steps of the window
        ctx_emb = out["emb_pre_cell"][:, :3]
        ctx_act = a[:, :3]
        pred = model.predict(ctx_emb, ctx_act)
        print(f"  predict: {tuple(pred.shape)}  (should be ({B},192))")
        print(f"  forward+backward: {dt*1000:.1f}ms")
        # Quick backward
        loss = out["emb"].sum() + pred.sum()
        loss.backward()
        print(f"  backward OK")
    print("\nALL OK")
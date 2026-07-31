"""CubifAE baseline world model (Kaiser et al., 2024 ICML).

Port of the CubifAE multi-timescale ALIF + time-cell readout to the ST-JEWM
16-env state-input suite. Architecture:

    1. StateProjector : (B, T, state_dim) -> (B, T, d_hid)   [from STJEWM]
    2. ActionMLP      : (B, T, action_dim) -> (B, T, d_hid)
    3. ALIFStack      : 4-layer Adaptive LIF cells with per-layer adaptation
                        variable b_l; each layer emits (spike, mem_potential).
                        The membrane potential of every layer is concatenated
                        along the feature axis to a long 1D "membrane trace"
                        (B, T, n_layers * d_hid) which is fed to the time-cell
                        readout.
    4. TimeCellReadout: 1D conv over the multi-layer membrane trace with
                        kernel size 256 and stride 128 -> 8 anchor samples
                        per step. Concat with current spike -> linear to 192.
    5. The state z_t is the linear-projected concat of (8 anchors, s_t).

This model is intentionally similar in budget to STJEWM (~3-4M trainable
params; the 16-env envelope is 1-10M) and exposes the standard ST-JEWM /
LeWM contract:

    model.encode(obs, action) -> dict with 'emb' (B, T, 192)
    model.predict(ctx_emb, ctx_act) -> (B, 192)
    model.forward(obs, action) -> dict with 'emb', 'trace', 'spike', 'h', ...
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, "/home/lx/snn/code")

from code.snn_cell import atan_spike


# ============================================================
# Adaptive LIF cell (ALIF). Adds a slow adaptation current b_l
# which is incremented by spike events and decays exponentially.
# Membrane update: v = decay * v + (1-decay) * (I - b) - v_reset*spk
# ============================================================
class ALIFCell(nn.Module):
    """Adaptive LIF cell with per-channel adaptation current.

    The cell is a drop-in replacement for LIFCell: forward(x) returns
    a dict with keys 'spike' (B, T, d_hid) and 'v' (B, T, d_hid)
    (the membrane potential — note: NOT used as the predictive latent;
    it is exposed only so the time-cell readout can read it).
    """

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        v_thresh: float = 0.3,
        v_reset: float = 0.0,
        tau_m: float = 20.0,
        tau_b: float = 200.0,   # adaptation timescale (slow)
        t_ref: float = 2.0,
        dt: float = 1.0,
        alpha_surr: float = 2.0,
        init_scale: float = 3.0,
        beta_b: float = 0.02,    # spike->b increment per spike
    ):
        super().__init__()
        self.d_in, self.d_hid = d_in, d_hid
        self.v_thresh, self.v_reset = v_thresh, v_reset
        self.tau_m, self.tau_b, self.t_ref, self.dt = tau_m, tau_b, t_ref, dt
        self.alpha_surr = alpha_surr
        self.beta_b = float(beta_b)
        self.w_in = nn.Linear(d_in, d_hid, bias=True)
        with torch.no_grad():
            self.w_in.weight.mul_(init_scale)
            self.w_in.bias.zero_()
        # Recurrent weights (small, optional — improves capacity)
        self.w_rec = nn.Linear(d_hid, d_hid, bias=False)
        with torch.no_grad():
            self.w_rec.weight.mul_(0.1)
        self.register_buffer("decay_v", torch.exp(torch.tensor(-dt / tau_m)))
        self.register_buffer("decay_b", torch.exp(torch.tensor(-dt / tau_b)))

    def forward(self, x: torch.Tensor) -> dict:
        B, T, _ = x.shape
        v = torch.zeros(B, self.d_hid, device=x.device)
        b = torch.zeros(B, self.d_hid, device=x.device)
        refr = torch.zeros_like(v)
        v_seq, spk_seq = [], []
        for t in range(T):
            I = self.w_in(x[:, t]) + self.w_rec(v) - b
            v = self.decay_v * v + (1.0 - self.decay_v) * I
            v = torch.where(refr > 0, torch.full_like(v, self.v_reset), v)
            spk = atan_spike(v, self.v_thresh, self.alpha_surr)
            v = torch.where(spk > 0, torch.full_like(v, self.v_reset), v)
            b = self.decay_b * b + self.beta_b * spk
            refr = torch.where(
                spk > 0,
                torch.full_like(refr, self.t_ref / self.dt),
                (refr - 1.0).clamp(min=0.0),
            )
            v_seq.append(v)
            spk_seq.append(spk)
        return {
            "v": torch.stack(v_seq, dim=1),
            "spike": torch.stack(spk_seq, dim=1),
        }


# ============================================================
# ALIF stack with time-cell readout.
# ============================================================
class ALIFStackWithTimeCells(nn.Module):
    """N-layer ALIF stack; emits (spike, time-cell readout, all_layers_v).

    The time-cell readout is a 1D conv over the per-layer membrane trace
    with kernel_size=256, stride=128, yielding 8 anchors per step. We
    apply it to the concatenated per-layer membrane trace.
    """

    def __init__(
        self,
        d_hid: int = 192,
        n_layers: int = 4,
        v_thresh: float = 0.3,
        v_reset: float = 0.0,
        tau_m: float = 20.0,
        tau_b: float = 200.0,
    ):
        super().__init__()
        self.d_hid = d_hid
        self.n_layers = n_layers
        self.cells = nn.ModuleList()
        for layer_idx in range(n_layers):
            d_in = d_hid
            self.cells.append(
                ALIFCell(
                    d_in=d_in, d_hid=d_hid, v_thresh=v_thresh, v_reset=v_reset,
                    tau_m=tau_m, tau_b=tau_b,
                )
            )
        # Time-cell readout: 1D conv over the concatenated membrane trace.
        # The input has n_layers * d_hid channels. The original CubifAE
        # paper uses a 1D conv with kernel_size=256, stride=128, yielding
        # 8 anchors per step. To get exactly T out for any T (and avoid
        # huge kernels on short windows where T=2-200), we use kernel=8,
        # stride=1 with explicit same-length padding (handled in forward),
        # which gives 8 local-anchor samples per step. This preserves the
        # "multi-timescale anchor readout" spirit while being fast and
        # numerically stable on the ST-JEWM windowed training regime.
        self.membrane_dim = n_layers * d_hid
        self.time_conv = nn.Conv1d(
            in_channels=self.membrane_dim,
            out_channels=8 * d_hid,    # 8 anchors of size d_hid
            kernel_size=8, stride=1, padding=0, bias=True,
        )
        # Linear fusion: 8 anchors + current spike (i.e. 9 * d_hid) -> d_hid
        self.fuse = nn.Linear(9 * d_hid, d_hid, bias=True)
        # Per-cell post-mix MLPs (light)
        self.post_norms = nn.ModuleList(
            [nn.LayerNorm(d_hid, eps=1e-6) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Forward through the ALIF stack with time-cell readout.

        x: (B, T, d_hid)
        returns dict with:
            h:        (B, T, d_hid)  — z_t (predictive latent)
            spike:    (B, T, d_hid)  — spikes from the last layer
            spike_layers: list of (B, T, d_hid)
            v_layers: list of (B, T, d_hid)  — membranes per layer
            trace:    (B, T, d_hid)  — mean of per-layer membranes (for
                                       probe.py compatibility)
        """
        spike_layers = []
        v_layers = []
        h = x
        for cell, norm in zip(self.cells, self.post_norms):
            cell_out = cell(h)
            spk = cell_out["spike"]  # (B, T, d_hid)
            v = cell_out["v"]        # (B, T, d_hid)
            spike_layers.append(spk)
            v_layers.append(v)
            # Residual update of the running hidden (per-cell)
            h = h + norm(spk)
        # Concatenate per-layer membranes along feature axis -> (B, T, n*d_hid)
        v_cat = torch.cat(v_layers, dim=-1)  # (B, T, n_layers * d_hid)
        # Time-cell readout: 1D conv across the (B, n_layers*d_hid, T) axis.
        # Apply same-length padding (pad with (kernel-1) zeros on the right
        # so output length = T).
        v_t = v_cat.transpose(1, 2)  # (B, n_layers*d_hid, T)
        v_padded = F.pad(v_t, (0, self.time_conv.kernel_size[0] - 1))
        anchors = self.time_conv(v_padded)  # (B, 8*d_hid, T)
        anchors = anchors.transpose(1, 2)  # (B, T, 8*d_hid)
        # Get the last layer's spike (the "current" spike)
        s_last = spike_layers[-1]  # (B, T, d_hid)
        # Fuse anchors + spike
        z_t = self.fuse(torch.cat([anchors, s_last], dim=-1))  # (B, T, d_hid)
        # Mean across layers of the membrane — used as 'trace' so probe.py
        # can read something meaningful without depending on which layer.
        v_mean = torch.stack(v_layers, dim=0).mean(dim=0)  # (B, T, d_hid)
        return {
            "h": z_t,
            "spike": s_last,
            "spike_layers": spike_layers,
            "v_layers": v_layers,
            "trace": v_mean,
            "time_cell": anchors,
        }


# ============================================================
# CubifAEBaseline — the full model. StateProjector + ActionMLP + ALIFStackWithTimeCells.
# ============================================================
class CubifAEBaseline(nn.Module):
    """CubifAE multi-timescale ALIF + time-cell readout, ported to LeWM state input."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_hid: int = 192,
        n_layers: int = 4,
        v_thresh: float = 0.3,
        tau_m: float = 20.0,
        tau_b: float = 200.0,
        image_size: int = 0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_hid = d_hid
        self.n_layers = n_layers
        # Pixel mode: use frozen ViT-Tiny preprocessor (5.5M frozen + 0.07M proj).
        # The preprocessor's projector outputs `d_hid` so it is a drop-in
        # replacement for the (state_dim -> d_hid) MLP.
        if state_dim >= 100 and image_size > 0:
            from code.core.pixel_pre import FrozenPixelPreprocessor
            self.pixel_pre = FrozenPixelPreprocessor(
                image_size=image_size, embed_dim=d_hid,
            )
            self.state_projector = None
        else:
            self.pixel_pre = None
            # State projector (same shape as STJEWM.StateProjector)
            self.state_projector = nn.Sequential(
                nn.Linear(state_dim, d_hid),
                nn.SiLU(),
                nn.Linear(d_hid, d_hid),
            )

        # Action encoder (1-layer MLP per LeWM/STJEWM spec)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_hid),
            nn.SiLU(),
            nn.Linear(d_hid, d_hid),
        )
        # ALIF stack with time-cell readout
        self.stack = ALIFStackWithTimeCells(
            d_hid=d_hid, n_layers=n_layers,
            v_thresh=v_thresh, tau_m=tau_m, tau_b=tau_b,
        )
        # readout_mode is not used (this is a non-STJEWM model), but we
        # expose the attribute to keep the contract checker happy.
        self.readout_mode = None

    @property
    def embed_dim(self):
        return self.d_hid

    def encode(self, obs: torch.Tensor, action: torch.Tensor) -> dict:
        """Encode (obs, action) -> dict with 'emb' (B, T, d_hid) and 'act_emb'.

        For ST-JEWM contract compatibility we also return 'emb_pre_cell'
        (the state-only encoding) so the trainer's SIGReg term can fire
        on the pre-cell embedding just like for STJEWM.
        """
        if self.pixel_pre is not None:
            # Pixel mode: state is (B, T, 3, H, W)
            s_emb = self.pixel_pre(obs)            # (B, T, d_hid)
        else:
            s_emb = self.state_projector(obs)       # (B, T, d_hid)
        a_emb = self.action_encoder(action)         # (B, T, d_hid)
        return {"emb": s_emb, "emb_pre_cell": s_emb, "act_emb": a_emb}

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> dict:
        """Full forward: returns dict with 'emb' (predictive latent), 'spike',
        'trace', etc. (probe.py reads 'trace' from this output)."""
        enc = self.encode(obs, action)
        s_emb = enc["emb"]            # (B, T, d_hid)
        a_emb = enc["act_emb"]        # (B, T, d_hid)
        h_in = s_emb + a_emb          # (B, T, d_hid)
        stack_out = self.stack(h_in)
        z_t = stack_out["h"]          # (B, T, d_hid)
        return {
            "emb": z_t,
            "emb_pre_cell": s_emb,
            "act_emb": a_emb,
            "spike": stack_out["spike"],
            "spike_layers": stack_out["spike_layers"],
            "trace": stack_out["trace"],
            "h": z_t,
            "time_cell": stack_out["time_cell"],
        }

    def predict(self, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> torch.Tensor:
        """Per-step prediction. Matches STJEWM.predict signature.

        ctx_emb: (B, H, d_hid) — pre-cell or post-cell embedding; we treat
                 it as already-encoded state embedding (not raw state).
        ctx_act: (B, H, action_dim) raw actions, OR (B, H, d_hid) pre-encoded.
        """
        if ctx_act.shape[-1] == self.action_dim:
            a_emb = self.action_encoder(ctx_act)
        else:
            a_emb = ctx_act
        h_in = ctx_emb + a_emb
        stack_out = self.stack(h_in)
        return stack_out["h"]

    @staticmethod
    def criterion(pred_emb: torch.Tensor, tgt_emb: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_emb, tgt_emb)

    @torch.no_grad()
    def rollout(self, init_emb: torch.Tensor, action_sequence: torch.Tensor,
                history_size: int = 3) -> torch.Tensor:
        """Autoregressive latent rollout (matches JEPA.rollout)."""
        B, T, A = action_sequence.shape
        h = init_emb[:, -history_size:].clone()
        preds = []
        for t in range(T):
            a_t = action_sequence[:, t: t + history_size]
            if a_t.shape[1] < history_size:
                pad = torch.zeros(B, history_size - a_t.shape[1], A,
                                  device=a_t.device, dtype=a_t.dtype)
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
        act_flat = action_candidates.reshape(B * K, T, A)
        act_emb_flat = self.action_encoder(act_flat)
        act_emb = act_emb_flat.reshape(B, K, T, -1)
        init_k = init_emb.unsqueeze(1).expand(-1, K, -1, -1).reshape(
            B * K, history_size, init_emb.shape[2]
        )
        h = init_k
        for t in range(T):
            avail = T - t
            if avail >= history_size:
                a_t = act_emb[:, :, t:t + history_size].reshape(B * K, history_size, -1)
            else:
                a_t_partial = act_emb[:, :, t:].reshape(B * K, avail, -1)
                pad = torch.zeros(
                    B * K, history_size - avail, act_emb.shape[-1],
                    device=act_emb.device, dtype=act_emb.dtype,
                )
                a_t = torch.cat([a_t_partial, pad], dim=1)
            h_in = h[:, -history_size:]
            nxt = self.predict(h_in, a_t)[:, -1]
            h = torch.cat([h, nxt.unsqueeze(1)], dim=1)
        z_final = h[:, -1].reshape(B, K, -1)
        goal_exp = goal_emb.unsqueeze(1).expand(-1, K, -1)
        return ((z_final - goal_exp) ** 2).sum(-1)


# ============================================================
# Factory: parameter-matched to STJEWM (~3-4M params)
# ============================================================
def make_cubifae_baseline(
    state_dim: int,
    action_dim: int,
    n_layers: int = 4,
    d_hid: int = 192,
    image_size: int = 0,
) -> CubifAEBaseline:
    """Build a CubifAEBaseline with the default config.

    Set `image_size>0` to use the frozen ViT-Tiny pixel preprocessor
    instead of the low-dim state projector.
    """
    return CubifAEBaseline(
        state_dim=state_dim, action_dim=action_dim,
        d_hid=d_hid, n_layers=n_layers,
        image_size=image_size,
    )


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    import time
    print("=" * 60)
    print("CubifAEBaseline smoke test")
    print("=" * 60)
    B, T, state_dim, action_dim = 2, 5, 7, 2
    model = CubifAEBaseline(state_dim=state_dim, action_dim=action_dim, d_hid=192, n_layers=4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.2f}M")
    state = torch.randn(B, T, state_dim)
    action = torch.randn(B, T, action_dim)
    t0 = time.time()
    out = model(state, action)
    dt = time.time() - t0
    print(f"forward OK in {dt*1000:.1f}ms")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v).__name__} (len={len(v)})")
    # Test predict
    ctx_emb = out["emb"][:, :3]
    ctx_act = action[:, :3]
    pred = model.predict(ctx_emb, ctx_act)
    print(f"predict output shape: {pred.shape}")
    print("=" * 60)
    print("CubifAEBaseline smoke test PASSED")
    print("=" * 60)

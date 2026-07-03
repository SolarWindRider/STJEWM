"""SpikeDreamer (Hong et al., 2024 AAAI) baseline world model.

Hybrid: 2-layer LIF encoder + Transformer world predictor.
Architecture (per §3.2 of docs/SNN_WORLD_MODEL_SURVEY.md):

  1. State encoder: state_dim -> d_snn=128
  2. 2-layer LIF encoder (atan surrogate, beta=0.9) -> (B,T,d_snn) binary spikes
  3. Linear spike -> (B,T,d_tx=192)  [s_proj]
  4. Action encoder: 1-MLP -> a_emb (B,T,192)
  5. Transformer predictor: 4-layer pre-norm causal Transformer with
     AdaLN-zero conditioning on a_emb (matches LeWM)
  6. z_t = fuser([s_proj, h_tx]) where fuser is a 1-layer MLP -> (B,T,192)

Contract (matches STJEWM / GRU / MLP baselines):

  model.encode(obs, action) -> dict with 'emb' (B,T,192), 'emb_pre_cell' (B,T,192)
  model.predict(ctx_emb, ctx_act) -> (B,192)
  model.forward(obs, action) returns the same dict that probe.py expects
      (with 'emb', 'emb_pre_cell', 'act_emb', 'spike', 'trace' (= s_proj),
       'h' (= h_tx), 'spike_layers')
"""
from __future__ import annotations
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/lx/snn/code")
from snn_cell import LIFCell, atan_spike, _build_trace  # noqa: E402


class _CausalMask(nn.Module):
    """Cache-friendly causal attention mask. Returns (T,T) bool mask with
    True for positions that may attend (j <= i)."""

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.ones(T, T, device=device, dtype=torch.bool).tril(0)


class StateProjector(nn.Module):
    """state_dim -> d_snn (LIF encoder input dim)."""

    def __init__(self, state_dim: int, d_snn: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim, d_snn),
            nn.LayerNorm(d_snn),
        )

    def forward(self, x):
        return self.proj(x)


class ActionMLP(nn.Module):
    """action_dim -> d_tx (action embedding)."""

    def __init__(self, action_dim: int, d_tx: int):
        super().__init__()
        self.proj = nn.Linear(action_dim, d_tx)

    def forward(self, a):
        return self.proj(a)


class _LIFStack2(nn.Module):
    """Two stacked LIF cells. Returns dict with 'spike' (B,T,d_snn), 'v' (B,T,d_snn),
    'spike_layers' (list of per-layer spikes for probing)."""

    def __init__(self, d_in: int, d_hid: int, v_thresh: float = 0.3,
                 beta: float = 0.9, alpha_surr: float = 2.0):
        super().__init__()
        self.cell1 = LIFCell(d_in, d_hid, v_thresh=v_thresh,
                             trace_beta=beta, alpha_surr=alpha_surr)
        self.cell2 = LIFCell(d_hid, d_hid, v_thresh=v_thresh,
                             trace_beta=beta, alpha_surr=alpha_surr)

    def forward(self, x):
        out1 = self.cell1(x)
        out2 = self.cell2(out1["spike"])
        return {
            "spike": out2["spike"],         # (B,T,d_snn) — the encoder's spike output
            "v": out2["v"],                 # (B,T,d_snn) — last membrane (probe convenience)
            "spike_layers": [out1["spike"], out2["spike"]],
            "trace": _build_trace(
                list(out2["spike"].unbind(dim=1)),
                beta=self.cell2.trace_beta,
            ),
        }


class _AdaLNZeroBlock(nn.Module):
    """Pre-norm Transformer block with AdaLN-zero conditioning (LeWM-style).

    Uses a learned causal mask via batch_first MultiheadAttention with attn_mask.
    AdaLN: scale/shift/gate for both attention and MLP; zero-init output.
    """

    def __init__(self, d: int, num_heads: int, mlp_ratio: float = 4.0,
                 max_T: int = 256, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, num_heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d * mlp_ratio), d),
        )
        # AdaLN modulation: produce (shift_msa, scale_msa, gate_msa,
        #                          shift_mlp, scale_mlp, gate_mlp)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 6 * d, bias=True),
        )
        # AdaLN-zero: zero-init the modulation output so the block is identity at init.
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        # Register a causal mask buffer up to a max length.
        mask = torch.ones(max_T, max_T, dtype=torch.bool).tril(0)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN(cond).chunk(6, dim=-1)
        )
        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_mask = self.causal_mask[:T, :T]
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + gate_msa * attn_out
        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(h)
        return x


class SpikeDreamerBaseline(nn.Module):
    """SpikeDreamer: 2-layer LIF encoder + Transformer world predictor.

    Returns emb of shape (B,T,192) where d_tx = embed_dim = 192 (matches STJEWM).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_snn: int = 128,
        d_tx: int = 192,
        num_layers: int = 4,
        num_heads: int = 8,
        beta: float = 0.9,
        v_thresh: float = 0.3,
        alpha_surr: float = 2.0,
        max_T: int = 256,
        history_size: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_snn = d_snn
        self.d_tx = d_tx
        self.embed_dim = d_tx
        self.d_hid = d_tx
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.history_size = history_size
        # required by closed_loop / encode.py compatibility checks
        self.readout_mode = None

        # Encoders
        self.state_proj = StateProjector(state_dim, d_snn)
        self.lif_stack = _LIFStack2(
            d_in=d_snn, d_hid=d_snn,
            v_thresh=v_thresh, beta=beta, alpha_surr=alpha_surr,
        )
        self.spike_proj = nn.Linear(d_snn, d_tx)
        self.action_encoder = ActionMLP(action_dim, d_tx)

        # Positional embedding for the Transformer
        self.pos_embed = nn.Parameter(torch.zeros(1, max_T, d_tx))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            _AdaLNZeroBlock(d_tx, num_heads, max_T=max_T)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_tx)

        # Fuser: MLP on [s_proj, h_tx] -> (B,T,192)
        self.fuser = nn.Sequential(
            nn.Linear(2 * d_tx, d_tx),
            nn.GELU(),
            nn.Linear(d_tx, d_tx),
        )

    # ---------------- core ----------------
    def _encode_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run SNN encoder. Returns (s_proj, h_tx_or_None, spike, lif_out).

        h_tx is None here; computed after the Transformer pass.
        """
        x = self.state_proj(obs)  # (B,T,d_snn)
        lif_out = self.lif_stack(x)
        spike = lif_out["spike"]  # (B,T,d_snn)
        s_proj = self.spike_proj(spike)  # (B,T,d_tx)
        return s_proj, spike, lif_out

    # ---------------- API ----------------
    def encode(self, obs: torch.Tensor, action: torch.Tensor) -> dict:
        return self.forward(obs, action)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> dict:
        """Encode (obs, action) over a window.

        obs: (B,T,state_dim), action: (B,T,action_dim)
        Returns dict with:
            emb           (B,T,d_tx)            — fused predictor output (planner reads this)
            emb_pre_cell  (B,T,d_tx)            — s_proj (pre-Transformer); for SIGReg loss
            act_emb       (B,T,d_tx)
            spike         (B,T,d_snn)
            trace         (B,T,d_snn)
            spike_layers  list of (B,T,d_snn)
            h             (B,T,d_tx)            — final Transformer hidden (h_tx)
        """
        B, T, _ = obs.shape
        s_proj, spike, lif_out = self._encode_obs(obs)  # s_proj: (B,T,d_tx)
        act_emb = self.action_encoder(action)  # (B,T,d_tx)

        # Transformer input: spike proj + action emb + positional embed
        x = s_proj + act_emb + self.pos_embed[:, :T]
        # AdaLN conditioning: condition on the action embedding (per spec)
        cond = act_emb + self.pos_embed[:, :T]
        for block in self.blocks:
            x = block(x, cond)
        h_tx = self.norm_out(x)  # (B,T,d_tx)

        # Fuse: concat(s_proj, h_tx) -> MLP -> (B,T,d_tx)
        z = self.fuser(torch.cat([s_proj, h_tx], dim=-1))

        return {
            "emb": z,
            "emb_pre_cell": s_proj,           # pre-Transformer activation for SIGReg
            "act_emb": act_emb,
            "spike": spike,
            "trace": lif_out["trace"],        # (B,T,d_snn) — surrogate 'trace' for probe.py
            "spike_layers": lif_out["spike_layers"],
            "h": h_tx,
        }

    def predict(self, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> torch.Tensor:
        """Per-step prediction over a window. ctx_emb: (B,H,d_tx), ctx_act: (B,H,action_dim).

        SpikeDreamer's predict re-runs the SNN encoder-free path:
        since ctx_emb already is the fused latent (from forward()), we run
        a lightweight version that re-applies only the Transformer + fuser
        with the spike-projection term estimated as a function of the ctx.

        However the canonical contract used by the trainer (see
        code/train/train.py) is to call predict(ctx_emb=H steps of `emb`,
        ctx_act=H steps of action) and compare against the next H steps of
        `emb`. To stay numerically consistent with forward() we run a
        mini forward pass on ctx_obs reconstructed from ctx_emb is not
        possible — instead we re-use ctx_emb directly:

            h_tx_pred = Transformer_decoder(ctx_emb)        (no AdaLN-zero init available)
            z_pred    = fuser([ctx_emb, h_tx_pred])

        This is a standard "latent-space predictor" pattern: the trainer
        passes the fused latents as ctx, and we decode one step ahead.
        """
        # ctx_emb: (B,H,d_tx); we treat it as the spiking projection's
        # "rate-coded surrogate" so we can drive the Transformer.
        # ctx_act may be either (B,H,action_dim) or (B,H,d_tx) when the
        # caller passes pre-encoded actions.
        if ctx_act.shape[-1] == self.action_dim:
            act_emb = self.action_encoder(ctx_act)
        else:
            act_emb = ctx_act
        B, H, _ = ctx_emb.shape
        x = ctx_emb + act_emb + self.pos_embed[:, :H]
        cond = act_emb + self.pos_embed[:, :H]
        for block in self.blocks:
            x = block(x, cond)
        h_tx = self.norm_out(x)
        z = self.fuser(torch.cat([ctx_emb, h_tx], dim=-1))
        return z

    @staticmethod
    def criterion(pred_emb: torch.Tensor, tgt_emb: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_emb, tgt_emb)

    # The methods below match STJEWM / GRU / MLP baselines — needed by
    # code/eval/closed_loop.py which calls rollout() / get_cost().
    @torch.no_grad()
    def rollout(self, init_emb: torch.Tensor, action_sequence: torch.Tensor,
                history_size: int = 3) -> torch.Tensor:
        """Autoregressive latent rollout (matches JEPA / STJEWM.rollout).

        Matches the STJEWM pattern: predict() takes a window of length
        history_size and returns a window of the same length; we keep `h`
        fixed at history_size and slide a new prediction in.
        """
        B, T, A = action_sequence.shape
        H = history_size
        h = init_emb[:, -H:].clone()
        preds = []
        for t in range(T):
            a_t = action_sequence[:, t: t + H]
            if a_t.shape[1] < H:
                pad = torch.zeros(B, H - a_t.shape[1], A,
                                  device=a_t.device, dtype=a_t.dtype)
                a_t = torch.cat([a_t, pad], dim=1)
            nxt = self.predict(h, a_t)[:, -1]
            preds.append(nxt)
            # Slide: drop oldest, append newest (keep window of size H)
            h = torch.cat([h[:, 1:], nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def cost(self, pred_emb: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
        return ((pred_emb - goal_emb) ** 2).flatten(1).sum(-1)

    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Cost for CEM action candidates (matches STJEWM.get_cost / GRUBaseline.get_cost)."""
        goal_emb = info_dict["goal_emb"]      # (B, D)
        init_emb = info_dict["init_emb"]      # (B, H, D)
        B, K, T, A = action_candidates.shape
        H = init_emb.shape[1]
        # Embed actions: (B,K,T,A) -> (B,K,T,D)
        act_flat = action_candidates.reshape(B * K, T, A)
        act_emb_flat = self.action_encoder(act_flat)
        act_emb = act_emb_flat.reshape(B, K, T, -1)
        # Replicate init across K
        init_k = init_emb.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, H, init_emb.shape[-1])
        h = init_k
        for t in range(T):
            avail = T - t
            if avail >= H:
                a_t = act_emb[:, :, t:t + H].reshape(B * K, H, -1)
            else:
                a_t_partial = act_emb[:, :, t:].reshape(B * K, avail, -1)
                pad = torch.zeros(B * K, H - avail, act_emb.shape[-1],
                                  device=action_candidates.device,
                                  dtype=action_candidates.dtype)
                a_t = torch.cat([a_t_partial, pad], dim=1)
            h_in = h[:, -H:]
            nxt = self.predict(h_in, a_t)[:, -1]
            h = torch.cat([h, nxt.unsqueeze(1)], dim=1)
        z_final = h[:, -1].reshape(B, K, -1)
        goal_exp = goal_emb.unsqueeze(1).expand(-1, K, -1)
        return ((z_final - goal_exp) ** 2).sum(-1)


def make_spikedreamer(state_dim: int, action_dim: int, **kwargs) -> SpikeDreamerBaseline:
    """Factory: parameter-matched SpikeDreamer baseline."""
    defaults = dict(
        d_snn=128, d_tx=192, num_layers=4, num_heads=8,
        beta=0.9, v_thresh=0.3, alpha_surr=2.0,
    )
    defaults.update(kwargs)
    return SpikeDreamerBaseline(
        state_dim=state_dim, action_dim=action_dim, **defaults,
    )


if __name__ == "__main__":
    import time
    B, T = 2, 8
    state_dim, action_dim = 9, 6
    model = make_spikedreamer(state_dim, action_dim)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SpikeDreamer baseline: {n/1e6:.2f}M trainable params")

    state = torch.randn(B, T, state_dim)
    action = torch.randn(B, T, action_dim)
    out = model(state, action)
    print("emb:", out["emb"].shape)
    print("emb_pre_cell:", out["emb_pre_cell"].shape)
    print("act_emb:", out["act_emb"].shape)
    print("spike:", out["spike"].shape)
    print("trace:", out["trace"].shape)
    print("h:", out["h"].shape)
    print("spike_layers:", [s.shape for s in out["spike_layers"]])

    # Test predict
    ctx_emb = out["emb"][:, :3]
    ctx_act = action[:, :3]
    pred = model.predict(ctx_emb, ctx_act)
    print("predict:", pred.shape)

    # Test criterion + rollout + get_cost
    tgt = torch.randn(B, 3, 192)
    print("criterion:", model.criterion(pred, tgt).item())
    rollout = model.rollout(out["emb"][:, :3], action, history_size=3)
    print("rollout:", rollout.shape)
    info = {"goal_emb": torch.randn(B, 192),
            "init_emb": out["emb"][:, :3]}
    K = 5
    a_cands = torch.randn(B, K, 6, action_dim)
    c = model.get_cost(info, a_cands)
    print("get_cost:", c.shape)
    print("SpikeDreamer smoke test OK")
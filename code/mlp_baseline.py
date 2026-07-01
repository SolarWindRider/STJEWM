"""MLP baseline world model — no historical state, just per-step FFN.

This is the 'no-memory' floor: a per-step FFN mapping (obs, action) -> latent.
There is no recurrence, no trace, no cross-time aggregation. STJEWM-trace should
win if memory matters, and rate-only should win over this if any temporal
aggregation at all helps.

API (matches code/gru_baseline.py and code/stjewm.py):
    model.encode(obs, action) -> dict with 'emb' (B, T, D) and 'emb_pre_cell' (B, T, D)
    model.predict(ctx_emb, ctx_act) -> (B, H, D) next-latent per window position
    model.cost(pred_emb, goal_emb) -> (B,) cost
    model.get_cost(info_dict, action_candidates) -> (B, K) cost for CEM
    model.criterion(pred, tgt) -> scalar
    model.rollout(init_emb, action_sequence, history_size) -> (B, T, D)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """Per-step FFN world model. No recurrence. (obs, action) -> latent."""

    def __init__(self, state_dim, action_dim, hidden_dim=576, num_layers=4,
                 emb_dim=192, history_size=3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.history_size = history_size

        # State projector: raw state -> emb_dim token. Used by forward() to
        # produce a (B, T, emb_dim) token stream that the FFN consumes.
        # Also produces the 'emb_pre_cell' tensor that train.py feeds to SIGReg.
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

        # Per-step FFN. Input is a (B, T, emb_dim + action_dim) token stream;
        # the FFN is a stack of Linear+SiLU that maps each position
        # independently to a (B, T, emb_dim) next-latent. The lack of any
        # cross-time mixing is the whole point of this baseline.
        in_dim = emb_dim + action_dim
        layers = []
        d = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim
        layers.append(nn.Linear(hidden_dim, emb_dim))
        self.net = nn.Sequential(*layers)

    @property
    def embed_dim(self):
        return self.emb_dim

    @property
    def d_hid(self):
        return self.emb_dim

    @property
    def readout_mode(self):
        # Baseline has no ReadoutMode contract; closed_loop.py tolerates missing attr.
        return None

    # ---------- core per-step FFN (the no-memory "world model") ----------
    def _ffn(self, token, action):
        """Apply the per-step FFN. Input: (..., emb_dim + action_dim). Output: (..., emb_dim)."""
        if action.shape[-1] != self.action_dim:
            # Action may already be embedded; replace with zeros in the expected dim.
            action = torch.zeros(*token.shape[:-1], self.action_dim,
                                 device=token.device, dtype=token.dtype)
        x = torch.cat([token, action], dim=-1)
        return self.net(x)

    # ---------- API ----------
    def encode(self, state, action):
        return self.forward(state, action)

    def forward(self, state, action):
        """Encode (state, action) over a window.

        state:  (B, T, state_dim)
        action: (B, T, action_dim)
        Returns: dict with 'emb' (B, T, emb_dim) and 'emb_pre_cell' (B, T, emb_dim).
        """
        # 'emb_pre_cell' is the state-only projection (used by SIGReg in train.py).
        emb_pre_cell = self.state_proj(state)
        # The per-step FFN consumes the state token and the action.
        emb = self._ffn(emb_pre_cell, action)
        return {"emb": emb, "emb_pre_cell": emb_pre_cell}

    def predict(self, ctx_emb, ctx_act):
        """Per-step prediction over a window.

        ctx_emb: (B, H, emb_dim) — previous latents (already in emb_dim space).
        ctx_act: (B, H, action_dim) — actions at each window position.
        Returns: (B, H, emb_dim) — next-latent prediction at each position.

        For the no-memory MLP, each prediction depends only on
        (ctx_emb[t], ctx_act[t]) — there is no aggregation across time.
        This is exactly the 'no memory' floor: cross-time context is fed in
        but the model is structurally forbidden from using it.
        """
        return self._ffn(ctx_emb, ctx_act)

    @staticmethod
    def criterion(pred_emb, tgt_emb):
        return F.mse_loss(pred_emb, tgt_emb)

    @torch.no_grad()
    def rollout(self, init_emb, action_sequence, history_size=3):
        """Autoregressive latent rollout (matches STJEWM.rollout).

        init_emb: (B, H, emb_dim) — initial history of latents.
        action_sequence: (B, T, action_dim) — action plan.
        Returns: (B, T, emb_dim) — predicted latents.
        """
        B, T, A = action_sequence.shape
        h = init_emb[:, -history_size:].clone()
        preds = []
        for t in range(T):
            a_t = action_sequence[:, t:t + history_size]
            if a_t.shape[1] < history_size:
                pad = torch.zeros(B, history_size - a_t.shape[1], A,
                                  device=a_t.device, dtype=a_t.dtype)
                a_t = torch.cat([a_t, pad], dim=1)
            nxt = self.predict(h[:, -history_size:], a_t)[:, -1]
            preds.append(nxt)
            h = torch.cat([h, nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def cost(self, pred_emb, goal_emb):
        return ((pred_emb - goal_emb) ** 2).flatten(1).sum(-1)

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        """Cost for action candidates (matches STJEWM.get_cost / GRUBaseline.get_cost).

        info_dict: {'goal_emb': (B, D), 'init_emb': (B, H, D)}
        action_candidates: (B, K, T, A)
        Returns: (B, K) cost per candidate sequence.
        """
        goal_emb = info_dict["goal_emb"]
        init_emb = info_dict["init_emb"]
        B, K, T, A = action_candidates.shape
        hist = init_emb.shape[1]
        init_k = init_emb.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, hist, -1)
        h = init_k
        for t in range(T):
            avail = T - t
            if avail >= hist:
                a_t = action_candidates[:, :, t:t + hist].reshape(B * K, hist, A)
            else:
                a_part = action_candidates[:, :, t:].reshape(B * K, avail, A)
                pad = torch.zeros(B * K, hist - avail, A,
                                  device=action_candidates.device,
                                  dtype=action_candidates.dtype)
                a_t = torch.cat([a_part, pad], dim=1)
            nxt = self.predict(h[:, -hist:], a_t)[:, -1]
            h = torch.cat([h, nxt.unsqueeze(1)], dim=1)
        z_final = h[:, -1].reshape(B, K, -1)
        return ((z_final - goal_emb.unsqueeze(1).expand(-1, K, -1)) ** 2).sum(-1)


def make_mlp_baseline(state_dim, action_dim, hidden_dim=576, num_layers=4,
                      emb_dim=192):
    """Factory: parameter-matched MLP baseline."""
    return MLPBaseline(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=hidden_dim, num_layers=num_layers,
        emb_dim=emb_dim,
    )


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("MLPBaseline smoke test")
    print("=" * 60)

    # Default config: small state / action dims
    B, T = 2, 5
    state_dim, action_dim = 9, 1
    model = make_mlp_baseline(state_dim=state_dim, action_dim=action_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"MLP baseline params: trainable={n_params:,}  total={n_total:,}")

    s = torch.randn(B, T, state_dim)
    a = torch.randn(B, T, action_dim)
    out = model(s, a)
    assert out["emb"].shape == (B, T, 192), out["emb"].shape
    assert out["emb_pre_cell"].shape == (B, T, 192), out["emb_pre_cell"].shape
    print(f"emb shape:          {tuple(out['emb'].shape)}")
    print(f"emb_pre_cell shape: {tuple(out['emb_pre_cell'].shape)}")

    # Predict
    H = 3
    ctx_emb = torch.randn(B, H, 192)
    ctx_act = torch.randn(B, H, action_dim)
    p = model.predict(ctx_emb, ctx_act)
    assert p.shape == (B, H, 192), p.shape
    print(f"predict shape:      {tuple(p.shape)}")

    # Rollout
    init_emb = torch.randn(B, H, 192)
    actions = torch.randn(B, 10, action_dim)
    r = model.rollout(init_emb, actions, history_size=H)
    assert r.shape == (B, 10, 192), r.shape
    print(f"rollout shape:      {tuple(r.shape)}")

    # get_cost
    K = 4
    info = {
        "goal_emb": torch.randn(B, 192),
        "init_emb": init_emb,
    }
    cands = torch.randn(B, K, 10, action_dim)
    cost = model.get_cost(info, cands)
    assert cost.shape == (B, K), cost.shape
    print(f"get_cost shape:     {tuple(cost.shape)}")

    # Larger state/action dims (pusht-like)
    print("-" * 60)
    print("Larger config (state_dim=20, action_dim=2)")
    model_big = make_mlp_baseline(state_dim=20, action_dim=2, hidden_dim=576,
                                  num_layers=4, emb_dim=192)
    n_params_big = sum(p.numel() for p in model_big.parameters() if p.requires_grad)
    print(f"MLP baseline params (big): trainable={n_params_big:,}")

    # Timing
    t0 = time.time()
    for _ in range(20):
        _ = model(s, a)
    dt = time.time() - t0
    print(f"forward x20: {dt:.3f}s ({dt/20*1000:.1f}ms/iter)")

    print("MLP smoke test OK")

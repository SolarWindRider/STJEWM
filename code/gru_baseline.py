"""GRU baseline world model — parameter-matched to STJEWM (~5M)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUBaseline(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=576, num_layers=3,
                 history_size=3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.history_size = history_size

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True)
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def embed_dim(self): return self.hidden_dim
    @property
    def d_hid(self): return self.hidden_dim
    @property
    def readout_mode(self): return None

    def encode(self, state, action):
        return self.forward(state, action)

    def forward(self, state, action):
        B, T, _ = state.shape
        s_emb = self.state_proj(state)
        a_emb = self.action_proj(action)
        x = s_emb + a_emb
        h, _ = self.gru(x)
        h = self.norm_out(h)
        return {"emb": self.proj_out(h), "emb_pre_cell": s_emb}

    def predict(self, ctx_emb, ctx_act):
        if ctx_act.shape[-1] == self.action_dim:
            a_emb = self.action_proj(ctx_act)
        else:
            a_emb = ctx_act
        x = ctx_emb + a_emb
        h, _ = self.gru(x)
        h = self.norm_out(h)
        return self.proj_out(h)

    @staticmethod
    def criterion(pred_emb, tgt_emb):
        return F.mse_loss(pred_emb, tgt_emb)

    @torch.no_grad()
    def rollout(self, init_emb, action_sequence, history_size=3):
        B, T, A = action_sequence.shape
        h = init_emb[:, -history_size:].clone()
        preds = []
        for t in range(T):
            a_t = action_sequence[:, t:t + history_size]
            if a_t.shape[1] < history_size:
                pad = torch.zeros(B, history_size - a_t.shape[1], A,
                                  device=a_t.device, dtype=a_t.dtype)
                a_t = torch.cat([a_t, pad], dim=1)
            nxt = self.predict(h, a_t)[:, -1]
            preds.append(nxt); h = torch.cat([h, nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def cost(self, pred_emb, goal_emb):
        return ((pred_emb - goal_emb) ** 2).flatten(1).sum(-1)

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        goal_emb = info_dict["goal_emb"]
        init_emb = info_dict["init_emb"]
        B, K, T, A = action_candidates.shape
        hist = init_emb.shape[1]
        act_flat = action_candidates.reshape(B * K, T, A)
        act_emb = self.action_proj(act_flat).reshape(B, K, T, -1)
        init_k = init_emb.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, hist, -1)
        h = init_k
        for t in range(T):
            avail = T - t
            if avail >= hist:
                a_t = act_emb[:, :, t:t + hist].reshape(B * K, hist, -1)
            else:
                a_part = act_emb[:, :, t:].reshape(B * K, avail, -1)
                pad = torch.zeros(B * K, hist - avail, act_emb.shape[-1],
                                  device=act_emb.device, dtype=act_emb.dtype)
                a_t = torch.cat([a_part, pad], dim=1)
            nxt = self.predict(h[:, -hist:], a_t)[:, -1]
            h = torch.cat([h, nxt.unsqueeze(1)], dim=1)
        z_final = h[:, -1].reshape(B, K, -1)
        return ((z_final - goal_emb.unsqueeze(1).expand(-1, K, -1)) ** 2).sum(-1)


if __name__ == "__main__":
    B, T = 2, 5
    model = GRUBaseline(state_dim=9, action_dim=1)
    n = sum(p.numel() for p in model.parameters())
    print(f"GRU baseline: {n/1e6:.2f}M params (target ~5M)")
    s, a = torch.randn(B, T, 9), torch.randn(B, T, 1)
    out = model(s, a)
    assert out['emb'].shape == (B, T, 576), out['emb'].shape
    p = model.predict(torch.randn(B, 3, 576), torch.randn(B, 3, 1))
    assert p.shape == (B, 3, 576)
    print("GRU smoke test OK")

"""Single canonical CEM (Cross-Entropy Method) planner.

Strictly follows LeWM App. B + App. F.1:
  - 300 samples, 30 elites, 10-30 iterations, sigma_init=1.0
  - receding-horizon execution

Bug fixes vs. the 9 originals:
  - goal encoding is always the caller's responsibility (no more "init state as
    goal" bug from stage35)
  - history_size is explicit (no more inheriting from init_emb.shape[1] which
    was a fragile implicit assumption)
  - sigma is `std` not `var` (the originals conflated these in different ways,
    producing non-comparable results)
"""
from __future__ import annotations

import torch


class CEM:
    """Cross-Entropy Method planner for any model with `predict(ctx_emb, ctx_act)`.

    The model must expose:
        model.predict(ctx_emb: (B, H, D), ctx_act: (B, H, A)) -> (B, D) next latent

    This is satisfied by:
        - STJEWM (code/stjewm.py)
        - LeWMTransformerBaseline (code/lewm_transformer_baseline.py)
    """

    def __init__(
        self,
        model,
        action_dim: int,
        horizon: int = 5,
        n_samples: int = 300,
        n_elites: int = 30,
        n_iters: int = 10,
        history_size: int = 3,
        sigma_init: float = 1.0,
        device: str | torch.device = "cuda",
    ):
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.history_size = history_size
        self.sigma_init = sigma_init
        self.device = device

    @torch.no_grad()
    def _rollout_cost(self, z_init: torch.Tensor, z_goal: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll out (N, H, A) actions through model.predict and compute cost.

        Args:
            z_init: (D,) initial latent (single episode)
            z_goal: (D,) goal latent (single episode)
            actions: (N, H, A) candidate action sequences

        Returns:
            (N,) cost for each candidate.
        """
        N, H, A = actions.shape
        # Expand z_init to (N, history_size, D) for batched rollout
        h = z_init.unsqueeze(0).expand(N, -1).unsqueeze(1).expand(N, self.history_size, -1).contiguous()
        for t in range(H):
            avail = H - t
            if avail >= self.history_size:
                a_window = actions[:, t:t + self.history_size]
            else:
                a_partial = actions[:, t:]
                pad = torch.zeros(N, self.history_size - avail, A, device=actions.device, dtype=actions.dtype)
                a_window = torch.cat([a_partial, pad], dim=1)
            h_in = h[:, -self.history_size:]
            nxt = self.model.predict(h_in, a_window)  # (N, history_size, D)
            # Take only the last step as the next latent
            nxt = nxt[:, -1]  # (N, D)
            h = torch.cat([h[:, 1:], nxt.unsqueeze(1)], dim=1)
        z_final = h[:, -1]  # (N, D)
        return ((z_final - z_goal.unsqueeze(0)) ** 2).sum(-1)  # (N,)

    @torch.no_grad()
    def plan(self, z_init: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Run CEM optimization, return best action sequence (H, A)."""
        H, A = self.horizon, self.action_dim
        N, K, T = self.n_samples, self.n_elites, self.n_iters
        mu = torch.zeros(H, A, device=self.device)
        sigma = torch.ones(H, A, device=self.device) * self.sigma_init
        for _ in range(T):
            eps = torch.randn(N, H, A, device=self.device)
            candidates = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (N, H, A)
            costs = self._rollout_cost(z_init, z_goal, candidates)
            topk = torch.topk(costs, K, largest=False).indices
            elites = candidates[topk]
            mu = elites.mean(dim=0)
            sigma = elites.std(dim=0).clamp_min(1e-4)
        # Final round: sample once more from the converged distribution, return best
        eps = torch.randn(N, H, A, device=self.device)
        candidates = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
        costs = self._rollout_cost(z_init, z_goal, candidates)
        return candidates[costs.argmin()]

    @torch.no_grad()
    def first_action(self, z_init: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """MPC mode: return only the first action of the optimized plan."""
        return self.plan(z_init, z_goal)[0]

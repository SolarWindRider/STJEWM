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

from collections.abc import Callable

import torch


def make_native_action_predict_hook(
    model_action_dim: int,
    action_low,
    action_high,
    device: str | torch.device,
    predict_hook: Callable[[object, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
):
    """Adapt native CEM controls to the bounded, zero-padded training input.

    Construct CEM with the native action dimension. Apply the same native
    bounds when executing its returned sequence in the environment.
    """
    low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
    high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
    if low.ndim != 1 or high.shape != low.shape or not torch.all(low < high):
        raise ValueError("Native action bounds must be ordered, matching 1D arrays")
    padding = model_action_dim - low.numel()
    if padding < 0:
        raise ValueError("Model action dimension is smaller than the native action dimension")

    def predict_native_actions(model, ctx_emb, ctx_act):
        if ctx_act.shape[-1] != low.numel():
            raise ValueError("CEM action dimension does not match the native action bounds")
        actions = torch.clamp(ctx_act, min=low, max=high)
        if padding:
            actions = torch.nn.functional.pad(actions, (0, padding))
        if predict_hook is not None:
            return predict_hook(model, ctx_emb, actions)
        return model.predict(ctx_emb, actions)

    return predict_native_actions


class CEM:
    """Cross-Entropy Method planner for any model with `predict(ctx_emb, ctx_act)`.

    The model must expose:
        model.predict(ctx_emb: (B, H, D), ctx_act: (B, H, A)) -> (B, H, D)

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
        predict_hook: Callable[[object, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
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
        # Optional input adapter or experiment-specific predictor intervention.
        self.predict_hook = predict_hook

    @torch.no_grad()
    def _rollout_cost(self, z_init: torch.Tensor, z_goal: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll out (N, H, A) actions through model.predict and compute cost.

        Args:
            z_init: (D,) initial latent or (history_size, D) observed context
            z_goal: (D,) goal latent (single episode)
            actions: (N, H, A) candidate action sequences

        Returns:
            (N,) cost for each candidate.
        """
        N, H, A = actions.shape
        if z_init.ndim == 1:
            context = z_init.unsqueeze(0).expand(self.history_size, -1)
        elif z_init.ndim == 2 and z_init.shape[0] == self.history_size:
            context = z_init
        else:
            raise ValueError("Initial latent must be (D,) or (history_size, D)")
        h = context.unsqueeze(0).expand(N, -1, -1).contiguous()
        for t in range(H):
            avail = H - t
            if avail >= self.history_size:
                a_window = actions[:, t:t + self.history_size]
            else:
                a_partial = actions[:, t:]
                pad = torch.zeros(N, self.history_size - avail, A, device=actions.device, dtype=actions.dtype)
                a_window = torch.cat([a_partial, pad], dim=1)
            h_in = h[:, -self.history_size:]
            if self.predict_hook is None:
                nxt = self.model.predict(h_in, a_window)  # (N, history_size, D)
            else:
                nxt = self.predict_hook(self.model, h_in, a_window)
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

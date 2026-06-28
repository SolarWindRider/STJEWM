"""Single canonical encode helper.

The model API contract (ST-JEWM + LeWM-style baseline both satisfy this):

    model.encode(obs: Tensor, action: Tensor) -> dict with key 'emb' (B, T, D)
    model.predict(ctx_emb: Tensor, ctx_act: Tensor) -> Tensor (B, D)
"""
from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def encode_obs(model, obs: torch.Tensor, action_dim: int, device: str | torch.device = "cuda") -> torch.Tensor:
    """Encode a single observation (D_obs,) into a latent embedding (D,).

    Returns:
        (D,) tensor on the specified device.
    """
    s = obs.reshape(1, 1, -1).to(device).float()
    a = torch.zeros(1, 1, action_dim, device=device)
    enc = model.encode(s, a)
    return enc["emb"][0, 0]


@torch.no_grad()
def encode_batch(model, obs: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Encode a batch of observations (B, D_obs) into latents (B, D).

    Returns:
        (B, D) tensor on the same device as obs.
    """
    s = obs.unsqueeze(1)  # (B, 1, D_obs)
    a = torch.zeros(s.shape[0], 1, action_dim, device=obs.device)
    enc = model.encode(s, a)
    return enc["emb"][:, 0, :]  # (B, D)


@torch.no_grad()
def encode_history(
    model,
    obs_list: list[torch.Tensor],
    action_dim: int,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Encode a list of `history_size` observations into a stacked history (H, D).

    Args:
        obs_list: list of `history_size` tensors, each of shape (D_obs,).
    """
    z_list = [encode_obs(model, o, action_dim, device) for o in obs_list]
    return torch.stack(z_list, dim=0)  # (H, D)


# ============================================================
# Model API contract checker (used by train.py and eval.py to
# verify a checkpoint exposes the expected interface before use).
# ============================================================
REQUIRED_MODEL_METHODS = ("encode", "predict")


def assert_model_compatible(model) -> None:
    """Raise RuntimeError if `model` does not expose the required API.

    The required API is:

        model.encode(obs: Tensor, action: Tensor) -> dict with 'emb' key
        model.predict(ctx_emb: Tensor, ctx_act: Tensor) -> Tensor

    Both STJEWM and LeWMTransformerBaseline satisfy this.
    """
    for m in REQUIRED_MODEL_METHODS:
        if not hasattr(model, m):
            raise RuntimeError(
                f"Model {type(model).__name__} is missing required method "
                f"`{m}()`. Required: model.encode(obs, action) -> dict with 'emb'; "
                f"model.predict(ctx_emb, ctx_act) -> next_emb. See "
                f"code/core/encode.py for the contract."
            )

    # Smoke test: signature must accept at least 2 positional parameters
    # (e.g. STJEWM uses (x, a), LeWMTransformerBaseline uses (state, action),
    #  both with 2 positional args). We don't enforce specific param names
    #  because they vary by model.
    import inspect
    sig = inspect.signature(model.encode)
    if len(sig.parameters) < 2:
        raise RuntimeError(
            f"model.encode() must accept at least 2 positional parameters. "
            f"Got signature: {sig}"
        )
    sig = inspect.signature(model.predict)
    if len(sig.parameters) < 2:
        raise RuntimeError(
            f"model.predict() must accept at least 2 positional parameters. "
            f"Got signature: {sig}"
        )

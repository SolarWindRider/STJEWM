#!/usr/bin/env python3
"""Shared strict checkpoint builder for utility drivers.

Replaces the per-driver build_model_from_ckpt variants whose hardcoded
defaults (GRU hidden 192, MLP width, missing ALIF/SLIF branches) predate the
5M-aligned training and crash on strict load. Uses the trainer's own
build_model so every family loads with the exact dims it was trained on.
"""
from __future__ import annotations

import torch


def build_strict(ckpt_path: str, device: str = "cpu"):
    """Load a 5M-aligned checkpoint and return (model, ck_args)."""
    from code.train.train import build_model

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    state_dim = ck_args.get("pad_obs_to") or 128
    action_dim = ck_args.get("action_dim") or 56
    model = build_model(
        ck_args.get("model", "stjewm"),
        obs_dim=state_dim,
        action_dim=action_dim,
        n_layers=ck_args.get("n_layers", 4),
        readout_mode=ck_args.get("readout_mode", "hidden_leak"),
        embed_dim=ck_args.get("embed_dim"),
        hidden_dim=ck_args.get("hidden_dim"),
        mlp_hidden=ck_args.get("mlp_hidden"),
        mlp_layers=ck_args.get("mlp_layers"),
        image_size=ck_args.get("image_size", 0),
    )
    model.load_state_dict(ck["model"])  # strict — mismatch must crash loudly
    model.to(device).eval()
    return model, ck_args

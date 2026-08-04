#!/usr/bin/env python3
"""Measure per-step dense/effective FLOPs for the 5M-aligned world models.

This script intentionally uses an architecture-level FLOP ledger rather than a
backend profiler.  That keeps the comparison stable across PyTorch/CUDA
versions and makes the event-driven discount explicit.  It loads each trained
checkpoint, executes several real random batches, and measures STJEWM soma
spike sparsity from ``spike_layers`` returned by the model.

The ledger counts a dense Linear matmul as ``2 * in_features * out_features``
(one multiply and one add per weight).  Bias adds, LayerNorm, nonlinearities,
membrane updates, trace arithmetic, softmax, and tensor copies are not counted.
The frozen ViT image encoder is deliberately excluded in pixel mode because it
is shared by all five models; its trainable projection is counted.  State
projectors and action encoders are counted.  STJEWM's SNN stack, gated-trace
linear, and mode-specific readout projection are the event-discounted part.

Usage (from /home/lx/snn):
  PYTHONPATH=/home/lx/snn /home/lx/miniconda3/envs/snn/bin/python \
    code/scripts/generalist_v0_7_5_5m/measure_energy.py --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Make direct invocation independent of the caller's cwd/PYTHONPATH.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from code.train.train import build_model


MODEL_NAMES = (
    "stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only", "stjewm_no_trace",
    "stjewm_hidden_leak", "stjewm_membrane_readout", "cubifae_baseline",
    "slt_lif_mpc_trace", "slt_lif_mpc_free", "spikedreamer_baseline",
    "gru_baseline", "mlp_baseline", "lewm_baseline_v2",
)
STJEWM_VARIANTS = {name for name in MODEL_NAMES if name.startswith("stjewm_")}
SPIKING_BASELINES = {"cubifae_baseline", "slt_lif_mpc_trace", "slt_lif_mpc_free", "spikedreamer_baseline"}
BASELINE_NAMES = {"gru_baseline", "mlp_baseline", "lewm_baseline_v2"} | SPIKING_BASELINES
DEFAULT_ROOT = Path("/home/lx/snn")
DEFAULT_OUT = DEFAULT_ROOT / "results" / "journal_prep" / "P11_energy"


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _linear_flops(layer: nn.Linear) -> int:
    """Dense multiply-add FLOPs for one token through a Linear layer."""
    return 2 * int(layer.in_features) * int(layer.out_features)


def _linears_flops(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(_linear_flops(m) for m in module.modules() if isinstance(m, nn.Linear))


def _module_param_count(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(int(p.numel()) for p in module.parameters())


def _safe_mode(model: nn.Module) -> str:
    mode = getattr(model, "readout_mode", "")
    value = getattr(mode, "value", mode)
    return str(value)


def _input_and_action_flops(model: nn.Module, model_kind: str) -> Tuple[int, Dict[str, int]]:
    """Return always-dense input/action FLOPs per token and a breakdown."""
    parts: Dict[str, int] = {}
    if model_kind == "stjewm":
        if getattr(model, "state_projector", None) is not None:
            parts["state_projector"] = _linears_flops(model.state_projector)
        else:
            # STJEWM pixel mode: encoder is excluded; this projector is not.
            parts["pixel_projector"] = _linears_flops(getattr(model, "projector", None))
        parts["action_encoder"] = _linears_flops(getattr(model, "action_encoder", None))
    elif model_kind == "gru_baseline":
        if getattr(model, "pixel_pre", None) is not None:
            parts["pixel_projector"] = _linears_flops(model.pixel_pre.proj)
        else:
            parts["state_projector"] = _linears_flops(getattr(model, "state_proj", None))
        parts["action_encoder"] = _linears_flops(getattr(model, "action_proj", None))
    elif model_kind == "mlp_baseline":
        if getattr(model, "pixel_pre", None) is not None:
            parts["pixel_projector"] = _linears_flops(model.pixel_pre.proj)
        else:
            parts["state_projector"] = _linears_flops(getattr(model, "state_proj", None))
        # The FFN itself belongs to the dense dynamic path below.
    elif model_kind == "lewm_baseline":
        if getattr(model, "pixel_pre", None) is not None:
            parts["pixel_projector"] = _linears_flops(model.pixel_pre.proj)
        else:
            parts["state_encoder"] = _linears_flops(getattr(model, "state_encoder", None))
        parts["action_encoder"] = _linears_flops(getattr(model, "action_encoder", None))
    elif model_kind in {"cubifae_baseline", "slt_lif_mpc_trace", "slt_lif_mpc_free"}:
        if getattr(model, "pixel_pre", None) is not None:
            parts["pixel_projector"] = _linears_flops(model.pixel_pre.proj)
        else:
            parts["state_projector"] = _linears_flops(getattr(model, "state_projector", None))
        parts["action_encoder"] = _linears_flops(getattr(model, "action_encoder", None))
    elif model_kind == "spikedreamer_baseline":
        if getattr(model, "pixel_pre", None) is not None:
            parts["pixel_projector"] = _linears_flops(model.pixel_pre.proj)
        else:
            parts["state_projector"] = _linears_flops(getattr(model, "state_proj", None))
        parts["action_encoder"] = _linears_flops(getattr(model, "action_encoder", None))
    else:
        raise ValueError(f"Unsupported model kind for energy ledger: {model_kind}")
    return sum(parts.values()), parts


def _stjewm_dynamic_flops(model: nn.Module) -> Tuple[int, Dict[str, int]]:
    """Count STJEWM stack + trace + selected readout for one time step."""
    stack_cells = 0
    post_mlps = 0
    for cell in model.stack.cells:
        # MultiCompartmentCell has four synaptic Linear transforms.  Counting
        # all linears also remains correct if a future cell gains a projection.
        stack_cells += _linears_flops(cell)
    for post in model.stack.post_mlps:
        post_mlps += _linears_flops(post)
    trace_gate = _linears_flops(getattr(model.gated_trace, "gate", None))
    readout = 0
    mode = _safe_mode(model)
    if mode in {"hidden_leak", "trace_only"}:
        readout = _linears_flops(getattr(model, "trace_proj", None))
    elif mode == "raw_spike":
        readout = _linears_flops(getattr(model, "raw_spike_proj", None))
    # membrane_readout, spike_gated, rate_only, and no_trace have no readout
    # matmul in the implementation (their elementwise operations are omitted).
    parts = {
        "lif_cells": stack_cells,
        "post_mlp": post_mlps,
        "gated_trace": trace_gate,
        "readout_projection": readout,
    }
    return sum(parts.values()), parts


def _gru_dynamic_flops(model: nn.Module) -> Tuple[int, Dict[str, int]]:
    """Count GRU gate matmuls and output projection for one sequence step."""
    recurrent = 0
    for layer_idx in range(int(model.gru.num_layers)):
        w_ih = getattr(model.gru, f"weight_ih_l{layer_idx}")
        w_hh = getattr(model.gru, f"weight_hh_l{layer_idx}")
        # The first dimension is 3*hidden (reset/update/new gates).
        recurrent += 2 * (int(w_ih.numel()) + int(w_hh.numel()))
    output = _linears_flops(getattr(model, "proj_out", None))
    parts = {"gru_gates": recurrent, "readout_projection": output}
    return sum(parts.values()), parts


def _mlp_dynamic_flops(model: nn.Module) -> Tuple[int, Dict[str, int]]:
    ff = _linears_flops(getattr(model, "net", None))
    return ff, {"mlp_ffn": ff}


def _lewm_dynamic_flops(model: nn.Module, sequence_len: int) -> Tuple[int, Dict[str, int]]:
    """Count LeWM blocks, amortized to one token of a length-L window."""
    blocks_linear = 0
    attention_interactions = 0
    for block in model.blocks:
        # AdaLN modulation linear (D -> 6D).
        blocks_linear += _linears_flops(getattr(block, "adaLN", None))
        # Fused QKV plus output projection.  Handle both fused and split forms.
        attn = block.attn
        if getattr(attn, "in_proj_weight", None) is not None:
            blocks_linear += 2 * int(attn.in_proj_weight.numel())
        else:
            for name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attn, name, None)
                if proj is not None:
                    blocks_linear += _linears_flops(proj)
        blocks_linear += _linears_flops(getattr(attn, "out_proj", None))
        blocks_linear += _linears_flops(getattr(block, "mlp", None))
        # QK^T and AV each cost 2*L*D for a whole L-token sequence;
        # divide by L to obtain 4*L*D per token.
        d = int(block.attn.embed_dim)
        attention_interactions += 4 * int(sequence_len) * d
    output = _linears_flops(getattr(model, "proj_out", None))
    parts = {
        "transformer_linears": blocks_linear,
        "attention_interactions": attention_interactions,
        "readout_projection": output,
    }
    return sum(parts.values()), parts


def _spiking_baseline_dynamic_flops(model: nn.Module, model_kind: str, sequence_len: int) -> Tuple[int, int, Dict[str, int]]:
    """Return total dynamic FLOPs, event-discountable FLOPs, and breakdown."""
    if model_kind in {"slt_lif_mpc_trace", "slt_lif_mpc_free"}:
        lif = _linears_flops(model.stack)
        readout = _linears_flops(model.readout)
        parts = {"lif_stack": lif, "readout_projection": readout}
        return lif + readout, lif + readout, parts
    if model_kind == "cubifae_baseline":
        lif = sum(_linears_flops(cell) for cell in model.stack.cells)
        conv = model.stack.time_conv
        time_cells = 2 * int(conv.in_channels) * int(conv.out_channels) * int(conv.kernel_size[0])
        fuse = _linears_flops(model.stack.fuse)
        parts = {"alif_stack": lif, "time_cell_conv": time_cells, "readout_fusion": fuse}
        return lif + time_cells + fuse, lif, parts
    if model_kind == "spikedreamer_baseline":
        lif = _linears_flops(model.lif_stack)
        spike_proj = _linears_flops(model.spike_proj)
        tx = interactions = 0
        for block in model.blocks:
            tx += _linears_flops(block.adaLN) + _linears_flops(block.mlp)
            tx += 2 * int(block.attn.in_proj_weight.numel()) + _linears_flops(block.attn.out_proj)
            interactions += 4 * sequence_len * int(block.attn.embed_dim)
        fuser = _linears_flops(model.fuser)
        parts = {"lif_stack": lif, "spike_projection": spike_proj, "transformer_linears": tx,
                 "attention_interactions": interactions, "readout_fusion": fuser}
        return lif + spike_proj + tx + interactions + fuser, lif + spike_proj, parts
    raise ValueError(model_kind)

def _flop_ledger(model: nn.Module, model_kind: str, sequence_len: int) -> Dict[str, Any]:
    input_flops, input_parts = _input_and_action_flops(model, model_kind)
    if model_kind == "stjewm":
        dynamic, dynamic_parts = _stjewm_dynamic_flops(model)
        dynamic_label = "snn_stack_readout"
    elif model_kind == "gru_baseline":
        dynamic, dynamic_parts = _gru_dynamic_flops(model)
        dynamic_label = "gru_recurrence_readout"
    elif model_kind == "mlp_baseline":
        dynamic, dynamic_parts = _mlp_dynamic_flops(model)
        dynamic_label = "mlp_ffn_readout"
    elif model_kind == "lewm_baseline":
        dynamic, dynamic_parts = _lewm_dynamic_flops(model, sequence_len)
        dynamic_label = "transformer_readout"
    elif model_kind in SPIKING_BASELINES:
        dynamic, event_dynamic, dynamic_parts = _spiking_baseline_dynamic_flops(model, model_kind, sequence_len)
        dynamic_label = "spiking_or_hybrid_predictor"
    else:
        raise ValueError(model_kind)
    return {
        "input_action_flops": int(input_flops),
        "input_action_breakdown": {k: int(v) for k, v in input_parts.items()},
        "dynamic_flops": int(dynamic),
        "dynamic_label": dynamic_label,
        "event_discountable_flops": int(event_dynamic if model_kind in SPIKING_BASELINES else dynamic if model_kind == "stjewm" else 0),
        "dynamic_breakdown": {k: int(v) for k, v in dynamic_parts.items()},
        "dense_flops_per_step": int(input_flops + dynamic),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _measure_sparsity(
    model: nn.Module,
    *,
    pixel: bool,
    obs_dim: int,
    action_dim: int,
    image_size: int,
    sequence_len: int,
    batches: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Run random real forwards and aggregate STJEWM spike activity."""
    _set_seed(seed)
    model.eval()
    total = 0
    nonzero = 0
    layer_total: List[int] = []
    layer_nonzero: List[int] = []
    timings: List[float] = []
    with torch.no_grad():
        for _ in range(batches):
            if pixel:
                obs = torch.rand(
                    batch_size, sequence_len, 3, image_size, image_size,
                    device=device,
                )
            else:
                obs = torch.randn(batch_size, sequence_len, obs_dim, device=device)
            # Actions use the common normalized control range.
            action = torch.empty(
                batch_size, sequence_len, action_dim, device=device
            ).uniform_(-1.0, 1.0)
            start = time.perf_counter()
            out = model(obs, action)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
            spikes = out.get("spike_layers") if isinstance(out, dict) else None
            if isinstance(spikes, (list, tuple)) and spikes:
                while len(layer_total) < len(spikes):
                    layer_total.append(0)
                    layer_nonzero.append(0)
                for idx, spike in enumerate(spikes):
                    layer_total[idx] += int(spike.numel())
                    layer_nonzero[idx] += int(torch.count_nonzero(spike).item())
                    total += int(spike.numel())
                    nonzero += int(torch.count_nonzero(spike).item())
    if total:
        active_fraction = nonzero / total
        sparsity = 1.0 - active_fraction
        per_layer = [
            {
                "layer": i,
                "elements": layer_total[i],
                "nonzero": layer_nonzero[i],
                "active_fraction": layer_nonzero[i] / layer_total[i],
                "sparsity": 1.0 - layer_nonzero[i] / layer_total[i],
            }
            for i in range(len(layer_total))
        ]
        source = "measured from all STJEWM soma spike_layers on random forwards"
    else:
        # Dense baselines have no event tensor.  Keep a numeric zero so the
        # effective-flop formula is total and unambiguous, while noting that it
        # is a convention rather than a measured spike sparsity.
        active_fraction = 1.0
        sparsity = 0.0
        per_layer = []
        source = "not applicable: dense baseline (no spike_layers)"
    return {
        "sparsity": float(sparsity),
        "active_fraction": float(active_fraction),
        "spike_elements": int(total),
        "spike_nonzero": int(nonzero),
        "per_layer": per_layer,
        "source": source,
        "batches": int(batches),
        "batch_size": int(batch_size),
        "sequence_len": int(sequence_len),
        "mean_forward_seconds": float(sum(timings) / len(timings)) if timings else None,
    }


def _checkpoint_args(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint is not a mapping")
    args = ckpt.get("args", {}) or {}
    if not isinstance(args, dict):
        args = vars(args) if hasattr(args, "__dict__") else {}
    return ckpt, dict(args)


def _build_from_checkpoint(
    model_name: str,
    args: Dict[str, Any],
    *,
    pixel: bool,
    obs_dim: int,
    action_dim: int,
    image_size: int,
) -> Tuple[nn.Module, str]:
    model_kind = str(args.get("model", "stjewm"))
    if model_name.startswith("stjewm_"):
        model_kind = "stjewm"
    # State STJEWM checkpoints were trained with the retained frozen 224px
    # ViT geometry even though state forwards bypass it.  Building that exact
    # geometry permits an integrity check of the complete checkpoint while the
    # forward measurement still excludes the unused encoder.
    default_image_size = 84 if pixel else (224 if model_kind == "stjewm" else 0)
    n_layers = _as_int(args.get("n_layers"), 4 if pixel and model_kind == "stjewm" else 2)
    readout_mode = str(args.get("readout_mode", "hidden_leak"))
    embed_dim = _as_optional_int(args.get("embed_dim"))
    hidden_dim = _as_optional_int(args.get("hidden_dim"))
    mlp_hidden = _as_optional_int(args.get("mlp_hidden"))
    mlp_layers = _as_optional_int(args.get("mlp_layers"))
    if model_kind == "stjewm" and not pixel:
        # build_model intentionally routes image_size>0 to pixel mode.  The
        # state checkpoints nevertheless retain a 224px frozen encoder, so
        # instantiate STJEWM directly to preserve both that buffer and the
        # active state projector without modifying the shared factory.
        from code.stjewm import STJEWM
        model = STJEWM(
            d_hid=192, embed_dim=192, action_dim=action_dim,
            action_emb_dim=192, state_dim=obs_dim, cell_n_layers=n_layers,
            n_d=3, trace_beta=0.9, freeze_encoder=True,
            image_size=224, patch_size=14, readout_mode=readout_mode,
        )
    else:
        model = build_model(
            model_kind,
            obs_dim,
            action_dim,
            n_layers,
            readout_mode,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            mlp_hidden=mlp_hidden,
            mlp_layers=mlp_layers,
            slt_layers=_as_optional_int(args.get("slt_layers")),
            slt_din=_as_optional_int(args.get("slt_din")),
            image_size=image_size if pixel else 0,
        )
    return model, model_kind


def _measure_one(
    model_name: str,
    modality: str,
    path: Path,
    *,
    device: torch.device,
    batches: int,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    pixel = modality == "pixel"
    if not path.exists():
        return {
            "model": model_name,
            "modality": modality,
            "checkpoint": str(path),
            "status": "missing",
            "error": f"checkpoint missing: {path}",
        }
    try:
        ckpt, args = _checkpoint_args(path)
        default_obs = 21168 if pixel else 128
        obs_dim = _as_int(args.get("pad_obs_to"), default_obs)
        action_dim = _as_int(args.get("action_dim"), 56)
        image_size = _as_int(args.get("image_size"), 84 if pixel else 224 if str(args.get("model", "stjewm")) == "stjewm" else 0)
        sequence_len = _as_int(args.get("history_size"), 1 if pixel else 3)
        sequence_len = max(1, sequence_len)
        model, model_kind = _build_from_checkpoint(
            model_name, args, pixel=pixel, obs_dim=obs_dim,
            action_dim=action_dim, image_size=image_size,
        )
        state_dict = ckpt.get("model")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint has no model state_dict")
        # Build the exact checkpoint geometry above, then validate all keys.
        # The state-mode forward bypasses the retained frozen ViT, so its FLOPs
        # remain intentionally excluded from the ledger.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"state_dict mismatch (missing={list(missing)[:5]}, "
                f"unexpected={list(unexpected)[:5]})"
            )
        if not pixel and model_kind == "stjewm":
            args = dict(args)
            args["unused_frozen_encoder_loaded"] = True
        model.to(device).eval()
        ledger = _flop_ledger(model, model_kind, sequence_len)
        sparsity = _measure_sparsity(
            model, pixel=pixel, obs_dim=obs_dim, action_dim=action_dim,
            image_size=image_size, sequence_len=sequence_len,
            batches=batches, batch_size=batch_size, device=device,
            seed=seed,
        )
        active = float(sparsity["active_fraction"])
        event_dynamic = float(ledger["event_discountable_flops"])
        if model_kind == "stjewm" or model_kind in SPIKING_BASELINES:
            effective_dynamic = ledger["dynamic_flops"] - event_dynamic + event_dynamic * active
            effective_total = ledger["input_action_flops"] + effective_dynamic
        else:
            effective_dynamic = float(ledger["dynamic_flops"])
            effective_total = float(ledger["dense_flops_per_step"])
        total_params = sum(int(p.numel()) for p in model.parameters())
        trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        result = {
            "model": model_name,
            "model_kind": model_kind,
            "modality": modality,
            "checkpoint": str(path),
            "status": "ok",
            "checkpoint_args": args,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "image_size": image_size,
            "sequence_len": sequence_len,
            "ledger": ledger,
            "sparsity_measurement": sparsity,
            "dense_flops_per_step": int(ledger["dense_flops_per_step"]),
            "effective_dynamic_flops_per_step": float(effective_dynamic),
            "effective_flops_per_step": float(effective_total),
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "frozen_params": int(frozen_params),
            "shared_pixel_encoder_excluded": bool(pixel),
        }
        return result
    except Exception as exc:  # retain missing/load errors in the summary
        return {
            "model": model_name,
            "modality": modality,
            "checkpoint": str(path),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        try:
            del model
        except UnboundLocalError:
            pass
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    return f"{int(round(float(value))):,}"


def _fmt_mflops(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) / 1e6:,.3f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.3f}%"


def _fmt_ratio(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.4f}×"


def _ok_rows(data: Dict[str, Any], modality: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = [r for r in data.get("measurements", []) if r.get("status") == "ok"]
    if modality is not None:
        rows = [r for r in rows if r.get("modality") == modality]
    return rows


def _render_summary(data: Dict[str, Any]) -> str:
    rows = data.get("measurements", [])
    lines: List[str] = []
    lines.append("# P1-1 Energy / Efficiency Measurement")
    lines.append("")
    lines.append("This report is generated from `measurements.json`; all numeric entries below are rendered from that file after the checkpoint forwards completed.")
    lines.append("")
    lines.append("## Scope and reproducibility")
    lines.append("")
    lines.append(f"- Checkpoint split: `{data['split']}`; seed directory: `seed_0`.")
    lines.append(f"- Random forward protocol: `{data['batches']}` batches × `{data['batch_size']}` samples, sequence length from each checkpoint's `history_size`, seed `{data['seed']}`.")
    lines.append(f"- Device: `{data['device']}`; Python/PyTorch execution used the repository environment.")
    lines.append("- State inputs are `(B,T,128)` and actions are `(B,T,56)`. Pixel inputs are `(B,T,3,84,84)` and actions are `(B,T,56)`.")
    lines.append("- A Linear with shape `(din,dout)` contributes `2×din×dout` FLOPs per token. Biases, activations, LayerNorm, membrane/trace elementwise updates, softmax, and tensor adds are excluded consistently.")
    lines.append("- Counted always-dense path: state/pixel projection and action encoder. Counted dynamic path: STJEWM MultiCompartment cell linears, post-cell MLPs, gated-trace gate, and mode-specific readout; GRU gates/output; MLP FFN; or LeWM AdaLN/QKV/output/MLP/attention interactions/output projection.")
    lines.append("- Pixel-mode frozen ViT backbone is excluded from every row because it is shared across the comparison. Its trainable projection is included. Thus pixel numbers are predictor-side FLOPs, not end-to-end camera encoding FLOPs.")
    lines.append("- STJEWM sparsity is measured as `1 − nonzero/total` over every layer's returned binary soma `spike_layers` tensor on the random forwards. The prescribed effective estimate is `always_dense + active_fraction × dynamic_SNN/readout`; dense baselines have no spike tensor and receive sparsity `0` / active fraction `1`.")
    lines.append("")
    lines.append("## Per-model FLOP table")
    lines.append("")
    lines.append("`Params` is trainable parameters; `total` additionally includes frozen parameters (notably the excluded pixel ViT). FLOPs are per input token/step, with transformer attention amortized over the reported sequence length.")
    lines.append("")
    lines.append("| Modality | Model | Status | T | Trainable params | Total params | Dense input/action MFLOPs | Dense dynamic MFLOPs | Dense total MFLOPs/step | Sparsity | Effective dynamic MFLOPs | Effective MFLOPs/step |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| {row.get('modality','—')} | {row.get('model','—')} | **{row.get('status')}** | — | — | — | — | — | — | — | — | — |")
            continue
        led = row["ledger"]
        sm = row["sparsity_measurement"]
        lines.append(
            f"| {row['modality']} | {row['model']} | ok | {row['sequence_len']} | "
            f"{_fmt_int(row['trainable_params'])} | {_fmt_int(row['total_params'])} | "
            f"{_fmt_mflops(led['input_action_flops'])} | {_fmt_mflops(led['dynamic_flops'])} | "
            f"{_fmt_mflops(row['dense_flops_per_step'])} | {_fmt_pct(sm['sparsity'])} | "
            f"{_fmt_mflops(row['effective_dynamic_flops_per_step'])} | {_fmt_mflops(row['effective_flops_per_step'])} |"
        )
    lines.append("")
    lines.append("## Explicit STJEWM effective-vs-dense ratios")
    lines.append("")
    lines.append("Each ratio is `STJEWM effective FLOPs/step ÷ comparator dense FLOPs/step` within the same modality. Values below 1.0 indicate a lower estimated predictor-side cost under the event discount.")
    lines.append("")
    lines.append("| Modality | STJEWM variant | vs GRU dense | vs MLP dense | vs LeWM-v2 dense |")
    lines.append("|---|---|---:|---:|---:|")
    for modality in ("state", "pixel"):
        mrows = {r["model"]: r for r in _ok_rows(data, modality)}
        for st_name in ("stjewm_trace_only", "stjewm_spike_only"):
            st = mrows.get(st_name)
            vals = []
            for baseline in ("gru_baseline", "mlp_baseline", "lewm_baseline_v2"):
                b = mrows.get(baseline)
                ratio = (st["effective_flops_per_step"] / b["dense_flops_per_step"]) if st and b else None
                vals.append(_fmt_ratio(ratio))
            lines.append(f"| {modality} | {st_name} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines.append("")
    lines.append("## Measured spike activity")
    lines.append("")
    lines.append("| Modality | Model | Spike elements | Nonzero spikes | Active fraction | Sparsity source | Per-layer sparsity |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for row in _ok_rows(data):
        sm = row["sparsity_measurement"]
        layer_text = ", ".join(f"L{x['layer']}={100*x['sparsity']:.3f}%" for x in sm.get("per_layer", [])) or "—"
        lines.append(
            f"| {row['modality']} | {row['model']} | {_fmt_int(sm['spike_elements'])} | {_fmt_int(sm['spike_nonzero'])} | "
            f"{_fmt_pct(sm['active_fraction'])} | {sm['source']} | {layer_text} |"
        )
    lines.append("")
    errors = [r for r in rows if r.get("status") != "ok"]
    lines.append("## Missing or failed inputs")
    lines.append("")
    if errors:
        for row in errors:
            lines.append(f"- `{row.get('modality')}/{row.get('model')}`: {row.get('error', row.get('status'))}")
    else:
        lines.append("- None; all requested state and pixel checkpoints were present and loaded.")
    lines.append("")
    lines.append("## Component interpretation")
    lines.append("")
    lines.append("The comparison is deliberately about the learned world-model predictor after observation projection. The shared frozen pixel ViT is reported in `total_params` for transparency but excluded from FLOPs; including the identical ViT once on both sides would add a common constant and not change the relative predictor ranking. The SNN discount is an analytical event-driven estimate, not a hardware benchmark: it discounts the counted STJEWM stack/readout matmuls by measured active soma fraction while leaving input/action projections dense.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="cross_benchmark_F1")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    parser.add_argument("--modalities", nargs="+", default=["state", "pixel"], choices=["state", "pixel"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if args.batches < 1 or args.batch_size < 1:
        raise ValueError("--batches and --batch-size must be positive")

    started = time.time()
    measurements: List[Dict[str, Any]] = []
    for modality in args.modalities:
        base = root / ("results/5m_pixel" if modality == "pixel" else "results/5m")
        for model_name in args.models:
            ckpt = base / args.split / model_name / "seed_0" / "final.pt"
            print(f"[measure_energy] {modality}/{model_name}: {ckpt}", flush=True)
            row = _measure_one(
                model_name, modality, ckpt, device=device,
                batches=args.batches, batch_size=args.batch_size,
                seed=args.seed + len(measurements),
            )
            measurements.append(row)
            if row.get("status") == "ok":
                print(
                    f"  dense={row['dense_flops_per_step']/1e6:.3f} MFLOPs "
                    f"effective={row['effective_flops_per_step']/1e6:.3f} MFLOPs "
                    f"sparsity={100*row['sparsity_measurement']['sparsity']:.3f}%",
                    flush=True,
                )
            else:
                print(f"  {row.get('status')}: {row.get('error')}", flush=True)

    data: Dict[str, Any] = {
        "experiment": "P1-1_energy",
        "split": args.split,
        "root": str(root),
        "device": str(device),
        "seed": int(args.seed),
        "batches": int(args.batches),
        "batch_size": int(args.batch_size),
        "models": list(args.models),
        "modalities": list(args.modalities),
        "started_unix": started,
        "finished_unix": time.time(),
        "measurements": measurements,
    }
    json_path = out_dir / "measurements.json"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    # Render by re-reading the persisted measurements: the markdown cannot
    # silently diverge from the actual output file.
    persisted = json.loads(json_path.read_text())
    summary_path = out_dir / "energy_summary.md"
    summary_path.write_text(_render_summary(persisted))
    print(f"[measure_energy] wrote {json_path}", flush=True)
    print(f"[measure_energy] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

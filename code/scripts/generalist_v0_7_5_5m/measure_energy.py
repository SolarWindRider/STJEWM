#!/usr/bin/env python3
"""G3/P11 random-input soma sparsity and analytic operation estimates.

The fixed comparison is 13 retained models x state/pixel on cross_benchmark_F1.
Every checkpoint must pass the final TrainingAudit before strict reconstruction
with the training factory and its saved arguments.

Actual random forwards measure binary soma activity. The separate dense ledger
counts Linear/GRU/Conv multiply-adds and nominal dense attention interactions.
Biases, nonlinearities, normalization, membrane/trace arithmetic, pooling,
softmax, tensor copies and the pixel ViT backbone are excluded.

The prescribed sparsity-weighted quantity is a HYPOTHETICAL PROXY: it applies
soma activity to selected operation counts, including continuous computations.
It is not measured energy, hardware speed, or rigorously executed FLOPs; the
current implementations use dense kernels even when their soma spikes are zero.

Usage:
  python -m code.scripts.generalist_v0_7_5_5m.measure_energy \
    --training-manifest /path/to/training_final_repair_manifest.json \
    --out-dir /path/to/fresh/P11_energy --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make direct invocation independent of the caller's cwd/PYTHONPATH.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from code.train.train import build_model
from code.scripts.audited_results import TrainingAudit, require, sha256, write_new_json
from code.scripts.generalist_v0_7_5_5m_pixel.cross_modality_table import MODEL_MAP


MODEL_NAMES = (
    "stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only", "stjewm_no_trace",
    "stjewm_hidden_leak", "stjewm_membrane_readout", "alif_timecell_baseline",
    "stacked_lif_trace", "stacked_lif_free", "lif_transformer_baseline",
    "gru_baseline", "mlp_baseline", "lewm_baseline_v2",
)
STJEWM_VARIANTS = {name for name in MODEL_NAMES if name.startswith("stjewm_")}
SPIKING_BASELINES = {"alif_timecell_baseline", "stacked_lif_trace", "stacked_lif_free", "lif_transformer_baseline"}
MODALITIES = ("state", "pixel")
DEFAULT_ROOT = Path("/home/lx/snn")
DEFAULT_OUT = DEFAULT_ROOT / "results" / "journal_prep" / "P11_energy"
PROTOCOL = "G3_P11_random_input_sparsity_proxy_v2"
PROXY_INTERPRETATION = (
    "Hypothetical soma-sparsity-weighted proxy, not measured hardware energy, "
    "speed, or rigorously executed FLOPs. The weighted paths include continuous "
    "membrane, residual, trace, gate and MLP computations; soma activity does not "
    "establish that these operations can be skipped."
)
LEDGER_EXCLUSIONS = (
    "frozen pixel ViT backbone", "bias additions", "nonlinearities", "normalization",
    "membrane and trace elementwise updates", "pooling", "softmax", "tensor additions and copies",
)




def _linear_flops(layer: nn.Linear) -> int:
    """Dense multiply-add FLOPs for one token through a Linear layer."""
    return 2 * int(layer.in_features) * int(layer.out_features)


def _linears_flops(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(_linear_flops(m) for m in module.modules() if isinstance(m, nn.Linear))




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
    elif model_kind in {"alif_timecell_baseline", "stacked_lif_trace", "stacked_lif_free"}:
        if getattr(model, "pixel_pre", None) is not None:
            parts["pixel_projector"] = _linears_flops(model.pixel_pre.proj)
        else:
            parts["state_projector"] = _linears_flops(getattr(model, "state_projector", None))
        parts["action_encoder"] = _linears_flops(getattr(model, "action_encoder", None))
    elif model_kind == "lif_transformer_baseline":
        if getattr(model, "pixel_pre", None) is not None:
            parts["pixel_projector"] = _linears_flops(model.pixel_pre.proj)
        else:
            parts["state_projector"] = _linears_flops(getattr(model, "state_proj", None))
        parts["action_encoder"] = _linears_flops(getattr(model, "action_encoder", None))
    else:
        raise ValueError(f"Unsupported model kind for operation ledger: {model_kind}")
    return sum(parts.values()), parts


def _stjewm_dynamic_flops(model: nn.Module) -> Tuple[int, Dict[str, int]]:
    """Count STJEWM stack + trace + selected readout for one time step."""
    stack_cells = 0
    post_mlps = 0
    for cell in model.stack.cells:
        # MultiCompartmentCell executes all four transforms, including
        # continuous membrane feedback. Soma sparsity does not skip them.
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
        # QK^T and AV together cost a nominal 4*L*L*D for the window,
        # or 4*L*D per token. This dense estimate does not credit a causal kernel.
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
    """Return dense dynamic count, hypothetically weighted count, and breakdown."""
    if model_kind in {"stacked_lif_trace", "stacked_lif_free"}:
        lif = _linears_flops(model.stack)
        readout = _linears_flops(model.readout)
        parts = {"lif_stack": lif, "readout_projection": readout}
        return lif + readout, lif + readout, parts
    if model_kind == "alif_timecell_baseline":
        lif = sum(_linears_flops(cell) for cell in model.stack.cells)
        conv = model.stack.time_conv
        time_cells = 2 * int(conv.weight.numel())
        fuse = _linears_flops(model.stack.fuse)
        parts = {"alif_stack": lif, "time_cell_conv": time_cells, "readout_fusion": fuse}
        return lif + time_cells + fuse, lif, parts
    if model_kind == "lif_transformer_baseline":
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
    """Analytic dense work and the prescribed proxy partition, never a profiler."""
    input_flops, input_parts = _input_and_action_flops(model, model_kind)
    weighted = 0
    if model_kind == "stjewm":
        dynamic, dynamic_parts = _stjewm_dynamic_flops(model)
        weighted = dynamic
        weighted_parts = dict(dynamic_parts)
    elif model_kind == "gru_baseline":
        dynamic, dynamic_parts = _gru_dynamic_flops(model)
        weighted_parts = {}
    elif model_kind == "mlp_baseline":
        dynamic, dynamic_parts = _mlp_dynamic_flops(model)
        weighted_parts = {}
    elif model_kind == "lewm_baseline":
        dynamic, dynamic_parts = _lewm_dynamic_flops(model, sequence_len)
        weighted_parts = {}
    elif model_kind in SPIKING_BASELINES:
        dynamic, weighted, dynamic_parts = _spiking_baseline_dynamic_flops(model, model_kind, sequence_len)
        names = {
            "alif_timecell_baseline": ("alif_stack",),
            "stacked_lif_trace": ("lif_stack", "readout_projection"),
            "stacked_lif_free": ("lif_stack", "readout_projection"),
            "lif_transformer_baseline": ("lif_stack", "spike_projection"),
        }[model_kind]
        weighted_parts = {name: dynamic_parts[name] for name in names}
    else:
        raise ValueError(model_kind)
    require(sum(weighted_parts.values()) == weighted, "Inconsistent hypothetical proxy partition")
    return {
        "kind": "analytic_dense_operation_estimate",
        "unit": "multiply_add_flop_equivalents_per_sample_token",
        "input_action_dense_flops_per_step": int(input_flops),
        "input_action_breakdown": input_parts,
        "predictor_dense_flops_per_step": int(dynamic),
        "predictor_breakdown": dynamic_parts,
        "analytic_dense_flops_per_step": int(input_flops + dynamic),
        "hypothetically_weighted_dense_flops_per_step": int(weighted),
        "hypothetically_weighted_breakdown": weighted_parts,
        "always_dense_predictor_flops_per_step": int(dynamic - weighted),
        "always_dense_flops_per_step": int(input_flops + dynamic - weighted),
        "always_dense_interpretation": (
            "Unweighted terms in the hypothetical proxy, not the only dense "
            "operations in the implementation; all counted kernels are dense."
        ),
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
    model_kind: str,
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
    """Measure all returned binary soma spikes, not hidden/trace activation zeros."""
    _set_seed(seed)
    model.eval()
    expects_spikes = model_kind == "stjewm" or model_kind in SPIKING_BASELINES
    layer_total: List[int] = []
    layer_nonzero: List[int] = []
    layer_shapes: List[List[int]] = []
    with torch.no_grad():
        for batch in range(batches):
            if pixel:
                obs = torch.rand(
                    batch_size, sequence_len, 3, image_size, image_size,
                    device=device,
                )
            else:
                obs = torch.randn(batch_size, sequence_len, obs_dim, device=device)
            action = torch.empty(
                batch_size, sequence_len, action_dim, device=device
            ).uniform_(-1.0, 1.0)
            out = model(obs, action)
            require(isinstance(out, dict), "Random forward did not return the model output mapping")
            emb = out.get("emb")
            require(isinstance(emb, torch.Tensor) and emb.ndim == 3
                    and tuple(emb.shape[:2]) == (batch_size, sequence_len)
                    and emb.shape[-1] == model.embed_dim, "Invalid random-forward latent dimensions")
            require(bool(torch.isfinite(emb).all()), "Non-finite random-forward latent")
            spikes = out.get("spike_layers")
            if not expects_spikes:
                require(spikes is None or isinstance(spikes, (list, tuple)) and not spikes,
                        "Dense baseline unexpectedly returned soma spike layers")
                continue
            require(isinstance(spikes, (list, tuple)) and spikes,
                    f"{model_kind} did not return required soma spike_layers")
            if batch == 0:
                layer_total = [0] * len(spikes)
                layer_nonzero = [0] * len(spikes)
            require(len(spikes) == len(layer_total), "Spike-layer count changed between batches")
            for idx, spike in enumerate(spikes):
                require(isinstance(spike, torch.Tensor) and spike.ndim == 3
                        and tuple(spike.shape[:2]) == (batch_size, sequence_len)
                        and spike.shape[-1] > 0, f"Invalid soma spike dimensions at layer {idx}")
                require(bool(((spike == 0) | (spike == 1)).all()),
                        f"Non-binary or non-finite soma spikes at layer {idx}")
                shape = list(spike.shape)
                if batch == 0:
                    layer_shapes.append(shape)
                require(shape == layer_shapes[idx], f"Spike dimensions changed at layer {idx}")
                layer_total[idx] += int(spike.numel())
                layer_nonzero[idx] += int(torch.count_nonzero(spike).item())
    total = sum(layer_total)
    nonzero = sum(layer_nonzero)
    per_layer = [
        {
            "layer": i,
            "shape_per_batch": layer_shapes[i],
            "elements": layer_total[i],
            "nonzero": layer_nonzero[i],
            "active_fraction": layer_nonzero[i] / layer_total[i],
            "sparsity": 1.0 - layer_nonzero[i] / layer_total[i],
        }
        for i in range(len(layer_total))
    ]
    return {
        "status": "measured" if expects_spikes else "not_applicable",
        "sparsity": 1.0 - nonzero / total if expects_spikes else None,
        "active_fraction": nonzero / total if expects_spikes else None,
        "spike_elements": total,
        "spike_nonzero": nonzero,
        "per_layer": per_layer,
        "source": (
            "all returned binary soma spike_layers on random forwards"
            if expects_spikes else "dense baseline: no soma spike measurement"
        ),
        "seed": seed,
        "batches": batches,
        "batch_size": batch_size,
        "sequence_len": sequence_len,
    }


def _planned_cells(audit: TrainingAudit, split: str, seed: int) -> List[Dict[str, Any]]:
    """Resolve the retained grid; never substitute a nearby checkpoint directory."""
    cells = []
    for modality in MODALITIES:
        for model_name in MODEL_NAMES:
            family = "5m_pixel" if modality == "pixel" else "5m_5mpar" if model_name in STJEWM_VARIANTS else "5m"
            checkpoint_model = MODEL_MAP.get(model_name, model_name) if modality == "pixel" else model_name
            path = audit.results_root / family / split / checkpoint_model / "seed_0" / "final.pt"
            cells.append({
                "id": f"{modality}/{model_name}",
                "model": model_name,
                "display_name": "LeWM" if model_name == "lewm_baseline_v2" else model_name,
                "modality": modality,
                "family": family,
                "checkpoint_model": checkpoint_model,
                "checkpoint": str(path.resolve()),
                "training_seed": 0,
                "random_seed": seed + len(cells),
            })
    require(len(cells) == 26 and len({cell["checkpoint"] for cell in cells}) == 26,
            "G3/P11 requires 26 distinct retained checkpoint cells")
    return cells


def _source_hashes(audit: TrainingAudit) -> Dict[str, str]:
    """Bind this run to the final training/loader sources and its own producer."""
    hashes: Dict[str, str] = {}
    for registry in ("training_source_sha256", "loader_source_sha256"):
        for relative, expected in audit.payload[registry].items():
            actual = hashes.get(relative)
            if actual is None:
                actual = sha256(ROOT / relative)
            require(actual == expected, f"Source differs from the final training audit: {relative}")
            hashes[relative] = actual
    for relative in (
        str(Path(__file__).resolve().relative_to(ROOT)),
        "code/scripts/audited_results.py",
        "code/scripts/generalist_v0_7_5_5m/repair_eval_grid.py",
        "code/scripts/generalist_v0_7_5_5m_pixel/cross_modality_table.py",
    ):
        hashes[relative] = sha256(ROOT / relative)
    return hashes


def _checkpoint_args(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    require(isinstance(ckpt, dict), "Checkpoint is not a mapping")
    args = ckpt.get("args")
    require(isinstance(args, dict), "Checkpoint has no saved argument mapping")
    require(isinstance(ckpt.get("model"), dict), "Checkpoint has no model state_dict")
    for key in ("pad_obs_to", "action_dim", "n_layers", "history_size"):
        require(type(args.get(key)) is int and args[key] > 0, f"Invalid saved {key}")
    require(type(args.get("image_size")) is int and args["image_size"] >= 0, "Invalid saved image_size")
    require(type(args.get("seed")) is int, "Missing saved training seed")
    return ckpt, dict(args)


def _build_from_checkpoint(
    model_name: str,
    args: Dict[str, Any],
    *,
    pixel: bool,
) -> Tuple[nn.Module, str]:
    """Use the canonical training factory, including the saved frozen geometry."""
    model_kind = args["model"]
    expected_kind = "stjewm" if model_name in STJEWM_VARIANTS else MODEL_MAP.get(model_name, model_name)
    require(model_kind == expected_kind, f"Saved model {model_kind!r} does not match cell {model_name}")
    obs_dim, image_size = args["pad_obs_to"], args["image_size"]
    is_pixel = image_size > 0 and obs_dim == 3 * image_size * image_size
    require(pixel == is_pixel, "Saved geometry does not match the planned modality")
    if model_kind == "stjewm":
        require(args["readout_mode"] == model_name.removeprefix("stjewm_"),
                "Saved STJEWM readout does not match the planned variant")
    model = build_model(
        model_kind, obs_dim, args["action_dim"], args["n_layers"], args["readout_mode"],
        embed_dim=args.get("embed_dim"),
        hidden_dim=args.get("hidden_dim"),
        mlp_hidden=args.get("mlp_hidden"),
        mlp_layers=args.get("mlp_layers"),
        stacked_lif_layers=args.get("stacked_lif_layers"),
        stacked_lif_din=args.get("stacked_lif_din"),
        image_size=image_size,
    )
    return model, model_kind


def _measure_one(
    cell: Dict[str, Any],
    audit: TrainingAudit,
    *,
    device: torch.device,
    batches: int,
    batch_size: int,
    measurement_source_sha256: str,
) -> Dict[str, Any]:
    path = Path(cell["checkpoint"])
    pixel = cell["modality"] == "pixel"
    result = dict(cell, status="error", measurement_source_sha256=measurement_source_sha256,
                  batches=batches, batch_size=batch_size)
    stage = "training_audit_admission"
    model = None
    try:
        result["repair_provenance"] = audit.provenance(path)
        stage = "strict_saved_geometry_load"
        ckpt, args = _checkpoint_args(path)
        require(args["seed"] == cell["training_seed"], "Checkpoint training seed differs from the planned cell")
        obs_dim, action_dim = args["pad_obs_to"], args["action_dim"]
        image_size, sequence_len = args["image_size"], args["history_size"]
        result.update(
            checkpoint_args=args, checkpoint_step=ckpt["step"],
            checkpoint_training_protocol_version=ckpt.get("training_protocol_version"),
            checkpoint_data_protocol_version=ckpt.get("data_protocol_version"),
            obs_dim=obs_dim, action_dim=action_dim, image_size=image_size,
            sequence_len=sequence_len, dtype="float32",
            observation_shape=[batch_size, sequence_len, 3, image_size, image_size]
            if pixel else [batch_size, sequence_len, obs_dim],
            action_shape=[batch_size, sequence_len, action_dim],
            sample_tokens=batches * batch_size * sequence_len,
        )
        model, model_kind = _build_from_checkpoint(cell["model"], args, pixel=pixel)
        model.load_state_dict(ckpt["model"], strict=True)
        del ckpt
        require(sha256(path) == result["repair_provenance"]["checkpoint_sha256"],
                "Checkpoint changed after training-audit admission")
        result.update(model_kind=model_kind, weights_loaded_strict=True,
                      model_class=type(model).__name__, embedding_dim=model.embed_dim)
        model.to(device).eval()
        stage = "analytic_ledger"
        ledger = _flop_ledger(model, model_kind, sequence_len)
        stage = "random_input_forward"
        sparsity = _measure_sparsity(
            model, model_kind=model_kind, pixel=pixel, obs_dim=obs_dim, action_dim=action_dim,
            image_size=image_size, sequence_len=sequence_len,
            batches=batches, batch_size=batch_size, device=device, seed=cell["random_seed"],
        )
        active = sparsity["active_fraction"] if sparsity["status"] == "measured" else 1.0
        weighted = ledger["hypothetically_weighted_dense_flops_per_step"]
        proxy_dynamic = ledger["always_dense_predictor_flops_per_step"] + active * weighted
        proxy_total = ledger["always_dense_flops_per_step"] + active * weighted
        stage = "checkpoint_stability"
        require(sha256(path) == result["repair_provenance"]["checkpoint_sha256"],
                "Checkpoint changed during the measurement")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        result.update(
            status="ok", ledger=ledger, sparsity_measurement=sparsity,
            analytic_dense_flops_per_step=ledger["analytic_dense_flops_per_step"],
            hypothetical_sparsity_weighted_proxy={
                "unit": "multiply_add_flop_equivalents_per_sample_token",
                "interpretation": PROXY_INTERPRETATION,
                "formula": "always_dense_flops_per_step + active_fraction * hypothetically_weighted_dense_flops_per_step",
                "active_fraction": active,
                "active_fraction_source": (
                    "measured pooled binary soma activity" if sparsity["status"] == "measured"
                    else "unweighted dense-baseline convention; not a sparsity measurement"
                ),
                "predictor_per_step": proxy_dynamic,
                "total_per_step": proxy_total,
            },
            trainable_params=trainable_params, total_params=total_params,
            frozen_params=total_params - trainable_params,
            frozen_pixel_encoder_excluded_from_ledger=pixel,
        )
    except Exception as exc:
        result.update(status="error", failure_stage=stage, error_type=type(exc).__name__, error=str(exc))
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


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
    return f"{float(value):.4f}"


def _ok_rows(data: Dict[str, Any], modality: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = [r for r in data.get("measurements", []) if r.get("status") == "ok"]
    if modality is not None:
        rows = [r for r in rows if r.get("modality") == modality]
    return rows


def _render_summary(data: Dict[str, Any]) -> str:
    rows = data["measurements"]
    lines = [
        "# G3/P11 Random-input soma sparsity and analytic proxy",
        "",
        f"Run status: **{data['status']}**; {data['completed_rows']}/{data['planned_rows']} planned rows succeeded.",
        "",
        data["interpretation"],
        "",
        "All numbers below come from `measurements.json`. Failed rows are retained, not dropped from the planned grid.",
        "",
        "## Protocol and provenance",
        "",
        f"- Split: `{data['split']}`; 13 retained models x state/pixel; training seed `0`.",
        f"- Final training manifest: `{data['training_manifest']}`; SHA-256 `{data['training_manifest_sha256']}`.",
        f"- Random forwards: `{data['batches']}` batches x `{data['batch_size']}` samples; T is the saved `history_size`.",
        f"- Seed base `{data['seed']}` plus the fixed state-then-pixel/model-order row index; each exact seed and input shape is recorded in JSON.",
        f"- Device: `{data['device']}`. Device identity is execution provenance, not an energy or speed measurement.",
        "- State inputs are standard normal, pixel inputs uniform [0,1), and actions uniform [-1,1). Every call starts from the model's reset recurrent state.",
        "- State STJEWM uses `5m_5mpar`; other state models use `5m`. Pixel uses `5m_pixel` with the audited `stjewm`/`lewm_baseline` directory mapping.",
        "- Saved arguments are passed to the canonical training factory; every weight/buffer is loaded with `strict=True`, including the unused frozen state-STJEWM encoder.",
        "- Checkpoint hashes, producer/training/loader source hashes, runtime versions, and batch dimensions are recorded in JSON.",
        "",
        "## Analytic dense ledger and hypothetical partition",
        "",
        "Counts use 2*din*dout per Linear, twice the GRU gate weight elements, twice the Conv1d weight elements per output position, and nominal 4*T*D attention interactions per token/block. Causal masking does not reduce this dense estimate.",
        "",
        "Excluded: " + "; ".join(data["ledger_exclusions"]) + ". Pixel forwards execute the frozen ViT to measure actual spike activity, but its work is excluded from the ledger. These are not end-to-end camera costs.",
        "",
        "The hypothetical formula is `always_dense + measured_soma_active_fraction * hypothetically_weighted_dense`. Always-dense denotes terms left unweighted in that formula, not all operations executed densely by the model.",
        "",
        "The prescribed weighted partition is: STJEWM cell linears, post-cell MLPs, gated-trace gate and selected readout; Stacked-LIF stack/readout; ALIF cell linears only; LIF-Transformer LIF stack/spike projection only. The remaining input/action, convolution, fusion and transformer terms stay unweighted. Dense baselines have no soma measurement (null, not measured zero), and their proxy equals their dense estimate.",
        "",
        "STJEWM feedback reads continuous membranes, its residual/gate context and trace are continuous, and its post-cell MLPs include continuous activations. Pooled soma sparsity is not operand sparsity for all these operations. No runtime skipping or energy saving is demonstrated.",
        "",
        "## Complete retained grid",
        "",
        "Operation columns are millions of multiply-add FLOP-equivalents per sample token. Params include inactive/frozen modules; total additionally includes frozen parameters.",
        "",
        "| Modality | Model | Status | T | Trainable params | Total params | Dense input/action | Dense predictor | Dense total | Always-dense in proxy | Measured soma sparsity | Hypothetical weighted proxy total |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {row['modality']} | {row['display_name']} | **error** | — | — | — | — | — | — | — | — | — |")
            continue
        led = row["ledger"]
        sm = row["sparsity_measurement"]
        proxy = row["hypothetical_sparsity_weighted_proxy"]
        lines.append(
            f"| {row['modality']} | {row['display_name']} | ok | {row['sequence_len']} | "
            f"{_fmt_int(row['trainable_params'])} | {_fmt_int(row['total_params'])} | "
            f"{_fmt_mflops(led['input_action_dense_flops_per_step'])} | {_fmt_mflops(led['predictor_dense_flops_per_step'])} | "
            f"{_fmt_mflops(row['analytic_dense_flops_per_step'])} | {_fmt_mflops(led['always_dense_flops_per_step'])} | "
            f"{_fmt_pct(sm['sparsity'])} | {_fmt_mflops(proxy['total_per_step'])} |"
        )
    lines.extend([
        "",
        "## Hypothetical proxy / analytic dense ratios",
        "",
        "Dimensionless STJEWM proxy divided by comparator dense estimate within modality. These ratios are NOT speedups, energy ratios, or executed-operation reductions.",
        "",
        "| Modality | STJEWM variant | / GRU dense | / MLP dense | / LeWM dense |",
        "|---|---|---:|---:|---:|",
    ])
    for modality in MODALITIES:
        mrows = {r["model"]: r for r in _ok_rows(data, modality)}
        for st_name in ("stjewm_trace_only", "stjewm_spike_only"):
            st = mrows.get(st_name)
            vals = []
            for baseline in ("gru_baseline", "mlp_baseline", "lewm_baseline_v2"):
                other = mrows.get(baseline)
                ratio = (st["hypothetical_sparsity_weighted_proxy"]["total_per_step"]
                         / other["analytic_dense_flops_per_step"]) if st and other else None
                vals.append(_fmt_ratio(ratio))
            lines.append(f"| {modality} | {st_name} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines.extend([
        "",
        "## Actual random-input soma activity",
        "",
        "| Modality | Model | Spike elements | Nonzero spikes | Active fraction | Source | Per-layer sparsity |",
        "|---|---|---:|---:|---:|---|---|",
    ])
    for row in _ok_rows(data):
        sm = row["sparsity_measurement"]
        layer_text = ", ".join(f"L{x['layer']}={100*x['sparsity']:.3f}%" for x in sm["per_layer"]) or "—"
        lines.append(
            f"| {row['modality']} | {row['display_name']} | {_fmt_int(sm['spike_elements'])} | {_fmt_int(sm['spike_nonzero'])} | "
            f"{_fmt_pct(sm['active_fraction'])} | {sm['source']} | {layer_text} |"
        )
    lines.extend(["", "## Missing, failed, or unstable inputs", ""])
    errors = [row for row in rows if row["status"] != "ok"]
    for row in errors:
        lines.append(f"- `{row['modality']}/{row['display_name']}` ({row['failure_stage']}): {row['error_type']}: {row['error']}")
    for error in data["integrity_errors"]:
        lines.append(f"- Run integrity: {error}")
    if not errors and not data["integrity_errors"]:
        lines.append("- None; all 26 planned checkpoints were admitted, strictly loaded and measured.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--split", default="cross_benchmark_F1", choices=["cross_benchmark_F1"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help="Must not exist; cannot be inside checkpoint/archive/staging roots")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(args.batches > 0 and args.batch_size > 0, "--batches and --batch-size must be positive")
    require(0 <= args.seed <= 2**32 - 26, "The 26 per-row NumPy seeds must fit uint32")
    audit = TrainingAudit(args.training_manifest)
    out_dir = args.out_dir.resolve()
    audit.protect_output(out_dir)
    source_hashes = _source_hashes(audit)
    cells = _planned_cells(audit, args.split, args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    measurements: List[Dict[str, Any]] = []
    producer_hash = source_hashes[str(Path(__file__).resolve().relative_to(ROOT))]
    for cell in cells:
        print(f"[measure_energy] {cell['modality']}/{cell['display_name']}: {cell['checkpoint']}", flush=True)
        row = _measure_one(
            cell, audit, device=device, batches=args.batches, batch_size=args.batch_size,
            measurement_source_sha256=producer_hash,
        )
        measurements.append(row)
        if row["status"] == "ok":
            print(
                f"  analytic_dense={row['analytic_dense_flops_per_step']/1e6:.3f} MFlop-equivalents/token "
                f"hypothetical_proxy={row['hypothetical_sparsity_weighted_proxy']['total_per_step']/1e6:.3f} "
                f"soma_sparsity={_fmt_pct(row['sparsity_measurement']['sparsity'])}",
                flush=True,
            )
        else:
            print(f"  error [{row['failure_stage']}]: {row['error_type']}: {row['error']}", flush=True)
    integrity_errors = []
    try:
        require(_source_hashes(audit) == source_hashes, "Measurement source changed during the run")
        require(sha256(audit.path) == audit.digest, "Training manifest changed during the run")
    except Exception as exc:
        integrity_errors.append(f"{type(exc).__name__}: {exc}")
    require(len(measurements) == 26 and [row["id"] for row in measurements] == [cell["id"] for cell in cells],
            "Incomplete or reordered G3/P11 output")
    completed = sum(row["status"] == "ok" for row in measurements)
    data: Dict[str, Any] = {
        "experiment": "G3/P11",
        "protocol": PROTOCOL,
        "status": "completed" if completed == 26 and not integrity_errors else "failed",
        "interpretation": PROXY_INTERPRETATION,
        "hardware_energy_measured": False,
        "hardware_speed_benchmarked": False,
        "executed_flops_measured": False,
        "ledger_exclusions": list(LEDGER_EXCLUSIONS),
        "split": args.split,
        "results_root": str(audit.results_root),
        "training_manifest": str(audit.path),
        "training_manifest_sha256": audit.digest,
        "source_sha256": source_hashes,
        "training_source_sha256": audit.payload["training_source_sha256"],
        "loader_source_sha256": audit.payload["loader_source_sha256"],
        "device": str(device),
        "runtime": {
            "python": sys.version, "python_executable": sys.executable,
            "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "command": sys.argv,
        "seed": args.seed,
        "seed_assignment": "base seed + index in planned_cells; original fixed model/modality order",
        "batches": args.batches,
        "batch_size": args.batch_size,
        "random_input_distributions": {
            "state": "standard normal", "pixel": "uniform [0,1)", "actions": "uniform [-1,1)",
        },
        "models": list(MODEL_NAMES),
        "modalities": list(MODALITIES),
        "planned_rows": 26,
        "completed_rows": completed,
        "failed_rows": 26 - completed,
        "planned_cells": cells,
        "started_unix": started,
        "finished_unix": time.time(),
        "integrity_errors": integrity_errors,
        "measurements": measurements,
    }
    json_path = out_dir / "measurements.json"
    write_new_json(json_path, data)
    persisted = json.loads(json_path.read_text())
    summary_path = out_dir / "energy_summary.md"
    with summary_path.open("x") as handle:
        handle.write(_render_summary(persisted))
    print(f"[measure_energy] {data['status']}: {completed}/26; wrote {json_path} and {summary_path}", flush=True)
    return 0 if data["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

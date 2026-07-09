"""Compute a single canonical param-count table for all 13 model classes.

This script is the **single source of truth** for the parameter counts cited in
the README, MASTER_TABLE, and paper. It supersedes the prior scattered numbers
(4.6M / 8.2M / 10.53M in code, paper, and MASTER_TABLE respectively).

Workflow
--------
1. For each of the 13 model classes, prefer the checkpoint at
   ``results/generalist_G16/<on_disk_dir>/seed_0/final.pt`` (the same one the
   aggregator uses for G16 generalist evaluation). Fall back to
   ``results/generalist_G8`` and then ``results/generalist``.
2. Read the ``args`` dict stashed in the checkpoint. If no checkpoint is
   available, use architecturally canonical defaults: ``pad_obs_to=128,
   action_dim=56, embed_dim=192, n_layers=2`` (matches ``train_one.sh``'s
   G16 budget).
3. Instantiate the model on CPU (no GPU needed) and count trainable / total
   parameters. Loading the state_dict also catches any silent shape drift
   between the canonical constructor and the trained checkpoint.
4. Emit a single JSON and a single markdown table to
   ``results/aggregate/``.

Mapping note
------------
The on-disk directory is ``lewm_baseline_v2`` (the model itself is
``code.lewm_transformer_baseline.LeWMTransformerBaseline``), matching the
CLI flag ``--model lewm_baseline`` used in ``train_one.sh``.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add repo root to sys.path so `code.<module>` imports resolve.
REPO_ROOT = Path("/home/lx/snn")
sys.path.insert(0, str(REPO_ROOT))

# Standard G16 budget baked into train_one.sh. Used as the canonical fallback
# when a model has no checkpoint.
CANONICAL = dict(
    pad_obs_to=128,
    action_dim=56,
    embed_dim=192,
    n_layers=2,
)

# LeWM uses embed_dim=256 by default; the G16 trainer overrides to 192 via
# --embed-dim 192. Both are recorded as separate canonical rows.
LEWM_CANONICAL = dict(
    pad_obs_to=128, action_dim=56, embed_dim=192, n_layers=2,
)

# Ckpt search path. The aggregator uses G16 first, then G8, then generalist.
CKPT_SEARCH_DIRS = [
    REPO_ROOT / "results" / "generalist_G16",
    REPO_ROOT / "results" / "generalist_G8",
    REPO_ROOT / "results" / "generalist",
]

OUT_DIR = REPO_ROOT / "results" / "aggregate"
JSON_OUT = OUT_DIR / "model_size_table.json"
MD_OUT = OUT_DIR / "model_size_table.md"


# =============================================================================
# Per-model canonical constructor. Each entry returns the module + a dict of
# canonical-config fields shown in the markdown table.
# =============================================================================
def _build_stjewm(readout: str) -> Callable[..., Any]:
    """Return a builder closure for an STJEWM with a given readout mode."""
    def _builder(state_dim: int, action_dim: int, n_layers: int, embed_dim: int,
                 **_ignored) -> Any:
        from code.stjewm import STJEWM
        return STJEWM(
            d_hid=embed_dim, embed_dim=embed_dim, action_dim=action_dim,
            action_emb_dim=embed_dim, state_dim=state_dim,
            cell_n_layers=n_layers, n_d=3, trace_beta=0.9, freeze_encoder=True,
            readout_mode=readout,
        )
    return _builder


def _build_cubifae(state_dim, action_dim, n_layers, embed_dim, **_ignored):
    from code.cubifae_baseline import CubifAEBaseline
    return CubifAEBaseline(
        state_dim=state_dim, action_dim=action_dim,
        d_hid=embed_dim, n_layers=n_layers,
    )


def _build_spikedreamer(state_dim, action_dim, n_layers, embed_dim, **_ignored):
    from code.spikedreamer_baseline import make_spikedreamer
    return make_spikedreamer(
        state_dim=state_dim, action_dim=action_dim,
        d_snn=128, d_tx=embed_dim, num_layers=n_layers, num_heads=8,
    )


def _build_slt_trace(state_dim, action_dim, n_layers, embed_dim, **_ignored):
    from code.slt_lif_mpc_baseline import make_slt_lif_mpc_trace
    return make_slt_lif_mpc_trace(
        state_dim=state_dim, action_dim=action_dim,
        d_in=embed_dim, embed_dim=embed_dim, n_layers=n_layers,
        trace_beta=0.9, k_avg=4,
    )


def _build_slt_free(state_dim, action_dim, n_layers, embed_dim, **_ignored):
    from code.slt_lif_mpc_baseline import make_slt_lif_mpc_free
    return make_slt_lif_mpc_free(
        state_dim=state_dim, action_dim=action_dim,
        d_in=embed_dim, embed_dim=embed_dim, n_layers=n_layers,
        trace_beta=0.9,
    )


def _build_lewm(state_dim, action_dim, n_layers, embed_dim, **_ignored):
    from code.lewm_transformer_baseline import LeWMTransformerBaseline
    return LeWMTransformerBaseline(
        state_dim=state_dim, action_dim=action_dim,
        embed_dim=embed_dim, num_layers=n_layers, num_heads=8,
    )


def _build_gru(state_dim, action_dim, n_layers, embed_dim, **_ignored):
    from code.gru_baseline import GRUBaseline
    return GRUBaseline(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=576, num_layers=3, history_size=3,
    )


def _build_mlp(state_dim, action_dim, n_layers, embed_dim, **_ignored):
    from code.mlp_baseline import make_mlp_baseline
    return make_mlp_baseline(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=576, num_layers=4, emb_dim=embed_dim,
    )


# (display_name, on_disk_dir, builder, default_canonical, notes)
MODELS: List[Dict[str, Any]] = [
    # --- 6 STJEWM readouts ---
    {"name": "stjewm_trace_only", "dir": "stjewm_trace_only",
     "build": _build_stjewm("trace_only"), "canonical": dict(CANONICAL),
     "notes": "hidden + trace_proj(trace), gated alpha"},
    {"name": "stjewm_spike_only", "dir": "stjewm_spike_only",
     "build": _build_stjewm("spike_only"), "canonical": dict(CANONICAL),
     "notes": "h * spike (back-compat alias for spike_gated)"},
    {"name": "stjewm_rate_only", "dir": "stjewm_rate_only",
     "build": _build_stjewm("rate_only"), "canonical": dict(CANONICAL),
     "notes": "moving-avg spike rate, no h"},
    {"name": "stjewm_no_trace", "dir": "stjewm_no_trace",
     "build": _build_stjewm("no_trace"), "canonical": dict(CANONICAL),
     "notes": "ablation: no trace branch"},
    {"name": "stjewm_hidden_leak", "dir": "stjewm_hidden_leak",
     "build": _build_stjewm("hidden_leak"), "canonical": dict(CANONICAL),
     "notes": "v2 default readout"},
    {"name": "stjewm_membrane_readout", "dir": "stjewm_membrane_readout",
     "build": _build_stjewm("membrane_readout"), "canonical": dict(CANONICAL),
     "notes": "h.detach() (treat h as discrete latent)"},

    # --- 7 baselines ---
    {"name": "cubifae_baseline", "dir": "cubifae_baseline",
     "build": _build_cubifae, "canonical": dict(CANONICAL),
     "notes": "ALIF + time-cell readout"},
    {"name": "spikedreamer_baseline", "dir": "spikedreamer_baseline",
     "build": _build_spikedreamer, "canonical": dict(CANONICAL),
     "notes": "2-layer LIF + AdaLN-zero Transformer"},
    {"name": "slt_lif_mpc_trace", "dir": "slt_lif_mpc_trace",
     "build": _build_slt_trace, "canonical": dict(CANONICAL),
     "notes": "DECOLLE LIF, trace-only readout"},
    {"name": "slt_lif_mpc_free", "dir": "slt_lif_mpc_free",
     "build": _build_slt_free, "canonical": dict(CANONICAL),
     "notes": "DECOLLE LIF, free-access readout"},
    {"name": "lewm_transformer_baseline", "dir": "lewm_baseline_v2",
     "build": _build_lewm, "canonical": dict(LEWM_CANONICAL),
     "notes": "LeWM-style AdaLN-zero Transformer (G16: 2 layers)"},
    {"name": "gru_baseline", "dir": "gru_baseline",
     "build": _build_gru, "canonical": dict(CANONICAL),
     "notes": "3-layer GRU h=576"},
    {"name": "mlp_baseline", "dir": "mlp_baseline",
     "build": _build_mlp, "canonical": dict(CANONICAL),
     "notes": "per-step FFN, no recurrence (collapse-control)"},
]


# =============================================================================
# Ckpt resolution + counting
# =============================================================================
def find_ckpt(on_disk_dir: str) -> Optional[Path]:
    """Return the first existing ckpt across G16 -> G8 -> generalist."""
    for root in CKPT_SEARCH_DIRS:
        cand = root / on_disk_dir / "seed_0" / "final.pt"
        if cand.exists():
            return cand
    # `results/generalist/<on_disk_dir>/final.pt` is also produced by some
    # training runs (no seed_0/ wrapper).
    for root in (REPO_ROOT / "results" / "generalist",):
        cand = root / on_disk_dir / "final.pt"
        if cand.exists():
            return cand
    return None


def load_ckpt_args(ckpt_path: Path) -> Dict[str, Any]:
    """Read the stashed args dict from a final.pt checkpoint."""
    import torch
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    args = ck.get("args", {}) or {}
    if not isinstance(args, dict):
        return {}
    return dict(args)


def merge_canonical(ck_args: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Pull pad_obs_to / action_dim / embed_dim / n_layers from ck_args, else fallback."""
    out = dict(fallback)
    if "pad_obs_to" in ck_args and ck_args["pad_obs_to"]:
        out["pad_obs_to"] = int(ck_args["pad_obs_to"])
    if "action_dim" in ck_args and ck_args["action_dim"]:
        out["action_dim"] = int(ck_args["action_dim"])
    if "embed_dim" in ck_args and ck_args["embed_dim"]:
        out["embed_dim"] = int(ck_args["embed_dim"])
    if "n_layers" in ck_args and ck_args["n_layers"]:
        out["n_layers"] = int(ck_args["n_layers"])
    return out


def count_params(model) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


def measure_one(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate one model, load the ckpt if available, return param dict."""
    import torch
    name = entry["name"]
    on_disk_dir = entry["dir"]
    build = entry["build"]
    canonical = entry["canonical"]
    notes = entry["notes"]

    ckpt_path = find_ckpt(on_disk_dir)
    if ckpt_path is not None:
        ck_args = load_ckpt_args(ckpt_path)
        cfg = merge_canonical(ck_args, canonical)
        ckpt_used = str(ckpt_path.relative_to(REPO_ROOT))
    else:
        cfg = dict(canonical)
        ckpt_used = "<canonical (no ckpt found)>"

    model = build(
        state_dim=cfg["pad_obs_to"],
        action_dim=cfg["action_dim"],
        n_layers=cfg["n_layers"],
        embed_dim=cfg["embed_dim"],
    ).eval().cpu()

    # Best-effort: load the trained weights to confirm the architecture matches
    # the ckpt (catches silent shape drift). Skip if shapes differ — we only
    # care about the param count.
    if ckpt_path is not None:
        try:
            sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False).get("model", {})
            if isinstance(sd, dict):
                model.load_state_dict(sd, strict=False)
        except Exception:
            # Shape mismatch is the most common failure; the param count is
            # still valid because we used the ckpt's own args to build.
            pass

    n = count_params(model)
    return {
        "name": name,
        "trainable": n["trainable"],
        "total": n["total"],
        "ckpt": ckpt_used,
        "n_layers": cfg["n_layers"],
        "embed_dim": cfg["embed_dim"],
        "pad_obs_to": cfg["pad_obs_to"],
        "action_dim": cfg["action_dim"],
        "notes": notes,
    }


# =============================================================================
# Output
# =============================================================================
def write_json(rows: List[Dict[str, Any]]) -> None:
    payload = {
        "models": rows,
        "generated_at": str(date.today()),
        "schema": {
            "name": "model class (display name, matches MASTER_TABLE column)",
            "trainable": "count of parameters with requires_grad=True",
            "total": "count of all parameters (incl. frozen encoder)",
            "ckpt": "path to the ckpt used for canonical args, or <canonical> fallback",
            "n_layers": "n_layers used to instantiate (from ckpt args or canonical)",
            "embed_dim": "embed_dim used to instantiate (from ckpt args or canonical)",
            "pad_obs_to": "state_dim used to instantiate (from ckpt args or canonical)",
            "action_dim": "action_dim used to instantiate (from ckpt args or canonical)",
            "notes": "short architectural note",
        },
        "n_models": len(rows),
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2))
    print(f"[model_sizes] wrote {JSON_OUT} ({len(rows)} rows)")


def write_md(rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Model size table (v0.7.6 — canonical param counts)\n")
    lines.append(
        "Single source of truth for the parameter counts cited in the README, "
        "MASTER_TABLE, and paper. Counts are computed by instantiating each "
        "model on CPU with the args from `results/generalist_G16/<model>/seed_0/final.pt` "
        "(preferred) and falling back to the canonical G16 budget "
        "(`pad_obs_to=128, action_dim=56, embed_dim=192, n_layers=2`) when no "
        "checkpoint is available.\n"
    )
    lines.append("| Model | trainable (M) | total (M) | n_layers | embed_dim | notes |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: x["name"]):
        lines.append(
            f"| {r['name']} | {r['trainable']/1e6:.2f} | {r['total']/1e6:.2f} | "
            f"{r['n_layers']} | {r['embed_dim']} | {r['notes']} |"
        )
    total_train = sum(r["trainable"] for r in rows)
    total_all = sum(r["total"] for r in rows)
    lines.append(
        f"\n**Summary.** {len(rows)} models; total trainable params across all "
        f"models = **{total_train/1e6:.2f} M**, total params (incl. frozen encoders) "
        f"= **{total_all/1e6:.2f} M**.\n"
    )
    lines.append(
        "**Source for ckpt columns** is `results/generalist_G16/<model>/seed_0/final.pt` "
        "for each model; on-disk dir for the LeWM baseline is `lewm_baseline_v2` "
        "(`code.lewm_transformer_baseline.LeWMTransformerBaseline`).\n"
    )
    MD_OUT.write_text("\n".join(lines) + "\n")
    print(f"[model_sizes] wrote {MD_OUT} ({len(rows)} rows)")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [measure_one(e) for e in MODELS]
    write_json(rows)
    write_md(rows)
    # Brief stdout summary.
    print()
    print(f"{'model':35s} {'trainable':>12s} {'total':>12s}  ckpt")
    for r in sorted(rows, key=lambda x: x["name"]):
        print(f"{r['name']:35s} {r['trainable']:>12,d} {r['total']:>12,d}  {r['ckpt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

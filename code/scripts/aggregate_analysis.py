#!/usr/bin/env python
"""Aggregate analysis results into markdown tables.

Reads:
  - results/probe/<env>_<model>_<target>.json
  - results/event_align/<env>_<model>.json
  - results/flops/<model>.json

Writes:
  - results/aggregate/probe_table.md
  - results/aggregate/event_align_table.md
  - results/aggregate/flops_table.md
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from collections import defaultdict

PROBE_DIR = Path("/home/lx/snn/results/probe")
EVENT_DIR = Path("/home/lx/snn/results/event_align")
FLOPS_DIR = Path("/home/lx/snn/results/flops")
AGG_DIR = Path("/home/lx/snn/results/aggregate")
AGG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(p: Path):
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def aggregate_probe() -> None:
    """Build a markdown table: rows = (env, model, target), cells = R²."""
    rows = defaultdict(dict)  # (env, model) -> {target: r2}
    targets_set: set = set()
    models_set: set = set()
    envs_set: set = set()
    for p in sorted(PROBE_DIR.glob("*.json")):
        # filename: <env>_<model>_<target>.json
        # envs and model names have well-known forms; targets are
        # {position, velocity, contact, future_k, goal_direction}
        known_targets = {"position", "velocity", "contact", "future_k", "goal_direction"}
        stem = p.stem
        for tgt in known_targets:
            if stem.endswith("_" + tgt):
                env_model = stem[: -len(tgt) - 1]
                target = tgt
                break
        else:
            continue
        data = load_json(p)
        if data is None:
            continue
        if data.get("skipped"):
            r2 = float("nan")
        else:
            r2 = data.get("r2", float("nan"))
        rows[(env_model)][target] = r2
        targets_set.add(target)
        # split env_model -> env / model by the known model names
        for m in ("stjewm_v2", "stjewm_nogoal", "lewm_baseline_v2", "lewm_baseline_no_goal"):
            if env_model.endswith("_" + m):
                envs_set.add(env_model[: -len(m) - 1])
                models_set.add(m)
                break

    targets = sorted(targets_set)
    models = sorted(models_set)
    envs = sorted(envs_set)
    lines = ["# Linear Probing Results (R² score, higher = better)\n"]
    lines.append(f"Envs: {', '.join(envs)}")
    lines.append(f"Targets: {', '.join(targets)}\n")
    # Build a pivot: rows = env, cols = (model, target)
    cols = [(m, t) for m in models for t in targets]
    lines.append("| env | " + " | ".join(f"{m}\\{t}" for m, t in cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for env in envs:
        cells = []
        for m, t in cols:
            v = rows.get(f"{env}_{m}", {}).get(t, float("nan"))
            cells.append(f"{v:.3f}" if v == v else "—")
        lines.append(f"| {env} | " + " | ".join(cells) + " |")


    out = AGG_DIR / "probe_table.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[probe] wrote {out} ({len(rows)} rows)")


def aggregate_event_align() -> None:
    rows = []
    for p in sorted(EVENT_DIR.glob("*.json")):
        data = load_json(p)
        if data is None:
            continue
        rows.append((p.stem, data))
    lines = ["# Event Boundary Alignment\n"]
    lines.append("Pearson correlations between obs event strength and (latent / spike-rate).\n")
    lines.append("| env_model | corr_obs_latent | corr_obs_rate | n_steps |")
    lines.append("|---|---|---|---|")
    for name, d in rows:
        lines.append(f"| {name} | {d.get('corr_obs_latent', 'n/a'):.3f} | {d.get('corr_obs_rate', 'n/a'):.3f} | {d.get('n_steps', 'n/a')} |")
    out = AGG_DIR / "event_align_table.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[event] wrote {out} ({len(rows)} rows)")


def aggregate_flops() -> None:
    rows = []
    for p in sorted(FLOPS_DIR.glob("*.json")):
        data = load_json(p)
        if data is None:
            continue
        rows.append((p.stem, data))
    lines = ["# FLOPs / Efficiency\n"]
    lines.append("| model | dense (GMACs) | sparse (GMACs) | sparsity | n_params (M) |")
    lines.append("|---|---|---|---|---|")
    for name, d in rows:
        dense = d.get("dense_gmacs", float("nan"))
        sparse = d.get("sparse_gmacs", float("nan"))
        spar = d.get("sparsity_assumed", float("nan"))
        nparams = d.get("n_params", 0) / 1e6
        lines.append(f"| {name} | {dense:.2f} | {sparse:.2f} | {spar:.2f} | {nparams:.2f} |")
    out = AGG_DIR / "flops_table.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[flops] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    aggregate_probe()
    aggregate_event_align()
    aggregate_flops()
    print("\n[aggregate_analysis] done")

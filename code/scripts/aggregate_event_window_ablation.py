#!/usr/bin/env python
"""Aggregate event-window ablation results into a markdown table.

Reads results/aggregate/event_window_ablation/<env>_<model>.json files and
produces a single markdown table with one row per (env, model) cell and
columns for each ablation mode.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True,
                   help="Directory containing per-cell JSON files")
    p.add_argument("--out-md", required=True,
                   help="Output markdown table path")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all (env, model) cells
    cells: List[Tuple[str, str, dict]] = []
    for f in sorted(in_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        env = data.get("env", "?")
        model = data.get("model", "?")
        if data.get("skipped"):
            cells.append((env, model, {"skipped": True, "reason": data.get("reason", "skip")}))
        else:
            cells.append((env, model, data))

    # Build markdown
    lines = []
    lines.append("# Event-Window Causal Ablation")
    lines.append("")
    lines.append("**Causal claim**: trace components at event windows (steps where "
                 "||Δobs|| > median + 1·MAD) are used by the planner.")
    lines.append("")
    lines.append("**Test**: zero r_t at the post-step history-update `predict` call. "
                 "Compare env-SR (drop in pp) across 5 modes:")
    lines.append("- **baseline**: no ablation")
    lines.append("- **event_window**: zero trace at event-aligned steps")
    lines.append("- **non_event_window**: zero trace at matched low-Δobs steps")
    lines.append("- **random_window**: zero trace at random steps (same count as event_window)")
    lines.append("- **ablate_all**: zero trace at every step (sanity check: hook works?)")
    lines.append("")
    lines.append("**Causal claim supported** if event-window drop > non_event and random drops.")
    lines.append("")

    # Per-cell table
    lines.append("## Per-cell results")
    lines.append("")
    lines.append("| env | model | baseline env-SR | event_window Δ | non_event_window Δ | random_window Δ | ablate_all Δ | claim_supported |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for env, model, data in cells:
        if data.get("skipped"):
            lines.append(f"| {env} | {model} | — (skipped: {data['reason']}) | — | — | — | — | — |")
            continue
        res = data.get("results", {})
        if not res or "baseline" not in res:
            lines.append(f"| {env} | {model} | — (no baseline) | — | — | — | — | — |")
            continue
        baseline_env_sr = res["baseline"]["success_rate_env"]
        baseline_lewm_sr = res["baseline"]["success_rate_lewm"]
        baseline_cos = res["baseline"]["mean_cos_dist"]
        drops = data.get("drops", {})
        ev_d = drops.get("event_window", {}).get("env_sr_drop_pp", float("nan"))
        ne_d = drops.get("non_event_window", {}).get("env_sr_drop_pp", float("nan"))
        rd_d = drops.get("random_window", {}).get("env_sr_drop_pp", float("nan"))
        aa_d = drops.get("ablate_all", {}).get("env_sr_drop_pp", float("nan"))
        claim = data.get("causal_claim_supported", False)
        claim_str = "✓" if claim else "✗"
        lines.append(f"| {env} | {model} | {baseline_env_sr*100:.1f}% | {ev_d:+.2f}pp | "
                     f"{ne_d:+.2f}pp | {rd_d:+.2f}pp | {aa_d:+.2f}pp | {claim_str} |")

    lines.append("")
    lines.append("Where Δ = (mode_env_sr − baseline_env_sr) × 100 (negative = ablation hurts).")
    lines.append("")

    # Per-cell cos_dist table
    lines.append("## Per-cell cos_dist (continuous metric)")
    lines.append("")
    lines.append("| env | model | baseline cos_dist | event_window Δ | non_event Δ | random Δ | ablate_all Δ |")
    lines.append("|---|---|---|---|---|---|---|")
    for env, model, data in cells:
        if data.get("skipped"):
            lines.append(f"| {env} | {model} | — | — | — | — | — |")
            continue
        res = data.get("results", {})
        if not res or "baseline" not in res:
            continue
        baseline_cos = res["baseline"]["mean_cos_dist"]
        drops = data.get("drops", {})
        ev_d = drops.get("event_window", {}).get("cos_dist_increase", float("nan"))
        ne_d = drops.get("non_event_window", {}).get("cos_dist_increase", float("nan"))
        rd_d = drops.get("random_window", {}).get("cos_dist_increase", float("nan"))
        aa_d = drops.get("ablate_all", {}).get("cos_dist_increase", float("nan"))
        lines.append(f"| {env} | {model} | {baseline_cos:.4f} | {ev_d:+.4f} | "
                     f"{ne_d:+.4f} | {rd_d:+.4f} | {aa_d:+.4f} |")
    lines.append("")
    lines.append("Δ = mode_cos_dist − baseline_cos_dist (positive = ablation hurts).")
    lines.append("")

    # Per-cell lewm_sr table
    lines.append("## Per-cell lewm-SR drop (latent-space success)")
    lines.append("")
    lines.append("| env | model | baseline lewm-SR | event_window Δ | non_event Δ | random Δ | ablate_all Δ |")
    lines.append("|---|---|---|---|---|---|---|")
    for env, model, data in cells:
        if data.get("skipped"):
            lines.append(f"| {env} | {model} | — | — | — | — | — |")
            continue
        res = data.get("results", {})
        if not res or "baseline" not in res:
            continue
        baseline = res["baseline"]["success_rate_lewm"]
        drops = data.get("drops", {})
        ev_d = drops.get("event_window", {}).get("lewm_sr_drop_pp", float("nan"))
        ne_d = drops.get("non_event_window", {}).get("lewm_sr_drop_pp", float("nan"))
        rd_d = drops.get("random_window", {}).get("lewm_sr_drop_pp", float("nan"))
        aa_d = drops.get("ablate_all", {}).get("lewm_sr_drop_pp", float("nan"))
        lines.append(f"| {env} | {model} | {baseline*100:.1f}% | {ev_d:+.2f}pp | "
                     f"{ne_d:+.2f}pp | {rd_d:+.2f}pp | {aa_d:+.2f}pp |")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Read the table as follows:")
    lines.append("- A negative Δ in event_window with non-positive Δ in non_event_window/random_window "
                 "would support the causal claim (event-specific trace use).")
    lines.append("- A negative Δ in non_event_window or random_window (>= event_window) suggests the trace "
                 "stores generic / non-event information that is needed throughout.")
    lines.append("- A near-zero Δ in ablate_all indicates the trace has small effect on this model's readout "
                 "for this env (small ablation budget relative to readout weight).")
    lines.append("- A near-zero Δ across all modes is consistent with the trace being either uninformative "
                 "or redundant given the cell hidden state h.")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"[aggregate_event_window_ablation] wrote {out_path}")
    print(f"[aggregate_event_window_ablation] cells aggregated: {len(cells)}")


if __name__ == "__main__":
    main()
"""Compare complete G4/G8/G16 protocol-2 diagnostics from one audited run.

This source-only diagnostic comparison does not reuse historical env-SR tables
or carry forward pre-audit model-regime conclusions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from code.scripts.audited_results import TrainingAudit, fmt, load_auxiliary, metric, require, summarize, write_new_json

FAMILIES = ("generalist_G4", "generalist_G8", "generalist_G16")


def collect(records):
    models = sorted({row["model"] for row in records})
    envs = sorted({row["env"] for row in records})
    require(len(models) == 12 and len(envs) == 7 and len(records) == 3 * 12 * 7,
            "Scaling comparison requires all 36 checkpoints and seven planned diagnostic environments")
    require({row["family"] for row in records} == set(FAMILIES), "Wrong scaling families")
    expected = {(family, model, env) for family in FAMILIES for model in models for env in envs}
    actual = {(row["family"], row["model"], row["env"]) for row in records}
    require(actual == expected, "Scaling cells are missing, duplicated, or unpaired")
    rows = []
    for family in FAMILIES:
        for model in models:
            cells = [row for row in records if row["family"] == family and row["model"] == model]
            rows.append({"family": family, "model": model,
                         "metrics": {key: summarize(metric(cell, key, nullable=key != "divergence") for cell in cells)
                                     for key in ("divergence", "responsiveness", "event_rho")},
                         "sources": [{"path": cell["source"], "sha256": cell["source_sha256"]} for cell in cells]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = TrainingAudit(args.training_manifest)
    records = load_auxiliary(args.diagnostic_run, audit, ("scale",))
    rows = collect(records)
    lines = ["# Audited diagnostic scaling: G4 / G8 / G16", "",
             "Primary measurement: forward.emb. All seven planned environments are paired across 36 checkpoints.",
             "These are diagnostic means, not re-evaluated env-SR. No pre-audit regime classifications are carried forward.",
             "Undefined ratios/correlations are excluded only from their metric mean, with the denominator shown explicitly.",
             "One training seed: no seed confidence interval is estimable.", "",
             "| suite | model | divergence | responsiveness | response defined/expected | event_rho | rho defined/expected |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        metrics = row["metrics"]
        response, rho = metrics["responsiveness"], metrics["event_rho"]
        lines.append(f"| {row['family']} | {row['model']} | {fmt(metrics['divergence']['mean'], 5)} | "
                     f"{fmt(response['mean'])} | {response['n_defined']}/{response['n_expected']} | "
                     f"{fmt(rho['mean'])} | {rho['n_defined']}/{rho['n_expected']} |")
    audit.protect_output(args.out)
    audit.protect_output(args.out.with_suffix(".json"))
    write_new_json(args.out.with_suffix(".json"), {"status": "completed", "protocol_version": 2,
                   "measurement_object": "forward.emb", "training_manifest": str(audit.path),
                   "training_manifest_sha256": audit.digest, "planned_cells": 252, "seed_std": None,
                   "env_sr_evidence": "not evaluated by this diagnostic grid", "rows": rows})
    with args.out.open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

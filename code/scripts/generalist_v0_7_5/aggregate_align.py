"""Aggregate canonical event_rho from a complete audited diagnostic run."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from code.scripts.audited_results import TrainingAudit, fmt, load_auxiliary, metric, require, summarize, write_new_json


def collect(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["family"], record["split"], record["model"], record["env"])].append(record)
    rows = []
    for (family, split, model, env), cells in sorted(grouped.items()):
        require(len({cell["seed"] for cell in cells}) == len(cells), "Duplicate diagnostic seed")
        rows.append({"family": family, "split": split, "model": model, "env": env,
                     "event_rho": summarize(metric(cell, "event_rho", nullable=True) for cell in cells),
                     "seeds": [cell["seed"] for cell in cells],
                     "sources": [{"path": cell["source"], "sha256": cell["source_sha256"]} for cell in cells]})
    return rows


def write_table(records, output, audit):
    rows = collect(records)
    require(rows, "No planned alignment cells")
    lines = ["# Audited event alignment", "",
             "Primary measurement: forward.emb. Correlation: temporal Pearson event_rho.",
             "Undefined correlations remain null; defined/expected counts are explicit. One seed has no seed std estimate.", "",
             "| family | split | model | env | event_rho | seed std | defined/expected |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        stats = row["event_rho"]
        lines.append(f"| {row['family']} | {row['split']} | {row['model']} | {row['env']} | "
                     f"{fmt(stats['mean'])} | {fmt(stats['std'])} | {stats['n_defined']}/{stats['n_expected']} |")
    audit.protect_output(output)
    audit.protect_output(output.with_suffix(".json"))
    write_new_json(output.with_suffix(".json"), {"status": "completed", "protocol_version": 2,
                   "measurement_object": "forward.emb", "training_manifest": str(audit.path),
                   "training_manifest_sha256": audit.digest, "planned_cells": len(records), "rows": rows})
    with output.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-run", type=Path, required=True)
    parser.add_argument("--group", required=True, choices=("state", "scale", "rollouts"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = TrainingAudit(args.training_manifest)
    write_table(load_auxiliary(args.diagnostic_run, audit, (args.group,)), args.out, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

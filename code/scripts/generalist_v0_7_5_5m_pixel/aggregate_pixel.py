"""Summarize the exact audited 130-checkpoint, 13-environment pixel grid."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from code.scripts.audited_results import (
    TrainingAudit, fmt, metric, require, sha256, summarize, validate_pixel_grid, write_new_json,
)

METRICS = ("success_rate_env", "success_rate_lewm_005", "mean_cos_dist", "mean_phys_dist")


def collect(pixel_run, audit):
    return validate_pixel_grid(Path(pixel_run) / "grid_status.json", audit)


def summarize_rows(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["split"], row["model"])].append(row)
    rows = []
    for (split, model), cells in sorted(groups.items()):
        require(len(cells) == len({cell["env"] for cell in cells}) == 13, "Unpaired pixel environment coverage")
        rows.append({"split": split, "model": model, "n_envs": 13, "training_seeds": [0],
                     "training_seed_std": None,
                     "metrics": {key: summarize(metric(cell["metrics"], key) for cell in cells) for key in METRICS}})
    require(len(rows) == 130, "Incomplete pixel checkpoint coverage")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--pixel-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = TrainingAudit(args.training_manifest)
    records = collect(args.pixel_run, audit)
    rows = summarize_rows(records)
    lines = ["# Audited pixel evaluation", "",
             "All 130 checkpoints and 1690 environment cells passed final-generation, process-exit and output-hash checks.",
             "Models retain their actual pixel checkpoint identities, including stjewm and lewm_baseline.",
             "Means use all 13 planned environments. Environment dispersion is not training-seed uncertainty; only one training seed exists.", "",
             "| split | pixel model | environments | env-SR | latent SR@0.05 | cosine distance | physical distance |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        values = " | ".join(fmt(row["metrics"][key]["mean"]) for key in METRICS)
        lines.append(f"| {row['split']} | {row['model']} | 13 | {values} |")
    audit.protect_output(args.out)
    audit.protect_output(args.out.with_suffix(".json"))
    write_new_json(args.out.with_suffix(".json"), {
        "status": "completed", "training_manifest": str(audit.path), "training_manifest_sha256": audit.digest,
        "grid_status": str(args.pixel_run / "grid_status.json"),
        "grid_status_sha256": sha256(args.pixel_run / "grid_status.json"),
        "planned_cells": 1690, "rows": rows,
    })
    with args.out.open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

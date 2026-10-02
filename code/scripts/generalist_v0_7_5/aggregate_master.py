"""Summarize complete state-evaluation manifests; never scan checkpoint trees."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from code.scripts.audited_results import TrainingAudit, fmt, load_state_grid, metric, require, summarize, write_new_json

METRICS = ("success_rate_env", "success_rate_lewm_005", "mean_cos_dist", "mean_phys_dist")


def collect_runs(run_dirs, audit):
    rows = [row for run_dir in run_dirs for row in load_state_grid(run_dir, audit)]
    require(len({row["id"] for row in rows}) == len(rows), "Overlapping state manifests would duplicate cells")
    return rows


def summarize_rows(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["experiment"], row["split"], row["model"])].append(row)
    rows, paired = [], {}
    for (experiment, split, model), cells in sorted(groups.items()):
        lattice = {(cell["seed"], cell["env"]) for cell in cells}
        require(len(lattice) == len(cells), "Duplicate state model/seed/environment")
        reference = paired.setdefault((experiment, split), lattice)
        require(lattice == reference, f"Unpaired model coverage: {experiment}/{split}/{model}")
        seeds = sorted({cell["seed"] for cell in cells})
        env_sets = [{cell["env"] for cell in cells if cell["seed"] == seed} for seed in seeds]
        require(all(envs == env_sets[0] for envs in env_sets), "Unpaired seed environments")
        per_seed = [{"seed": seed, **{key: summarize(metric(cell["metrics"], key) for cell in cells if cell["seed"] == seed)["mean"]
                                      for key in METRICS}} for seed in seeds]
        rows.append({"experiment": experiment, "split": split, "model": model, "n_envs": len(env_sets[0]),
                     "seeds": seeds, "per_seed": per_seed,
                     "metrics": {key: summarize(row[key] for row in per_seed) for key in METRICS}})
    return rows


def write_summary(records, output, audit, title="Audited generalist state summary"):
    rows = summarize_rows(records)
    lines = ["# " + title, "", "All input manifests are complete and checkpoint/output hashes have been verified.",
             "Each environment is paired across models and seeds within its planned split. Mean/std are over training-seed means.",
             "A single training seed has undefined seed std; it does not have a zero-width confidence interval.", "",
             "| experiment | split | model | environments | training seeds | env-SR | latent SR@0.05 | cosine distance |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        values = " | ".join(fmt(row["metrics"][key]["mean"]) for key in METRICS[:3])
        lines.append(f"| {row['experiment']} | {row['split']} | {row['model']} | {row['n_envs']} | {row['seeds']} | {values} |")
    audit.protect_output(output)
    audit.protect_output(output.with_suffix(".json"))
    write_new_json(output.with_suffix(".json"), {"status": "completed", "training_manifest": str(audit.path),
                   "training_manifest_sha256": audit.digest, "planned_cells": len(records), "rows": rows,
                   "sources": [{"id": row["id"], "path": row["source"], "sha256": row["source_sha256"]} for row in records]})
    with output.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def parser(description=__doc__):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--training-manifest", type=Path, required=True)
    result.add_argument("--state-run", type=Path, action="append", required=True)
    result.add_argument("--out", type=Path, required=True)
    return result


def main():
    args = parser().parse_args()
    audit = TrainingAudit(args.training_manifest)
    write_summary(collect_runs(args.state_run, audit), args.out, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

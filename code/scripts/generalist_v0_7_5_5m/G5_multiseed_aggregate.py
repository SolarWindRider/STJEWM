"""Aggregate the complete paired 13-model, three-split, three-training-seed grid."""
from __future__ import annotations

import math
import statistics

from code.scripts.audited_results import TrainingAudit, fmt, metric, require, summarize, write_new_json
from code.scripts.generalist_v0_7_5.aggregate_master import collect_runs, parser
from code.scripts.generalist_v0_7_5_5m.repair_eval_grid import MODELS, SEED_SPLITS

METRICS = ("mean_cos_dist", "success_rate_lewm_005", "success_rate_env")
SEEDS = (0, 1, 2)


def seed_statistics(values):
    require(len(values) == 3, "A three-seed interval requires all three paired seeds")
    result = summarize(values)
    radius = 4.302652729911275 * result["std"] / math.sqrt(3)
    return {**result, "ci95_low": result["mean"] - radius, "ci95_high": result["mean"] + radius,
            "interval": "two-sided Student-t over three training-seed means, df=2"}


def aggregate(records, models):
    cells = {}
    for row in records:
        if row["experiment"] not in ("E1", "E2") or row["split"] not in SEED_SPLITS or row["model"] not in models:
            continue
        require((row["experiment"] == "E1" and row["seed"] == 0)
                or (row["experiment"] == "E2" and row["seed"] in (1, 2)), "Wrong logical seed source")
        key = (row["split"], row["model"], row["seed"], row["env"])
        require(key not in cells, f"Duplicate multiseed cell: {key}")
        cells[key] = row
    per_split, seed_means = [], {}
    for split in SEED_SPLITS:
        expected_envs = {key[3] for key in cells if key[:3] == (split, models[0], 0)}
        require(expected_envs, f"Missing seed-zero reference for {split}")
        for model in models:
            for seed in SEEDS:
                present = {key[3] for key in cells if key[:3] == (split, model, seed)}
                require(present == expected_envs, f"Unpaired/missing multiseed environments: {split}/{model}/seed{seed}")
                seed_means[(split, model, seed)] = {
                    metric_name: statistics.mean(metric(cells[(split, model, seed, env)]["metrics"], metric_name)
                                                 for env in sorted(expected_envs)) for metric_name in METRICS}
            per_split.append({"split": split, "model": model, "n_envs": len(expected_envs), "seeds": list(SEEDS),
                              "metrics": {key: seed_statistics([seed_means[(split, model, seed)][key] for seed in SEEDS])
                                          for key in METRICS}})
    per_model = []
    for model in models:
        per_seed = [{"seed": seed, **{key: statistics.mean(seed_means[(split, model, seed)][key] for split in SEED_SPLITS)
                                      for key in METRICS}} for seed in SEEDS]
        per_model.append({"model": model, "seeds": list(SEEDS), "per_seed": per_seed,
                          "metrics": {key: seed_statistics([row[key] for row in per_seed]) for key in METRICS}})
    return {"per_split_model": per_split, "per_model_aggregate": per_model,
            "splits": list(SEED_SPLITS), "models": list(models), "seeds": list(SEEDS), "planned_cells": len(cells)}


def publish(args, models, title):
    audit = TrainingAudit(args.training_manifest)
    records = collect_runs(args.state_run, audit)
    result = aggregate(records, models)
    result.update(status="completed", training_manifest=str(audit.path), training_manifest_sha256=audit.digest,
                  sources=[{"id": row["id"], "path": row["source"], "sha256": row["source_sha256"]} for row in records])
    lines = ["# " + title, "", "All three training seeds and all planned environments are paired before averaging.",
             "Environment means are averaged across the three splits within each seed; uncertainty is computed over the three seed means.", "",
             "| model | cosine mean | seed std | 95% Student-t interval | latent SR@0.05 | env-SR |",
             "|---|---|---|---|---|---|"]
    for row in result["per_model_aggregate"]:
        stats = row["metrics"]["mean_cos_dist"]
        lines.append(f"| {row['model']} | {fmt(stats['mean'])} | {fmt(stats['std'])} | "
                     f"[{fmt(stats['ci95_low'])}, {fmt(stats['ci95_high'])}] | "
                     f"{fmt(row['metrics']['success_rate_lewm_005']['mean'])} | {fmt(row['metrics']['success_rate_env']['mean'])} |")
    audit.protect_output(args.out)
    audit.protect_output(args.out.with_suffix(".json"))
    write_new_json(args.out.with_suffix(".json"), result)
    with args.out.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    publish(parser(__doc__).parse_args(), tuple(MODELS), "Audited G5 three-seed comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

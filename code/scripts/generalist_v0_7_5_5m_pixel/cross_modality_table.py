"""Describe paired state/pixel cells from complete audited primary manifests.

The goals, initialization and planning protocols differ; this is not a
controlled modality-only comparison and emits no prewritten model verdicts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from code.scripts.audited_results import TrainingAudit, fmt, metric, require, sha256, write_new_json
from code.scripts.generalist_v0_7_5.aggregate_master import collect_runs
from code.scripts.generalist_v0_7_5_5m_pixel.aggregate_pixel import collect

MODEL_MAP = {"stjewm_trace_only": "stjewm", "lewm_baseline_v2": "lewm_baseline"}
ENV_MAP = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum"}
METRICS = ("success_rate_env", "success_rate_lewm_005", "mean_cos_dist")


def paired_records(state_runs, pixel_run, audit):
    state = collect_runs(state_runs, audit)
    pixel = collect(pixel_run, audit)
    pixel_envs = {row["env"] for row in pixel}
    pixels = {(row["split"], row["model"], row["env"]): row for row in pixel}
    require(len(pixels) == len(pixel), "Duplicate pixel comparison cell")
    pairs, excluded_state, used_pixel = [], [], set()
    for row in state:
        if row["experiment"] != "E1":
            continue
        require(row["seed"] == 0, "Primary comparison requires the original seed-zero state grid")
        env = ENV_MAP.get(row["env"], row["env"])
        if env not in pixel_envs:
            excluded_state.append({"id": row["id"], "reason": "no equivalent native pixel condition in the planned grid"})
            continue
        model = MODEL_MAP.get(row["model"], row["model"])
        key = (row["split"], model, env)
        require(key in pixels and key not in used_pixel, f"Missing/duplicate paired pixel cell: {key}")
        used_pixel.add(key)
        other = pixels[key]
        pairs.append({"split": row["split"], "state_model": row["model"], "pixel_model": model,
                      "env": env, "state_env": row["env"], "seed": 0,
                      "state": {key: metric(row["metrics"], key) for key in METRICS},
                      "pixel": {key: metric(other["metrics"], key) for key in METRICS},
                      "state_budget": {key: row["metrics"][key] for key in ("horizon", "eval_budget", "n_episodes")},
                      "pixel_budget": {key: other["metrics"][key] for key in ("horizon", "eval_budget", "n_episodes")},
                      "sources": {"state": row["source"], "state_sha256": row["source_sha256"],
                                  "pixel": other["source"], "pixel_sha256": other["source_sha256"]}})
    require(pairs, "The supplied complete state manifests have no primary E1 comparison cells")
    lattices = {}
    for model in sorted({row["state_model"] for row in pairs}):
        lattices[model] = {(row["split"], row["env"]) for row in pairs if row["state_model"] == model}
    require(len(lattices) == 13 and all(cells == next(iter(lattices.values())) for cells in lattices.values()),
            "Unpaired model/environment/split comparison coverage")
    exclusions = {"state": excluded_state,
                  "pixel": [{"split": split, "model": model, "env": env,
                             "reason": "no matching E1 state cell in its planned split"}
                            for split, model, env in sorted(set(pixels) - used_pixel)]}
    return pairs, exclusions


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--training-manifest", type=Path, required=True)
    result.add_argument("--state-run", type=Path, action="append", required=True)
    result.add_argument("--pixel-run", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    return result


def publish(args, audit, pairs, exclusions, lines, summaries=None):
    audit.protect_output(args.out)
    audit.protect_output(args.out.with_suffix(".json"))
    write_new_json(args.out.with_suffix(".json"), {
        "status": "completed", "training_manifest": str(audit.path), "training_manifest_sha256": audit.digest,
        "pixel_grid_sha256": sha256(args.pixel_run / "grid_status.json"),
        "comparison": "descriptive paired native environments; different goal/init/planning protocols",
        "paired_cells": len(pairs), "excluded_unpaired_scope": exclusions,
        "training_seed_std": None, "rows": pairs, "summaries": summaries,
    })
    with args.out.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parser().parse_args()
    audit = TrainingAudit(args.training_manifest)
    pairs, exclusions = paired_records(args.state_run, args.pixel_run, audit)
    lines = ["# Audited paired state/pixel cells", "",
             "Descriptive only: state uses offline future-state goals; pixel uses fixed rendered physical goals.",
             "Initialization, horizon and replanning protocols differ. These results do not isolate the effect of modality.",
             "Unpaired planned conditions are enumerated in the JSON companion, not silently averaged together.",
             "One training seed: no confidence interval is estimable.", "",
             "| split | state model | pixel model | env | state env-SR | pixel env-SR | state cos | pixel cos |",
             "|---|---|---|---|---|---|---|---|"]
    for row in pairs:
        lines.append(f"| {row['split']} | {row['state_model']} | {row['pixel_model']} | {row['env']} | "
                     f"{fmt(row['state']['success_rate_env'])} | {fmt(row['pixel']['success_rate_env'])} | "
                     f"{fmt(row['state']['mean_cos_dist'])} | {fmt(row['pixel']['mean_cos_dist'])} |")
    publish(args, audit, pairs, exclusions, lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run or aggregate the complete audited 12-model latent-env gradient sweep."""
from __future__ import annotations

import math
import statistics
import argparse
from pathlib import Path

from code.scripts.audited_results import TrainingAudit, fmt, load_json, metric, require, write_new_json

G16_CKPTS = (
    "stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only",
    "stjewm_no_trace", "stjewm_hidden_leak", "stjewm_membrane_readout",
    "alif_timecell_baseline", "stacked_lif_trace", "stacked_lif_free",
    "lewm_baseline_v2", "gru_baseline", "mlp_baseline",
)
ENVS = ("cheetah", "walker", "reacher", "finger")


def read_cell(path, audit, model, env):
    payload = load_json(path)
    require(payload.get("status") == "completed" and payload.get("protocol_version") == 2
            and payload.get("measurement_object") == "forward.emb" and payload.get("weights_loaded_strict") is True,
            f"Obsolete latent-env-gradient producer output: {path}")
    require((payload["model"], payload["env"]) == (model, env), f"Identity mismatch: {path}")
    checkpoint = audit.results_root / "generalist_G16" / model / "seed_0" / "final.pt"
    audit.validate_provenance(payload, checkpoint)
    require(str(Path(payload["ckpt"]).resolve()) == payload["repair_provenance"]["checkpoint"], "Wrong checkpoint")
    require(payload.get("n_steps", 0) > 1, "Missing correlation sample count")
    for field in ("mean_abs_corr", "mean_corr"):
        metric(payload, field, nullable=True)
    values = payload["all_corrs"]
    require(len(values) == payload["n_steps"], "Gradient coverage mismatch")
    valid = [value for value in values if value is not None]
    require(all(type(value) in (int, float) and math.isfinite(value) and -1.000001 <= value <= 1.000001
                for value in valid), "Invalid gradient correlation")
    require(payload["n_valid"] == len(valid) and payload["n_undefined"] == len(values) - len(valid),
            "Undefined-gradient counts disagree with observations")
    for key, expected in (("mean_corr", statistics.mean(valid) if valid else None),
                          ("mean_abs_corr", statistics.mean(map(abs, valid)) if valid else None)):
        actual = metric(payload, key, nullable=True)
        require(actual is None if expected is None else actual is not None and math.isclose(actual, expected, abs_tol=1e-12),
                f"Gradient aggregate mismatch: {key}")
    return payload


def aggregate(out_dir, table_path, audit, out_json_path):
    rows, cells = [], {}
    for model in G16_CKPTS:
        for env in ENVS:
            path = Path(out_dir) / model / f"{env}.json"
            payload = read_cell(path, audit, model, env)
            rows.append(payload)
            cells[(model, env)] = (payload["mean_abs_corr"], payload["mean_corr"], payload["n_undefined"])
    lines = ["# Audited latent-environment gradient correlation", "",
             "Both gradients are evaluated at the same restored native state with a strict-load audited checkpoint.",
             "The physical objective is negative native goal-state L2 distance, not environment reward.",
             "Zero-gradient windows are reported as undefined counts, never as zero correlation.", "",
             "| model | " + " | ".join(f"{env}: mean absolute / signed / undefined" for env in ENVS) + " |",
             "|---|" + "|".join(["---"] * len(ENVS)) + "|"]
    for model in G16_CKPTS:
        values = " | ".join(f"{fmt(cells[(model, env)][0])} / {fmt(cells[(model, env)][1])} / {cells[(model, env)][2]}" for env in ENVS)
        lines.append(f"| {model} | {values} |")
    audit.protect_output(table_path)
    audit.protect_output(out_json_path)
    write_new_json(out_json_path, {"status": "completed", "training_manifest": str(audit.path),
                   "training_manifest_sha256": audit.digest, "planned_cells": len(rows), "rows": rows})
    Path(table_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(table_path).open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--table-path", type=Path, required=True)
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    audit = TrainingAudit(args.training_manifest)
    if not args.aggregate_only:
        from code.scripts.utility.latent_env_grad import run_one
        audit.protect_output(args.out_root)
        for model in G16_CKPTS:
            checkpoint = audit.results_root / "generalist_G16" / model / "seed_0" / "final.pt"
            for env in ENVS:
                run_one(checkpoint, model, env, args.n_steps, args.device,
                        args.out_root / model / f"{env}.json", args.training_manifest)
    aggregate(args.out_root, args.table_path, audit, args.json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

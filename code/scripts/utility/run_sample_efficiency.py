"""Run or aggregate the complete audited 12-model sample-efficiency sweep.

Cells missing from a completed producer run fail the table; there is no
silently skipped checkpoint, environment, or fraction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from code.scripts.audited_results import TrainingAudit, load_json, metric, require, write_new_json

G16_CKPTS = (
    "stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only",
    "stjewm_no_trace", "stjewm_hidden_leak", "stjewm_membrane_readout",
    "alif_timecell_baseline", "stacked_lif_trace", "stacked_lif_free",
    "lewm_baseline_v2", "gru_baseline", "mlp_baseline",
)
ENVS = ("cheetah", "walker", "reacher", "finger")
FRACTION_KEYS = ("0.010", "0.050", "0.100", "0.250", "1.000")


def read_cell(path, audit, model, env):
    payload = load_json(path)
    require(payload.get("status") == "completed" and payload.get("protocol_version") == 2
            and payload.get("measurement_object") == "forward.emb" and payload.get("weights_loaded_strict") is True,
            f"Obsolete sample-efficiency producer output: {path}")
    require((payload["model"], payload["env"]) == (model, env), f"Identity mismatch: {path}")
    checkpoint = audit.results_root / "generalist_G16" / model / "seed_0" / "final.pt"
    audit.validate_provenance(payload, checkpoint)
    require(str(Path(payload["ckpt"]).resolve()) == payload["repair_provenance"]["checkpoint"], "Wrong checkpoint")
    fractions = payload["per_fraction"]
    require(set(fractions) == set(FRACTION_KEYS), f"Incomplete fraction sweep: {path}")
    heldout = payload.get("heldout_indices")
    require(isinstance(heldout, list) and heldout and len(set(heldout)) == len(heldout) == payload["n_steps"],
            "Missing/invalid fixed held-out indices")
    n_total = payload["n_total"]
    pool_size = n_total - len(heldout)
    require(payload["training_pool_size"] == pool_size > 0
            and payload["fraction_denominator"] == "nonheldout_training_windows", "Wrong fraction denominator")
    require(all(type(index) is int and 0 <= index < n_total for index in heldout), "Invalid held-out index")
    training = payload.get("training_indices_by_fraction", {})
    require(set(training) == set(FRACTION_KEYS), "Missing training-index audit")
    for key, indices in training.items():
        require(len(indices) == len(set(indices)) == max(1, int(pool_size * float(key))),
                f"Wrong training size for fraction {key}")
        require(all(type(index) is int and 0 <= index < n_total for index in indices), "Invalid training index")
        require(not set(indices) & set(heldout), f"Fraction {key} leaked held-out samples into training")
    require(set(training["1.000"]) | set(heldout) == set(range(n_total)), "Full fraction omits legal training windows")
    for earlier, later in zip(FRACTION_KEYS, FRACTION_KEYS[1:]):
        require(set(training[earlier]) <= set(training[later]), "Unpaired training fractions")
    for key, row in fractions.items():
        require((row["data_fraction"], row["n_train"]) == (float(key), len(training[key])),
                f"Fraction identity mismatch: {path}/{key}")
        require(row["n_eval"] == len(heldout) and row["training_pool_size"] == pool_size, "Fraction evaluation coverage mismatch")
        require(0 <= metric(row, "env_success") <= 1 and 0 <= metric(row, "mean_cos_dist_terminal") <= 1
                and metric(row, "mean_phys_dist") >= 0, "Invalid held-out policy metric")
        for field in ("env_success", "mean_phys_dist", "mean_cos_dist_terminal"):
            require(metric(row, field) is not None, f"Undefined {field}: {path}/{key}")
    return payload


def aggregate(out_dir, table_path, audit, out_json_path):
    rows, lines = [], ["# Audited frozen-encoder sample efficiency", "",
                       "Every cell uses a strict-load audited checkpoint, an env-native policy rollout, and a fixed",
                       "disjoint held-out set drawn once before any training fraction. Success is env-native.", "",
                       "| env | model | fraction | env-native SR | physical distance | terminal cosine distance |",
                       "|---|---|---|---|---|---|"]
    paired_holdouts = {}
    for env in ENVS:
        for model in G16_CKPTS:
            payload = read_cell(Path(out_dir) / model / f"{env}.json", audit, model, env)
            signature = (payload["heldout_indices"], payload["training_indices_by_fraction"], payload["n_total"])
            require(env not in paired_holdouts or paired_holdouts[env] == signature, "Models used different train/holdout partitions")
            paired_holdouts[env] = signature
            rows.append(payload)
            for key in FRACTION_KEYS:
                row = payload["per_fraction"][key]
                lines.append(f"| {env} | {model} | {key} | {row['env_success']:.3f} | "
                             f"{row['mean_phys_dist']:.4f} | {row['mean_cos_dist_terminal']:.4f} |")
    audit.protect_output(table_path)
    audit.protect_output(out_json_path)
    write_new_json(out_json_path, {"status": "completed", "training_manifest": str(audit.path),
                   "training_manifest_sha256": audit.digest, "planned_cells": len(G16_CKPTS) * len(ENVS), "rows": rows})
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
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--fractions", default="0.01,0.05,0.1,0.25,1.0")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    audit = TrainingAudit(args.training_manifest)
    if not args.aggregate_only:
        from code.scripts.utility.sample_efficiency import run_one
        audit.protect_output(args.out_root)
        fractions = tuple(float(value) for value in args.fractions.split(","))
        require({f"{value:.3f}" for value in fractions} == set(FRACTION_KEYS), "The complete five-fraction sweep is required")
        for model in G16_CKPTS:
            checkpoint = audit.results_root / "generalist_G16" / model / "seed_0" / "final.pt"
            for env in ENVS:
                run_one(checkpoint, model, env, args.n_steps, fractions, args.device,
                        args.out_root / model / f"{env}.json", args.training_manifest)
    aggregate(args.out_root, args.table_path, audit, args.json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run or aggregate the complete audited 3-model x 3-data-budget grid."""
from __future__ import annotations

import argparse
from pathlib import Path

from code.scripts.audited_results import TrainingAudit, fmt, load_json, metric, read_diagnostic, require, sha256, validate_closed_loop, write_new_json
from code.scripts.utility.budget_scaling import DMC_ENVS, aggregate, evaluate_budget, native_env_id

MODELS = ("stjewm_trace_only", "stjewm_spike_only", "mlp_baseline")
FRACS = (0.5, 1.0, 2.0)


def aggregate_table(out_root, table_path, audit):
    rows = []
    for model in MODELS:
        for frac in FRACS:
            path = out_root / model / str(frac) / "summary.json"
            row = load_json(path)
            require(row.get("status") == "completed" and row.get("protocol_version") == 2
                    and row.get("measurement_object") == "forward.emb", f"Incomplete/obsolete budget cell: {path}")
            require((row["model"], row["frac"]) == (model, frac), f"Wrong budget cell: {path}")
            family = Path("generalist_G16") / model if frac == 1.0 else Path("generalist_G16_compression") / model / str(frac)
            checkpoint = audit.results_root / family / f"seed_{row['seed']}" / "final.pt"
            audit.validate_provenance(row, checkpoint)
            for cell in row["per_env"]:
                for prefix in ("eval", "diagnostic"):
                    require(sha256(cell[prefix + "_source"]) == cell[prefix + "_sha256"], f"Changed {prefix} result")
                evaluation = load_json(cell["eval_source"])
                validate_closed_loop(evaluation, episodes=row["repair_provenance"]["budget"]["episodes"])
                audit.validate_provenance(evaluation, checkpoint)
                require(evaluation["env_id"] == native_env_id(cell["env"])
                        and evaluation["protocol"].get("checkpoint") == str(checkpoint), "Budget evaluation identity mismatch")
                require(metric(evaluation, "success_rate_env") == cell["env-SR"], "Copied budget success disagrees with raw result")
                diagnostic = read_diagnostic(cell["diagnostic_source"], audit, checkpoint,
                                             model=model, env=cell["env"], seed=row["seed"])
                require(all(cell[short] == diagnostic[long] for short, long in
                            (("div", "divergence"), ("resp", "responsiveness"), ("event_rho", "event_rho"))),
                        "Copied budget diagnostic disagrees with raw result")
            require(row["n_envs"] == len(DMC_ENVS) and row["metrics"] == aggregate(row["per_env"]), "Budget coverage/metrics mismatch")
            rows.append(row)
    require(len({row["seed"] for row in rows}) == 1, "Unpaired training seeds")
    lines = ["# Audited training-data-budget scaling", "",
             "Means use the six planned environments. Undefined diagnostics remain null and their counts are shown.",
             "Environment dispersion is not a confidence interval; one training seed has no seed uncertainty estimate.", "",
             "| model | fraction | env-SR | divergence | responsiveness | event_rho | rho defined/expected |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        metrics = row["metrics"]
        values = " | ".join(fmt(metrics[key]["mean"]) for key in ("env-SR", "div", "resp", "event_rho"))
        rho = metrics["event_rho"]
        lines.append(f"| {row['model']} | {row['frac']} | {values} | {rho['n_defined']}/{rho['n_expected']} |")
    audit.protect_output(table_path)
    json_path = table_path.with_suffix(".json")
    audit.protect_output(json_path)
    write_new_json(json_path, {"status": "completed", "training_manifest": str(audit.path),
                               "training_manifest_sha256": audit.digest, "planned_cells": 9,
                               "seed_std": None, "rows": rows})
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--table-path", type=Path, required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    audit = TrainingAudit(args.training_manifest)
    if not args.aggregate_only:
        audit.protect_output(args.out_root)
        for model in MODELS:
            for frac in FRACS:
                if frac == 1.0:
                    checkpoint = audit.results_root / "generalist_G16" / model / f"seed_{args.seed}" / "final.pt"
                else:
                    checkpoint = audit.results_root / "generalist_G16_compression" / model / str(frac) / f"seed_{args.seed}" / "final.pt"
                evaluate_budget(model, frac, checkpoint, args.out_root / model / str(frac), audit,
                                seed=args.seed, device=args.device)
    aggregate_table(args.out_root, args.table_path, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

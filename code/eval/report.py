"""Render explicitly selected, complete audited state manifests (never archive scans)."""
from code.scripts.audited_results import TrainingAudit, fmt, metric, write_new_json
from code.scripts.generalist_v0_7_5.aggregate_master import collect_runs, parser


def format_table(results):
    lines = ["# Audited per-environment state evaluation", "",
             "Every row is selected by a completed plan and verified against its checkpoint and output hashes.",
             "Training model/seed identity is manifest-derived. Per-row rollout variation is not a training-seed confidence interval.", "",
             "| experiment | split | environment | model | training seed | episodes | latent SR@0.05 | env-SR | cosine distance | physical distance |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in results:
        payload = row["metrics"]
        values = " | ".join(fmt(metric(payload, key)) for key in
                            ("success_rate_lewm_005", "success_rate_env", "mean_cos_dist", "mean_phys_dist"))
        lines.append(f"| {row['experiment']} | {row['split']} | {row['env']} | {row['model']} | {row['seed']} | "
                     f"{payload['n_episodes']} | {values} |")
    return "\n".join(lines) + "\n"


def main():
    args = parser(__doc__).parse_args()
    audit = TrainingAudit(args.training_manifest)
    results = collect_runs(args.state_run, audit)
    audit.protect_output(args.out)
    audit.protect_output(args.out.with_suffix(".json"))
    write_new_json(args.out.with_suffix(".json"), {"status": "completed", "training_manifest": str(audit.path),
                   "training_manifest_sha256": audit.digest, "planned_cells": len(results), "rows": results})
    with args.out.open("x") as handle:
        handle.write(format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

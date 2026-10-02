"""Render every planned state cell from complete audited evaluation manifests."""
from code.scripts.audited_results import TrainingAudit, fmt, metric, write_new_json
from code.scripts.generalist_v0_7_5.aggregate_master import collect_runs, parser, summarize_rows


def main():
    args = parser(__doc__).parse_args()
    audit = TrainingAudit(args.training_manifest)
    records = collect_runs(args.state_run, audit)
    summaries = summarize_rows(records)
    lines = ["# Audited state evaluation cells", "",
             "Checkpoint locations and logical training seeds come from the manifests, not the evaluation directory names.",
             "All planned cells are complete. A single training seed has no seed confidence interval.", "",
             "| experiment | split | model | training seed | env | env-SR | latent SR@0.05 | cosine distance | physical distance |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in records:
        values = " | ".join(fmt(metric(row["metrics"], key)) for key in
                            ("success_rate_env", "success_rate_lewm_005", "mean_cos_dist", "mean_phys_dist"))
        lines.append(f"| {row['experiment']} | {row['split']} | {row['model']} | {row['seed']} | {row['env']} | {values} |")
    audit.protect_output(args.out)
    audit.protect_output(args.out.with_suffix(".json"))
    write_new_json(args.out.with_suffix(".json"), {"status": "completed", "training_manifest": str(audit.path),
                   "training_manifest_sha256": audit.digest, "planned_cells": len(records),
                   "evals": records, "summaries": summaries})
    with args.out.open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

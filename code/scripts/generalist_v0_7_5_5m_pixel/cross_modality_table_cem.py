"""Summarize paired native-environment cells; never read obsolete CEM summary files."""
from collections import defaultdict

from code.scripts.audited_results import TrainingAudit, fmt, summarize
from code.scripts.generalist_v0_7_5_5m_pixel.cross_modality_table import METRICS, paired_records, parser, publish


def main():
    args = parser().parse_args()
    audit = TrainingAudit(args.training_manifest)
    pairs, exclusions = paired_records(args.state_run, args.pixel_run, audit)
    groups = defaultdict(list)
    for row in pairs:
        groups[(row["state_model"], row["pixel_model"])].append(row)
    rows = []
    lines = ["# Audited paired state/pixel CEM summary", "",
             "Both sides use CEM, but their goals, initialization, horizon and replanning protocols differ.",
             "This is descriptive, not a controlled modality-only experiment. Every mean uses the same paired split/environment cells.",
             "Cell dispersion is not a seed confidence interval; there is one training seed per modality.", "",
             "| state model | pixel model | paired cells | state env-SR | pixel env-SR | state cosine | pixel cosine |",
             "|---|---|---|---|---|---|---|"]
    for (state_model, pixel_model), cells in sorted(groups.items()):
        metrics = {modality: {key: summarize(cell[modality][key] for cell in cells) for key in METRICS}
                   for modality in ("state", "pixel")}
        rows.append({"state_model": state_model, "pixel_model": pixel_model,
                     "paired_cells": len(cells), "metrics": metrics, "training_seed_std": None})
        lines.append(f"| {state_model} | {pixel_model} | {len(cells)} | "
                     f"{fmt(metrics['state']['success_rate_env']['mean'])} | {fmt(metrics['pixel']['success_rate_env']['mean'])} | "
                     f"{fmt(metrics['state']['mean_cos_dist']['mean'])} | {fmt(metrics['pixel']['mean_cos_dist']['mean'])} |")
    publish(args, audit, pairs, exclusions, lines, summaries=rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

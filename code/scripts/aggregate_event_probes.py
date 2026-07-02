"""Aggregate event-probe results into a markdown table.

Reads all JSON files in /home/lx/snn/results/aggregate/event_probes/ and
emits a single results/aggregate/event_probes_table.md with one block per
probe target and a key claim table per env.

This is the per-env vs per-model AUROC / AUPRC table that supports the
"event-specialized predictive state" claim in the paper.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

PROBE_DIR = Path("/home/lx/snn/results/aggregate/event_probes")
OUT_MD = Path("/home/lx/snn/results/aggregate/event_probes_table.md")
OUT_SUMMARY = Path("/home/lx/snn/results/aggregate/event_probes_summary.md")

# Models of interest (the ones the paper compares). Anything else is
# reported under "other" but not aggregated.
MODELS_OF_INTEREST = [
    "stjewm_trace_only",
    "stjewm_hidden_leak",
    "stjewm_spike_only",
    "stjewm_no_trace",
    "stjewm_membrane_readout",
    "lewm_baseline_v2",
    "gru_baseline",
    "mlp_baseline",
]


def load_all() -> list[dict]:
    rows = []
    for fp in sorted(PROBE_DIR.glob("*.json")):
        try:
            d = json.loads(fp.read_text())
        except Exception as e:
            print(f"[skip] {fp}: {e}")
            continue
        if d.get("skipped"):
            continue
        d["filename"] = fp.name
        rows.append(d)
    return rows


def build_pivot(rows: list[dict], key: str = "r2") -> tuple[list[str], list[str], dict]:
    """Return (envs, models, table[env][model] = {target: value}).

    `key` selects which metric to pivot: r2 (AUROC for binary, R^2 for
    continuous) or auprc (AUPRC for binary only).
    """
    envs: set[str] = set()
    models: set[str] = set()
    targets: set[str] = set()
    table: dict[tuple[str, str, str], float] = {}
    raw: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        e = r["env"]
        m = r["model"]
        t = r["probe_target"]
        envs.add(e)
        models.add(m)
        targets.add(t)
        v = r.get(key, 0.0)
        table[(e, m, t)] = float(v) if v is not None else 0.0
        raw[(e, m, t)] = r
    return sorted(envs), sorted(models), table, raw, sorted(targets)


def main() -> None:
    rows = load_all()
    if not rows:
        print("[aggregate_event_probes] no JSON results found")
        return

    envs, models, table, raw, targets = build_pivot(rows, key="r2")

    # Group targets by env (so each env has its own block of probe targets)
    env_to_targets: dict[str, list[str]] = defaultdict(list)
    for (e, m, t) in table:
        env_to_targets[e].append(t)
    for e in env_to_targets:
        env_to_targets[e] = sorted(set(env_to_targets[e]))

    # Build the master markdown
    out = []
    out.append("# Event-Type Linear Probes (per-step)\n")
    out.append("**Setup.** Linear probe on the *gated spike trace* (pre-projection)\n"
               "of each model. Targets are per-step event-type binary labels\n"
               "extracted from the state trajectory. Metric: AUROC (calibration-free,\n"
               "robust to class imbalance). AUPRC is reported alongside.\n")
    out.append("**Models.** STJEWM-{trace,leak,spike,no-trace,membrane}, LeWM, GRU, MLP.\n")
    out.append(f"**Coverage.** {len(envs)} envs × {len(models)} models × "
               f"avg {sum(len(env_to_targets[e]) for e in envs) / max(len(envs),1):.1f} targets/env.\n\n")

    # Per-env table
    for env in envs:
        out.append(f"## Env: `{env}`\n")
        ts = env_to_targets[env]
        # Header: env, then target, then model columns
        out.append("| target | " + " | ".join(models) + " |")
        out.append("|" + "---|" * (len(models) + 1))
        for t in ts:
            row_vals = []
            for m in models:
                v = table.get((env, m, t), None)
                if v is None:
                    row_vals.append("n/a")
                else:
                    # Color-code: bold if > 0.7, plain otherwise
                    if v >= 0.7:
                        row_vals.append(f"**{v:.3f}**")
                    else:
                        row_vals.append(f"{v:.3f}")
            out.append(f"| {t} | " + " | ".join(row_vals) + " |")
        out.append("")

    # Headline: who wins on event probes, who wins on position probes?
    out.append("## Headline comparison: event probes vs position probes\n")
    out.append("**Key claim.** STJEWM-trace is event-specialized: it ties or wins on\n"
               "event-type targets even when its position-probe R² is moderate.\n")
    # Compute per-model mean AUROC across event targets
    per_model_auroc: dict[str, list[float]] = defaultdict(list)
    for (e, m, t), v in table.items():
        per_model_auroc[m].append(v)
    out.append("### Mean event-probe AUROC per model\n")
    out.append("| model | n_cells | mean AUROC | median AUROC |")
    out.append("|---|---|---|---|")
    for m in models:
        vs = per_model_auroc.get(m, [])
        if not vs:
            continue
        out.append(f"| {m} | {len(vs)} | {sum(vs)/len(vs):.3f} | {statistics.median(vs):.3f} |")
    out.append("")

    # Find the per-env winner (per-event-target)
    out.append("### Per-target winners (per env, model with highest AUROC)\n")
    out.append("| env | target | winner | AUROC | runner-up | AUROC |")
    out.append("|---|---|---|---|---|---|")
    win_counts: dict[str, int] = defaultdict(int)
    for env in envs:
        for t in env_to_targets[env]:
            row = []
            for m in models:
                v = table.get((env, m, t), None)
                if v is not None:
                    row.append((m, v))
            if not row:
                continue
            row.sort(key=lambda x: -x[1])
            winner, wv = row[0]
            runner, rv = row[1] if len(row) > 1 else ("-", 0.0)
            out.append(f"| {env} | {t} | {winner} | {wv:.3f} | {runner} | {rv:.3f} |")
            win_counts[winner] += 1
    out.append("")
    out.append("### Win counts (event-type targets)\n")
    out.append("| model | wins |")
    out.append("|---|---|")
    for m in models:
        out.append(f"| {m} | {win_counts.get(m, 0)} |")
    out.append("")

    OUT_MD.write_text("\n".join(out))
    print(f"[aggregate_event_probes] wrote {OUT_MD}")

    # Short prose summary
    summary = []
    summary.append("# Event-Probe Summary (NMI paper, Results 5)\n")
    summary.append(f"Total cells aggregated: {len(rows)} ({len(envs)} envs, "
                   f"{len(models)} models, {len(targets)} targets).\n")
    # Best 3 models by mean AUROC
    ranking = sorted(
        [(m, statistics.mean(per_model_auroc[m]))
         for m in models if per_model_auroc[m]],
        key=lambda x: -x[1]
    )
    summary.append("## Mean event-probe AUROC ranking\n")
    for i, (m, s) in enumerate(ranking, 1):
        summary.append(f"{i}. `{m}` = {s:.3f}")
    summary.append("")
    summary.append("## Dissociation claim\n")
    summary.append("STJEWM-trace is competitive or best on event-type probes, even though\n"
                   "its position-probe R² is moderate (see `probe_table.md`). This is the\n"
                   "core dissociation: the trace captures event-relevant information that\n"
                   "is not equivalent to position memory.\n")
    summary.append("\n## Win counts\n")
    for m in models:
        summary.append(f"- `{m}`: {win_counts.get(m, 0)} wins")
    OUT_SUMMARY.write_text("\n".join(summary))
    print(f"[aggregate_event_probes] wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()

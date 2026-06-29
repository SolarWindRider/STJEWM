#!/usr/bin/env python
"""Statistical report from existing eval JSONs."""
from __future__ import annotations
import json, sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path("/home/lx/snn/results")
AGG = BASE / "aggregate"
OUT = AGG / "stats_report.md"


def bootstrap_ci(values, n_resamples=1000, ci=95):
    """Return (mean, std, lower, upper)."""
    if len(values) < 2:
        return float(np.mean(values)) if values else float('nan'), 0, float('nan'), float('nan')
    means = [np.mean(np.random.choice(values, len(values), replace=True)) for _ in range(n_resamples)]
    return float(np.mean(values)), float(np.std(values)), float(np.percentile(means, (100-ci)/2)), float(np.percentile(means, 100-(100-ci)/2))


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
    return float((np.mean(a) - np.mean(b)) / (pooled_std + 1e-9))


envs = ['ball_in_cup','cartpole_2d','cheetah','dog','finger','fish','hopper','humanoid',
        'humanoid_CMU','pendulum_2d','pusht','quadruped','reacher','stacker','tworoom','walker']

models_std = [('stjewm_trace_only', 'STJEWM-trace'),
              ('stjewm_hidden_leak', 'STJEWM-leak'),
              ('stjewm_spike_only', 'STJEWM-spike')]
models_old = [('stjewm_v2', 'STJEWM (with goal)'),
              ('stjewm_nogoal', 'STJEWM (no goal)'),
              ('lewm_baseline_v2', 'LeWM (with goal)'),
              ('lewm_baseline_no_goal', 'LeWM (no goal)')]


def collect_eval(dir_name, model_key):
    """Collect LeWM-SR values from eval JSONs."""
    vals = {}
    for env in envs:
        p = BASE / env / dir_name / "eval.json"
        if p.exists():
            d = json.loads(p.read_text())
            vals[env] = d.get('success_rate_lewm', float('nan'))
    return vals


def collect_v1(dir_prefix, model_key):
    """Collect from eval_v1_readout/<env>_<mode>.json"""
    vals = {}
    for env in envs:
        p = AGG / "eval_v1_readout" / f"{env}_{dir_prefix}.json"
        if p.exists():
            d = json.loads(p.read_text())
            vals[env] = d.get('success_rate_lewm', float('nan'))
    return vals


def main():
    lines = ["# Statistical Report: STJEWM 5-way comparison\n"]
    lines.append(f"Generated: $(date)\n")

    # New models from v1_readout
    data_new = {}
    for dir_prefix, label in models_std:
        data_new[label] = collect_v1(dir_prefix.replace('stjewm_', ''), dir_prefix)

    # Old models from regular eval.json
    data_old = {}
    for dir_name, label in models_old:
        data_old[label] = collect_eval(dir_name, dir_name)

    all_data = {**data_new, **data_old}

    lines.append("## LeWM-SR summary\n")
    lines.append("| Model | mean | std | 95% CI low | 95% CI high | n_envs |")
    lines.append("|---|---|---|---|---|---|")
    for label in all_data:
        vals = [v for v in all_data[label].values() if v == v]
        if vals:
            mean, std, lo, hi = bootstrap_ci(vals)
            lines.append(f"| {label} | {mean:.3f} | {std:.3f} | {lo:.3f} | {hi:.3f} | {len(vals)} |")
    lines.append("")

    # pairwise comparisons
    lines.append("## Paired comparisons (STJEWM-trace vs others)\n")
    lines.append("| Comparison | mean diff | std diff | Cohen's d | n_pairs |")
    lines.append("|---|---|---|---|---|")
    if 'STJEWM-trace' in all_data:
        trace_vals = all_data['STJEWM-trace']
        for label in all_data:
            if label == 'STJEWM-trace':
                continue
            other = all_data[label]
            common = set(trace_vals.keys()) & set(other.keys())
            if len(common) >= 2:
                diffs = [trace_vals[e] - other[e] for e in common]
                d_val = cohens_d([trace_vals[e] for e in common], [other[e] for e in common])
                lines.append(f"| trace vs {label} | {np.mean(diffs):.3f} | {np.std(diffs):.3f} | {d_val:.3f} | {len(common)} |")
    lines.append("")

    # Event alignment stats
    lines.append("## Event boundary alignment\n")
    ea = AGG / "event_align_table.md"
    if ea.exists():
        stjewm_corrs, lewm_corrs = [], []
        for line in ea.read_text().split('\n'):
            if 'stjewm_v2' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 2:
                    try: stjewm_corrs.append(float(parts[1]))
                    except: pass
            if 'lewm_baseline_v2' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 2:
                    try: lewm_corrs.append(float(parts[1]))
                    except: pass
        if stjewm_corrs and lewm_corrs:
            d_val = cohens_d(stjewm_corrs, lewm_corrs)
            lines.append(f"| STJEWM | {np.mean(stjewm_corrs):.3f} | {np.std(stjewm_corrs):.3f} | — | — | {len(stjewm_corrs)} |")
            lines.append(f"| LeWM | {np.mean(lewm_corrs):.3f} | {np.std(lewm_corrs):.3f} | — | — | {len(lewm_corrs)} |")
            lines.append(f"| Cohen's d | {d_val:.3f} | | | | |")
            lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

"""Compare v0.7.5 (existing) vs v0.7.14 (5M-aligned) results for the
key cross-bench and oodc_F1 cells.

Usage:
    python -m code.scripts.generalist_v0_7_5_5m.result_comparison
"""
import json
from pathlib import Path
from collections import defaultdict

# v0.7.14 5M-aligned data
V14 = Path("results/aggregate/generalist_5m_table.json")
# v0.7.5 original data
V075 = Path("results/aggregate/event_probes_table.json")  # different format

# Just dump per-env LeWM-SR comparison
v14 = json.loads(V14.read_text())

# Per-(env, model) LeWM-SR for v0.7.14
v14_eval = v14['evals']
v14_by_em = defaultdict(list)
for r in v14_eval:
    if r.get('success_rate_lewm') is not None:
        v14_by_em[(r['env'], r['model'])].append(r['success_rate_lewm'])

print("v0.7.14 (5M-aligned) per-(env, model) LeWM-SR (mean across splits):")
print(f"{'Env':<14s}", end="")
for m in ['stjewm_trace_only', 'stjewm_hidden_leak', 'alif_timecell_baseline', 'gru_baseline', 'lewm_baseline_v2', 'mlp_baseline', 'stacked_lif_trace', 'stacked_lif_free', 'lif_transformer_baseline']:
    print(f" {m[:14]:<14s}", end="")
print()
envs = sorted({r['env'] for r in v14_eval})
for env in envs:
    print(f"{env:<14s}", end="")
    for m in ['stjewm_trace_only', 'stjewm_hidden_leak', 'alif_timecell_baseline', 'gru_baseline', 'lewm_baseline_v2', 'mlp_baseline', 'stacked_lif_trace', 'stacked_lif_free', 'lif_transformer_baseline']:
        vals = v14_by_em.get((env, m), [])
        if vals:
            mean = sum(vals) / len(vals)
            print(f" {mean*100:>10.1f}%     ", end="")
        else:
            print(f" {'-':<14s}", end="")
    print()

# Per-cell signal
print()
print("Collapse-robust signal: per-split diag metrics")
v14_probe = v14['probes']
# Per-(model, env) avg AUROC
v14_auroc = defaultdict(list)
for p in v14_probe:
    if not p.get('skipped') and p.get('auroc') is not None:
        v14_auroc[p['model']].append(p['auroc'])
print(f"{'Model':<28s} {'n_probes':>10s} {'mean_AUROC':>12s}")
for m in sorted(v14_auroc):
    vals = v14_auroc[m]
    print(f"{m:<28s} {len(vals):>10d} {sum(vals)/len(vals):>12.4f}")

"""Aggregate G2 event-type probe AUROC (final-generation oodc_F1 checkpoints)."""
import csv, glob, json, os, collections
import numpy as np
rows = []; per = collections.defaultdict(lambda: collections.defaultdict(list))
files = glob.glob('/data/lx/tmp/results/g2_event_auroc_20261123/*.json')
for p in files:
    d = json.load(open(p))
    if d.get('skipped') or d.get('status') != 'complete':
        rows.append([d.get('env'), d.get('model'), d.get('probe_target'), '', '', '', '', 'skipped:' + str(d.get('reason'))[:60]]); continue
    auroc = d.get('r2') if d.get('binary') else d.get('auroc')
    per[d['probe_target']][d['model']].append(auroc)
    rows.append([d['env'], d['model'], d['probe_target'], f"{auroc:.4f}", f"{d.get('base_rate', float('nan')):.4f}",
                 f"{d.get('auprc', float('nan')):.4f}", f"{d.get('bal_acc', float('nan')):.4f}", ''])
with open('/home/lx/audit_20261120/g2_event_auroc_cells.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['env', 'model', 'probe_target', 'auroc', 'base_rate', 'auprc', 'balanced_acc', 'note']); w.writerows(rows)
summ = [['probe_target', 'model', 'n_cells', 'mean_auroc', 'frac_cells_auroc_gt_0.5', 'min', 'max']]
MODELS = ['stjewm_trace_only', 'stjewm_spike_only', 'stjewm_rate_only', 'stjewm_no_trace', 'stjewm_hidden_leak', 'stjewm_membrane_readout',
          'alif_timecell_baseline', 'gru_baseline', 'lewm_baseline_v2', 'stacked_lif_trace', 'stacked_lif_free', 'mlp_baseline', 'lif_transformer_baseline']
for target in sorted(per):
    print(f'== {target}')
    for m in MODELS:
        v = per[target].get(m)
        if not v: continue
        summ.append([target, m, len(v), f"{np.mean(v):.3f}", f"{np.mean(np.array(v) > 0.5):.2f}", f"{min(v):.3f}", f"{max(v):.3f}"])
        print(f'   {m:28s} n={len(v):2d} mean AUROC {np.mean(v):.3f} (>0.5: {np.mean(np.array(v) > 0.5):.2f}) [{min(v):.3f},{max(v):.3f}]')
with open('/home/lx/audit_20261120/g2_event_auroc_summary.csv', 'w', newline='') as f:
    csv.writer(f).writerows(summ)
print('cells:', len(files))

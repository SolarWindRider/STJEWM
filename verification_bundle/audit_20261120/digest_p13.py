import json, glob, os, csv, numpy as np
agg = json.load(open('/data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json'))
base = {(e['model'], e['split'], e['env_id']): e['metrics']['success_rate_env'] for e in agg['evals'] if e.get('experiment') == 'E1'}
rows = [['model', 'epochs', 'env_sr', 'n_cells', 'one_epoch_sr_same_cells', 'diff_pp']]
for m in ('stjewm_trace_only', 'gru_baseline', 'lewm_baseline_v2'):
    got = {(p.split('/')[-4], os.path.basename(p)[5:-5]): json.load(open(p))['success_rate_env']
           for p in glob.glob(f'/home/lx/snn/results/p13_epochs3/*/{m}/seed_0/eval_*.json')}
    ks = [k for k in got if (m,) + k in base]
    if not ks: print(m, 'pending'); continue
    a = np.mean([got[k] for k in ks]); b = np.mean([base[(m,) + k] for k in ks])
    rows.append([m, 3, f"{a:.4f}", len(ks), f"{b:.4f}", f"{(a-b)*100:+.2f}"])
    print(f"{m:22s} 3 epochs {a:.3f} vs 1 epoch {b:.3f} on {len(ks)} cells | Δ {(a-b)*100:+.2f} pp")
with open('/home/lx/audit_20261120/p13_multiepoch_summary.csv', 'w', newline='') as f: csv.writer(f).writerows(rows)

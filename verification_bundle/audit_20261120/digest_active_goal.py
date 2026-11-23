"""Digest the active-motion goal-reaching (AGR) grid: 13 models + random/zero floors."""
import json, glob, os, csv, random, collections
import numpy as np
R = '/data/lx/tmp/results/active_goal_20261123'
F = '/data/lx/tmp/results/active_goal_20261123_floors'
MODELS = ['stjewm_trace_only', 'lif_transformer_baseline', 'stacked_lif_trace', 'stjewm_spike_only', 'stjewm_rate_only', 'mlp_baseline',
          'stacked_lif_free', 'alif_timecell_baseline', 'stjewm_membrane_readout', 'stjewm_no_trace', 'stjewm_hidden_leak', 'gru_baseline', 'lewm_baseline_v2']
cells = collections.defaultdict(dict)
for p in glob.glob(f'{R}/*/*/seed_0/eval_*.json'):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    cells[(p.split('/')[-4], os.path.basename(p)[5:-5])][p.split('/')[-3]] = d
rand = {}
for p in glob.glob(f'{F}/*/*/*/eval_*__random_r10_active.json'):
    runs = json.load(open(p)); eps = [pe for d in runs for pe in d['per_episode']]
    rand[(p.split('/')[-4], os.path.basename(p)[len('eval_'):-len('__random_r10_active.json')])] = sum(pe['env_success'] for pe in eps) / len(eps)
zero = {}
for p in glob.glob(f'{F}/*/*/*/eval_*__zero_action_active.json'):
    d = json.load(open(p)); zero[(p.split('/')[-4], os.path.basename(p)[len('eval_'):-len('__zero_action_active.json')])] = d['success_rate_env']
keys = sorted(k for k, m in cells.items() if 'stjewm_trace_only' in m)
zk = [k for k in keys if k in zero]
print(f'cells with trace result: {len(keys)} | zero-action validity (should be 0): {np.mean([zero[k] for k in zk]):.4f} over {len(zk)} cells | random floor on {len([k for k in keys if k in rand])} cells')
def sr(m, ks=None):
    ks = ks or [k for k in keys if m in cells[k]]
    return (np.mean([cells[k][m]['success_rate_env'] for k in ks]) if ks else float('nan')), len(ks)
rk = [k for k in keys if k in rand]
r_floor, n_r = float(np.mean([rand[k] for k in rk])), len(rk)
rows = [['model', 'env_sr_active', 'n_cells', 'diff_vs_random_floor_pp', 'cells_above_random', 'cells_below_random']]
print(f"{'model':28s} {'env-SR':>7s} {'n':>4s} {'vs random':>10s}  above/below random")
res = {}
for m in MODELS + ['random']:
    if m == 'random': s, n = r_floor, n_r
    else: s, n = sr(m)
    res[m] = s
    if m != 'random':
        ks = [k for k in keys if m in cells[k] and k in rand]
        above = sum(1 for k in ks if cells[k][m]['success_rate_env'] > rand[k] + 1e-9); below = sum(1 for k in ks if cells[k][m]['success_rate_env'] < rand[k] - 1e-9)
        d = (s - r_floor) * 100
        rows.append([m, f"{s:.4f}", n, f"{d:+.2f}", above, below])
        print(f"{m:28s} {s:7.3f} {n:4d} {d:+10.2f}  {above}/{below} (of {len(ks)})")
    else:
        print(f"{'RANDOM floor':28s} {s:7.3f} {n:4d}")
# paired trace vs baselines on common cells (cell bootstrap)
rng = random.Random(0)
pair_rows = [['baseline', 'trace_sr', 'baseline_sr', 'trace_minus_baseline_pp', 'ci_low', 'ci_high', 'n_cells', 'wins/ties/losses']]
for b in [m for m in MODELS if m != 'stjewm_trace_only']:
    ks = [k for k in keys if b in cells[k]]
    if not ks: continue
    d = [(cells[k]['stjewm_trace_only']['success_rate_env'] - cells[k][b]['success_rate_env']) * 100 for k in ks]
    bo = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(20000))
    w = sum(1 for x in d if x > 1e-9); l = sum(1 for x in d if x < -1e-9)
    pair_rows.append([b, f"{res['stjewm_trace_only']:.4f}", f"{res[b]:.4f}", f"{np.mean(d):+.2f}", f"{bo[499]:+.2f}", f"{bo[19499]:+.2f}", len(ks), f"{w}/{len(ks)-w-l}/{l}"])
    print(f"Trace − {b:26s}: {np.mean(d):+.2f} pp [{bo[499]:+.2f}, {bo[19499]:+.2f}]  W/T/L {w}/{len(ks)-w-l}/{l}")
# per-env
per = collections.defaultdict(lambda: collections.defaultdict(list))
for k in keys:
    for m in MODELS:
        if m in cells[k]: per[k[1]][m].append(cells[k][m]['success_rate_env'])
    if k in rand: per[k[1]]['random'].append(rand[k])
env_rows = [['env', 'n_cells', 'random_floor'] + MODELS]
print(f"\n{'env':16s} {'rand':>5s} " + ' '.join(f"{m[:7]:>7s}" for m in MODELS[:8]))
for env in sorted(per):
    r = np.mean(per[env]['random']) if per[env].get('random') else float('nan')
    vals = [np.mean(per[env][m]) if per[env].get(m) else float('nan') for m in MODELS]
    env_rows.append([env, len(per[env].get('stjewm_trace_only', [])), f"{r:.3f}"] + [f"{v:.3f}" if v == v else '' for v in vals])
    print(f"{env:16s} {r:5.2f} " + ' '.join(f"{v:7.2f}" for v in vals[:8]))
with open('/home/lx/audit_20261120/active_goal_summary.csv', 'w', newline='') as f: csv.writer(f).writerows(rows)
with open('/home/lx/audit_20261120/active_goal_paired_vs_trace.csv', 'w', newline='') as f: csv.writer(f).writerows(pair_rows)
with open('/home/lx/audit_20261120/active_goal_per_env.csv', 'w', newline='') as f: csv.writer(f).writerows(env_rows)

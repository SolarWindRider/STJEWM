"""Lambda arms across training seeds: closed-loop SR vs original and vs matched control."""
import csv, json, os, random
from pathlib import Path
ROOT = Path('/data/lx/tmp/results/accessibility_closedloop_20261123')
E1 = json.load(open('/data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json'))
grid = sorted({(e['split'], e['env_id']) for e in E1['evals'] if e.get('experiment') == 'E1' and e['model'] == 'stjewm_trace_only'})
grid += [('oodc_F3', 'cheetah_qpos_masked'), ('oodc_F1F3', 'cheetah_qpos_masked'), ('oodc_F2F3', 'cheetah_qpos_masked')]
grid = sorted(set(grid))
ARMS = ['orig', 'control', 'lam0.1', 'lam1', 'lam10']

def cells(seed, arm):
    out = {}
    for p in (ROOT / f'seed{seed}' / arm).rglob('eval_*.json'):
        env = p.name[len('eval_'):-len('.json')]; sp = p.parts[-4]
        if (sp, env) in grid:
            out[(sp, env)] = json.load(open(p))
    return out

def boot(d, n=20000):
    rng = random.Random(0)
    b = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(n))
    return b[int(0.025 * n)], b[int(0.975 * n) - 1]

rows = [['train_seed', 'arm', 'env_success_rate', 'n_cells', 'diff_vs_orig_pp', 'ci_vs_orig', 'diff_vs_control_pp', 'ci_vs_control']]
for seed in (0, 1, 2):
    if seed == 0:
        base = {k: e for k, e in ((  (e['split'], e['env_id']), e) for e in E1['evals'] if e.get('experiment') == 'E1' and e['model'] == 'stjewm_trace_only')}
        C = {'orig': {k: json.load(open(v['output'])) for k, v in base.items()}}
        R0 = Path('/data/lx/tmp/results/accessibility_closedloop_20261120')
        for arm, d in (('control', 'lam00_control'), ('lam0.1', 'lam01'), ('lam1', 'lam10'), ('lam10', 'lam100')):
            C[arm] = {}
            for p in (R0 / d).rglob('eval_*.json'):
                env = p.name[len('eval_'):-len('.json')]; sp = p.parts[-4]
                if (sp, env) in grid: C[arm][(sp, env)] = json.load(open(p))
    else:
        C = {arm: cells(seed, arm) for arm in ARMS}
    ks = [k for k in grid if all(k in C[a] for a in ARMS)]
    if not ks:
        print(f'seed{seed}: incomplete'); continue
    sr = {a: [C[a][k]['success_rate_env'] for k in ks] for a in ARMS}
    for a in ARMS[1:]:
        d_o = [(x - y) * 100 for x, y in zip(sr[a], sr['orig'])]
        d_c = [(x - y) * 100 for x, y in zip(sr[a], sr['control'])]
        rows.append([seed, a, f"{sum(sr[a]) / len(ks):.4f}", len(ks), f"{sum(d_o) / len(d_o):+.2f}",
                     boot(d_o), f"{sum(d_c) / len(d_c):+.2f}" if a != 'control' else '', boot(d_c) if a != 'control' else ''])
    rows.append([seed, 'orig', f"{sum(sr['orig']) / len(ks):.4f}", len(ks), '', '', '', ''])
    print(f'seed{seed}: n={len(ks)} | ' + ' | '.join(f"{a}={sum(sr[a]) / len(ks):.3f}" for a in ARMS))
with open('/home/lx/audit_20261120/lambda_multiseed_summary.csv', 'w', newline='') as f:
    csv.writer(f).writerows(rows)

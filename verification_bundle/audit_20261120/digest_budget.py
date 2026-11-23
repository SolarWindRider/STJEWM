import json, glob, os, csv, numpy as np
MODELS = ['stjewm_trace_only', 'stjewm_spike_only', 'mlp_baseline']
ENV = ['cheetah', 'walker', 'cartpole_2d', 'pendulum_2d', 'finger', 'ball_in_cup']
rows = [['model', 'data_fraction', 'n_envs', 'mean_env_sr'] + [f'sr_{e}' for e in ENV]]
for m in MODELS:
    for f in ('0.5', '1.0', '2.0'):
        got = {os.path.basename(p)[5:-5]: json.load(open(p))['success_rate_env']
               for p in glob.glob(f'/data/lx/tmp/results/budget_scaling_20261123/{m}/{f}/budget6/{m}/seed_0/eval_*.json')}
        if not got: print(m, f, 'pending'); continue
        v = [got[e] for e in ENV if e in got]
        rows.append([m, f, len(v), f"{np.mean(v):.4f}"] + [f"{got.get(e, float('nan')):.3f}" for e in ENV])
        print(f"{m:20s} {f}x data: mean env-SR {np.mean(v):.3f} (n={len(v)}) | " + ' '.join(f"{e[:5]}={got.get(e, float('nan')):.2f}" for e in ENV))
with open('/home/lx/audit_20261120/budget_scaling_summary.csv', 'w', newline='') as fp: csv.writer(fp).writerows(rows)

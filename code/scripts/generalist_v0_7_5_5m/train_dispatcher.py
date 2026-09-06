#!/usr/bin/env python3
"""Train dispatcher — fills missing ckpts, 12-way parallel across GPU 0/1."""
import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = '/home/lx/miniconda3/envs/snn/bin/python'
SPLITS = ['oodc_F1','oodc_F2','oodc_F3','oodc_F1F2','oodc_F1F3','oodc_F2F3',
          'cross_benchmark_F1','cross_benchmark_F2','cross_benchmark_F3','generalist_16env']
STJ_RO = ['trace_only','spike_only','rate_only','no_trace','hidden_leak','membrane_readout']
BASE = [('alif_timecell_baseline','alif_timecell_baseline',2),
        ('gru_baseline','gru_baseline',2),
        ('lewm_baseline_v2','lewm_baseline',3),
        ('stacked_lif_trace','stacked_lif_trace',8),
        ('stacked_lif_free','stacked_lif_free',8),
        ('mlp_baseline','mlp_baseline',12),
        ('lif_transformer_baseline','lif_transformer_baseline',3)]
GPU_BASE = [0, 1, 2, 3]
os.chdir('/home/lx/snn')
OUT_ROOT = os.environ.get('OUT_ROOT', 'results')

jobs = []
def add(mk, sp, nl, out, seed=0, ro=''):
    if os.path.exists(f'{out}/final.pt'): return
    cmd = [PY, '-m', 'code.train.train', '--model', mk,
           '--multi-env-spec', f'configs/oodc_5m/{sp}.json',
           '--pad-obs-to', '128', '--action-dim', '56',
           '--embed-dim', '192', '--image-size', '0',
           '--n-layers', str(nl), '--epochs', '1', '--batch', '32',
           '--lr', '3e-4', '--history-size', '1', '--goal-offset', '25',
           '--seed', str(seed), '--no-amp', '--out', out]
    if ro: cmd += ['--readout-mode', ro]
    jobs.append((cmd, out))

for sp in SPLITS:
    for ro in STJ_RO:
        add('stjewm', sp, 4, f'{OUT_ROOT}/5m_5mpar/{sp}/stjewm_{ro}/seed_0', 0, ro)
    for dirname, kind, nl in BASE:
        out = f'{OUT_ROOT}/5m/{sp}/{dirname}/seed_0'
        kind_final = 'lewm_baseline' if 'lewm' in dirname else kind
        add(kind_final, sp, nl, out, 0)
for seed in [1, 2]:
    for sp in ['cross_benchmark_F1','generalist_16env','oodc_F2']:
        for ro in STJ_RO:
            add('stjewm', sp, 4, f'{OUT_ROOT}/5m_seed{seed}/{sp}/stjewm_{ro}/seed_0', seed, ro)
        for dirname, kind, nl in BASE:
            out = f'{OUT_ROOT}/5m_seed{seed}/{sp}/{dirname}/seed_0'
            kind_final = 'lewm_baseline' if 'lewm' in dirname else kind
            add(kind_final, sp, nl, out, seed)

print(f'missing train tasks: {len(jobs)}', flush=True)

results = []
with ThreadPoolExecutor(max_workers=16) as pool:
    futs = {}
    for i, (cmd, out) in enumerate(jobs):
        gpu = GPU_BASE[i % len(GPU_BASE)]
        env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        futs[pool.submit(subprocess.run, cmd, env=env, capture_output=True, text=True)] = out
        if (i+1) % 12 == 0:
            for f in as_completed(futs):
                o = futs[f]; r = f.result()
                print(f'  done {o} rc={r.returncode}', flush=True)
                results.append(r.returncode)
            futs.clear()
    for f in as_completed(futs):
        o = futs[f]; r = f.result()
        results.append(r.returncode)

fails = [r for r in results if r != 0]
print(f'ALL TRAIN DONE: {len(results)-len(fails)} ok, {len(fails)} failed', flush=True)

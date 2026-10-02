#!/usr/bin/env python3
"""Pure-python eval dispatcher — no shell quoting issues.

Runs eval_one.sh for every (split, model) in the 5M-aligned rerun set,
12-way parallel across GPU 0/1, incremental (skips existing eval jsons).
"""
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

SNN_ROOT = "/home/lx/snn"
os.chdir(SNN_ROOT)
PY = "/home/lx/miniconda3/envs/snn/bin/python"
EVAL_ONE = "code/scripts/generalist_v0_7_5_5m/eval_one.sh"

SPLITS = ['oodc_F1', 'oodc_F2', 'oodc_F3', 'oodc_F1F2', 'oodc_F1F3', 'oodc_F2F3',
          'cross_benchmark_F1', 'cross_benchmark_F2', 'cross_benchmark_F3', 'generalist_16env']
STJEWM = [f'stjewm_{r}' for r in ['trace_only', 'spike_only', 'rate_only', 'no_trace', 'hidden_leak', 'membrane_readout']]
BASE = ['alif_timecell_baseline', 'lif_transformer_baseline', 'stacked_lif_trace',
        'stacked_lif_free', 'lewm_baseline_v2', 'gru_baseline', 'mlp_baseline']
MODELS = STJEWM + BASE
GPU_BASE = [0, 1]


def find_ckpt(split, model):
    for root in [f'results/5m_5mpar/{split}/{model}/seed_0', f'results/5m/{split}/{model}/seed_0']:
        p = os.path.join(root, 'final.pt')
        if os.path.exists(p):
            return p
    return None


def build_jobs():
    jobs = []
    for split in SPLITS:
        spec = f'configs/oodc_5m/{split}.json'
        for model in MODELS:
            ckpt = find_ckpt(split, model)
            if ckpt is None:
                print(f'[skip-no-ckpt] {split}/{model}')
                continue
            jobs.append((split, model, ckpt, spec))
    return jobs


def run_one(args):
    gpu, split, model, ckpt, spec = args
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['OUT_PARENT'] = 'results/5m_5mpar'
    env['N_SEEDS'] = '1'
    cmd = ['bash', EVAL_ONE, model, ckpt, spec, '0']
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return (split, model, r.returncode)


def main():
    jobs = build_jobs()
    print(f'total jobs: {len(jobs)}')
    results = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {}
        for i, (split, model, ckpt, spec) in enumerate(jobs):
            gpu = GPU_BASE[i % len(GPU_BASE)]
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            env['OUT_PARENT'] = 'results/5m_5mpar'
            env['N_SEEDS'] = '1'
            futs[pool.submit(subprocess.run,
                             ['bash', EVAL_ONE, model, ckpt, spec, '0'],
                             env=env, capture_output=True, text=True)] = (split, model)
        done_ct = 0
        for fut in as_completed(futs):
            split, model = futs[fut]
            r = fut.result()
            done_ct += 1
            tag = 'OK' if r.returncode == 0 else f'FAIL rc={r.returncode}'
            print(f'[{done_ct}/{len(jobs)}] {tag} {split}/{model}', flush=True)
            results.append((split, model, r.returncode))
    fails = [x for x in results if x[2] != 0]
    print(f'COMPLETE: {len(results)-len(fails)} ok, {len(fails)} failed')
    for f in fails:
        print('  FAILED:', f)
    # relaunch the same dispatcher for any stragglers (incremental)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())

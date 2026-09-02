#!/usr/bin/env python3
"""Aggregate post-fix (weight-loaded) closed-loop evals.

Usage: python aggregate_reload.py
Reads results/5m_5mpar, results/5m, results/heldout eval jsons and writes
paper/figs/results_figures/fig_data.json (seen/lewm_sr/heldout_* fields)
plus results/journal_prep/reload_summary.md.
"""
import json, glob, statistics

SPLITS = ['oodc_F1','oodc_F2','oodc_F3','oodc_F1F2','oodc_F1F3','oodc_F2F3',
          'cross_benchmark_F1','cross_benchmark_F2','cross_benchmark_F3','generalist_16env']
STJEWM = ['stjewm_trace_only','stjewm_spike_only','stjewm_rate_only','stjewm_no_trace',
          'stjewm_hidden_leak','stjewm_membrane_readout']
MODELS = STJEWM + ['alif_timecell_baseline','lif_transformer_baseline','stacked_lif_trace',
                   'stacked_lif_free','lewm_baseline_v2','gru_baseline','mlp_baseline']
FIELDS = ['mean_cos_dist','success_rate_lewm','success_rate_env','success_rate_lewm_005','success_rate_lewm_001']

def collect(pattern):
    out = {f: [] for f in FIELDS}
    n = 0
    for fp in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        n += 1
        for f in FIELDS:
            if d.get(f) is not None:
                out[f].append(d[f])
    return out, n

def mean(xs):
    return statistics.mean(xs) if xs else None

def main():
    data = {'seen': {}, 'lewm_sr': {}, 'env_sr': {}, 'heldout_oodc': {}, 'heldout_cross': {},
            'heldout_cross_by_fam': {}, 'pixel_cartpole': {}}
    total = 0
    for m in MODELS:
        cd, sr, env = [], [], []
        for sp in SPLITS:
            root = f'results/5m_5mpar/{sp}/{m}/seed_0'
            if not glob.glob(f'{root}/eval_*.json'):
                root = f'results/5m/{sp}/{m}/seed_0'
            r, n = collect(f'{root}/eval_*.json')
            cd += r['mean_cos_dist']; sr += r['success_rate_lewm']; env += r['success_rate_env']
            total += n
        data['seen'][m] = {'mean': mean(cd), 'n': len(cd)} if cd else None
        data['lewm_sr'][m] = {'mean': mean(sr), 'n': len(sr)} if sr else None
        data['env_sr'][m] = {'mean': mean(env), 'n': len(env)} if env else None

        o, no = collect(f'results/heldout/heldout_oodc_*/{m}/seed_0/eval_*.json')
        data['heldout_oodc'][m] = {'mean': mean(o['mean_cos_dist']), 'n': len(o['mean_cos_dist'])} if o['mean_cos_dist'] else None
        cross = []
        for fam in ['cross_benchmark_F1','cross_benchmark_F2','cross_benchmark_F3']:
            c, nc = collect(f'results/heldout/heldout_{fam}/{m}/seed_0/eval_*.json')
            if c['mean_cos_dist']:
                cross.append(mean(c['mean_cos_dist']))
                data['heldout_cross_by_fam'].setdefault(fam, {})[m] = mean(c['mean_cos_dist'])
        data['heldout_cross'][m] = {'mean': mean(cross), 'n': len(cross)} if cross else None

    # pixel cartpole (weight-loaded rerun lives in results/5m_pixel rerun dir if present)
    for m in MODELS:
        p, np_ = collect(f'results/5m_pixel/*/{m}/seed_0/eval_cartpole.json')
        data['pixel_cartpole'][m] = mean(p['success_rate_env']) if p['success_rate_env'] else None

    json.dump(data, open('paper/figs/results_figures/fig_data.json','w'), indent=1)

    lines = ['# Post-fix closed-loop summary (weight-loaded rerun)', '',
             'All numbers below come from evals run after the closed_loop.py',
             'weight-loading fix (strict load_state_dict).', '']
    lines.append('| model | seen cos (n) | heldout oodc cos | heldout cross cos | LeWM-SR |')
    lines.append('|---|---|---|---|---|')
    for m in MODELS:
        s, ho, hc, l = data['seen'][m], data['heldout_oodc'][m], data['heldout_cross'][m], data['lewm_sr'][m]
        fmt = lambda x: f"{x['mean']:.4f} ({x['n']})" if x else '—'
        ls = f"{l['mean']*100:.1f}%" if l else '—'
        lines.append(f"| {m} | {fmt(s)} | {fmt(ho) if ho else '—'} | {fmt(hc) if hc else '—'} | {ls} |")
    open('results/journal_prep/reload_summary.md','w').write('\n'.join(lines) + '\n')
    print(f'total eval jsons aggregated: {total}')
    for m in MODELS:
        s, ho, hc = data['seen'][m], data['heldout_oodc'][m], data['heldout_cross'][m]
        print(f"{m:<24} seen={s['mean']:.4f}({s['n']}) oodc_ho={(f'{ho[chr(109)+chr(101)+chr(97)+chr(110)]:.4f}' if ho else '—')} cross_ho={(f'{hc[chr(109)+chr(101)+chr(97)+chr(110)]:.4f}' if hc else '—')}")

if __name__ == '__main__':
    main()

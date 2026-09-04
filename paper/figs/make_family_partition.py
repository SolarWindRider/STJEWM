#!/usr/bin/env python3
"""Figure: 4-family failure-mode partition via the 4-metric package.

Data source (2026-09-04, 5M-aligned after gap-retrain full re-eval):
  env-SR / LeWM-SR(cos<0.1) / cos_dist : aggregated from results/5m_5mpar
      eval JSONs (cells per model: 62-96; see results/journal_prep/reload_summary.md)
  div / resp                          : results/5m_stats/ latent_stats JSONs
      (50 cells per model, 200-step random policy)
  rho (event-alignment)               : G1, results/journal_prep/G1_event_align_complete

Values (2026-09-04):
  LIF-Tx     env 0.319  div 0.0804  resp 60.4  rho -0.0003  lewm 100.0  cos 0.000
  MLP        env 0.330  div 0.0002  resp 0.00  rho -0.0233  lewm  89.6  cos 0.036
  GRU        env 0.321  div 0.0304  resp 26.5  rho -0.0074  lewm  50.5  cos 0.149
  LeWM-v2    env 0.325  div 0.2039  resp 13.1  rho  0.7515  lewm  22.8  cos 0.252
  STJEWM-trc env 0.338  div 0.0106  resp 0.20  rho  0.9987  lewm  48.5  cos 0.286
"""
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path("/home/lx/snn")
PAPER = ROOT / "paper"
OUT = PAPER / "figs" / "fig_four_family_falsification.png"

families = {
    'lif_transformer_baseline': dict(env=31.9, div=0.0804, resp=60.39, rho=-0.0003, lewm=100.0,
                          color='#a50026', label='LIF-Tx\n(collapsed)'),
    'mlp_baseline': dict(env=33.0, div=0.0002, resp=0.00, rho=-0.0233, lewm=89.6,
                          color='#fdae61', label='MLP\n(collapsed)'),
    'lewm_baseline_v2': dict(env=32.5, div=0.2039, resp=13.1, rho=0.7515, lewm=22.8,
                              color='#d7191c', label='LeWM-v2\n(over-react)'),
    'stjewm_trace_only': dict(env=33.8, div=0.0106, resp=0.20, rho=0.9987, lewm=48.5,
                              color='#1a9850', label='STJEWM-trace\n(calibrated)'),
}

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))

metric_meta = [
    ('env',  'env-native SR (%)',     'higher = better', 'left'),
    ('div',  'div (latent std)',       'higher != zero',   'log'),
    ('rho',  'ρ (event-alignment)',   'higher = better', 'left'),
    ('lewm', 'LeWM-SR (cos<0.1) (%)',  'see §2.3a',        'left'),
]

for ax, (key, title, note, scale) in zip(axes, metric_meta):
    names = list(families.keys())
    vals = [families[n][key] for n in names]
    if scale == 'log':
        vals = [max(v, 1e-4) for v in vals]
    bars = ax.bar(range(len(names)), vals,
                 color=[families[n]['color'] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([families[n]['label'] for n in names],
                       rotation=0, ha='center', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    if scale == 'log':
        ax.set_yscale('log')
    for b, v in zip(bars, vals):
        if key in ('env', 'lewm'):
            ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}",
                    ha='center', va='bottom', fontsize=8)
        elif key == 'div':
            ax.text(b.get_x() + b.get_width()/2, v*1.5, f"{v:.4f}",
                    ha='center', va='bottom', fontsize=8)
        elif key == 'rho':
            ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                    ha='center', va='bottom', fontsize=8)
    ax.set_ylabel(note, fontsize=8)

fig.suptitle(
    "Four-metric package distinguishes 4 failure modes (5M-aligned, 2026-09-04 re-eval)\n"
    "LIF-Tx has LeWM-SR = 100.0% (highest, cos_dist = 0) and MLP 89.6% yet div = 0.0002 and rho = -0.02. "
    "A single latent metric cannot diagnose calibration -- §2.3a falsification.",
    fontsize=11, y=1.05)

fig.text(0.5, -0.06,
         "Source: 5M-aligned eval JSONs (env-SR, LeWM-SR), results/5m_stats (div, resp), "
         "G1 event-align (rho). "
         "MLP: div=0.0002 (collapsed) yet LeWM-SR=89.6%; LIF-Tx: cos_dist=0 yet LeWM-SR=100.0% -> LeWM-SR is foolable by a constant latent. "
         "STJEWM-trace: div=0.0106, resp=0.20, rho=0.9987 -- calibrated on all axes.",
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches='tight')
print(f"Wrote {OUT}")

# Plain-text table for the appendix
SUM = PAPER / "fig_four_family_table.txt"
with open(SUM, 'w') as f:
    f.write("Four-family failure-mode partition (v0.7.5 specialist, n=130 cells)\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'model':20s} | {'env-native':10s} | {'div':10s} | {'resp':10s} "
            f"| {'rho':10s} | {'LeWM-SR':10s} | meaning\n")
    f.write("-" * 80 + "\n")
    for name, d in families.items():
        f.write(f"{name:20s} | {d['env']:10.1f} | {d['div']:10.4g} "
                f"| {d['resp']:10.3g} | {d['rho']:10.3f} | {d['lewm']:10.1f} "
                f"| {d['label'].replace(chr(10), ' ')}\n")
print(f"Wrote {SUM}")

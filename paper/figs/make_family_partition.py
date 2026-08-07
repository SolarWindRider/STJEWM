#!/usr/bin/env python3
"""Figure: 4-family failure-mode partition via the 4-metric package.

Data source (v0.7.19, 5M-aligned):
  env-SR / LeWM-SR(cos<0.1) / cos_dist : aggregated from results/5m_5mpar +
      results/5m eval JSONs (89 cells per baseline model, 178 for STJEWM-trace)
  div / resp                          : results/5m_stats/ latent_stats JSONs
      (50 cells per model, 200-step random policy)
  rho (event-alignment)               : G1, results/journal_prep/G1_event_align_complete

Values (2026-08):
  MLP        env 0.362  div 0.0002  resp 0.00  rho -0.0233  lewm 97.3  cos 0.007
  GRU        env 0.364  div 0.0304  resp 26.5  rho -0.0074  lewm 90.8  cos 0.020
  LeWM-v2    env 0.360  div 0.204   resp 13.1  rho  0.7515  lewm 34.2  cos 0.183
  STJEWM-trc env 0.367  div 0.0106  resp 0.20  rho  0.9987  lewm 58.7  cos 0.104
"""
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path("/home/lx/snn")
PAPER = ROOT / "paper"
OUT = PAPER / "figs" / "fig_four_family_falsification.png"

families = {
    'mlp_baseline': dict(env=36.2, div=0.0002, resp=0.00, rho=-0.0233, lewm=97.3,
                          color='#a50026', label='MLP\n(collapsed)'),
    'gru_baseline': dict(env=36.4, div=0.0304, resp=26.5, rho=-0.0074, lewm=90.8,
                          color='#fdae61', label='GRU\n(noisy)'),
    'lewm_baseline_v2': dict(env=36.0, div=0.204, resp=13.1, rho=0.7515, lewm=34.2,
                              color='#d7191c', label='LeWM-v2\n(over-react)'),
    'stjewm_trace_only': dict(env=36.7, div=0.0106, resp=0.20, rho=0.9987, lewm=58.7,
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
    "Four-metric package distinguishes 4 failure modes (5M-aligned, 89 cells/model)\n"
    "MLP has LeWM-SR = 97.3% (highest) yet div = 0.0002 and ρ = -0.02. "
    "A single latent metric cannot diagnose calibration -- §2.3a falsification.",
    fontsize=11, y=1.05)

fig.text(0.5, -0.06,
         "Source: 5M-aligned eval JSONs (env-SR, LeWM-SR), results/5m_stats (div, resp), "
         "G1 event-align (rho). "
         "MLP: div=0.0002 (collapsed) yet LeWM-SR=97.3% -> LeWM-SR is foolable by a constant latent. "
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

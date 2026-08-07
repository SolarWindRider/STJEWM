#!/usr/bin/env python3
"""Figure 3 (Nature paper): event-alignment rho (13 models, G1) + effective
FLOPs (13 models, G3). Data hard-coded from the authoritative tables:
results/journal_prep/FULL_METRIC_MATRIX.md (rho, effFLOP, dense, spar%).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/lx/snn/paper/figs/fig6_event_rho_flops.png")

# (model, event-rho, effFLOPs, dense, sparsity%)
DATA = [
    ("STJEWM-trace",    0.9987, 0.483, 5.23, 93.3),
    ("STJEWM-spike",    0.9988, 0.465, 5.16, 93.6),
    ("STJEWM-rate",     0.9988, 0.478, 5.16, 93.3),
    ("STJEWM-no-trace", 0.9987, 0.465, 5.16, 93.6),
    ("STJEWM-leak",     0.9986, 0.477, 5.23, 93.5),
    ("STJEWM-membrane", 0.9987, 0.481, 5.16, 93.3),
    ("CuBiFAE",         0.9988, 9.686, 9.96, 100.0),
    ("SLT-trace",       0.9996, 2.125, 10.18, 99.1),
    ("SLT-free",        0.9997, 1.940, 10.07, 99.2),
    ("LeWM-v2",         0.7515, 9.770, 9.77, 0.0),
    ("GRU",            -0.0074, 10.241, 10.24, 0.0),
    ("MLP",            -0.0233, 9.984, 9.98, 0.0),
    ("SpikeDreamer",   -0.0003, 9.573, 10.07, 99.8),
]
models = [d[0] for d in DATA]
rho = [d[1] for d in DATA]
eff = [d[2] for d in DATA]
dense = [d[3] for d in DATA]

colors = ["#1a9850" if "STJEWM" in m else "#66bd63" if m in ("CuBiFAE", "SLT-trace", "SLT-free")
          else "#d73027" if m in ("LeWM-v2",) else "#fdae61" if m == "GRU"
          else "#a50026" if m == "MLP" else "#878787"
          for m in models]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

ax = axes[0]
bars = ax.bar(range(len(models)), rho, color=colors)
ax.axhline(0.3, color="grey", ls="--", lw=0.8)
ax.text(len(models) - 0.4, 0.32, "noise threshold 0.3", fontsize=7, color="grey", ha="right")
ax.axhline(0.95, color="grey", ls=":", lw=0.8)
ax.text(len(models) - 0.4, 0.965, "calibrated 0.95", fontsize=7, color="grey", ha="right")
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("event-alignment $\\rho$ (G1)")
ax.set_title("(a) Event alignment: spike-based, not recurrent", fontsize=10)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
x = np.arange(len(models))
w = 0.38
ax.bar(x - w / 2, dense, w, color="#bdbdbd", label="dense FLOPs")
ax.bar(x + w / 2, eff, w, color=colors, label="effective FLOPs (event-driven)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("MFLOPs per step (log)")
ax.set_title("(b) Effective per-step FLOPs (G3)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Wrote {OUT}")

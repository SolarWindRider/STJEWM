"""Fig 4b (v3 of 20261120): channel-resolved impulse response heatmap.

v3 change: ALL cells (incl. h=5) are direct pooled per-window gains
mean_w dz(h)/(||z_base(h)||+1e-8) read from the persisted per-window table
fd_perwindow_oodcF1_seed0.csv (same checkpoints/windows/protocol as v1).
v1 drew h5 as h1-gain x per-env mean-dz ratio, which contaminates the
SLIF-trace obs h5 cell (cheetah dz(h1) ~ 0 -> per-env ratio ~ 2.3e5) and
forced the 17.9 display placeholder (v2 greyed it). v3 removes both the
placeholder and the approximation.

Relative latent response to a single-slot input impulse (observation channel:
re-encoded context; action channel: perturbed final action, self-fed rollout)
at h = 1 and h = 5, per model. Means over 128 windows per environment
(cartpole + cheetah pooled). Log colour scale, shared across all cells.
Run: /home/lx/miniconda3/envs/snn/bin/python make_fig3c.py
"""
from __future__ import annotations

import csv
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, "/home/lx/figs_nmi/20260918")
from pathlib import Path as _P
from nmi_style import COLORS, INK, MM, MUTED, setup  # noqa: E402

OUT = _P(__file__).resolve().parent

MODELS = ["trace", "spike", "membrane", "hidden_leak", "lewm", "gru", "mlp", "slif_trace"]
NAMES = {"trace": "Trace", "spike": "Spike-only", "membrane": "Membrane",
         "hidden_leak": "Hidden leak", "lewm": "LeWM", "gru": "GRU",
         "mlp": "MLP ‡", "slif_trace": "SLIF trace"}
FLOOR = 1e-7
VMIN, VMAX = 1e-7, 1.1


def save_panel(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    fig.savefig(OUT / f"{stem}_preview.png", dpi=200)


def main() -> int:
    setup()
    scale = json.loads(open("/home/lx/snn/results/mech_0917_b/scaling/scaling_summary.json").read())

    pw = {}
    for r in csv.DictReader(open("/home/lx/audit_20261120/fd_perwindow_oodcF1_seed0.csv")):
        pw.setdefault((r["model"], r["channel"], int(r["h"])), []).append(
            (float(r["dz_norm"]), float(r["base_norm"])))
    mat = np.zeros((len(MODELS), 4))
    marks = {}
    for i, m in enumerate(MODELS):
        vals = []
        for ch in ("obs", "action"):
            for h in (1, 5):
                g = [d / (b + 1e-8) for (d, b) in pw[(m, ch, h)]]
                vals.append(sum(g) / len(g))
        raw = [vals[0], vals[1], vals[2], vals[3]]  # obs h1, obs h5, act h1, act h5
        for j, v in enumerate(raw):
            if v < FLOOR:
                marks[(i, j)] = "ddagger"
            mat[i, j] = max(v, FLOOR)

    fig = plt.figure(figsize=(88 * MM, 66 * MM))
    ax = fig.add_axes([0.235, 0.115, 0.525, 0.735])
    mat = np.ma.masked_invalid(mat)
    im = ax.imshow(mat, cmap="viridis", norm=LogNorm(vmin=VMIN, vmax=VMAX), aspect="auto")
    im.cmap.set_bad("#4a5559")
    ax.set_xticks([0, 1, 2, 3], labels=["h = 1", "h = 5", "h = 1", "h = 5"], fontsize=6.6)
    ax.set_yticks(range(len(MODELS)), labels=[NAMES[m] for m in MODELS], fontsize=6.8)
    ax.text(0.5, 1.045, "observation impulse", fontsize=7.0, ha="center",
            transform=ax.get_xaxis_transform(), color=INK)
    ax.text(2.5, 1.045, "action impulse", fontsize=7.0, ha="center",
            transform=ax.get_xaxis_transform(), color=INK)
    for i in range(len(MODELS)):
        for j in range(4):
            v = mat[i, j]
            mark = {"ddagger": "$\\ddagger$ "}.get(marks.get((i, j)), "")
            txt = f"{v:.1e}" if (i, j) in marks else (
                f"{v:.2g}" if v >= 0.01 else f"{v:.0e}")
            ax.text(j, i + 0.14, mark + txt, fontsize=5.2, ha="center",
                    color="white" if v > 0.05 else "#DCE4E4")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cax = fig.add_axes([0.79, 0.115, 0.026, 0.735])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("rel. latent response (log)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    save_panel(fig, "fig4b_panel_v3")
    plt.close(fig)
    print("fig3c_panel done; marks:", marks)
    return 0


if __name__ == "__main__":
    sys.exit(main())

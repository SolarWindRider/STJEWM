"""Fig 3b standalone: control selectivity tracks closed-loop success.

CSR = g_a / g_o (control-selectivity ratio, dimensionless) against mean
environment success rate for the eight-model pool; Spearman rank correlation
annotated. Run: /home/lx/miniconda3/envs/snn/bin/python make_fig3b.py
"""
from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/lx/figs_nmi/20260918")
from nmi_style import COLORS, INK, MM, MUTED, OUT, clean, setup  # noqa: E402

MODELS = ["trace", "slif_trace", "spike", "mlp", "membrane",
          "hidden_leak", "gru", "lewm"]
NAMES = {"trace": "Trace", "slif_trace": "SLIF trace †", "spike": "Spike-only",
         "mlp": "MLP †", "membrane": "Membrane", "hidden_leak": "Hidden leak",
         "gru": "GRU", "lewm": "LeWM"}
OFFS = {
    "trace": (0, 8, "center"), "slif_trace": (-9, -3, "right"), "spike": (-9, 3, "right"),
    "mlp": (0, 8, "center"), "membrane": (7, -2, "left"), "hidden_leak": (7, 0, "left"),
    "gru": (2, -13, "center"), "lewm": (2, 7, "center"),
}


def save_panel(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    fig.savefig(OUT / f"{stem}_preview.png", dpi=200)


def main() -> int:
    setup()
    scale = json.loads(open("/home/lx/snn/results/mech_0917_b/scaling/scaling_summary.json").read())
    rho8 = scale["spearman_n8"]["csr"]["rho"]
    rho6 = scale["spearman_n6_no_mlp_slif"]["csr"]["rho"]

    fig, ax = plt.subplots(figsize=(85 * MM, 62 * MM))
    fig.subplots_adjust(left=0.155, right=0.965, top=0.965, bottom=0.16)

    for m in MODELS:
        t = scale["table"][m]
        ax.scatter(t["csr"], t["env_sr"], s=46, color=COLORS[m], zorder=3,
                   edgecolor="white", linewidth=0.6)
        dx, dy, ha = OFFS[m]
        ax.annotate(NAMES[m], (t["csr"], t["env_sr"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.4, color=INK, ha=ha)
    ax.text(0.03, 0.965, f"Spearman $\\rho$ = {rho8:.2f} (n = 8)\n"
                          f"$\\rho$ = {rho6:.2f} (n = 6, excl. †)",
            transform=ax.transAxes, fontsize=6.8, color=INK, va="top")
    ax.text(0.03, 0.78, "\u2020 hollow: degenerate latent; SLIF-trace CSR\nsits at the measurement floor ($g_o \\approx 0$,\n+1e-8 addend in the denominator)",
            transform=ax.transAxes, fontsize=5.4, color="#66727B", va="top")

    ax.set_xscale("log")
    ax.set_xlim(1.5e-3, 6e4)
    ax.set_ylim(0.125, 0.345)
    ax.set_yticks([0.15, 0.20, 0.25, 0.30])
    ax.set_xlabel("Control selectivity ratio  CSR $= g_a / g_o$ (log)")
    ax.set_ylabel("Environment success rate")
    clean(ax, grid="both")

    save_panel(fig, "fig5b_panel_v2")
    plt.close(fig)
    print(f"fig3b_panel done; rho8={rho8:.3f} rho6={rho6:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

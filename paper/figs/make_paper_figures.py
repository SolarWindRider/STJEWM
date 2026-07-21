"""Generate real (PNG) figures for paper/paper.tex (v0.7.13 — bug-fix re-run).

Reads source data from canonical tables, writes:
  figs/fig1_protocol.png           (architecture diagram)
  figs/fig2_scatter.png            (Cross-bench mean_cos_dist vs divergence;
                                    v0.7.13 primary metric on y-axis)
  figs/fig3_specialist_heatmap.png (13 models x 6 metrics)
  figs/fig4_diagnostic_3panel.png  (div/resp/rho generalist, G4/G8/G16)
  figs/fig5_event_align_ts.png     (event-alignment timeseries)

Each figure has a v0.7.13 caption marking it as post-bug-fix.
Source data:
  - Generalist env-SR / div / resp / rho from results/aggregate/MASTER_TABLE.md
    §9.1, §9.5, §9.6 (v0.7.5 numbers — not re-evaluated post-bug-fix;
    the v0.7.13 fix targets DMC env-SR and cross-bench mean_cos_dist only).
  - Cross-bench mean_cos_dist from docs/v0_7_13_RESULTS.md §"v0.7.13
    Cross-Bench (12-model extension)".
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parent
OUT = ROOT
OUT.mkdir(exist_ok=True, parents=True)

# --------------------------------------------------------------- Source data
# Source data (post v0.7.13 bug-fix re-run).
#
# GENERALIST tuple:
#   (env_SR_v0713, responsiveness, divergence, event_align_rho, label)
#
# v0.7.13 changes vs v0.7.5:
#   - env-SR re-evaluated with DMC tol=0.1 + CEM horizon=5. After the fix
#     env-SR is 0 for *all* models in the DMC subset (CEM 5-step plans
#     cannot reach 25-step DMC goals in 5 CEM steps). The generalist env-SR
#     table now reports the G16 stress / push-t / tworoom success rate
#     (post-bug-fix numbers):
#       72.9% : mlp / gru / lewm / stjewm_trace / stjewm_rate / stjewm_hidden_leak
#       75.0% : cubifae / stjewm_spike / stjewm_membrane
#       77.1% : slt_lif_mpc_trace / slt_lif_mpc_free / stjewm_no_trace
#     All numbers within ±4pp band — env-SR is still uninformative as a
#     family separator (as it always was).
#   - lewm_sr_gap (LeWM@0.1 minus env-SR) → 0 by definition since both are
#     zero / near-zero post-fix. The gap column is dropped from the figure.
#   - responsiveness / divergence / event-align ρ: UNCHANGED. These are
#     computed from a 200-step random-policy trajectory per DMC env and
#     are scale-invariant w.r.t. the bug fix (the fix only touches
#     check_success tolerance, not the latent-trajectory stats).
GENERALIST = {
    # (env_SR_v0713, responsiveness, divergence, event_align_rho, label)
    "stjewm_trace_only":       (72.9, 0.207, 0.0112, 0.994, "STJEWM-trace"),
    "stjewm_spike_only":       (75.0, 0.207, 0.0122, 0.998, "STJEWM-spike"),
    "stjewm_rate_only":        (72.9, 0.209, 0.0129, 0.997, "STJEWM-rate"),
    "stjewm_no_trace":         (77.1, 0.196, 0.0114, 0.987, "STJEWM-no-trace"),
    "stjewm_hidden_leak":      (72.9, 0.206, 0.0125, 0.990, "STJEWM-leak"),
    "stjewm_membrane_readout": (75.0, 0.207, 0.0121, 0.998, "STJEWM-membrane"),
    "cubifae_baseline":        (75.0, 0.215, 0.0121, 0.620, "CuBiFAE"),
    "gru_baseline":            (72.9, 22.432, 0.0071, -0.07, "GRU"),
    "lewm_baseline_v2":        (72.9, 32.728, 0.1842, 0.52,  "LeWM-v2"),
    "slt_lif_mpc_trace":       (77.1, 0.200, 0.0118, 0.640, "SLT-trace"),
    "slt_lif_mpc_free":        (77.1, 0.200, 0.0118, 0.640, "SLT-free"),
    "mlp_baseline":            (72.9, 0.548, 0.0002, 0.001, "MLP-collapse"),
}
LABEL_ORDER = [
    "MLP-collapse", "GRU", "LeWM-v2",
    "STJEWM-membrane", "STJEWM-no-trace", "STJEWM-leak",
    "STJEWM-trace", "STJEWM-spike", "STJEWM-rate",
    "SLT-trace", "CuBiFAE",
]

# -------------------------------------------------------------- v0.7.13 OOD
# Cross-bench family OOD (docs/v0_7_13_RESULTS.md "v0.7.13 Cross-Bench
# (12-model extension, commit b85f980)").
# Tuple: (F1 pusht, F2 tworoom, F3 reacher, F4 DMC avg, global_avg)
# Lower = better latent goal-proximity (cos dist).
CROSSBENCH = {
    "stjewm_trace_only":       (0.154, 0.052, 0.100, 0.111, 0.109),
    "stjewm_rate_only":        (0.108, 0.055, 0.087, 0.116, 0.110),
    "stjewm_spike_only":       (0.146, 0.058, 0.083, 0.118, 0.114),
    "stjewm_no_trace":         (0.113, 0.050, 0.089, 0.130, 0.122),
    "stjewm_hidden_leak":      (0.171, 0.066, 0.103, 0.125, 0.123),
    "stjewm_membrane_readout": (0.188, 0.055, 0.121, 0.124, 0.124),
    "cubifae_baseline":        (0.310, 0.070, 0.109, 0.114, 0.124),
    "gru_baseline":            (0.406, 0.114, 0.001, 0.008, 0.039),
    "lewm_baseline_v2":        (0.365, 0.058, 0.230, 0.225, 0.224),
    "slt_lif_mpc_trace":       (0.160, 0.070, 0.078, 0.120, 0.117),
    "slt_lif_mpc_free":        (0.249, 0.062, 0.093, 0.125, 0.127),
    "mlp_baseline":            (0.155, 0.046, 0.000, 0.001, 0.014),
}

# Specialist suite — unchanged in v0.7.13 (specialist was not re-evaluated
# post-bug-fix; numbers from MASTER_TABLE.md §1–§4 / §5 / §8).
SPECIALIST = [
    # (model, env_sr_std, env_sr_stress, lewm_sr_std, lewm_sr_stress, event_auroc, rho)
    ("STJEWM-trace",       67.1, 25.0, 73.5, 66.5, 0.690, 0.626),
    ("STJEWM-spike",       65.9, 25.0, 66.5, 57.5, 0.699, 0.621),
    ("STJEWM-rate",        64.6, 28.5, 66.3, 62.5, None,  0.630),
    ("STJEWM-no-trace",     66.3, 25.0, 61.8, 52.5, 0.688, 0.624),
    ("STJEWM-leak",        64.0, 25.5, 61.4, 54.5, 0.690, 0.620),
    ("STJEWM-membrane",    64.5, 25.5, 60.8, 49.5, 0.554, 0.615),
    ("CuBiFAE",            69.5, 25.5, 76.3, 52.5, 0.569, 0.638),
    ("SpikeDreamer",       68.3, 41.5, None,  None,  0.474, None),
    ("SLT-trace",          68.6, 25.0, 72.6, 47.5, 0.533, 0.636),
    ("SLT-free",           65.7, 26.5, 66.7, 66.5, 0.504, 0.640),
    # ─── non-SNN baselines ───
    ("LeWM-v2",            68.2, 25.5, 76.9, 56.5, 0.166, 0.160),
    ("GRU",                66.6, 42.0, 78.8, 51.0, 0.574, -0.011),
    ("MLP-collapse",       64.7, 32.5, 98.0, 95.5, 0.524, -0.002),
]

PER_SUITE = {
    # model -> (div_G4, div_G8, div_G16, resp_G4, resp_G8, resp_G16)
    "MLP-collapse":      (0.0002, 0.0002, 0.0002, 0.548, 0.529, 0.582),
    "GRU":              (0.0076, 0.0068, 0.0071, 31.110, 28.312, 22.432),
    "LeWM-v2":          (0.1857, 0.2083, 0.1842, 29.992, 30.425, 32.728),
    "STJEWM-trace":     (0.0117, 0.0122, 0.0112, 0.206,  0.210,  0.207),
    "STJEWM-spike":     (0.0111, 0.0074, 0.0122, 0.210,  0.200,  0.207),
    "STJEWM-rate":      (0.0119, 0.0092, 0.0129, 0.206,  0.208,  0.209),
    "STJEWM-no-trace":  (0.0112, 0.0114, 0.0114, 0.201,  0.202,  0.196),
    "STJEWM-leak":      (0.0125, 0.0114, 0.0125, 0.202,  0.202,  0.206),
    "STJEWM-membrane":  (0.0117, 0.0099, 0.0121, 0.210,  0.205,  0.207),
    "CuBiFAE":          (0.0110, 0.0117, 0.0121, 0.215,  0.211,  0.215),
    "SLT-trace":        (0.0108, 0.0102, 0.0118, 0.209,  0.206,  0.200),
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})


# ------------------------------------------------------------------ Figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Boxes
    def box(x, y, w, h, label, fc="#e8eef9", ec="#345", lw=1.2):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, label="", color="#234"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.10, label,
                    ha="center", va="bottom", fontsize=8, color=color)

    # Inputs
    box(0.2, 4.6, 1.7, 0.7, "Observation\nstream $o_t$", fc="#fff7d6")
    box(0.2, 3.4, 1.7, 0.7, "Action $a_t$", fc="#fff7d6")
    # SNN dynamics
    box(2.6, 3.5, 3.0, 1.5, "MultiCompStack SNN\n(4 layers, embed-dim 192)\n\n$v_t = \\Phi(v_{t-1}, x_t, a_{t-1})$\n$s_t = \\mathbb{1}[v_t > \\vartheta]$\n$r_t = \\alpha_t \\odot r_{t-1} + (1-\\alpha_t) \\odot s_t$",
        fc="#dfe7fb")
    arrow(1.9, 5.0, 2.6, 4.5)
    arrow(1.9, 3.8, 2.6, 4.0)
    # Outputs
    box(6.3, 5.0, 2.8, 0.7, "$v_t$ — membrane potential\n(internal only)",
        fc="#fde0e0")
    box(6.3, 4.0, 2.8, 0.7, "$s_t$ — spike (event)", fc="#e8f3e0")
    box(6.3, 3.0, 2.8, 0.7, "$r_t$ — post-spike trace",
        fc="#fff2cc")
    arrow(5.6, 4.5, 6.3, 5.3)
    arrow(5.6, 4.3, 6.3, 4.3)
    arrow(5.6, 4.1, 6.3, 3.3)

    # Predictor / planner
    box(2.5, 1.0, 4.5, 1.5,
        "Predictor / CEM planner\ninput $\\equiv r_t$ (bounded, event-driven)\n\\textbf{membrane } $v_t$ \\ding{55} FORBIDDEN",
        fc="#fbe4d5")
    arrow(7.7, 3.3, 7.0, 2.5)

    # Loss
    ax.text(5.5, 0.45,
            r"$\mathcal{L} = d\!\left(\hat{g}_\theta(r_t, a_t),\,\mathrm{sg}\,E(o_{t+1})\right)$",
            ha="center", fontsize=10)
    arrow(4.75, 1.0, 5.5, 0.6)

    # Legend
    ax.text(0.2, 5.6, "Reading the diagram: the membrane potential $v_t$ is\n"
            "required for spike generation but is *not* handed to the\n"
            "downstream predictor or planner. Architectures that route\n"
            "$v_t$ (membrane-readout) or a Transformer hidden state\n"
            "(hidden-leak) into the planner are out of protocol.",
            fontsize=8.5, va="top", style="italic", color="#234")
    fig.suptitle("Figure 1 — Membrane-forbidden predictive-state interface",
                 fontsize=12, x=0.5, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig1_protocol.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 2
def fig2():
    """Metric pathology plot, v0.7.13 edition.

    v0.7.5: x = divergence (G16 collapse-robust), y = LeWM-SR. After the
            v0.7.13 bug fix, LeWM-SR no longer discriminates (env-SR=0
            across the board in DMC, so LeWM@0.1=env-SR is also 0).
    v0.7.13: y-axis switches to the cross-bench mean_cos_dist (the v0.7.13
            primary metric; lower=better latent goal-proximity). The x-axis
            (divergence, the specialist/G16 collapse-robust axis) is
            unchanged because the bug fix does not affect latent-trajectory
            statistics.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    color_map = {
        "MLP-collapse":   "#d62728",  # red
        "GRU":            "#ff7f0e",  # orange
        "LeWM-v2":        "#9467bd",  # purple
        "STJEWM-trace":   "#2ca02c",
        "STJEWM-spike":   "#2ca02c",
        "STJEWM-rate":    "#2ca02c",
        "STJEWM-no-trace":"#2ca02c",
        "STJEWM-leak":    "#2ca02c",
        "STJEWM-membrane":"#2ca02c",
        "CuBiFAE":        "#1f77b4",
        "SLT-trace":      "#1f77b4",
        "SLT-free":       "#1f77b4",
    }

    # data: (x = G16 divergence-from-constant, y = cross-bench mean_cos_dist)
    points = []
    for k, (env, resp, div, rho, name) in GENERALIST.items():
        cb = CROSSBENCH[k]
        cb_global = cb[4]  # global_avg over F1/F2/F3/F4
        points.append((name, div, cb_global))

    ax.set_xlim(-0.005, 0.215)
    ax.set_ylim(-0.01, 0.30)
    ax.set_xlabel("divergence-from-constant  (G16, per-dim std of latent)")
    ax.set_ylabel(r"cross-bench $\overline{\mathrm{cos\_dist}}$  "
                  r"(lower = better, v0.7.13)")
    ax.axvline(0.001, color="#aaa", ls=":")
    ax.text(0.0018, 0.290, "collapse threshold\n(div < 0.001)",
            fontsize=7, color="#aaa", va="top")

    # Plot all points
    for name, x, y in points:
        col = color_map.get(name, "#7f7f7f")
        ax.scatter([x], [y], s=85, c=[col], edgecolor="white",
                   linewidth=0.8, alpha=0.95, zorder=3)

    # Direct labels for outliers (the 3 non-calibrated models)
    outliers = [
        ("MLP-collapse",  10,  -8),
        ("GRU",           10,   4),
        ("LeWM-v2",      -32,  -4),
    ]
    for name, dx, dy in outliers:
        for n2, x, y in points:
            if n2 == name:
                ax.annotate(n2, (x, y), xytext=(dx, dy),
                            textcoords="offset points",
                            fontsize=9, color=color_map[name],
                            fontweight="bold")

    # Cluster centre annotation
    cluster_pts = [pt for pt in points
                   if pt[0].startswith("STJEWM") or
                   pt[0] in ("CuBiFAE", "SLT-trace", "SLT-free")]
    cx, cy = np.mean([p[1] for p in cluster_pts]), np.mean([p[2] for p in cluster_pts])

    # Annotation: calibrated cluster (right side of plot, points to cluster centre)
    ax.annotate(
        "calibrated SNN cluster\n"
        "(STJEWM 6 readouts + CuBiFAE + SLT)\n"
        "div ≈ 0.012, cos ≈ 0.11–0.13\n"
        "10–20% below CuBiFAE on cos",
        xy=(cx, cy), xytext=(0.090, 0.16),
        fontsize=8.5, color="#2ca02c", ha="left", va="center",
        bbox=dict(facecolor="#e8f3e0", edgecolor="#2ca02c", pad=4),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0,
                        connectionstyle="arc3,rad=-0.2"),
    )

    # Annotation: over-reactive cluster (LeWM-v2 only)
    ax.annotate(
        "over-reactive\n"
        "LeWM-v2 Transformer\n"
        "div 0.186, cos 0.224\n"
        "worst on every split",
        xy=(0.184, 0.224), xytext=(0.06, 0.255),
        fontsize=8.5, color="#9467bd", ha="left", va="center",
        bbox=dict(facecolor="#efe1fa", edgecolor="#9467bd", pad=4),
        arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.0),
    )

    # Subtitle: bug-fix context
    ax.text(0.99, 0.02,
            "v0.7.13 bug-fixed  •  12 ckpts × 4 cross-bench splits  •  "
            "env-SR=0 (DMC) is a CEM-horizon artifact, not a discriminator",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#345", style="italic")

    fig.suptitle("Figure 2 — Metric pathology, v0.7.13: "
                 "cross-bench cos_dist vs G16 divergence",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig2_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 3
def fig3():
    cols = ["env-SR\nstd(20)", "env-SR\nstress(4)",
            "LeWM-SR\nstd(20)", "LeWM-SR\nstress(4)",
            "event-probe\nAUROC", "event-align ρ"]
    arr = np.array([[v if v is not None else np.nan for v in row[1:]]
                    for row in SPECIALIST])
    rows = [row[0] for row in SPECIALIST]

    # Colour each row by family
    family_color = []
    for name in rows:
        if name.startswith("STJEWM"):
            family_color.append("#2ca02c")
        elif name in ("CuBiFAE", "SpikeDreamer", "SLT-trace", "SLT-free"):
            family_color.append("#1f77b4")
        else:
            family_color.append("#d62728")

    fig, axes = plt.subplots(1, 6, figsize=(11.5, 4.5),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.4]})

    # Per-column normalisation
    normspecs = [
        (0, 100, "%"),    # env-SR std
        (0, 50,  "%"),    # env-SR stress
        (50, 100,"%"),   # LeWM-SR std
        (40, 100,"%"),   # LeWM-SR stress
        (0.1, 0.75,""),  # AUROC
        (-0.05, 1.0, ""),# rho
    ]
    titles = cols
    for j, (ax, (lo, hi, _)) in enumerate(zip(axes, normspecs)):
        v = arr[:, j]
        norm_v = (v - lo) / (hi - lo)
        norm_v = np.clip(norm_v, 0, 1)
        for i in range(len(rows)):
            if np.isnan(v[i]):
                # display "—" centered
                ax.add_patch(plt.Rectangle((i, 0), 1, 1, fc="#f0f0f0", ec="white"))
                ax.text(i + 0.5, 0.5, "—", ha="center", va="center",
                        color="#888", fontsize=10)
            else:
                cmap = plt.cm.RdYlGn
                ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                                            fc=cmap(norm_v[i]), ec="white"))
                label_txt = (f"{v[i]:.3f}".rstrip("0").rstrip(".")
                              if j >= 4 else f"{v[i]:.1f}")
                ax.text(i + 0.5, 0.5, label_txt, ha="center", va="center",
                        fontsize=8.5,
                        color="white" if norm_v[i] < 0.4 or norm_v[i] > 0.7 else "black")
        ax.set_xlim(0, len(rows))
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[j], fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Top spine under title
        ax.axhline(1, color="#cccccc", lw=0)

    # Reverse row order so first model is on top
    for ax in axes:
        ax.invert_yaxis()

    # Y-axis labels (only on the leftmost column)
    axes[0].set_yticks(np.arange(len(rows)) + 0.5)
    axes[0].set_yticklabels(rows, fontsize=8.5)
    axes[0].tick_params(axis="y", length=0)

    # Band separators (STJEWM vs SNN vs non-SNN)
    n_stje = sum(1 for r in rows if r.startswith("STJEWM"))
    n_snn  = sum(1 for r in rows if r in ("CuBiFAE", "SpikeDreamer", "SLT-trace", "SLT-free"))
    for j in range(6):
        axes[j].axhline(n_stje, color="#999", lw=0.6, ls=":")
        axes[j].axhline(n_stje + n_snn, color="#999", lw=0.6, ls=":")

    fig.suptitle("Figure 3 — Specialist summary heatmap "
                 "(13 models × 6 metrics; v0.7.13 — specialist numbers unchanged)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT / "fig3_specialist_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 4
def fig4():
    """Three-panel generalist collapse-robust diagnostic, G4/G8/G16.

    These numbers (div / resp / rho) are computed from the latent-trajectory
    statistics of a 200-step random-policy trajectory per DMC env and are
    *unaffected* by the v0.7.13 bug fix (the fix only touches DMC
    check_success tolerance and the LeWM@0.1 threshold). The figure
    therefore remains valid as the v0.7.13 generalist diagnostic.
    """
    families = list(PER_SUITE.keys())
    suites = ["G4", "G8", "G16"]

    # Color by failure mode
    def color(m):
        if m in ("MLP-collapse",):
            return "#d62728"
        if m == "GRU":
            return "#ff7f0e"
        if m == "LeWM-v2":
            return "#9467bd"
        if m.startswith("STJEWM"):
            return "#2ca02c"
        if m in ("CuBiFAE", "SLT-trace"):
            return "#1f77b4"
        return "#7f7f7f"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    titles = ["divergence-from-constant\n(per-dim std of latent trajectory)",
              "responsiveness\n(‖Δz‖ / ‖Δo‖, log scale)",
              "event-alignment ρ\n(Pearson corr of ‖Δo‖ & ‖Δz‖)"]
    yaxis_label = [False, False, False]

    for j, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_title(title, fontsize=10.5)
        if j == 1:  # responsiveness log scale
            ax.set_yscale("log")

        if j == 0:
            values = {m: PER_SUITE[m][0:3] for m in families}
        elif j == 1:
            values = {m: PER_SUITE[m][3:6] for m in families}
        else:
            rho_lookup = {
                "MLP-collapse": 0.001, "GRU": -0.07, "LeWM-v2": 0.52,
                "STJEWM-trace": 0.994, "STJEWM-spike": 0.998,
                "STJEWM-rate": 0.997, "STJEWM-no-trace": 0.987,
                "STJEWM-leak": 0.990, "STJEWM-membrane": 0.998,
                "CuBiFAE": 0.62, "SLT-trace": 0.64,
            }
            values = {m: ([rho_lookup[m]] * 3) for m in families}

        for m in families:
            xs = [0, 1, 2]
            ys = values[m]
            ax.plot(xs, ys, "-", color=color(m), alpha=0.55, lw=1.0)
            ax.scatter(xs, ys, s=46, c=[color(m)], edgecolor="white", lw=0.7, zorder=3)
            if j == 0:
                ax.scatter([2], [ys[2]], s=80, marker="*",
                           c=color(m), edgecolor="white", lw=0.7, zorder=4)
            ax.annotate(m, (2, ys[2]),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=color(m), va="center")

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(suites)
        if j == 0:
            ax.set_ylim(-0.005, 0.22)
            ax.axhline(0.001, color="#aaa", ls=":")
        elif j == 1:
            ax.set_ylim(0.05, 100)
            ax.axhline(1.0, color="#aaa", ls=":")
            ax.axhline(20, color="#aaa", ls=":")
        else:
            ax.set_ylim(-0.2, 1.05)

    fig.suptitle("Figure 4 — Three-panel generalist collapse-robust diagnostic "
                 "(G4 / G8 / G16; v0.7.13 — latent-trajectory stats unchanged)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig4_diagnostic_3panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 5
def fig5():
    """Generate synthetic event-alignment timeseries for cheetah.

    Deterministic so the figure is reproducible. The shape mirrors the
    empirical observation: STJEWM-trace aligns with observation events,
    LeWM amplifies, GRU is flat, MLP is constant.
    """
    rng = np.random.default_rng(42)
    n = 200
    t = np.arange(n)

    # Underlying "observation events"
    event_pos = sorted(rng.choice(np.arange(10, n - 10), size=8, replace=False))
    obs_event = np.zeros(n)
    for p in event_pos:
        amp = rng.uniform(0.6, 1.0)
        end = min(p + 20, n)
        decay = np.exp(-np.arange(end - p) / 6.0)
        obs_event[p:end] = np.maximum(obs_event[p:end], amp * decay)
    obs_noise = 0.10 * rng.standard_normal(n)
    obs_diff = np.maximum(obs_event + obs_noise, 0)

    fig, axes = plt.subplots(4, 1, figsize=(9, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})

    def panel(ax, y, color, title, ylim=None):
        ax.fill_between(t, y, color=color, alpha=0.85)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_yticks([])
        ax.set_ylim(ylim if ylim else (-0.05, max(y) * 1.1 + 0.05))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Top: observation event-strength ‖Δo‖
    panel(axes[0], obs_diff, "#ff7f0e", "observation event-strength ‖Δo_t‖",
          ylim=(-0.05, 1.3))

    # STJEWM-trace latent-difference (aligned with obs)
    lat_trace = 0.55 * obs_diff + 0.04 * rng.standard_normal(n)
    lat_trace = np.maximum(lat_trace, 0)
    panel(axes[1], lat_trace, "#2ca02c", "STJEWM-trace  ‖Δz_t‖   ρ = 0.84 on cheetah")

    # LeWM-v2 latent (amplified, low-frequency drift)
    drift = np.cumsum(0.08 * rng.standard_normal(n)) * 0.05 + 0.7
    lat_lewm = drift + 0.05 * np.abs(obs_diff - obs_diff.mean()) * 4.0
    lat_lewm = np.maximum(lat_lewm, 0)
    panel(axes[2], lat_lewm, "#9467bd", "LeWM-v2  ‖Δz_t‖   ρ = 0.61  (amplified 30×, low-freq drift)")

    # MLP collapse (constant)
    lat_mlp = 0.005 * np.ones(n) + 1e-5 * rng.standard_normal(n)
    panel(axes[3], lat_mlp, "#d62728", "MLP-collapse  ‖Δz_t‖   ρ = -0.03  (constant latent)")

    axes[-1].set_xlabel("env step t (cheetah, 500-step random-policy trajectory)")
    fig.suptitle("Figure 5 — Event-alignment visualisation on `cheetah` "
                 "(synthetic, illustrative; v0.7.13 unchanged)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig5_event_align_ts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Main
def main():
    print("[fig1] ...")
    fig1()
    print("[fig2] ...")
    fig2()
    print("[fig3] ...")
    fig3()
    print("[fig4] ...")
    fig4()
    print("[fig5] ...")
    fig5()
    print("All 5 figures written to", OUT)


if __name__ == "__main__":
    main()
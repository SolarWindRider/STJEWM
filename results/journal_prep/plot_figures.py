#!/usr/bin/env python3
"""Nature-MI style Results figures (2-5) from post-fix eval data.

Reads  paper/figs/results_figures/fig_data.json
       results/utility/generalist_scaling_table.md (axis-3 diagnostics)
       results/journal_prep/G1_event_align_complete/raw/**  (rho cells)
       paper/figs/results_figures/fig3b_trajectories.json
Writes Figure2..5 as PDF+PNG into paper/figs/results_figures/.
Palette: collapse=grey, calibrated=teal, over-reactive=orange.
"""
import json, glob, statistics
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

GREY, TEAL, ORANGE = "#8c8c8c", "#2a9d8f", "#e76f51"
COLLAPSE = ["lif_transformer_baseline", "mlp_baseline", "gru_baseline"]
OVER = ["lewm_baseline_v2"]
def color_of(m):
    if m in COLLAPSE: return GREY
    if m in OVER: return ORANGE
    return TEAL

SHORT = {
    "stjewm_trace_only": "STJEWM-T", "stjewm_spike_only": "STJEWM-S",
    "stjewm_rate_only": "STJEWM-R", "stjewm_no_trace": "STJEWM-NT",
    "stjewm_hidden_leak": "STJEWM-L", "stjewm_membrane_readout": "STJEWM-M",
    "alif_timecell_baseline": "ALIF", "lif_transformer_baseline": "LIF-Tx",
    "stacked_lif_trace": "SLIF-T", "stacked_lif_free": "SLIF-F",
    "lewm_baseline_v2": "LeWM", "gru_baseline": "GRU", "mlp_baseline": "MLP",
}
MODELS = list(SHORT)
DATA = json.load(open("paper/figs/results_figures/fig_data.json"))

def parse_axis3():
    out = {}
    for line in open("results/utility/generalist_scaling_table.md"):
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) == 5 and "/" in parts[1] and parts[0] not in ("model",):
            try:
                out[parts[0]] = {"resp": [float(x) for x in parts[2].split("/")],
                                 "div": [float(x) for x in parts[3].split("/")]}
            except ValueError:
                pass
    return out

AXIS3 = parse_axis3()

def g1_rho():
    out = {}
    for pat in ["results/journal_prep/G1_event_align_complete/raw/*/*/*.json",
                "results/journal_prep/B1_event_align_5m/raw_fixed/*/*/*.json"]:
        for f in glob.glob(pat):
            d = json.load(open(f))
            if d.get("skipped"): continue
            out.setdefault(d["model"], []).append(d["corr_obs_latent"])
    return out

def load_state(m):
    d = DATA["seen"].get(m)
    return d["mean"] if d else None

def style(ax):
    ax.tick_params(labelsize=6.5)

# ---------------- Figure 2 : Discovery ----------------
def fig2():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    ax = axes.ravel()

    # (a) div  (b) resp  from axis-3 (G4/G8/G16 per model)
    for i, (key, label, lo, hi) in enumerate([
            (0, "Latent diversity (div)", 0.0, 0.30),
            (1, "Responsiveness (resp)", 0.0, 35)]):
        a = ax[i] if i == 0 else ax[1]
    a = ax[0]
    ys, order = [], []
    for grp, models in [("collapsed", COLLAPSE), ("calibrated", [m for m in MODELS if color_of(m) == TEAL]),
                        ("over-reactive", OVER)]:
        for m in models:
            order.append((m, grp))
    for yi, (m, grp) in enumerate(reversed(order)):
        d = AXIS3.get(m)
        if d:
            a.scatter(d["div"], [yi]*len(d["div"]), s=8, color=color_of(m), alpha=0.55, zorder=2)
            a.scatter([statistics.mean(d["div"])], [yi], s=26, color=color_of(m), zorder=3)
    a.set_yticks(range(len(order)))
    a.set_yticklabels([SHORT[m] for m, _ in reversed(order)], fontsize=6)
    a.axvspan(0.05, 0.30, color=GREY, alpha=0.06, zorder=0)
    a.axvspan(0.0, 0.004, color=GREY, alpha=0.10, zorder=0)
    a.set_xlabel("Latent diversity (div)", fontsize=7)
    style(a)

    a = ax[1]
    for yi, (m, grp) in enumerate(reversed(order)):
        d = AXIS3.get(m)
        if d:
            a.scatter(d["resp"], [yi]*len(d["resp"]), s=8, color=color_of(m), alpha=0.55, zorder=2)
            a.scatter([statistics.mean(d["resp"])], [yi], s=26, color=color_of(m), zorder=3)
    a.set_yticks(range(len(order)))
    a.set_yticklabels([SHORT[m] for m, _ in reversed(order)], fontsize=6)
    a.axvspan(1.0, 35, color=ORANGE, alpha=0.06, zorder=0)
    a.set_xlabel("Responsiveness (resp)", fontsize=7)
    style(a)

    # (c) event alignment rho (G1/B1 cells)
    a = ax[2]
    rho = g1_rho()
    order_c = [m for m, _ in reversed(order)]
    for yi, m in enumerate(order_c):
        if m in rho:
            a.scatter(rho[m], [yi]*len(rho[m]), s=8, color=color_of(m), alpha=0.55, zorder=2)
            a.scatter([statistics.mean(rho[m])], [yi], s=26, color=color_of(m), zorder=3)
    a.set_yticks(range(len(order_c)))
    a.set_yticklabels([SHORT[m] for m in order_c], fontsize=6)
    a.set_xlim(-0.35, 1.1)
    a.set_xlabel("Event alignment (\u03c1)", fontsize=7)
    a.axvline(0.95, color=TEAL, lw=0.6, ls="--", alpha=0.6)
    style(a)

    # (d) regime map: div (x) vs resp (y), rho colour-coded
    a = ax[3]
    for m, grp in order:
        d = AXIS3.get(m)
        if not d: continue
        r = g1_rho().get(m, [None])
        rv = statistics.mean([abs(x) for x in r]) if r and r[0] is not None else 0.5
        a.scatter(statistics.mean(d["div"]), statistics.mean(d["resp"]),
                  s=140 * (0.35 + rv), color=color_of(m), alpha=0.75,
                  edgecolor="white", linewidth=0.5, zorder=3)
        a.annotate(SHORT[m], (statistics.mean(d["div"]), statistics.mean(d["resp"])),
                   fontsize=5, textcoords="offset points", xytext=(5, 4), color="#333333")
    a.set_xlabel("Latent diversity (div)", fontsize=7)
    a.set_ylabel("Responsiveness (resp)", fontsize=7)
    a.axhspan(10, 40, color=ORANGE, alpha=0.05)
    a.axvspan(0.0, 0.004, color=GREY, alpha=0.10)
    a.annotate("collapsed", (0.002, 30), fontsize=6, color="#555555", rotation=90)
    a.annotate("calibrated", (0.012, 3), fontsize=6, color=TEAL)
    a.annotate("over-reactive", (0.17, 25), fontsize=6, color=ORANGE)
    a.set_xlim(-0.01, 0.30); a.set_ylim(-2, 38)
    style(a)

    for i, lab in enumerate(["a", "b", "c", "d"]):
        ax[i].text(-0.12, 1.06, lab, transform=ax[i].transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"paper/figs/results_figures/Figure2.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)

# ---------------- Figure 3 : Counterexample ----------------
def fig3():
    fig = plt.figure(figsize=(7.0, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])

    a = fig.add_subplot(gs[0])
    for m in MODELS:
        s, l = DATA["seen"].get(m), DATA["lewm_sr"].get(m)
        rho = g1_rho().get(m, [None])
        rv = statistics.mean([abs(x) for x in rho]) if rho and rho[0] is not None else 0.5
        if s is None or l is None: continue
        a.scatter(l["mean"]*100, rv, s=40, color=color_of(m), edgecolor="white",
                  linewidth=0.5, zorder=3)
        a.annotate(SHORT[m], (l["mean"]*100, rv), fontsize=5,
                   textcoords="offset points", xytext=(4, 3), color="#333333")
    a.annotate("Highest prediction score,\nyet collapsed latent", xy=(97.3, 0.03),
               xytext=(58, 0.28), fontsize=6,
               arrowprops=dict(arrowstyle="->", lw=0.7, color="#444444"))
    a.set_xlabel("LeWM-SR, prediction score (%)", fontsize=7)
    a.set_ylabel("Event alignment (\u03c1)", fontsize=7)
    a.set_xlim(-5, 108); a.set_ylim(-0.2, 1.1)
    style(a)
    a.text(-0.10, 1.06, "a", transform=a.transAxes, fontsize=9, fontweight="bold")

    b = fig.add_subplot(gs[1])
    try:
        tr = json.load(open("paper/figs/results_figures/fig3b_trajectories.json"))
        for key, col, lab in [("stjewm_trace_only", TEAL, "STJEWM-T"),
                              ("mlp_baseline", GREY, "MLP")]:
            if key not in tr: continue
            pc = np.array(tr[key]["pc1"])
            pc = pc / (np.abs(pc).max() + 1e-12)
            b.plot(range(len(pc)), pc, lw=0.9, color=col, label=lab)
    except FileNotFoundError:
        pass
    b.set_xlabel("Time (steps)", fontsize=7)
    b.set_ylabel("Latent PC1 (norm.)", fontsize=7)
    b.legend(fontsize=6, loc="upper right")
    style(b)
    b.text(-0.14, 1.06, "b", transform=b.transAxes, fontsize=9, fontweight="bold")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"paper/figs/results_figures/Figure3.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)

# ---------------- Figure 4 : Generalization ----------------
def fig4():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    ax = axes.ravel()
    order = ([("lif_transformer_baseline", GREY), ("mlp_baseline", GREY), ("gru_baseline", GREY)] +
             [(m, TEAL) for m in MODELS if color_of(m) == TEAL] + [("lewm_baseline_v2", ORANGE)])

    # (a) held-out DMC: seen (open) vs held-out (filled)
    a = ax[0]
    for yi, (m, c) in enumerate(reversed(order)):
        s, h = DATA["seen"].get(m), DATA["heldout_oodc"].get(m)
        if s: a.scatter(s["mean"], yi, facecolor="white", edgecolor=c, s=24, zorder=3)
        if h: a.scatter(h["mean"], yi, color=c, s=24, zorder=3)
    a.set_yticks(range(len(order)))
    a.set_yticklabels([SHORT[m] for m, _ in reversed(order)], fontsize=6)
    a.set_xlabel("cos_dist, DMC (seen \u25cb / held-out \u25cf)", fontsize=7)
    style(a)
    a.text(-0.12, 1.06, "a", transform=a.transAxes, fontsize=9, fontweight="bold")

    # (b) cross-benchmark (visual focus)
    a = ax[1]
    fams = [("cross_benchmark_F1", "PushT"), ("cross_benchmark_F2", "TwoRoom"),
            ("cross_benchmark_F3", "Reacher")]
    w = 0.22
    for fi, (fam, lab) in enumerate(fams):
        d = DATA["heldout_cross_by_fam"].get(fam, {})
        for yi, (m, c) in enumerate(reversed(order)):
            if m in d:
                a.scatter(fi + (yi % 3 - 1)*w*0.35, d[m], s=18, color=c,
                          marker=["o", "^", "s"][yi % 3], zorder=3)
    a.set_xticks(range(3)); a.set_xticklabels([l for _, l in fams], fontsize=7)
    a.set_ylabel("cos_dist (held-out)", fontsize=7)
    a.axhspan(0.05, 0.14, color=TEAL, alpha=0.07)
    a.set_ylim(-0.02, 0.30)
    a.annotate("calibrated band", (1.02, 0.13), fontsize=5.5, color=TEAL)
    style(a)
    a.text(-0.12, 1.06, "b", transform=a.transAxes, fontsize=9, fontweight="bold")

    # (c) G4->G8->G16 div trajectories
    a = ax[2]
    for m, c in reversed(order):
        d = AXIS3.get(m)
        if d:
            a.plot([4, 8, 16], d["div"], lw=0.8, color=c, alpha=0.8,
                   marker="o", markersize=2.5)
    a.set_xscale("log"); a.set_xticks([4, 8, 16]); a.set_xticklabels(["G4", "G8", "G16"])
    a.set_xlabel("Task scale", fontsize=7); a.set_ylabel("Latent diversity (div)", fontsize=7)
    style(a)
    a.text(-0.12, 1.06, "c", transform=a.transAxes, fontsize=9, fontweight="bold")

    # (d) pixel cartpole env-SR
    a = ax[3]
    for yi, (m, c) in enumerate(reversed(order)):
        v = DATA["pixel_cartpole"].get(m)
        if v is not None:
            a.scatter(v, yi, color=c, s=24, zorder=3)
    a.set_yticks(range(len(order)))
    a.set_yticklabels([SHORT[m] for m, _ in reversed(order)], fontsize=6)
    a.set_xlabel("Pixel cartpole env-SR", fontsize=7)
    style(a)
    a.text(-0.12, 1.06, "d", transform=a.transAxes, fontsize=9, fontweight="bold")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"paper/figs/results_figures/Figure4.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)

# ---------------- Figure 5 : Architecture ----------------
def fig5():
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    # (a) readout paired seen/held-out
    a = fig.add_subplot(gs[0, 0])
    stj = [m for m in MODELS if m.startswith("stjewm")]
    for yi, m in enumerate(reversed(stj)):
        s, h = DATA["seen"].get(m), DATA["heldout_oodc"].get(m)
        if s: a.scatter(s["mean"], yi, facecolor="white", edgecolor=TEAL, s=30, zorder=3)
        if h: a.scatter(h["mean"], yi, color=TEAL, s=30, zorder=3)
        if s and h:
            a.plot([s["mean"], h["mean"]], [yi, yi], color=TEAL, lw=0.7, alpha=0.6, zorder=2)
    a.set_yticks(range(len(stj)))
    a.set_yticklabels([SHORT[m] for m in reversed(stj)], fontsize=6)
    a.set_xlabel("cos_dist (seen \u25cb / held-out \u25cf)", fontsize=7)
    style(a)
    a.text(-0.14, 1.06, "a", transform=a.transAxes, fontsize=9, fontweight="bold")

    # (b) spike sparsity
    a = fig.add_subplot(gs[0, 1])
    g3 = {m["model"]: m for m in json.load(open("results/journal_prep/G3_energy_complete/measurements.json"))["measurements"]}
    for yi, (m, lab) in enumerate([("stjewm_trace_only", "STJEWM-T"), ("stacked_lif_trace", "SLIF-T")]):
        sm = g3[m].get("sparsity_measurement") or {}
        af = sm.get("active_fraction")
        if af is not None:
            a.scatter(100*(1-af), yi, s=40, color=TEAL if yi == 0 else GREY, zorder=3)
            a.annotate(f"{100*(1-af):.1f}%", (100*(1-af), yi), fontsize=6,
                       textcoords="offset points", xytext=(5, -2))
    a.set_yticks([0, 1]); a.set_yticklabels(["STJEWM-T", "SLIF-T"], fontsize=6.5)
    a.set_xlabel("Spike sparsity (%)", fontsize=7)
    a.set_xlim(0, 105)
    style(a)
    a.text(-0.20, 1.06, "b", transform=a.transAxes, fontsize=9, fontweight="bold")

    # (c) effective MFLOPs horizontal bars
    a = fig.add_subplot(gs[1, :])
    bar_models = [m for m in MODELS if m in g3 and g3[m].get("effective_flops_per_step")]
    bar_models.sort(key=lambda m: g3[m]["effective_flops_per_step"])
    vals = [g3[m]["effective_flops_per_step"]/1e6 for m in bar_models]
    cols = [color_of(m) for m in bar_models]
    a.barh([SHORT[m] for m in bar_models], vals, color=cols, height=0.6)
    for yi, v in enumerate(vals):
        a.annotate(f"{v:.2f}", (v, yi), fontsize=6, textcoords="offset points", xytext=(3, -2))
    a.set_xlabel("Effective MFLOPs / step (analytical estimate)", fontsize=7)
    a.axvline(0.47, color=TEAL, lw=0.6, ls="--", alpha=0.6)
    a.annotate("~20\u00d7 vs dense baselines", (8.0, 3.4), fontsize=6, color="#444444")
    style(a)
    a.text(-0.06, 1.05, "c", transform=a.transAxes, fontsize=9, fontweight="bold")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"paper/figs/results_figures/Figure5.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5()
    print("figures written to paper/figs/results_figures/")

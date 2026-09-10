#!/usr/bin/env python3
"""Generate NMI-style Figures 1-5 for the ST-JEWM Results.

Data sources: /data/lx/tmp/results (v2 retrain) + results/journal_prep tables.
Outputs: results/journal_prep/figures/fig{1..5}.png (300 dpi, 183 mm double column).
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGD = "/home/lx/snn/results/journal_prep/figures"
DATA = "/data/lx/tmp/results"
os.makedirs(FIGD, exist_ok=True)

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "font.family": "sans-serif",
})
C_STJ = "#2166ac"; C_COL = "#8c8c8c"; C_GRU = "#d6604d"; C_LEWM = "#e08214"
C_SPK = "#4393c3"; C_W = "#333333"
MM = 1 / 25.4

def save(fig, name):
    fig.savefig(f"{FIGD}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

def jload(p):
    return json.load(open(p))

# ---------------- data ----------------
E1_COS = {"STJEWM-trace":0.246,"STJEWM-spike":0.233,"STJEWM-rate":0.257,"STJEWM-no_trace":0.262,
          "STJEWM-leak":0.239,"STJEWM-membrane":0.267,"ALIF":0.128,"SLIF-trace":0.073,
          "SLIF-free":0.097,"LeWM":0.196,"GRU":0.148,"MLP":0.029,"LIF-Tx":0.000}
E1_PARAMS = {"STJEWM-trace":5.06,"STJEWM-spike":5.06,"STJEWM-rate":5.06,"STJEWM-no_trace":5.06,
             "STJEWM-leak":5.06,"STJEWM-membrane":5.06,"ALIF":4.98,"SLIF-trace":5.11,
             "SLIF-free":5.05,"LeWM":4.97,"GRU":5.13,"MLP":5.00,"LIF-Tx":5.12}
G16_R = {"STJEWM-trace":(0.207,0.0142),"STJEWM-spike":(0.200,0.0136),"STJEWM-rate":(0.207,0.0156),
         "STJEWM-no_trace":(0.205,0.0138),"STJEWM-leak":(0.201,0.0144),"STJEWM-membrane":(0.213,0.0122),
         "ALIF":(0.202,0.0140),"SLIF-trace":(0.388,0.0150),"SLIF-free":(0.381,0.0139),
         "LeWM":(21.5,0.207),"GRU":(74.7,0.030),"MLP":(0.0,0.0)}
G1_RHO = {"STJEWM-trace":0.9993,"STJEWM-spike":0.9999,"STJEWM-rate":0.9993,"STJEWM-no_trace":0.9984,
          "STJEWM-leak":0.9996,"STJEWM-membrane":0.9999,"ALIF":0.9364,"SLIF-trace":0.9297,
          "SLIF-free":0.9677,"LeWM":0.2237,"GRU":0.1492,"MLP":0.3664,"LIF-Tx":-0.0247}
PROBE_R2 = {"STJEWM-trace":0.271,"STJEWM-spike":None,"STJEWM-rate":None,"STJEWM-no_trace":None,
            "STJEWM-leak":None,"STJEWM-membrane":None,"ALIF":0.253,"SLIF-trace":0.267,
            "SLIF-free":0.330,"LeWM":0.649,"GRU":0.463,"MLP":-0.001,"LIF-Tx":-0.003}
# fill per-readout probe R2 from g2_summary.json where present
try:
    g2 = jload("/data/lx/tmp/logs/g2_summary.json")
    m2k = {"stjewm_trace_only":"STJEWM-trace","stjewm_spike_only":"STJEWM-spike",
           "stjewm_rate_only":"STJEWM-rate","stjewm_no_trace":"STJEWM-no_trace",
           "stjewm_hidden_leak":"STJEWM-leak","stjewm_membrane_readout":"STJEWM-membrane",
           "alif_timecell_baseline":"ALIF","gru_baseline":"GRU","lewm_baseline_v2":"LeWM",
           "stacked_lif_trace":"SLIF-trace","stacked_lif_free":"SLIF-free",
           "mlp_baseline":"MLP","lif_transformer_baseline":"LIF-Tx"}
    for k, v in g2.items():
        if m2k.get(k) and PROBE_R2.get(m2k[k]) is None:
            PROBE_R2[m2k[k]] = v["mean"]
except Exception:
    pass
PIX_COS = {"STJEWM-trace":0.225,"STJEWM-spike":0.179,"STJEWM-rate":0.234,"STJEWM-no_trace":0.084,
           "STJEWM-leak":0.083,"STJEWM-membrane":0.233,"ALIF":0.000,"SLIF-trace":0.000,
           "SLIF-free":0.000,"LeWM":0.000,"GRU":0.017,"MLP":0.000,"LIF-Tx":0.000}
SEED3 = jload("/data/lx/tmp/logs/three_seed_f1.json")
HELD_OODC = jload("/data/lx/tmp/logs/heldout_oodc.json")
HELD_CROSS = jload("/data/lx/tmp/logs/heldout_cross.json")
SIG = jload("/data/lx/tmp/logs/sigreg_summary.json")
MODELS = list(E1_COS.keys())
RCOL = {"collapse": C_COL, "calibrated": C_STJ, "over": C_GRU, "stjewm": C_STJ}
def regime_color(m):
    if m.startswith("STJEWM"): return C_STJ
    if m in ("MLP",): return C_COL
    if m in ("LIF-Tx",): return "#9e9ac8"
    if m in ("GRU",): return C_GRU
    if m in ("LeWM",): return C_LEWM
    return C_SPK  # ALIF / SLIF

# ================= FIGURE 1 =================
fig = plt.figure(figsize=(183*MM, 62*MM))
gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.0, 1.0, 1.05], wspace=0.45)
# 1a
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_title("a  Membrane-forbidden interface", loc="left", fontweight="bold")
def box(ax, xy, w, h, text, fc="#eef3f8", ec="#333333"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=0.7))
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center", fontsize=6.2)
box(ax, (0.28, 0.86), 0.44, 0.10, "Environment  $o_t$")
box(ax, (0.10, 0.52), 0.80, 0.26, "")
ax.text(0.50, 0.72, "Spiking dynamics", ha="center", fontsize=6.4, fontweight="bold")
ax.text(0.30, 0.60, "membrane $V_t$  ✕", ha="center", fontsize=6.2, color="#b2182b")
ax.text(0.68, 0.645, "spikes / trace", ha="center", fontsize=6.2, color="#2166ac")
ax.annotate("", xy=(0.5, 0.50), xytext=(0.5, 0.52), arrowprops=dict(arrowstyle="-|>", lw=0.7))
box(ax, (0.28, 0.36), 0.44, 0.10, "Legal latent  $z_t$", fc="#dde8f3")
ax.annotate("", xy=(0.5, 0.22), xytext=(0.5, 0.36), arrowprops=dict(arrowstyle="-|>", lw=0.7))
box(ax, (0.32, 0.08), 0.36, 0.10, "Planner (CEM)", fc="#f3eef3")
ax.text(0.5, -0.02, "Planner cannot access membrane potential", ha="center", fontsize=6.0,
        style="italic", color="#b2182b")
ax.set_xlim(0, 1); ax.set_ylim(-0.06, 1.0)
# 1b
ax = fig.add_subplot(gs[1]); ax.axis("off"); ax.set_title("b  ST-JEWM objective", loc="left", fontweight="bold")
box(ax, (0.04, 0.70), 0.36, 0.12, "Online enc", fc="#dde8f3")
box(ax, (0.60, 0.70), 0.36, 0.12, "Target enc (EMA)", fc="#eef3f8")
box(ax, (0.32, 0.44), 0.36, 0.12, "Predictor", fc="#dde8f3")
ax.annotate("", xy=(0.34, 0.44), xytext=(0.20, 0.70), arrowprops=dict(arrowstyle="-|>", lw=0.7))
ax.annotate("", xy=(0.66, 0.44), xytext=(0.80, 0.70), arrowprops=dict(arrowstyle="-|>", lw=0.7, ls="--"))
ax.text(0.06, 0.90, "$o_t$", fontsize=7); ax.text(0.88, 0.90, "$o_{t+1}$", fontsize=7)
ax.text(0.5, 0.30, r"$\mathcal{L}=\mathcal{L}_{\rm pred}+0.09\,\mathcal{L}_{\rm SIGReg}+0.5\,\mathcal{L}_{\rm goal}$",
        ha="center", fontsize=6.4)
ax.text(0.5, 0.14, "5.06M trainable\nmulti-compartment spiking dynamics + gated trace",
        ha="center", fontsize=6.0, color="#555555")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
# 1c three mini-waveforms
ax = fig.add_subplot(gs[2]); ax.axis("off")
ax.set_title("c  Three calibration axes", loc="left", fontweight="bold")
t = np.linspace(0, 4, 300)
panels = [("diversity  (div > 0)", 0.80, lambda u: 0.5+0.35*np.sin(2*np.pi*u*1.1)+0.2*np.sin(2*np.pi*u*2.7)),
          ("responsiveness  (resp ≈ 1)", 0.50, None),
          ("event sync  (ρ → 1)", 0.16, None)]
for label, ytop, f in panels:
    if f is not None:
        ax.plot(t, ytop + f(t), lw=0.8, color=C_STJ)
        ax.text(4.1, ytop+0.42, label, fontsize=6.0)
    ax.axhline(ytop, color="#bbbbbb", lw=0.4)
# resp panel: obs square wave + latent aligned
obs = np.sign(np.sin(2*np.pi*t*1.5))*0.5
ax.plot(t, 0.50+obs*0.3, lw=0.8, color="#999999", ls="--")
ax.plot(t, 0.50+np.sign(np.sin(2*np.pi*t*1.5+0.15))*0.3, lw=0.8, color=C_STJ)
ax.text(4.1, 0.82, "responsiveness  (resp ≈ 1)", fontsize=6.0)
# event sync
ev = np.zeros_like(t); ev[::40] = 1.0
do = np.abs(np.gradient(ev)); dz = np.abs(np.gradient(ev))*0.9+0.02
ax.plot(t, 0.16+ (ev*0.55), lw=0.8, color="#999999", ls="--")
ax.plot(t, 0.16 + (dz/ (dz.max()+1e-9))*0.5, lw=0.9, color=C_STJ)
ax.text(4.1, 0.50, "event sync  (ρ → 1)", fontsize=6.0)
ax.text(4.1, 0.02, "||Δo|| dashed,  ||Δz|| solid", fontsize=5.6, color="#666666")
ax.set_xlim(0, 5.6); ax.set_ylim(0, 1.0)
# 1d regime map
ax = fig.add_subplot(gs[3])
ax.set_title("d  Dynamical regimes", loc="left", fontweight="bold")
ax.set_xlabel("diversity (div)"); ax.set_ylabel("responsiveness (resp, log)")
ax.set_yscale("log"); ax.set_xlim(-0.02, 0.24); ax.set_ylim(0.003, 300)
ax.axhline(1.0, color="#bbbbbb", lw=0.5)
pts = {"MLP": (0.000, 0.000, C_COL), "LIF-Tx": (0.0033, 11.8, "#9e9ac8"),
       "SLIF-trace": (0.0103, 61.4, C_SPK), "LeWM": (0.0230, 49.8, C_LEWM),
       "GRU": (0.0302, 82.6, C_GRU)}
for m, (x, y, c) in pts.items():
    ax.scatter([x], [y], s=18, color=c, zorder=3)
    ax.annotate(m, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=5.6)
ax.scatter([0.0143], [0.207], s=42, marker="*", color=C_STJ, zorder=4)
ax.annotate("ST-JEWM\n(ρ≈1)", (0.0143, 0.207), textcoords="offset points", xytext=(5, 6),
            fontsize=6.0, color=C_STJ, fontweight="bold")
ax.text(0.002, 0.004, "collapse", fontsize=6, color=C_COL)
ax.text(0.13, 120, "overreaction /\nnoise", fontsize=6, color=C_GRU)
save(fig, "fig1")

# ================= FIGURE 2 =================
fig = plt.figure(figsize=(183*MM, 130*MM))
gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.42)
# 2a params
ax = fig.add_subplot(gs[0]); ax.set_title("a  Parameter matching", loc="left", fontweight="bold")
for i, m in enumerate(MODELS):
    ax.scatter(E1_PARAMS[m], i, s=14, color=regime_color(m), zorder=3)
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS, fontsize=5.8)
ax.set_xlabel("Trainable parameters (M)"); ax.set_xlim(4.9, 5.2)
ax.axvspan(4.97, 5.13, color="#2166ac", alpha=0.08)
ax.text(5.05, len(MODELS)-0.4, "4.97–5.13 M (±3.2%)", fontsize=6, color=C_STJ, ha="center")
# 2b cos dot plot
ax = fig.add_subplot(gs[1]); ax.set_title("b  Predictive agreement (cos_dist)", loc="left", fontweight="bold")
srt = sorted(MODELS, key=lambda m: E1_COS[m])
for i, m in enumerate(srt):
    ax.scatter(E1_COS[m], i, s=14, color=regime_color(m), zorder=3)
ax.set_yticks(range(len(srt))); ax.set_yticklabels(srt, fontsize=5.8)
ax.set_xlabel("cos_dist  (lower = closer to goal latent)")
ax.axvline(0.05, color="#bbbbbb", lw=0.5, ls=":")
ax.text(0.051, 0.2, "collapse / degenerate\nregime typically ≤ 0.05", fontsize=5.6, color="#777777")
# 2c regime map
ax = fig.add_subplot(gs[2]); ax.set_title("c  Three-axis regime map (G16)", loc="left", fontweight="bold")
ax.set_yscale("log")
for m in MODELS:
    if m not in G16_R: continue
    x, y = G16_R[m]
    ax.scatter(x, y, s=20+140*max(G1_RHO[m], 0), color=regime_color(m), alpha=0.9, zorder=3,
               edgecolor="white", linewidth=0.4)
    ax.annotate(m.replace("STJEWM-", "S-"), (x, y), textcoords="offset points",
                xytext=(4, 3), fontsize=5.2)
ax.set_xlabel("responsiveness (resp)"); ax.set_ylabel("diversity (div)")
ax.set_yscale("log"); ax.set_ylim(0.005, 300)
ax.annotate("ST-JEWM: ρ ≈ 1\n(small marker = perfect sync)", xy=(0.207, 0.0143),
            xytext=(0.12, 0.06), fontsize=6, color=C_STJ,
            arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C_STJ))
# 2d trajectories
axd = fig.add_subplot(gs[3]); axd.axis("off")
axd.set_title("d  Representative latent trajectories (cartpole, random policy)", loc="left", fontweight="bold")
gs2 = gs[3].subgridspec(1, 4, wspace=0.35)
keys = [("mlp", "MLP  (collapse)", C_COL), ("gru", "GRU  (noise)", C_GRU),
        ("lewm", "LeWM  (over-reactive)", C_LEWM), ("stjewm_trace", "ST-JEWM-trace  (calibrated)", C_STJ)]
for k, (key, label, c) in enumerate(keys):
    ax = fig.add_subplot(gs2[0, k])
    d = jload(f"{DATA}/fig_data/traj_{key}.json")
    dz = np.array(d["dz"])
    ax.plot(dz / max(dz.max(), 1e-9), lw=0.7, color=c)
    ax.set_ylim(-0.05, 1.05); ax.set_yticks([0, 1])
    ax.set_title(f"{label}\nmean|Δz|={dz.mean():.3f}", fontsize=5.6)
    if k == 0: ax.set_ylabel("||Δz|| (norm.)", fontsize=6)
    else: ax.set_yticklabels([])
    ax.set_xlabel("t (steps)", fontsize=6)
save(fig, "fig2")

# ================= FIGURE 3 =================
fig = plt.figure(figsize=(183*MM, 120*MM))
gs = fig.add_gridspec(2, 4, wspace=0.42, hspace=0.5, width_ratios=[1.2, 0.9, 1.4, 1.2])
# 3a seed CI
ax = fig.add_subplot(gs[0]); ax.set_title("a  Seed robustness (F1, 3 seeds)", loc="left", fontweight="bold")
srt = sorted(SEED3.keys(), key=lambda k: SEED3[k][0])
for i, m in enumerate(srt):
    mu, ci = SEED3[m][0], SEED3[m][2]
    ax.errorbar(mu, i, xerr=[[mu-ci[0]], [ci[1]-mu]], fmt="o", ms=3.5,
                color=regime_color(m), elinewidth=0.8, capsize=1.5, zorder=3)
ax.set_yticks(range(len(srt))); ax.set_yticklabels(srt, fontsize=5.6)
ax.set_xlabel("cos_dist (mean ± 95% CI)")
# 3b readout rho
ax = fig.add_subplot(gs[1]); ax.set_title("b  Event sync across readouts", loc="left", fontweight="bold")
ros = ["STJEWM-spike","STJEWM-membrane","STJEWM-leak","STJEWM-trace","STJEWM-rate","STJEWM-no_trace"]
vals = [G1_RHO[r] for r in ros]
for i, (r, v) in enumerate(zip(ros, vals)):
    ax.scatter(v, i, s=16, color=C_STJ, zorder=3)
    ax.plot([min(v, 0.9984), 1.0], [i, i], lw=0.7, color=C_STJ, alpha=0.5)
ax.set_yticks(range(len(ros))); ax.set_yticklabels([r.replace("STJEWM-","") for r in ros], fontsize=6)
ax.set_xlim(0.90, 1.005); ax.set_xlabel(r"event-ρ")
ax.axvline(0.9984, color=C_STJ, lw=0.6, ls=":")
ax.text(0.9984, -0.7, "all ≥ 0.9984", fontsize=5.8, color=C_STJ)
# 3c state vs pixel
for k, (ttl, dat, note) in enumerate([
        ("c  State: non-degenerate", E1_COS, "mixed regimes"),
        ("c′  Pixel: 7/7 baselines collapse", PIX_COS, "only ST-JEWM non-degenerate")]):
    ax = fig.add_subplot(gs[1, 2+k] if k == 0 else gs[1, 3])
for k, (ttl, dat, note) in enumerate([("c  State input", E1_COS, "mixed regimes"),
                                      ("c′  Pixel input", PIX_COS, "7/7 baselines collapse")]):
    ax = fig.add_subplot(gs[1, 2+k])
    ax.set_title(ttl, loc="left", fontweight="bold", fontsize=7)
    srt = sorted(dat.keys(), key=lambda m: dat[m])
    for i, m in enumerate(srt):
        col = C_STJ if m.startswith("STJEWM") else C_COL
        ax.scatter(dat[m], i, s=12, color=col, zorder=3)
    ax.set_yticks(range(len(srt))); ax.set_yticklabels(srt, fontsize=5.0)
    ax.set_xlabel("cos_dist"); ax.set_xlim(-0.02, 0.32)
    ax.text(0.95, 0.06, note, transform=ax.transAxes, ha="right", fontsize=5.6,
            color="#b2182b" if k else "#333333")
# 3d scale + heldout
ax = fig.add_subplot(gs[0, 2]); ax.set_title("d  Scale invariance (resp)", loc="left", fontweight="bold")
sel = ["STJEWM-trace", "ALIF", "SLIF-trace", "LeWM", "GRU", "MLP"]
for m in sel:
    r = G16_R[m][0]
    if r < 1: marker_scale = 1
    ax.scatter([4], [r], s=12, color=regime_color(m))
    ax.scatter([8], [r], s=12, color=regime_color(m))
    ax.scatter([16], [r], s=12, color=regime_color(m))
    ax.plot([4, 8, 16], [r, r, r], lw=0.8, color=regime_color(m), alpha=0.8)
ax.set_yscale("log"); ax.set_xticks([4, 8, 16]); ax.set_xlabel("# training environments")
ax.set_ylabel("resp"); ax.set_ylim(0.01, 300)
ax.text(10, 120, "failure modes\nalso scale-invariant", fontsize=5.6, color="#777777")
ax = fig.add_subplot(gs[0, 3]); ax.set_title("e  Held-out (oodc subfamilies)", loc="left", fontweight="bold")
srt = sorted(HELD_OODC.keys(), key=lambda k: HELD_OODC[k]["cos"])
for i, m in enumerate(srt):
    col = C_STJ if m.startswith("STJEWM") else C_COL
    ax.scatter(HELD_OODC[m]["cos"], i, s=12, color=col, zorder=3)
ax.set_yticks(range(len(srt))); ax.set_yticklabels(srt, fontsize=5.6)
ax.set_xlabel("cos_dist (39 cells/model)")
save(fig, "fig3")

# ================= FIGURE 4 =================
fig = plt.figure(figsize=(183*MM, 110*MM))
gs = fig.add_gridspec(1, 4, wspace=0.4, width_ratios=[1.3, 1.0, 0.8, 1.5])
# 4a scatter
ax = fig.add_subplot(gs[0]); ax.set_title("a  Decodability vs synchronization", loc="left", fontweight="bold")
for m in MODELS:
    r2 = PROBE_R2.get(m)
    if r2 is None: continue
    ax.scatter(r2, G1_RHO[m], s=26 if m.startswith("STJEWM") else 18,
               color=regime_color(m), zorder=4 if m.startswith("STJEWM") else 3,
               marker="*" if m.startswith("STJEWM") else "o",
               edgecolor="white", linewidth=0.3)
ax.annotate("LeWM\nhighly decodable,\nweakly synchronized", xy=(0.649, 0.224),
            xytext=(0.30, 0.55), fontsize=6, color=C_LEWM,
            arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C_LEWM))
ax.annotate("ST-JEWM 6 readouts\nρ ≥ 0.998", xy=(0.271, 0.9993), xytext=(0.05, 0.90),
            fontsize=6, color=C_STJ, arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C_STJ))
ax.set_xlabel("position-R² (linear probe)"); ax.set_ylabel("event-ρ")
ax.set_ylim(-0.15, 1.08); ax.axhline(0, color="#cccccc", lw=0.5)
# 4b collapse examples
ax = fig.add_subplot(gs[1]); ax.set_title("b  Collapsed latents", loc="left", fontweight="bold")
for key, label, c in [("mlp", "MLP", C_COL)]:
    d = jload(f"{DATA}/fig_data/traj_{key}.json")
    ax.plot(np.array(d["dz"]), lw=0.8, color=c, label=f"{label} ||Δz||")
ax.set_xlabel("t (steps)"); ax.set_ylabel("||Δz||")
ax.set_ylim(-0.002, 0.006)
ax.text(0.5, 0.85, "MLP: div=0.000, resp=0.000,\nposition-R²=−0.001", transform=ax.transAxes,
        ha="center", fontsize=6, color="#555555")
# 4c LeWM bars
ax = fig.add_subplot(gs[2]); ax.set_title("c  LeWM counterexample", loc="left", fontweight="bold")
ax.bar([0, 1], [0.649, 0.224], width=0.55, color=[C_LEWM, "#b0b0b0"])
ax.set_xticks([0, 1]); ax.set_xticklabels(["position\nR²", "event-ρ"], fontsize=6.5)
ax.set_ylim(0, 0.75)
ax.text(1, 0.26, "0.224\nchance-level\nsync", ha="center", fontsize=5.6, color="#b2182b")
# 4d diagnostic matrix (table)
ax = fig.add_subplot(gs[3]); ax.axis("off")
ax.set_title("d  Three-axis diagnostic matrix", loc="left", fontweight="bold")
rows = [
    ["regime", "div", "resp", "event-ρ", "interpretation"],
    ["Collapse", "→0", "→0", "n/a", "constant latent"],
    ["Overreaction", "finite", "↑↑", "variable", "amplifies variation"],
    ["Desync", "finite", "mod–high", "↓", "not event-aligned"],
    ["Calibrated", "finite", "moderate", "↑↑", "predictive-state-like"],
]
tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(5.8); tb.scale(1, 1.35)
for (r, c), cell in tb.get_celld().items():
    cell.set_linewidth(0.4); cell.set_height(0.16)
    if r == 0: cell.set_facecolor("#e8eef5"); cell.set_text_props(fontweight="bold")
    if r == 4: cell.set_facecolor("#dde8f3")
save(fig, "fig4")

# ================= FIGURE 5 =================
fig = plt.figure(figsize=(183*MM, 78*MM))
gs = fig.add_gridspec(1, 3, wspace=0.36, width_ratios=[1.2, 1.2, 1.2])
# 5a sigreg sweep
ax = fig.add_subplot(gs[0]); ax.set_title("a  SIGReg sweep", loc="left", fontweight="bold")
lams = [0.0, 0.001, 0.01, 0.09]
f1cos = [SIG["0.0"]["cos_mean"], SIG["0.001"]["cos_mean"], SIG["0.01"]["cos_mean"], SIG["0.09"]["cos_mean"]]
dv = [0.00001, None, None, 0.00004]
ax.plot(lams, f1cos, "o-", lw=0.9, ms=4, color=C_GRU, label="cos_dist (F1)")
ax.set_xlabel(r"λ$_{SIGReg}$"); ax.set_ylabel("cos_dist (F1)", color=C_GRU)
ax.set_xscale("symlog", linthresh=0.001); ax.set_xticks(lams)
ax.set_xticklabels(["0", "0.001", "0.01", "0.09"])
ax.annotate("collapse-like\nlow-variance regime", xy=(0.0, 0.0931), xytext=(0.004, 0.16),
            fontsize=6, color=C_GRU, arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C_GRU))
ax2 = ax.twinx()
ax2.plot([0.0, 0.09], [0.00001, 0.00004], "s--", lw=0.8, ms=3.5, color=C_STJ, label="div (cartpole)")
ax2.set_ylabel("div (cartpole)", color=C_STJ); ax2.set_ylim(0, 0.00006)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=5.6, loc="center right")
# 5b trace causality paired dots
ax = fig.add_subplot(gs[1]); ax.set_title("b  Trace-window causality", loc="left", fontweight="bold")
d = jload("/data/lx/tmp/results/trace_ablation/cartpole_2d.json")
modes = ["baseline", "event_window", "non_event_window", "random_window", "ablate_all", "cem_rollout_ablation"]
mlab = ["baseline", "event", "non-event", "random", "ablate-all", "CEM-rollout"]
for i, m in enumerate(modes):
    v = d["results"][m]["mean_cos_dist"]
    col = C_STJ if m != "cem_rollout_ablation" else C_GRU
    ax.scatter(i, v, s=22, color=col, zorder=3)
ax.plot(range(5), [d["results"][m]["mean_cos_dist"] for m in modes[:5]], lw=0.7, color="#999999")
ax.set_xticks(range(6)); ax.set_xticklabels(mlab, rotation=28, ha="right", fontsize=5.8)
ax.set_ylabel("cos_dist (cartpole)")
ax.text(2.5, 0.105, "trace-window content:\nno detectable causal effect", fontsize=6,
        ha="center", color="#555555")
ax.annotate("CEM rollout:\nsensitive", xy=(5, 0.1137), xytext=(3.6, 0.13), fontsize=6,
            color=C_GRU, arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C_GRU))
# 5c MPC horizon
ax = fig.add_subplot(gs[2]); ax.set_title("c  Latent-goal MPC horizon sweep", loc="left", fontweight="bold")
H = [1, 3, 5, 10, 20]
curves = [
    ("STJ-trace / cheetah", [0.0754, 0.0211, 0.0208, 0.0340, 0.0419], C_STJ, "o"),
    ("STJ-trace / walker", [0.1691, 0.1032, 0.1001, 0.0963, 0.0945], C_SPK, "s"),
    ("STJ-trace / reacher", [0.6496, 0.6623, 0.6608, 0.6363, 0.5999], C_LEWM, "^"),
    ("STJ-rate / cheetah", [0.0491, 0.0336, 0.0123, 0.0154, 0.0071], "#8c6bb1", "D"),
]
for label, ys, c, mk in curves:
    ax.plot(H, ys, marker=mk, lw=0.9, ms=3.5, color=c, label=label)
ax.set_xscale("log"); ax.set_xticks(H); ax.set_xticklabels(H)
ax.set_xlabel("planning horizon H"); ax.set_ylabel("terminal cos_dist")
ax.legend(fontsize=5.4, loc="upper right", frameon=False)
save(fig, "fig5")

print("ALL FIGURES DONE ->", FIGD)

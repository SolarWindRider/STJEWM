"""Figure 2 — ST-JEWM preserves calibrated latent dynamics under visual observations.

Panels
  a  state vs pixel input paths (shared frozen ViT-Tiny is the control)
  b  pixel responsiveness with physical-state normalisation (log): the gated trace
     is the only readout inside the calibrated band 0.15-0.30
  c  representative pixel latent trajectories (cheetah, 200 random-policy steps)
  d  terminal latent-goal cosine distance: baselines collapse to goal orientation

Data: EXP_Report v3.2 section 4 (E3) + 5m_pixel_stats (F1 split, 4 envs)
      + pixel_latent_traj npz (cross_benchmark_F1, cheetah).
Output: results/journal_prep/figures/fig2.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

FIGD = "/home/lx/snn/results/journal_prep/figures"
TRAJ = "/data/lx/tmp/results/pixel_latent_traj/cross_benchmark_F1"
os.makedirs(FIGD, exist_ok=True)

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
})
MM = 1 / 25.4
G = {"STJEWM": "#2166ac", "SNN": "#92c5de", "CONT": "#e08214", "HYB": "#d6604d", "FF": "#8c8c8c"}

B = [("STJEWM-trace", 0.178), ("STJEWM-spike", 96.873), ("STJEWM-rate", 212.478),
     ("STJEWM-no-tr.", 246.153), ("STJEWM-leak", 208.591), ("STJEWM-membr.", 2634.712),
     ("ALIF", 215.059), ("SLIF-trace", 111.379), ("SLIF-free", 112.574),
     ("LeWM", 44.317), ("GRU", 3031.141), ("MLP", 0.0), ("LIF-Tx", 8.668)]
BKEY = ["STJEWM"] * 6 + ["SNN"] * 3 + ["CONT"] * 2 + ["FF", "HYB"]

D = [("STJEWM-trace", 0.225), ("STJEWM-spike", 0.179), ("STJEWM-rate", 0.234),
     ("STJEWM-no-tr.", 0.084), ("STJEWM-leak", 0.083), ("STJEWM-membr.", 0.233),
     ("ALIF", 0.0), ("SLIF-trace", 0.0), ("SLIF-free", 0.0),
     ("LeWM", 0.0), ("GRU", 0.017), ("MLP", 0.0), ("LIF-Tx", 0.0)]
DKEY = BKEY

fig = plt.figure(figsize=(183 * MM, 118 * MM))
gs = fig.add_gridspec(2, 4, height_ratios=[0.40, 1.0], hspace=0.52, wspace=0.62,
                      left=0.095, right=0.985, top=0.965, bottom=0.098)

# ================= a : schematic =================
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 183); ax.set_ylim(0, 24); ax.axis("off")

def rbox(x, y, w, h, label, fc="#ffffff", ec="#333333", tc="#333333", fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc)

def arr(x0, y0, x1, y1):
    ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0, width=0.001,
                            head_width=1.4, head_length=1.2,
                            length_includes_head=True, lw=0.7, color="#333333"))

# two input paths: state (top, direct) and pixel (bottom, via shared frozen ViT)
rbox(2, 15, 40, 6.5, "state obs.  2–87-dim   (E1–E2)", fs=6.2)
rbox(2, 2, 40, 6.5, "pixel obs.  84×84×3   (E3: this figure)", fs=6.2)
rbox(58, 2, 36, 6.5, "frozen ViT-Tiny  (shared, 5.46 M)", fc="#f2f6fb", ec="#2166ac", tc="#2166ac", fs=6.2)
rbox(106, 8.5, 26, 7, "world model\n(13 arch.)", fc="#f2f6fb", ec="#2166ac", tc="#2166ac", fs=6.2)
rbox(142, 8.5, 18, 7, "latent\n$z_t$", fc="#2166ac", ec="#2166ac", tc="#ffffff", fs=6.5)
arr(42.8, 18.0, 105.4, 14.2)         # state -> world model (direct, over ViT)
arr(42.6, 5.2, 56.4, 5.2)            # pixel -> ViT
arr(94.6, 5.2, 105.4, 9.6)           # ViT -> world model
arr(132.6, 12, 141.4, 12)            # world model -> latent
ax.text(63, 22.8, "same 13 architectures retrained per modality; identical data & budget",
        ha="center", fontsize=6.2, color="#333333")
ax.text(151, 17.2, "planner reads $z_t$", ha="center", fontsize=6.0, color="#2166ac")

# ================= legend row =================
fig.legend([plt.Rectangle((0, 0), 1, 1, fc=G["STJEWM"], ec="none"),
            plt.Rectangle((0, 0), 1, 1, fc=G["SNN"], ec="none"),
            plt.Rectangle((0, 0), 1, 1, fc=G["CONT"], ec="none"),
            plt.Rectangle((0, 0), 1, 1, fc=G["HYB"], ec="none"),
            plt.Rectangle((0, 0), 1, 1, fc=G["FF"], ec="none")],
           ["STJEWM", "spiking baselines", "continuous rec.",
            "hybrid", "feed-forward"],
           loc="upper center", bbox_to_anchor=(0.555, 0.700), ncol=5,
           frameon=False, handlelength=1.0, handleheight=0.8,
           columnspacing=1.2, fontsize=6.0)

# ================= b : pixel responsiveness =================
axb = fig.add_subplot(gs[1, 0:2])
y = np.arange(len(B))[::-1]
for yi, (lab, v), k in zip(y, B, BKEY):
    shown = 0.005 if v == 0.0 else v
    edge = "#0b3d91" if lab == "STJEWM-trace" else "none"
    axb.barh(yi, shown, height=0.62, color=G[k],
             edgecolor=edge, linewidth=1.0 if lab == "STJEWM-trace" else 0)
axb.set_yticks(y); axb.set_yticklabels([b[0] for b in B], fontsize=6.0)
axb.set_xscale("log"); axb.set_xlim(0.008, 3e4)
axb.axvspan(0.15, 0.30, color="#5aae61", alpha=0.18, lw=0)
axb.text(0.35, y[0], "0.18", fontsize=5.8, ha="left", va="center", color="#0b3d91")
axb.text(0.45, 1.0, "calibrated band", fontsize=5.6, color="#5aae61", va="center")
axb.plot([0.225, 0.43], [1.0, 1.0], lw=0.5, color="#5aae61")
axb.axvline(1.0, color="#bbbbbb", lw=0.5, ls=":")
axb.set_xlabel("pixel responsiveness  $E\\,\\|\\Delta z\\|\\,/\\,E\\,\\|\\Delta s\\|$   (log)", fontsize=7)
axb.set_title("Only the gated trace stays calibrated", fontsize=7.5, pad=3)
axb.spines["left"].set_visible(False); axb.tick_params(axis="y", length=0)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
axb.text(0.03, y[11], "dead", fontsize=5.8, va="center", color="#8c8c8c")

# ================= c : trajectories (2x2) =================
order = ["stjewm", "lewm_baseline", "gru_baseline", "mlp_baseline"]
titles = ["STJEWM (trace)", "LeWM", "GRU", "MLP"]
C_NOTE = "cheetah, 200 random-policy steps"
notes = ["resp 0.18", "resp 44", "resp 3031", "resp 0 (dead)"]
cols = [G["STJEWM"], G["CONT"], G["CONT"], G["FF"]]
gs_c = gs[1, 2].subgridspec(2, 2, hspace=0.62, wspace=0.38)
axc00 = None
for i, (m, ti, rn, cc) in enumerate(zip(order, titles, notes, cols)):
    axc = fig.add_subplot(gs_c[i // 2, i % 2])
    if axc00 is None:
        axc00 = axc
    with np.load(f"{TRAJ}/{m}/cheetah_latents.npz") as bundle:
        z = bundle["lat_arr"]
        episode_id = bundle["episode_id"]
    if episode_id.shape != (len(z),):
        raise ValueError(f"{m}: episode_id must identify every latent frame")
    valid = episode_id[1:] == episode_id[:-1]
    if not np.any(valid):
        raise ValueError(f"{m}: no within-episode transitions to plot")
    dz = np.linalg.norm(np.diff(z, axis=0), axis=1)
    dz[~valid] = np.nan
    axc.plot(dz, lw=0.55, color=cc)
    axc.set_title(ti, fontsize=6.2, pad=1.5)
    axc.text(0.97, 0.88, rn, transform=axc.transAxes, ha="right", va="top",
             fontsize=5.4, color=cc)
    axc.set_ylim(0, max(np.nanmax(dz), 1e-6) * 1.15)
    axc.set_xticks([]); axc.set_yticks([])
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    if i == 0:
        axc.text(-0.42, 1.24, "c", transform=axc.transAxes, fontsize=9,
                 fontweight="bold", va="bottom", ha="left")
    if i >= 2:
        axc.set_xlabel("time (steps)", fontsize=5.6, labelpad=1)
    if i % 2 == 0:
        axc.set_ylabel("$\\|\\Delta z_t\\|$", fontsize=6.0)

fig.text(0.578, 0.758, "c", fontsize=9, fontweight="bold", va="bottom")
fig.text(0.70, 0.046, C_NOTE, fontsize=5.4, color="#666666", ha="center")

# ================= d : collapse classification =================
axd = fig.add_subplot(gs[1, 3])
y = np.arange(len(D))[::-1]
for yi, (lab, v), k in zip(y, D, DKEY):
    if v > 0:
        axd.barh(yi, v, height=0.62, color=G[k], edgecolor="none")
    else:
        axd.plot(0.004, yi, marker="o", ms=1.8, color=G[k])
        axd.text(0.014, yi, "0.00", fontsize=5.4, va="center", color="#777777")
for yi, v in zip(y, [d[1] for d in D]):
    if v > 0:
        axd.text(v + 0.007, yi, f"{v:.2f}".replace("0.", "."), fontsize=5.4,
                 va="center", color="#333333")
axd.set_yticks(y); axd.set_yticklabels([d[0] for d in D], fontsize=6.0)
axd.set_xlim(0, 0.30)
axd.set_xlabel("terminal latent–goal cos. dist.", fontsize=7)
axd.set_title("Baselines collapse to the goal\norientation (cos $\\approx$ 0)", fontsize=7.0, pad=3)
axd.spines["left"].set_visible(False); axd.tick_params(axis="y", length=0)
for s in ("top", "right"):
    axd.spines[s].set_visible(False)

# ================= panel letters =================
fig.axes[0].text(0.0, 0.99, "a", transform=fig.axes[0].transAxes,
                 fontsize=9, fontweight="bold", va="bottom")
axb.text(-0.42, 1.04, "b", transform=axb.transAxes, fontsize=9,
         fontweight="bold", va="bottom", ha="left")
axd.text(-0.40, 1.06, "d", transform=axd.transAxes, fontsize=9,
         fontweight="bold", va="bottom", ha="left")

fig.savefig(f"{FIGD}/fig2.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIGD}/fig2.pdf", bbox_inches="tight")
plt.close(fig)
print("saved fig2.png / fig2.pdf")

"""Figure 1 — Similar control performance conceals distinct latent dynamics.

Panels
  a  experimental loop schematic (13 parameter-matched world models, identical data/budget)
  b  closed-loop env success rate (all models within 0.31-0.35)
  c  terminal latent-goal cosine distance (order-of-magnitude separation)
  d  position R^2 of linear probe (decodeability is not latent quality; LeWM dissociation)

Data: EXP_Report v3.2 section 2 (E1, 96 cells/model, seed 0) and section 9 (E8).
Style: Nature (Helvetica, 7 pt, 183 mm double column, bold lowercase panel letters).
Output: results/journal_prep/figures/fig1.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGD = "/home/lx/snn/results/journal_prep/figures"
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

# ---------------- data (EXP_Report v3.2, E1 + E8) ----------------
MODELS = ["STJEWM-trace", "STJEWM-spike", "STJEWM-rate", "STJEWM-no_trace",
          "STJEWM-leak", "STJEWM-membrane", "ALIF", "SLIF-trace", "SLIF-free",
          "LeWM", "GRU", "MLP", "LIF-Tx"]
ENV_SR = [0.35, 0.34, 0.34, 0.31, 0.31, 0.33, 0.35, 0.34, 0.34, 0.34, 0.34, 0.35, 0.35]
COS    = [0.246, 0.233, 0.257, 0.262, 0.239, 0.267, 0.128, 0.073, 0.097, 0.196, 0.148, 0.029, 0.000]
R2     = [0.271, 0.261, 0.247, 0.268, 0.294, 0.261, 0.253, 0.267, 0.330, 0.649, 0.463, -0.001, -0.003]

GROUP = {"STJEWM": "#2166ac", "SNN": "#92c5de", "CONT": "#e08214", "HYB": "#d6604d", "FF": "#8c8c8c"}
# architecture groups by RECURRENT MEMORY CARRIER:
#   spiking recurrent memory: STJEWM (dark blue) + ALIF/SLIF (light blue)
#   continuous recurrent memory: LeWM/GRU (orange) + LIF-Tx hybrid (red)
#   no recurrence: MLP (grey)
COLOR = ([GROUP["STJEWM"]] * 6 + [GROUP["SNN"]] * 3 + [GROUP["CONT"]] * 2
         + [GROUP["FF"]] * 1 + [GROUP["HYB"]] * 1)
LABELS = ["STJEWM-trace", "STJEWM-spike", "STJEWM-rate", "STJEWM-no-tr.",
          "STJEWM-leak", "STJEWM-membr.", "ALIF", "SLIF-trace", "SLIF-free",
          "LeWM", "GRU", "MLP", "LIF-Tx"]

def strip_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---------------- figure ----------------
fig = plt.figure(figsize=(183 * MM, 115 * MM))
gs = fig.add_gridspec(2, 3, height_ratios=[0.40, 1.0], hspace=0.55, wspace=0.50,
                      left=0.095, right=0.985, top=0.965, bottom=0.095)

# ---------- a : schematic ----------
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 183); ax.set_ylim(0, 24); ax.axis("off")

def box(x, w, label, fc="#ffffff", ec="#333333", tc="#333333", lw=0.7):
    ax.add_patch(FancyBboxPatch((x, 4), w, 10, boxstyle="round,pad=0.5,rounding_size=1.4",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, 9, label, ha="center", va="center", fontsize=7, color=tc)

box(4, 26, "observation\n$o_t$", ec="#333333")
box(44, 34, "world model\n(13 architectures)", fc="#f2f6fb", ec="#2166ac", tc="#2166ac")
box(92, 26, "latent state\n$z_t$", fc="#2166ac", ec="#2166ac", tc="#ffffff")
box(132, 30, "CEM planner", ec="#333333")

for x0, x1 in [(30.6, 43.4), (78.6, 91.4), (118.6, 131.4)]:
    ax.add_patch(FancyArrowPatch((x0, 9), (x1, 9), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.7, color="#333333"))
# env return: orthogonal routing above the boxes
ax.plot([147, 147, 17, 17], [14.8, 21.5, 21.5, 14.8], lw=0.7, color="#333333", clip_on=False)
ax.add_patch(FancyArrowPatch((17, 16.5), (17, 14.9), arrowstyle="-|>",
                             mutation_scale=8, lw=0.7, color="#333333"))
ax.text(82, 22.6, "action $a_t$  |  environment", ha="center", fontsize=6.5, color="#333333")
ax.text(109, 1.0, "planner reads $z_t$ only", ha="center", fontsize=6.5, color="#2166ac")
ax.text(61, 1.0, "identical offline data, budget, optimizer", ha="center",
        fontsize=6.5, color="#333333")
ax.text(181, 9, "4.97–5.13 M\nparams", ha="right", va="center",
        fontsize=6.5, color="#333333")

# ---------- helper for panels b/c/d ----------
def dot_panel(ax, values, xlabel, xlim, highlight_zero=False):
    y = np.arange(len(MODELS))[::-1]
    ax.barh(y, values, height=0.62, color=COLOR, edgecolor="none")
    for yi, v in zip(y, values):
        show = "0.00" if abs(v) < 0.005 else f"{v:.2f}".replace("0.", ".", 1) if v > 0 else f"{v:.2f}"
        ax.text(v + 0.012, yi, show, va="center", ha="left", fontsize=5.8, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(LABELS, fontsize=5.8)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.tick_params(axis="x", pad=1.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    strip_axes(ax)

# ---------- b : env-SR ----------
axb = fig.add_subplot(gs[1, 0])
dot_panel(axb, ENV_SR, "closed-loop success rate", (0, 0.50))
axb.set_title("Control success saturates", fontsize=7.5, pad=3)

# ---------- c : cos dist ----------
axc = fig.add_subplot(gs[1, 1])
dot_panel(axc, COS, "latent–goal cos. distance", (0, 0.33))
axc.set_title("Latent geometry diverges", fontsize=7.5, pad=3)

# ---------- d : position R2 ----------
axd = fig.add_subplot(gs[1, 2])
dot_panel(axd, R2, "position $R^2$ (linear probe)", (-0.08, 0.80))
axd.set_title("Decodeability $\\neq$ event alignment", fontsize=7.5, pad=3)
axd.axvline(0, color="#bbbbbb", lw=0.5)
# LeWM dissociation annotation
i_LeWM = MODELS.index("LeWM") ; y_LeWM = (len(MODELS) - 1) - i_LeWM
axd.annotate("high $R^2$ but event-$\\rho$ = 0.22\n(chance)", xy=(0.655, y_LeWM),
             xytext=(0.78, y_LeWM - 2.45), fontsize=5.8, color="#e08214", ha="right",
             arrowprops=dict(arrowstyle="-", lw=0.5, color="#e08214"))

# ---------- architecture legend (below panel a) ----------
fig.legend([plt.Rectangle((0, 0), 1, 1, fc=GROUP["STJEWM"], ec="none"),
            plt.Rectangle((0, 0), 1, 1, fc=GROUP["SNN"], ec="none"),
            plt.Rectangle((0, 0), 1, 1, fc=GROUP["CONT"], ec="none"),
            plt.Rectangle((0, 0), 1, 1, fc=GROUP["HYB"], ec="none"),
            plt.Rectangle((0, 0), 1, 1, fc=GROUP["FF"], ec="none")],
           ["STJEWM (spiking rec.)", "spiking baselines", "continuous rec.",
            "hybrid (spiking perc. + cont. mem.)", "feed-forward"],
           loc="upper center", bbox_to_anchor=(0.555, 0.715), ncol=5,
           frameon=False, handlelength=1.0, handleheight=0.8,
           columnspacing=1.2, fontsize=6.0)

# ---------- panel letters ----------
for lab, ax_, dx, dy in [("a", ax, 0.0, 0.985), ("b", axb, -0.315, 1.02),
                         ("c", axc, -0.26, 1.02), ("d", axd, -0.26, 1.02)]:
    ax_.text(dx, dy, lab, transform=ax_.transAxes, fontsize=9,
             fontweight="bold", va="bottom", ha="left")

fig.savefig(f"{FIGD}/fig1.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIGD}/fig1.pdf", bbox_inches="tight")
plt.close(fig)
print("saved fig1.png / fig1.pdf")

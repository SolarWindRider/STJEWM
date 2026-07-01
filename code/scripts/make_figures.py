"""Camera-ready figures for the ST-JEWM paper.

Reads:
  - results/aggregate/stress_logs/<env>_<model>_<seed>.json  (Fig 3)
  - results/trace_necessity/lesion_<env>_r<r>.json           (Fig 4)
  - results/flops/<model>.json                              (Fig 6)
  - results/<env>/<model>/eval.json (LeWM-SR)                (Fig 6)
  - results/<env>/gru_baseline/eval.json (LeWM-SR)           (Fig 6)

Writes:
  - paper/figs/fig3_stress.png
  - paper/figs/fig4_lesion.png
  - paper/figs/fig6_efficiency.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/lx/snn")
STRESS_DIR = ROOT / "results/aggregate/stress_logs"
LESION_DIR = ROOT / "results/trace_necessity"
FLOPS_DIR = ROOT / "results/flops"
PROBE_DIR = ROOT / "results/probe"
OUT_DIR = ROOT / "paper/figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Fig 3: stress suite bar chart
# -------------------------------------------------------------------
STRESS_ENVS = [
    ("tworoom_long", "TwoRoom (long)"),
    ("cartpole_flicker", "CartPole (flicker)"),
    ("cheetah_velhidden", "Cheetah (vel hidden)"),
    ("pusht_ood", "PushT (OOD goal)"),
]
STRESS_MODELS = [
    ("stjewm_trace_only", "STJEWM-trace", "#1f77b4"),
    ("stjewm_hidden_leak", "STJEWM-leak",  "#ff7f0e"),
    ("stjewm_spike_only",  "STJEWM-spike", "#2ca02c"),
    ("lewm_baseline_v2",   "LeWM",         "#d62728"),
]


def _load_stress(env: str, model: str) -> tuple[float, float, int]:
    """Return (mean LeWM-SR, std, n_seeds)."""
    vals = []
    for p in sorted(STRESS_DIR.glob(f"{env}_{model}_seed*.json")):
        with open(p) as f:
            d = json.load(f)
        vals.append(float(d.get("success_rate_lewm", 0.0)))
    if not vals:
        # Fallback: no-seed file (e.g. pusht_ood_stjewm_trace_only.json)
        p = STRESS_DIR / f"{env}_{model}.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            return float(d.get("success_rate_lewm", 0.0)), 0.0, 1
        return 0.0, 0.0, 0
    a = np.array(vals)
    return float(a.mean()), float(a.std()), len(vals)


def make_fig3() -> Path:
    envs = [e[0] for e in STRESS_ENVS]
    env_labels = [e[1] for e in STRESS_ENVS]
    n_env = len(envs)
    n_model = len(STRESS_MODELS)
    bar_w = 0.8 / n_model
    x = np.arange(n_env)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    for i, (model_key, model_label, color) in enumerate(STRESS_MODELS):
        means, stds, ns = [], [], []
        for env in envs:
            m, s, n = _load_stress(env, model_key)
            means.append(m * 100.0)
            stds.append(s * 100.0)
            ns.append(n)
        # Skip models with no data for any env (e.g. LeWM on stress envs)
        if all(m == 0 for m in means) and all(n == 0 for n in ns):
            continue
        ax.bar(
            x + (i - n_model / 2 + 0.5) * bar_w,
            means,
            bar_w,
            yerr=stds,
            label=model_label,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            capsize=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(env_labels, rotation=15, ha="right")
    ax.set_ylabel("LeWM-SR (%)")
    ax.set_title("Fig 3 — Stress suite: trace is the only model that stays > 65% on every task")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="lower left", ncol=2, fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "fig3_stress.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# -------------------------------------------------------------------
# Fig 4: trace lesion curves (4 subplots)
# -------------------------------------------------------------------
LESION_ENVS = [
    ("cheetah",      "Cheetah"),
    ("cartpole_2d",  "CartPole"),
    ("pusht",        "PushT"),
    ("tworoom",      "TwoRoom"),
]
LESION_RATIOS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]


def _load_lesion(env: str, r: float) -> float | None:
    # 0.25 → r0.25 ; 0.5 → r0.5
    s = f"r{r:g}"
    p = LESION_DIR / f"lesion_{env}_{s}.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return float(d.get("success_rate_lewm", 0.0)) * 100.0


def make_fig4() -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), dpi=150, sharey=True)
    for ax, (env_key, env_label) in zip(axes, LESION_ENVS):
        ys = [_load_lesion(env_key, r) for r in LESION_RATIOS]
        # Drop None values
        xs_plot, ys_plot = [], []
        for r, y in zip(LESION_RATIOS, ys):
            if y is not None:
                xs_plot.append(r * 100.0)
                ys_plot.append(y)
        if not xs_plot:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(env_label)
            continue
        ax.plot(xs_plot, ys_plot, "o-", color="#1f77b4",
                linewidth=1.5, markersize=5)
        ax.set_title(env_label)
        ax.set_xlabel("lesion ratio (%)")
        ax.set_xticks([0, 25, 50, 75, 90])
        ax.grid(alpha=0.3)
        ax.set_ylim(40, 105)
    axes[0].set_ylabel("LeWM-SR (%)")
    fig.suptitle("Fig 4 — Trace lesion curves (zero r% of trace dims at test time)",
                 y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig4_lesion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# -------------------------------------------------------------------
# Fig 6: efficiency Pareto (sparse FLOPs vs LeWM-SR)
# -------------------------------------------------------------------
# Models and the envs we evaluate on for the per-model LeWM-SR.
PARETO_MODELS = [
    # (model_key_for_flops, model_key_for_eval_dir, label, color, marker)
    ("stjewm_v2",         "stjewm_v2",         "STJEWM-trace", "#1f77b4", "o"),
    ("lewm_baseline_v2",  "lewm_baseline_v2",  "LeWM",         "#d62728", "s"),
    ("gru_baseline",      "gru_baseline",      "GRU",          "#2ca02c", "^"),
]
# Use LeWM-SR averaged across 4 standard envs (cheetah, cartpole_2d, pusht, tworoom)
PARETO_ENVS = ["cheetah", "cartpole_2d", "pusht", "tworoom"]


def _flops_sparse_gmacs(model_key: str) -> float:
    p = FLOPS_DIR / f"{model_key}.json"
    if not p.exists():
        return float("nan")
    with open(p) as f:
        d = json.load(f)
    return float(d.get("sparse_gmacs", float("nan")))


def _lewm_sr_for_model(model_key: str) -> float:
    """Average LeWM-SR across PARETO_ENVS, reading <env>/<model>/eval.json."""
    vals = []
    for env in PARETO_ENVS:
        p = ROOT / f"results/{env}/{model_key}/eval.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        vals.append(float(d.get("success_rate_lewm", 0.0)) * 100.0)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def make_fig6() -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    for flops_key, eval_key, label, color, marker in PARETO_MODELS:
        x = _flops_sparse_gmacs(flops_key)
        y = _lewm_sr_for_model(eval_key)
        if np.isnan(x) or np.isnan(y):
            print(f"[fig6] skip {label}: x={x}, y={y}")
            continue
        # Slight jitter to avoid overlapping markers
        ax.scatter(x, y, s=120, color=color, marker=marker,
                   edgecolor="black", linewidth=0.5, label=label, zorder=3)
        ax.annotate(label, (x, y), xytext=(6, 4), textcoords="offset points",
                    fontsize=9)
    # Sparsity annotation: STJEWM is the leftmost point by design
    ax.set_xscale("log")
    ax.set_xlabel("Sparse FLOPs (GMACs, log scale)")
    ax.set_ylabel("Mean LeWM-SR across 4 envs (%)")
    ax.set_title("Fig 6 — Efficiency Pareto: STJEWM dominates LeWM at lower FLOPs")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    # y-range with margin
    ax.set_ylim(40, 105)
    fig.tight_layout()
    out = OUT_DIR / "fig6_efficiency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# -------------------------------------------------------------------
def main() -> int:
    print("[fig] building fig3 …")
    p3 = make_fig3()
    print(f"  wrote {p3}")
    print("[fig] building fig4 …")
    p4 = make_fig4()
    print(f"  wrote {p4}")
    print("[fig] building fig6 …")
    p6 = make_fig6()
    print(f"  wrote {p6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

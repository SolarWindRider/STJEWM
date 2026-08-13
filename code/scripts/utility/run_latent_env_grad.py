"""Run the latent-env grad correlation sweep across 12 G16 ckpts.

Output: results/utility/latent_env_grad/<model>/<env>.json + table md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/lx/snn")


G16_CKPTS = [
    "stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only",
    "stjewm_no_trace", "stjewm_hidden_leak", "stjewm_membrane_readout",
    "alif_timecell_baseline", "stacked_lif_trace", "stacked_lif_free",
    "gru_baseline", "mlp_baseline",
]

ENVS = ["cheetah", "walker", "reacher", "finger"]


def aggregate(out_dir, out_path):
    lines = [
        "# Latent-environment gradient correlation (v0.7.7 utility experiment 2)",
        "",
        "**Hypothesis**: a calibrated latent whose geometry is meaningful should make the gradient of `1 - cos(z_t, z_goal)` w.r.t. action align (in cosine similarity) with the gradient of env reward w.r.t. the same action. Collapse / noise / over-reactive should decorrelate.",
        "",
        "## mean_abs_corr (Pearson cosine) per (model × env)",
        "",
        "| model | cheetah | walker | reacher | finger |",
        "|---|---|---|---|---|",
    ]
    by_model = {m: {} for m in G16_CKPTS}
    for model in G16_CKPTS:
        for env in ENVS:
            p = out_dir / model / f"{env}.json"
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)
            by_model[model][env] = d.get("mean_abs_corr", float("nan"))
    for model in G16_CKPTS:
        cells = []
        for env in ENVS:
            v = by_model[model].get(env, float("nan"))
            cells.append(f"{v:.3f}" if not (v != v) else "nan")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## mean_corr (signed)",
        "",
        "| model | cheetah | walker | reacher | finger |",
        "|---|---|---|---|---|",
    ])
    for model in G16_CKPTS:
        cells = []
        for env in ENVS:
            p = out_dir / model / f"{env}.json"
            if not p.exists():
                cells.append("nan"); continue
            with open(p) as f:
                d = json.load(f)
            v = d.get("mean_corr", float("nan"))
            cells.append(f"{v:.3f}" if not (v != v) else "nan")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    out_path = Path("results/utility/latent_env_grad_table.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[done] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/generalist_G16")
    parser.add_argument("--out-dir", default="results/utility/latent_env_grad")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_eval:
        from code.scripts.utility.latent_env_grad import run_one
        for model in G16_CKPTS:
            ckpt = f"{args.results_dir}/{model}/seed_0/final.pt"
            if not os.path.exists(ckpt):
                print(f"[skip] {model}")
                continue
            for env in ENVS:
                try:
                    run_one(ckpt, env, args.n_steps, "cpu", str(out_dir / model / f"{env}.json"))
                except Exception as e:
                    print(f"[err] {model} on {env}: {e}")

    aggregate(out_dir, None)


if __name__ == "__main__":
    main()

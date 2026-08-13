"""Run the frozen-encoder sample efficiency sweep across 12 G16 ckpts.

Output: results/utility/sample_efficiency/<model>/<env>.json + table md.
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
    fracs = ["0.010", "0.050", "0.100", "0.250", "1.000"]
    lines = [
        "# Frozen-encoder sample efficiency (v0.7.7 utility experiment 3)",
        "",
        "**Hypothesis**: a calibrated latent should be usable by a tiny linear policy even with little data. A collapse / noise / over-reactive latent should need more data.",
        "",
        "## mean_cos_dist_terminal per (model × env × data fraction)",
        "",
        "Lower is better. The collapse latent (MLP) gives ~0.0 at all fractions because the policy can't move in a constant latent space.",
        "",
    ]
    for env in ENVS:
        lines.append(f"### env = {env}")
        lines.append("")
        lines.append("| model | " + " | ".join([f"{f} data" for f in fracs]) + " |")
        lines.append("|---|" + "|".join(["---"] * len(fracs)) + "|")
        for model in G16_CKPTS:
            p = out_dir / model / f"{env}.json"
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)
            cells = []
            for fkey in fracs:
                v = d.get("per_fraction", {}).get(fkey, {}).get("mean_cos_dist_terminal", float("nan"))
                cells.append(f"{v:.4f}" if not (v != v) else "nan")
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines.append("")
    out_path = Path("results/utility/sample_efficiency_table.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[done] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/generalist_G16")
    parser.add_argument("--out-dir", default="results/utility/sample_efficiency")
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--fractions", type=str, default="0.01,0.05,0.1,0.25,1.0")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions = tuple(float(f) for f in args.fractions.split(","))

    if not args.skip_eval:
        from code.scripts.utility.sample_efficiency import run_one
        for model in G16_CKPTS:
            ckpt = f"{args.results_dir}/{model}/seed_0/final.pt"
            if not os.path.exists(ckpt):
                print(f"[skip] {model}")
                continue
            for env in ENVS:
                try:
                    run_one(ckpt, env, args.n_steps, fractions, "cpu", str(out_dir / model / f"{env}.json"))
                except Exception as e:
                    print(f"[err] {model} on {env}: {e}")

    aggregate(out_dir, None)


if __name__ == "__main__":
    main()

"""Run the latent-goal MPC horizon sweep across the 12 G16 generalist ckpts.

Outputs per-(model, env) JSONs to results/utility/latent_goal_mpc/<model>/<env>.json
and aggregates to results/utility/latent_goal_mpc_table.md.

Usage:
    python -m code.scripts.utility.run_latent_goal_mpc
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
    "stjewm_trace_only",
    "stjewm_spike_only",
    "stjewm_rate_only",
    "stjewm_no_trace",
    "stjewm_hidden_leak",
    "stjewm_membrane_readout",
    "alif_timecell_baseline",
    "stacked_lif_trace",
    "stacked_lif_free",
    "gru_baseline",
    "mlp_baseline",
]

ENVS = ["cheetah", "walker", "reacher", "finger"]
HORIZONS = [1, 3, 5, 10, 20]


def aggregate(out_dir: Path, args, all_results):
    """Read all per-(model, env) JSONs and write a markdown table."""
    lines = [
        "# Latent-goal MPC horizon sweep (v0.7.7 utility experiment 1)",
        "",
        f"**CEM config**: n_samples={args.cem_samples}, n_elites={args.cem_elites}, n_iters={args.cem_iters}, n_episodes={args.n_episodes}",
        "",
        "## mean_cos_dist_terminal per (model × env × horizon)",
        "",
        "Lower is better. A collapse latent gives ~1e-7; a calibrated latent gives ~0.05; over-reactive gives >0.10 and grows with H.",
        "",
        "| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in G16_CKPTS:
        for env in ENVS:
            json_path = out_dir / model / f"{env}.json"
            if not json_path.exists():
                continue
            with open(json_path) as f:
                d = json.load(f)
            ph = d.get("per_horizon", {})
            row = [model, env]
            for H in HORIZONS:
                v = ph.get(str(H), {}).get("mean_cos_dist_terminal", float("nan"))
                row.append(f"{v:.4f}")
            lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## env_success per (model × env × horizon)",
        "",
        "Env-native success: |state - goal| < per-env tol. The DMC tol is loose (1.0 for cheetah/walker) so most models get 100% trivially. The cos_dist table is the real signal.",
        "",
        "| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |",
        "|---|---|---|---|---|---|---|",
    ])
    for model in G16_CKPTS:
        for env in ENVS:
            json_path = out_dir / model / f"{env}.json"
            if not json_path.exists():
                continue
            with open(json_path) as f:
                d = json.load(f)
            ph = d.get("per_horizon", {})
            row = [model, env]
            for H in HORIZONS:
                v = ph.get(str(H), {}).get("env_success", float("nan"))
                row.append(f"{v:.2f}")
            lines.append("| " + " | ".join(row) + " |")

    out_path = Path("results/utility/latent_goal_mpc_table.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[done] Aggregated table: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/generalist_G16")
    parser.add_argument("--out-dir", default="results/utility/latent_goal_mpc")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--cem-samples", type=int, default=100)
    parser.add_argument("--cem-elites", type=int, default=10)
    parser.add_argument("--cem-iters", type=int, default=10)
    parser.add_argument("--skip-eval", action="store_true", help="only re-aggregate")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_eval:
        from code.scripts.utility.latent_goal_mpc import run_horizon_sweep

        all_results = []
        for model in G16_CKPTS:
            ckpt = f"{args.results_dir}/{model}/seed_0/final.pt"
            if not os.path.exists(ckpt):
                print(f"[skip] {model}: no ckpt at {ckpt}")
                continue
            for env in ENVS:
                t0 = time.time()
                try:
                    result = run_horizon_sweep(
                        ckpt_path=ckpt,
                        env_kind=env,
                        horizons=HORIZONS,
                        n_episodes=args.n_episodes,
                        cem_samples=args.cem_samples,
                        cem_elites=args.cem_elites,
                        cem_iters=args.cem_iters,
                        out_path=str(out_dir / model / f"{env}.json"),
                    )
                    result["wall_time_sec_total"] = time.time() - t0
                    all_results.append({"model": model, "env": env, "result": result})
                except Exception as e:
                    print(f"[err] {model} on {env}: {e}")
                    all_results.append({"model": model, "env": env, "error": str(e)})

        with open(out_dir / "_index.json", "w") as f:
            json.dump(all_results, f, indent=2)

    aggregate(out_dir, args, [])


if __name__ == "__main__":
    main()

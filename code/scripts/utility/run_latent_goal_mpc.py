"""Run the latent-goal MPC horizon sweep across the 12 G16 generalist ckpts.

Outputs per-(model, env) JSONs to results/utility/latent_goal_mpc/<model>/<env>.json
and aggregates to results/utility/latent_goal_mpc_table.md.

Usage:
    python -m code.scripts.utility.run_latent_goal_mpc
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
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
    "lewm_baseline_v2",
]

ENVS = ["cheetah", "walker", "reacher", "finger"]
HORIZONS = [1, 3, 5, 10, 20]


def aggregate(out_dir: Path, args):
    """Aggregate the complete repaired 12-model grid without mixing protocols."""
    all_results = []
    for model in G16_CKPTS:
        for env in ENVS:
            json_path = out_dir / model / f"{env}.json"
            with open(json_path) as f:
                result = json.load(f)
            if result.get("protocol_version") != 2:
                raise ValueError(f"Archive and rerun the legacy utility result: {json_path}")
            expected = {
                "horizons": HORIZONS, "n_episodes": args.n_episodes,
                "cem_samples": args.cem_samples, "cem_elites": args.cem_elites,
                "cem_iters": args.cem_iters, "cem_cost": "squared_l2",
                "goal_offset": 25, "history_size": 1, "eval_budget": 50,
            }
            for key, value in expected.items():
                if result.get(key) != value:
                    raise ValueError(f"Inconsistent {key} in {json_path}: {result.get(key)}")
            all_results.append({"model": model, "env": env, "result": result})

    lines = [
        "# Latent-goal MPC horizon sweep",
        "",
        f"**Grid**: {len(G16_CKPTS)} model configurations × {len(ENVS)} environments.",
        f"**CEM config**: n_samples={args.cem_samples}, n_elites={args.cem_elites}, n_iters={args.cem_iters}, n_episodes={args.n_episodes}",
        "**Protocol**: t→t+25 offline goals, 50-step budget, fresh qpos initialization with zero velocity; observed-state replanning after each H-step action chunk.",
        "**Planner objective**: canonical squared L2 in latent space; candidates are native-dimensional controls, bounded and zero-padded before model prediction.",
        "",
        "## mean_cos_dist_terminal per (model × env × horizon)",
        "",
        "Terminal metric: (1 − cosine_similarity) / 2. Interpret with native physical success; latent collapse can also produce small distances.",
        "",
        "| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in all_results:
        name = "LeWM" if record["model"] == "lewm_baseline_v2" else record["model"]
        row = [name, record["env"]]
        for H in HORIZONS:
            row.append(f'{record["result"]["per_horizon"][str(H)]["mean_cos_dist_terminal"]:.4f}')
        lines.append("| " + " | ".join(row) + " |")
    lines.extend([
        "",
        "## env_success per (model × env × horizon)",
        "",
        "Success uses unpadded native qpos RMS distance: thresholds cheetah/walker=0.1, reacher=0.05, finger=0.3.",
        "",
        "| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |",
        "|---|---|---|---|---|---|---|",
    ])
    for record in all_results:
        name = "LeWM" if record["model"] == "lewm_baseline_v2" else record["model"]
        row = [name, record["env"]]
        for H in HORIZONS:
            row.append(f'{record["result"]["per_horizon"][str(H)]["env_success"]:.2f}')
        lines.append("| " + " | ".join(row) + " |")

    index_path = out_dir / "_index.json"
    table_path = out_dir.parent / f"{out_dir.name}_table.md"
    for path in (index_path, table_path):
        if path.exists():
            raise FileExistsError(f"Archive the existing aggregate before rerunning: {path}")
    with open(index_path, "x") as f:
        json.dump(all_results, f, indent=2, allow_nan=False)
    with open(table_path, "x") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] Aggregated {len(all_results)} records: {index_path}, {table_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/generalist_G16")
    parser.add_argument("--out-dir", default="results/utility/latent_goal_mpc")
    parser.add_argument("--models", nargs="+", choices=G16_CKPTS, default=G16_CKPTS)
    parser.add_argument("--envs", nargs="+", choices=ENVS, default=ENVS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--cem-samples", type=int, default=100)
    parser.add_argument("--cem-elites", type=int, default=10)
    parser.add_argument("--cem-iters", type=int, default=10)
    parser.add_argument("--skip-eval", action="store_true", help="only re-aggregate the complete grid")
    parser.add_argument("--skip-aggregate", action="store_true", help="write only selected per-cell JSONs")
    args = parser.parse_args()
    if args.skip_eval and args.skip_aggregate:
        parser.error("--skip-eval and --skip-aggregate cannot be combined")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_aggregate:
        for path in (out_dir / "_index.json", out_dir.parent / f"{out_dir.name}_table.md"):
            if path.exists():
                raise FileExistsError(f"Archive the existing aggregate before rerunning: {path}")

    if not args.skip_eval:
        from code.scripts.utility.latent_goal_mpc import run_horizon_sweep

        for model in args.models:
            ckpt = Path(args.results_dir) / model / "seed_0" / "final.pt"
            if not ckpt.is_file():
                raise FileNotFoundError(f"No checkpoint for {model}: {ckpt}")
            for env in args.envs:
                path = out_dir / model / f"{env}.json"
                if path.exists():
                    raise FileExistsError(f"Archive the existing result before rerunning: {path}")
        failures = []
        for model in args.models:
            ckpt = Path(args.results_dir) / model / "seed_0" / "final.pt"
            for env in args.envs:
                print(f"[run] {model} / {env} on {args.device}", flush=True)
                try:
                    run_horizon_sweep(
                        ckpt_path=str(ckpt), env_kind=env, horizons=HORIZONS,
                        n_episodes=args.n_episodes,
                        cem_samples=args.cem_samples, cem_elites=args.cem_elites,
                        cem_iters=args.cem_iters, device=args.device,
                        out_path=str(out_dir / model / f"{env}.json"),
                    )
                except Exception as exc:
                    traceback.print_exc()
                    failures.append(f"{model}/{env}: {exc}")
        if failures:
            raise RuntimeError("Failed utility cells:\n" + "\n".join(failures))
    if not args.skip_aggregate:
        aggregate(out_dir, args)


if __name__ == "__main__":
    main()

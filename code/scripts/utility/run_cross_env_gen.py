"""Run the cross-environment generalisation suite.

Loops over the four requested minus-walker/humanoid checkpoints, then
aggregates them beside the existing full-G16 checkpoints.

Usage:
    /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.utility.run_cross_env_gen
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path("/home/lx/snn")
MODELS = [
    "stjewm_trace_only",
    "stjewm_spike_only",
    "mlp_baseline",
    "gru_baseline",
]
PYTHON = "/home/lx/miniconda3/envs/snn/bin/python"


def run(cmd: list[str]) -> None:
    print("[run-cross-env] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all cross-env generalisation models.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--latent-steps", type=int, default=200)
    p.add_argument("--align-steps", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--skip-full-baseline", action="store_true")
    p.add_argument("--eval-full-baseline", action="store_true")
    p.add_argument("--target-only-full-baseline", action="store_true",
                   help="Only refresh full-G16 metrics for the four target models, not all 12.")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--force-latent", action="store_true")
    p.add_argument("--force-align", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def common_args(args: argparse.Namespace) -> list[str]:
    out = [
        "--seed", str(args.seed),
        "--n-episodes", str(args.n_episodes),
        "--latent-steps", str(args.latent_steps),
        "--align-steps", str(args.align_steps),
        "--device", args.device,
    ]
    for flag in ("skip_train", "skip_eval", "skip_metrics", "force_train",
                 "force_eval", "force_latent", "force_align"):
        if getattr(args, flag):
            out.append("--" + flag.replace("_", "-"))
    return out


def main() -> int:
    args = parse_args()
    base = [PYTHON, "-m", "code.scripts.utility.cross_env_gen"]
    if args.aggregate_only:
        run(base + ["--aggregate-only", "--seed", str(args.seed)])
        return 0

    if not args.skip_full_baseline:
        full = base + ["--ensure-full-baseline"] + common_args(args)
        if not args.target_only_full_baseline:
            full.append("--all-full-models")
        if args.eval_full_baseline:
            full.append("--eval-full-baseline")
        run(full)

    for model in MODELS:
        run(base + ["--model", model] + common_args(args))

    run(base + ["--aggregate-only", "--seed", str(args.seed)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

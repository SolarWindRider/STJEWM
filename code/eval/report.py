"""Aggregate per-env eval JSONs into a single report table.

Reads all eval JSONs in `results/*/eval.json` and `results/*/eval_*.json`,
and produces a markdown table summarizing the 24-env benchmark.

Usage:
    python -m code.eval.report --results-dir /home/lx/snn/results --out /home/lx/snn/docs/report/EVAL_TABLE.md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def collect_results(results_dir: str) -> List[Dict[str, Any]]:
    """Walk results dir, load every .json that looks like an eval result."""
    results = []
    for path in Path(results_dir).rglob("*.json"):
        if "loss_log" in path.name:
            continue
        if "summary" in path.name.lower():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        # Heuristic: must have either success_rate_lewm or success_rate_env
        if "success_rate_lewm" in data or "success_rate_env" in data or "success_rate" in data:
            data["_path"] = str(path)
            results.append(data)
    return results


def format_table(results: List[Dict]) -> str:
    """Format results as a markdown table."""
    if not results:
        return "# No eval results found\n"

    # Group by (env_id, model_kind)
    by_env: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        env_id = r.get("env_id", "unknown")
        by_env[env_id].append(r)

    lines = ["# ST-JEWM vs LeWM-style baseline — 24-env evaluation summary\n"]
    lines.append("| Env | Model | n_eps | n_seeds | LeWM SR (cos<0.1) | Env-native SR | Mean cos_dist | Mean phys_dist |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for env_id in sorted(by_env.keys()):
        for r in by_env[env_id]:
            model_kind = r.get("_path", "").split("/")[-2] if "_path" in r else "?"
            n_eps = r.get("n_episodes", 0)
            n_seeds = r.get("n_seeds", 0)
            lewm_sr = r.get("success_rate_lewm", float("nan"))
            lewm_std = r.get("success_rate_lewm_std", 0.0)
            env_sr = r.get("success_rate_env", float("nan"))
            env_std = r.get("success_rate_env_std", 0.0)
            mean_cos = r.get("mean_cos_dist", float("nan"))
            mean_phys = r.get("mean_phys_dist", float("nan"))
            lines.append(
                f"| {env_id} | {model_kind} | {n_eps} | {n_seeds} | "
                f"{lewm_sr:.3f} ± {lewm_std:.3f} | {env_sr:.3f} ± {env_std:.3f} | "
                f"{mean_cos:.4f} | {mean_phys:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="/home/lx/snn/results")
    p.add_argument("--out", default="/home/lx/snn/docs/report/EVAL_TABLE.md")
    args = p.parse_args()
    results = collect_results(args.results_dir)
    print(f"Found {len(results)} eval result files in {args.results_dir}")
    table = format_table(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(table)
    print(f"Wrote {args.out}")
    print()
    print(table)


if __name__ == "__main__":
    main()

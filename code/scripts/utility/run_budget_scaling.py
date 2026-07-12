"""Driver for the data-budget scaling sweep (v0.7.8 utility experiment 2).

Loops over 3 models x 3 data budgets (0.5x, 1.0x, 2.0x). For each cell it:
  - trains a new ckpt (skips if final.pt already exists) — at 0.5x and 2.0x
  - measures latent_stats + event_align + closed_loop on 6 DMC envs
  - aggregates to results/utility/budget_scaling/<model>/<frac>.json

The final results/utility/budget_scaling_table.md is the paper-ready summary.

Usage:
    python -m code.scripts.utility.run_budget_scaling
    python -m code.scripts.utility.run_budget_scaling --skip-train
    python -m code.scripts.utility.run_budget_scaling --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, "/home/lx/snn")


MODELS = [
    "stjewm_trace_only",
    "stjewm_spike_only",
    "mlp_baseline",
]
FRACS = [0.5, 1.0, 2.0]
FRAC_LABELS = {0.5: "0.5", 1.0: "1.0", 2.0: "2.0"}


def aggregate_table(out_dir: Path, table_path: Path) -> None:
    """Read all per-cell JSONs and write the markdown table."""
    rows: List[Dict] = []
    for model in MODELS:
        for frac in FRACS:
            label = FRAC_LABELS[frac]
            jp = out_dir / model / f"{label}.json"
            if not jp.exists():
                continue
            d = json.loads(jp.read_text())
            rows.append(d)

    if not rows:
        print(f"[aggregate] no per-cell JSONs found under {out_dir}")
        return

    lines: List[str] = []
    lines.append("# Data-budget compression sweep\n")
    lines.append("Diagnotics drift when the per-env data budget is scaled by\n"
                 "frac in {0.5, 1.0, 2.0} where 1.0 = the existing G16 budget\n"
                 "(10K windows per env × 16 envs).\n")
    lines.append("| model | frac | env-SR (avg ± std) | div (avg) | resp (avg) | ρ (avg) |")
    lines.append("|-------|------|--------------------|-----------|------------|---------|")

    for d in rows:
        env_sr = d.get("env-SR_avg", float("nan"))
        env_sr_std = d.get("env-SR_std", 0.0)
        div = d.get("div_avg", float("nan"))
        resp = d.get("resp_avg", float("nan"))
        rho = d.get("rho_avg", float("nan"))
        lines.append(
            f"| {d['model']} | {d['frac_label']} | "
            f"{env_sr:.3f} ± {env_sr_std:.3f} | "
            f"{div:.4f} | {resp:.3f} | {rho:.3f} |"
        )

    lines.append("")
    lines.append("Cells with frac=1.0 reuse existing G16 outputs. Cells with frac≠1.0\n"
                 "train fresh ckpts with per-env max_windows = round(BASE_PER_ENV × frac).\n")

    # Highlight the win/loss rows
    lines.append("## Robustness narrative\n")
    lines.append("Lower absolute drift across the frac axis = more robust.\n")

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines) + "\n")
    print(f"[aggregate] wrote {table_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/utility/budget_scaling")
    ap.add_argument("--table-path", default="results/utility/budget_scaling_table.md")
    ap.add_argument("--skip-train", action="store_true",
                    help="don't retrain (use existing ckpts; useful to re-eval).")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="only re-aggregate from the existing per-cell JSONs.")
    ap.add_argument("--fracs", type=str, default=None,
                    help="comma-separated list of fracs to run (default: 0.5,1.0,2.0)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fracs = FRACS
    if args.fracs is not None:
        fracs = [float(x) for x in args.fracs.split(",")]

    if not args.aggregate_only:
        from code.scripts.utility.budget_scaling import (
            run_for_1x_baseline, run_for_new_ckpt,
        )

        for model in MODELS:
            for frac in fracs:
                label = FRAC_LABELS[frac]
                cell_out = out_dir / model / f"{label}.json"
                cell_out.parent.mkdir(parents=True, exist_ok=True)
                if cell_out.exists():
                    print(f"[skip-aggregate] {model}/{label}: {cell_out} already exists")
                    continue
                t0 = time.time()
                if frac == 1.0 or args.skip_train:
                    result = run_for_1x_baseline(model, cell_out)
                else:
                    result = run_for_new_ckpt(model, frac, label, cell_out)
                dt = time.time() - t0
                print(f"[cell] {model} frac={label}: {dt/60:.1f} min")

    aggregate_table(out_dir, Path(args.table_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

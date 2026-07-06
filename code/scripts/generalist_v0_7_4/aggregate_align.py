"""Aggregate event_align JSONs into a per-(env, model) table.

Reads:
  results/event_align/<env>_<model>_seed<s>.json   (v0.7.4 layout)
  results/event_align/<env>_<model>.json            (legacy v0.7.3 layout)

Writes:
  generalist_align_table.md (or generalist_align_table_<suite>.md with --out-name)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

MODELS = [
    "stjewm_trace_only",
    "stjewm_spike_only",
    "stjewm_rate_only",
    "stjewm_no_trace",
    "stjewm_hidden_leak",
    "stjewm_membrane_readout",
    "cubifae_baseline",
    "gru_baseline",
    "lewm_baseline_v2",
    "slt_lif_mpc_trace",
    "slt_lif_mpc_free",
    "mlp_baseline",
]

ENV_NAMES = ["cheetah", "walker", "cartpole_2d", "pendulum_2d", "finger", "ball_in_cup"]


def parse_filename(stem: str) -> tuple[str | None, str | None]:
    """Return (env, model) by trying multiple filename conventions.

    Environment names can contain underscores (e.g. cartpole_2d), so we
    match against a known list of env names rather than relying on regex
    greediness.
    """
    for env in ENV_NAMES:
        if not stem.startswith(env + "_"):
            continue
        model_part = stem[len(env) + 1:]
        m = re.match(r"^(?P<model>.+?)_seed\d+$", model_part)
        if m:
            return env, m.group("model")
        return env, model_part
    return None, None


def collect(align_dir: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    rows: dict[tuple[str, str], dict] = {}
    if not align_dir.exists():
        return rows
    for fp in sorted(align_dir.glob("*.json")):
        env, model = parse_filename(fp.stem)
        if env is None or model is None:
            continue
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        if d.get("skipped"):
            continue
        rho = d.get("corr_obs_latent")
        if rho is None:
            continue
        key = (env, model)
        rows.setdefault(key, {"rhos": [], "n_steps": 0})
        rows[key]["rhos"].append(float(rho))
        rows[key]["n_steps"] = max(rows[key]["n_steps"], int(d.get("n_steps", 0)))
    for k, v in rows.items():
        v["rho_mean"] = sum(v["rhos"]) / len(v["rhos"]) if v["rhos"] else None
    return rows


def write_md(rows: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# Event Boundary Alignment (Pearson ρ between obs-event and latent-event)\n\n")
        f.write("Pearson correlation between obs first-difference (event strength) and latent first-difference.\n")
        f.write("High ρ means the latent preserves obs-level event timing. Aggregated across seeds.\n\n")
        envs_present = sorted({env for (env, _) in rows.keys()})
        if not envs_present:
            f.write("(no data)\n")
            return
        f.write("| model | " + " | ".join(envs_present) + " | AVG |\n")
        f.write("|" + "---|" * (len(envs_present) + 2) + "\n")
        for m in MODELS:
            cells = []
            env_rhos = []
            for env in envs_present:
                v = rows.get((env, m))
                if v is None or v["rho_mean"] is None:
                    cells.append("-")
                else:
                    cells.append(f"{v['rho_mean']:.3f}")
                    env_rhos.append(v["rho_mean"])
            avg = f"{sum(env_rhos)/len(env_rhos):.3f}" if env_rhos else "-"
            f.write(f"| {m} | " + " | ".join(cells) + f" | **{avg}** |\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-dir", default="/home/lx/snn/results/event_align",
                    help="Directory containing event_align JSONs.")
    ap.add_argument("--out-dir", default="/home/lx/snn/results/aggregate")
    ap.add_argument("--out-name", default=None,
                    help="Override output filename (default: generalist_align_table.md).")
    args = ap.parse_args()
    rows = collect(Path(args.align_dir))
    out_name = args.out_name or "generalist_align_table.md"
    out_md = Path(args.out_dir) / out_name
    out_json = Path(args.out_dir) / out_name.replace(".md", ".json")
    write_md(rows, out_md)
    out_json.write_text(json.dumps({f"{env}__{model}": v for (env, model), v in rows.items()}, indent=2))
    print(f"[aggregate_align] {len(rows)} (env, model) pairs")
    print(f"[aggregate_align] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
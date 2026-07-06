"""Aggregate generalist eval JSONs into a single comparison table.

Reads:
  results/generalist/<MODEL>/eval_<ENV>.json   for each of N generalist models
  configs/generalist_<K>env.json               to know the env list

Output:
  results/aggregate/generalist_table.md
  results/aggregate/generalist_table.json

Usage:
    python -m code.scripts.aggregate_generalist
    python -m code.scripts.aggregate_generalist --spec configs/generalist_20env.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


GENERALIST_MODELS = [
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

# Stress envs get a section header in the markdown table.
STRESS_ENVS = {"pusht_ood", "tworoom_long", "cartpole_flicker", "cheetah_velhidden"}


def load_eval_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def row_for_model_env(eval_path: Path) -> Dict[str, Any]:
    """Pull env-SR / LeWM-SR / cos_dist from a single eval JSON.

    Tries both the flat path (`<model>/eval_<env>.json`) and the per-seed
    path (`<model>/seed_<s>/eval_<env>.json`). For per-seed layout, averages
    across the discovered seeds and reports std.
    """
    def _one(p: Path) -> Optional[Dict[str, Any]]:
        return load_eval_json(p)
    seeds: list[int] = []
    if eval_path.exists():
        ev = _one(eval_path)
        seeds_data = [ev] if ev is not None else []
    else:
        seeds_data = []
        # Look for any seed_<s>/eval_<env>.json under the same model dir.
        model_dir = eval_path.parent
        for sd in sorted(model_dir.glob("seed_*")):
            if not sd.is_dir():
                continue
            try:
                seed_n = int(sd.name.split("_", 1)[1])
            except ValueError:
                continue
            seeds.append(seed_n)
            ev = _one(sd / eval_path.name)
            if ev is not None:
                seeds_data.append(ev)
    if not seeds_data:
        return {"env_sr": None, "env_sr_std": None, "lewm_sr": None,
                "lewm_sr_std": None, "cos_dist": None, "n_episodes": None, "n_seeds": 0}
    import statistics
    env_srs = [e.get("success_rate_env") for e in seeds_data if e.get("success_rate_env") is not None]
    lewm_srs = [e.get("success_rate_lewm") for e in seeds_data if e.get("success_rate_lewm") is not None]
    cos_dists = [e.get("mean_cos_dist") for e in seeds_data if e.get("mean_cos_dist") is not None]
    return {
        "env_sr": statistics.mean(env_srs) if env_srs else None,
        "env_sr_std": statistics.stdev(env_srs) if len(env_srs) > 1 else 0.0,
        "lewm_sr": statistics.mean(lewm_srs) if lewm_srs else None,
        "lewm_sr_std": statistics.stdev(lewm_srs) if len(lewm_srs) > 1 else 0.0,
        "cos_dist": statistics.mean(cos_dists) if cos_dists else None,
        "n_episodes": seeds_data[0].get("n_episodes"),
        "n_seeds": len(seeds_data),
    }


def write_markdown(rows: List[Dict[str, Any]], out_path: Path, spec_path: Path) -> None:
    envs = [r["env"] for r in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# Generalist vs Specialist — comparison table\n\n")
        f.write(f"Spec: `{spec_path}`\n\n")
        f.write(f"Models: {', '.join(GENERALIST_MODELS)}\n\n")
        f.write("Cells: env-native success rate (%) and LeWM-SR (cos_dist<0.1) (%)\n")
        f.write("'-' = eval JSON not yet produced\n\n")
        f.write("| env | " + " | ".join(
            f"{m} env-SR | {m} LeWM-SR" for m in GENERALIST_MODELS
        ) + " |\n")
        f.write("|" + "---|" * (1 + 2 * len(GENERALIST_MODELS)) + "\n")
        for r in rows:
            env = r["env"]
            mark = " (stress)" if env in STRESS_ENVS else ""
            cells = []
            for m in GENERALIST_MODELS:
                d = r["models"].get(m, {})
                env_sr = d.get("env_sr")
                lewm_sr = d.get("lewm_sr")
                cells.append(
                    f"{env_sr*100:.1f}" if env_sr is not None else "-"
                )
                cells.append(
                    f"{lewm_sr*100:.1f}" if lewm_sr is not None else "-"
                )
            f.write(f"| {env}{mark} | " + " | ".join(cells) + " |\n")
        # Aggregate rows
        f.write("\n")
        for metric_key, metric_label in [("env_sr", "env-SR"), ("lewm_sr", "LeWM-SR")]:
            f.write(f"### {metric_label} AVG per model\n\n")
            f.write("| model | all 20 | std only | stress only |\n|---|---|---|---|\n")
            for m in GENERALIST_MODELS:
                vals_all, vals_std, vals_str = [], [], []
                for r in rows:
                    d = r["models"].get(m, {})
                    v = d.get(metric_key)
                    if v is None:
                        continue
                    vals_all.append(v)
                    if r["env"] in STRESS_ENVS:
                        vals_str.append(v)
                    else:
                        vals_std.append(v)
                f.write(
                    f"| {m} | "
                    f"{(np.mean(vals_all)*100 if vals_all else 0):.1f} | "
                    f"{(np.mean(vals_std)*100 if vals_std else 0):.1f} | "
                    f"{(np.mean(vals_str)*100 if vals_str else 0):.1f} |\n"
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/generalist_16env.json")
    ap.add_argument("--results-dir", default="/home/lx/snn/results/generalist")
    ap.add_argument("--out-dir", default="/home/lx/snn/results/aggregate")
    args = ap.parse_args()
    spec_path = Path(args.spec)
    if not spec_path.exists():
        raise SystemExit(f"Spec file not found: {spec_path}")
    specs = json.loads(spec_path.read_text())
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    rows = []
    for entry in specs:
        env = entry["env_id"]
        row = {"env": env, "models": {}}
        for m in GENERALIST_MODELS:
            eval_path = results_dir / m / f"eval_{env}.json"
            row["models"][m] = row_for_model_env(eval_path)
        rows.append(row)

    out_md = out_dir / "generalist_table.md"
    out_json = out_dir / "generalist_table.json"
    write_markdown(rows, out_md, spec_path)
    out_json.write_text(json.dumps({"spec": str(spec_path), "rows": rows}, indent=2))
    print(f"[aggregate_generalist] wrote {out_md}")
    print(f"[aggregate_generalist] wrote {out_json}")


if __name__ == "__main__":
    main()

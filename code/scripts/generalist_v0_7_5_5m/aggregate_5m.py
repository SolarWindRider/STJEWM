"""Aggregate 5M-aligned generalist eval JSONs across (split, model, env) into
master tables. Mirror of code/scripts/generalist_v0_7_5/aggregate_master.py
but pointed at results/5m/ and results/5m_stress/.

Reads:
  results/5m/<split>/<MODEL>/seed_0/eval_<env>.json   (ID envs)
  results/5m_stress/<split>/<MODEL>/seed_0/eval_<env>.json   (stress envs, if any)
  results/probe_5m/<env>_<model>_<target>.json  (event-AUROC probes)
  results/aggregate/event_probes_5m/<env>_<model>_<target>.json  (mirror)

Writes:
  results/aggregate/generalist_5m_table.md
  results/aggregate/generalist_5m_table.json
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

GENERALIST_MODELS = [
    "stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only",
    "stjewm_no_trace", "stjewm_hidden_leak", "stjewm_membrane_readout",
    "alif_timecell_baseline", "gru_baseline", "lewm_baseline_v2",
    "stacked_lif_trace", "stacked_lif_free", "mlp_baseline", "lif_transformer_baseline",
]
COLLAPSE_CONTROL = "mlp_baseline"

DEFAULT_RESULTS = Path("/home/lx/snn/results/5m")
DEFAULT_STRESS = Path("/home/lx/snn/results/5m_stress")
DEFAULT_PROBES = Path("/home/lx/snn/results/probe_5m")
DEFAULT_OUT = Path("/home/lx/snn/results/aggregate")
STRESS_ENVS = {"pusht_ood", "tworoom_long", "cartpole_flicker", "cheetah_velhidden"}


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def gather_evals(results_root: Path, models: List[str], splits: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for split in splits:
        for model in models:
            ckpt = results_root / split / model / "seed_0" / "final.pt"
            if not ckpt.exists():
                continue
            ev_dir = results_root / split / model / "seed_0"
            for f in sorted(ev_dir.glob("eval_*.json")):
                env = f.stem.replace("eval_", "")
                d = load_json(f)
                if d is None:
                    continue
                rows.append({
                    "split": split,
                    "model": model,
                    "env": env,
                    "success_rate_lewm": d.get("success_rate_lewm"),
                    "success_rate_lewm_005": d.get("success_rate_lewm_005"),
                    "success_rate_lewm_001": d.get("success_rate_lewm_001"),
                    "success_rate_env": d.get("success_rate_env"),
                    "mean_cos_dist": d.get("mean_cos_dist"),
                    "mean_phys_dist": d.get("mean_phys_dist"),
                })
    return rows


def gather_probes(probes_root: Path, models: List[str], envs: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for env in envs:
        for model in models:
            for f in sorted(probes_root.glob(f"{env}_{model}_*.json")):
                target = f.stem.split("_")[-1]
                d = load_json(f)
                if d is None:
                    continue
                rows.append({
                    "env": env,
                    "model": model,
                    "target": target,
                    "auroc": d.get("r2") if d.get("metric") == "auroc" else None,
                    "skipped": d.get("skipped", False),
                    "reason": d.get("reason", ""),
                })
    return rows


def fmt_pct(v):
    if v is None: return "—"
    return f"{v*100:.1f}%"


def fmt_f(v, n=3):
    if v is None: return "—"
    return f"{v:.{n}f}"


def mean_or_none(vals):
    return statistics.mean(vals) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--stress", type=Path, default=DEFAULT_STRESS)
    p.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    # Discover splits
    splits = sorted([p.name for p in args.results.iterdir() if p.is_dir() and p.name != "_logs"])
    envs = sorted({ev["env"] for ev in gather_evals(args.results, GENERALIST_MODELS, splits)})

    evals = gather_evals(args.results, GENERALIST_MODELS, splits)
    stress_evals = gather_evals(args.stress, GENERALIST_MODELS, splits)
    probes = gather_probes(args.probes, GENERALIST_MODELS, envs)

    print(f"Splits: {len(splits)}, models: {len(GENERALIST_MODELS)}, evals: {len(evals)}, probes: {len(probes)}")

    # Per-env model table: LeWM-SR per (env, model)
    args.out.mkdir(parents=True, exist_ok=True)
    out_md = args.out / "generalist_5m_table.md"
    out_json = args.out / "generalist_5m_table.json"

    rows = []
    with out_md.open("w") as f:
        f.write(f"# Generalist 5M-Aligned Table\n\n")
        f.write(f"Splits: {', '.join(splits)}\n")
        f.write(f"Models: {len(GENERALIST_MODELS)} ({', '.join(GENERALIST_MODELS)})\n\n")
        # Per-split env-SR (env native success rate averaged per model)
        f.write("## Per-split LeWM-SR (env-SR) per model\n\n")
        f.write("| Split | Model | n_envs | LeWM-SR | env-SR | cos_dist |\n")
        f.write("|---|---|---|---|---|---|\n")
        for split in splits:
            for model in GENERALIST_MODELS:
                split_rows = [r for r in evals if r["split"] == split and r["model"] == model]
                if not split_rows:
                    continue
                lewm = [r["success_rate_lewm"] for r in split_rows if r["success_rate_lewm"] is not None]
                env = [r["success_rate_env"] for r in split_rows if r["success_rate_env"] is not None]
                cos = [r["mean_cos_dist"] for r in split_rows if r["mean_cos_dist"] is not None]
                f.write(f"| {split} | {model} | {len(split_rows)} | "
                        f"{fmt_pct(statistics.mean(lewm) if lewm else None)} | "
                        f"{fmt_pct(statistics.mean(env) if env else None)} | "
                        f"{fmt_f(statistics.mean(cos) if cos else None)} |\n")
                rows.append({
                    "split": split, "model": model, "n_envs": len(split_rows),
                    "lewm_sr": statistics.mean(lewm) if lewm else None,
                    "env_sr": statistics.mean(env) if env else None,
                    "mean_cos_dist": statistics.mean(cos) if cos else None,
                })

        # Per-env LeWM-SR (all splits pooled)
        f.write("\n## Per-env LeWM-SR per model (all splits pooled)\n\n")
        all_envs = sorted({r["env"] for r in evals})
        f.write("| Env | " + " | ".join(GENERALIST_MODELS) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(GENERALIST_MODELS)) + "|\n")
        for env in all_envs:
            row_vals = []
            for model in GENERALIST_MODELS:
                rows_for = [r for r in evals if r["env"] == env and r["model"] == model]
                lewm = [r["success_rate_lewm"] for r in rows_for if r["success_rate_lewm"] is not None]
                row_vals.append(fmt_pct(mean_or_none(lewm)))
            f.write(f"| {env} | " + " | ".join(row_vals) + " |\n")

        # Probes: per-env per-model per-target AUROC
        f.write("\n## Probes (event-AUROC)\n\n")
        f.write(f"Total probes: {len(probes)} (skipped={sum(1 for p in probes if p['skipped'])}, OK={sum(1 for p in probes if not p['skipped'])})\n")
        # Per-target per-model mean
        f.write("\n| Target | " + " | ".join(GENERALIST_MODELS) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(GENERALIST_MODELS)) + "|\n")
        targets = sorted({p["target"] for p in probes})
        for tgt in targets:
            row_vals = []
            for model in GENERALIST_MODELS:
                vals = [p["auroc"] for p in probes
                        if p["target"] == tgt and p["model"] == model and not p["skipped"] and p["auroc"] is not None]
                row_vals.append(fmt_f(mean_or_none(vals), n=3))
            f.write(f"| {tgt} | " + " | ".join(row_vals) + " |\n")

    # JSON
    out_json.write_text(json.dumps({
        "splits": splits,
        "evals": evals,
        "stress_evals": stress_evals,
        "probes": probes,
    }, indent=2))
    print(f"Wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()

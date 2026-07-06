"""Aggregate generalist eval JSONs across models / seeds / envs into a
master table.

Reads:
  results/generalist/<MODEL>/seed_<s>/eval_<env>.json
  results/generalist_stress/<MODEL>/seed_<s>/eval_<env>.json

Writes:
  results/aggregate/generalist_master_table.md
  results/aggregate/generalist_master_table.json

Usage:
    python -m code.scripts.generalist_v0_7_4.aggregate_master
    python -m code.scripts.generalist_v0_7_4.aggregate_master --suite G16
    python -m code.scripts.generalist_v0_7_4.aggregate_master --probes --align
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

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
COLLAPSE_CONTROL = "mlp_baseline"

STRESS_ENVS = {"pusht_ood", "tworoom_long", "cartpole_flicker", "cheetah_velhidden"}

DEFAULT_RESULTS = Path("/home/lx/snn/results/generalist")
DEFAULT_STRESS = Path("/home/lx/snn/results/generalist_stress")
DEFAULT_OUT = Path("/home/lx/snn/results/aggregate")


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def collect_one(model: str, env: str, results_dir: Path, seed: int) -> Optional[Dict[str, Any]]:
    """Load one eval JSON. Returns None if missing."""
    fp = results_dir / model / f"seed_{seed}" / f"eval_{env}.json"
    return load_json(fp)


def aggregate_model_env(model: str, env: str, results_dir: Path, seeds: List[int]) -> Dict[str, Any]:
    """Mean and std of env-SR / LeWM-SR / cos_dist across seeds for one (model, env).
    Also reads the per-(model, env) latent_stats JSONs produced by
    measure_latent_stats.py and adds responsiveness / divergence.
    """
    env_srs: List[float] = []
    lewm_srs: List[float] = []
    cos_dists: List[float] = []
    responsives: List[float] = []
    divergences: List[float] = []
    for s in seeds:
        ev = collect_one(model, env, results_dir, s)
        if ev is None:
            continue
        if "success_rate_env" in ev:
            env_srs.append(float(ev["success_rate_env"]))
        if "success_rate_lewm" in ev:
            lewm_srs.append(float(ev["success_rate_lewm"]))
        if "mean_cos_dist" in ev:
            cos_dists.append(float(ev["mean_cos_dist"]))
        # Latent stats (collapse-robust responsiveness / divergence).
        # One file per (model, env, seed); we average across seeds.
        lat_fp = results_dir / model / f"seed_{s}" / f"latent_stats_{env}.json"
        lat = load_json(lat_fp) if lat_fp.exists() else None
        if lat is not None and not lat.get("skipped"):
            if "responsiveness" in lat:
                responsives.append(float(lat["responsiveness"]))
            if "divergence" in lat:
                divergences.append(float(lat["divergence"]))
    return {
        "env_sr_mean": statistics.mean(env_srs) if env_srs else None,
        "env_sr_std": statistics.stdev(env_srs) if len(env_srs) > 1 else 0.0,
        "lewm_sr_mean": statistics.mean(lewm_srs) if lewm_srs else None,
        "lewm_sr_std": statistics.stdev(lewm_srs) if len(lewm_srs) > 1 else 0.0,
        "cos_dist_mean": statistics.mean(cos_dists) if cos_dists else None,
        "n_seeds_with_data": len(env_srs),
        "responsiveness_mean": statistics.mean(responsives) if responsives else None,
        "divergence_mean": statistics.mean(divergences) if divergences else None,
    }


def discover_seeds(results_dir: Path, models: List[str]) -> List[int]:
    """Return sorted list of seed_<N> integers found under any model dir."""
    seeds = set()
    for m in models:
        mdir = results_dir / m
        if not mdir.exists():
            continue
        for child in mdir.iterdir():
            if child.is_dir() and child.name.startswith("seed_"):
                try:
                    seeds.add(int(child.name.split("_", 1)[1]))
                except Exception:
                    continue
    return sorted(seeds)


def write_section(
    title: str,
    rows: List[Dict[str, Any]],
    envs: List[str],
    models: List[str],
    out,
    collapse_marker: bool = True,
) -> None:
    out.write(f"\n### {title}\n\n")
    out.write("Cells: `env-SR` mean ± std across seeds, in [0, 100]. '-' = no data.\n\n")
    header = "| env | " + " | ".join(
        (f"{m}*" if collapse_marker and m == COLLAPSE_CONTROL else m) for m in models
    ) + " |"
    out.write(header + "\n")
    out.write("|" + "---|" * (1 + len(models)) + "\n")
    for env in envs:
        cells = []
        for m in models:
            row = next((r for r in rows if r["env"] == env and r["model"] == m), None)
            if row is None or row["env_sr_mean"] is None:
                cells.append("-")
            else:
                v = row["env_sr_mean"] * 100
                s = row["env_sr_std"] * 100
                cells.append(f"{v:.1f}±{s:.1f}" if s > 0 else f"{v:.1f}")
        out.write(f"| {env} | " + " | ".join(cells) + " |\n")


def write_summary(rows: List[Dict[str, Any]], envs_id: List[str], envs_stress: List[str],
                  models: List[str], out) -> None:
    """Per-model summary: AVG env-SR, LeWM-SR, worst-25%, collapse-gap,
    responsiveness, divergence (collapse-robust)."""
    out.write("\n### Summary per model\n\n")
    out.write("Columns: `mean` env-SR across ID envs, `lewm` LeWM-SR across ID envs, "
              "`worst25` mean of the bottom-25% envs' env-SR (interference proxy), "
              "`gap` LeWM-SR − env-SR (collapse-inflatable; large +ve = likely collapse), "
              "`resp` responsiveness (mean ‖Δlatent‖ / mean ‖Δobs‖, calibrated ~0.2), "
              "`div` divergence-from-constant (per-dim std, collapse < 0.001).\n\n")
    out.write("| model | mean_id | lewm_id | worst25_id | gap_id | mean_stress | lewm_stress | resp | div |\n")
    out.write("|---|---|---|---|---|---|---|---|---|\n")
    for m in models:
        id_env_srs: List[float] = []
        id_lewm_srs: List[float] = []
        id_responsives: List[float] = []
        id_divergences: List[float] = []
        stress_env_srs: List[float] = []
        stress_lewm_srs: List[float] = []
        for env in envs_id:
            row = next((r for r in rows if r["env"] == env and r["model"] == m), None)
            if row is None:
                continue
            if row["env_sr_mean"] is not None:
                id_env_srs.append(row["env_sr_mean"])
            if row["lewm_sr_mean"] is not None:
                id_lewm_srs.append(row["lewm_sr_mean"])
            if row.get("responsiveness_mean") is not None:
                id_responsives.append(row["responsiveness_mean"])
            if row.get("divergence_mean") is not None:
                id_divergences.append(row["divergence_mean"])
        for env in envs_stress:
            row = next((r for r in rows if r["env"] == env and r["model"] == m), None)
            if row is None:
                continue
            if row["env_sr_mean"] is not None:
                stress_env_srs.append(row["env_sr_mean"])
            if row["lewm_sr_mean"] is not None:
                stress_lewm_srs.append(row["lewm_sr_mean"])
        mean_id = (sum(id_env_srs) / len(id_env_srs) * 100) if id_env_srs else None
        lewm_id = (sum(id_lewm_srs) / len(id_lewm_srs) * 100) if id_lewm_srs else None
        worst25 = (
            sum(sorted(id_env_srs)[: max(1, len(id_env_srs) // 4)]) /
            max(1, len(id_env_srs) // 4) * 100
        ) if id_env_srs else None
        gap = (lewm_id - mean_id) if (mean_id is not None and lewm_id is not None) else None
        mean_stress = (sum(stress_env_srs) / len(stress_env_srs) * 100) if stress_env_srs else None
        lewm_stress = (sum(stress_lewm_srs) / len(stress_lewm_srs) * 100) if stress_lewm_srs else None
        resp = (sum(id_responsives) / len(id_responsives)) if id_responsives else None
        div = (sum(id_divergences) / len(id_divergences)) if id_divergences else None

        def fmt(v, dec=1):
            return f"{v:.{dec}f}" if v is not None else "-"

        marker = " (collapse-control)" if m == COLLAPSE_CONTROL else ""
        out.write(
            f"| {m}{marker} | {fmt(mean_id)} | {fmt(lewm_id)} | {fmt(worst25)} | "
            f"{fmt(gap)} | {fmt(mean_stress)} | {fmt(lewm_stress)} | "
            f"{fmt(resp, 3)} | {fmt(div, 4)} |\n"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all",
                    help="Suite name written as title. Default 'all'.")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--stress-dir", default=str(DEFAULT_STRESS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--probes", action="store_true", help="(Stage 5) Add probe section.")
    ap.add_argument("--align", action="store_true", help="(Stage 5) Add event-align section.")
    ap.add_argument("--probe-dir", default="/home/lx/snn/results/probe")
    ap.add_argument("--align-dir", default="/home/lx/snn/results/event_align")
    ap.add_argument("--merge-all", action="store_true",
                    help="Aggregate G4/G8/G16 into a single consolidated file (generalist_master_table.md).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_all:
        out_json = out_dir / "generalist_master_table.json"
        all_rows = []
        for suite in ("G4", "G8", "G16"):
            suite_results = Path(f"/home/lx/snn/results/generalist{'' if suite == 'G4' else '_' + suite}")
            suite_stress = Path(f"/home/lx/snn/results/generalist{'' if suite == 'G4' else '_' + suite}_stress")
            if not suite_results.exists():
                continue
            seeds_s = discover_seeds(suite_results, GENERALIST_MODELS)
            if not seeds_s: seeds_s = [0]
            for m in GENERALIST_MODELS:
                for env in ("pendulum_2d", "dog", "tworoom", "quadruped", "reacher", "hopper",
                            "fish", "stacker", "cheetah", "walker", "pusht", "humanoid",
                            "finger", "ball_in_cup", "cartpole_2d"):
                    agg = aggregate_model_env(m, env, suite_results, seeds_s)
                    if agg["env_sr_mean"] is not None:
                        all_rows.append({"suite": suite, "env": env, "model": m, **agg})
                for env in ("pusht_ood", "tworoom_long", "cartpole_flicker", "cheetah_velhidden"):
                    agg = aggregate_model_env(m, env, suite_stress, seeds_s)
                    if agg["env_sr_mean"] is not None:
                        all_rows.append({"suite": suite, "env": env, "model": m, **agg})
        out_json.write_text(json.dumps({
            "suites": ["G4", "G8", "G16"],
            "models": GENERALIST_MODELS,
            "rows": all_rows,
        }, indent=2))
        print(f"[aggregate_master] merged G4/G8/G16 into {out_json} ({len(all_rows)} cells)")
        return 0

    stress_dir = Path(args.stress_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = discover_seeds(results_dir, GENERALIST_MODELS)
    if not seeds:
        seeds = [0]
    print(f"[aggregate_master] seeds discovered: {seeds}")

    # Discover all envs from the union of available eval JSONs.
    envs_id: List[str] = []
    envs_stress: List[str] = []
    for m in GENERALIST_MODELS:
        mdir = results_dir / m
        if not mdir.exists():
            continue
        for sd in mdir.iterdir():
            if not sd.is_dir() or not sd.name.startswith("seed_"):
                continue
            for fp in sd.glob("eval_*.json"):
                env = fp.stem.replace("eval_", "", 1)
                if env not in envs_id:
                    envs_id.append(env)
    for m in GENERALIST_MODELS:
        mdir = stress_dir / m
        if not mdir.exists():
            continue
        for sd in mdir.iterdir():
            if not sd.is_dir() or not sd.name.startswith("seed_"):
                continue
            for fp in sd.glob("eval_*.json"):
                env = fp.stem.replace("eval_", "", 1)
                if env not in envs_stress:
                    envs_stress.append(env)
    print(f"[aggregate_master] ID envs: {len(envs_id)}, stress envs: {len(envs_stress)}")

    # Build rows.
    rows: List[Dict[str, Any]] = []
    for m in GENERALIST_MODELS:
        for env in envs_id:
            agg = aggregate_model_env(m, env, results_dir, seeds)
            rows.append({"env": env, "model": m, **agg})
        for env in envs_stress:
            agg = aggregate_model_env(m, env, stress_dir, seeds)
            rows.append({"env": env, "model": m, **agg})

    if args.merge_all:
        # Combine results from G4/G8/G16 into a single output file.
        out_md = out_dir / "generalist_master_table.md"
        out_json = out_dir / "generalist_master_table.json"
    else:
        out_md = out_dir / f"generalist_master_table_{args.suite}.md"
        out_json = out_dir / f"generalist_master_table_{args.suite}.json"
    with out_md.open("w") as f:
        f.write(f"# Generalist World-Model Evaluation — Master Table\n\n")
        f.write(f"Suite: **{args.suite}**\n\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write(f"Models: {len(GENERALIST_MODELS)} (1 collapse-control marked with `*`)\n\n")
        f.write("`mlp_baseline*` is the negative control for latent collapse — its "
                "high LeWM-SR with low env-SR is the expected signature.\n\n")
        write_section(f"{args.suite} — In-Distribution", rows, envs_id, GENERALIST_MODELS, f)
        if envs_stress:
            write_section(f"{args.suite} — Stress", rows, envs_stress, GENERALIST_MODELS, f)
        write_summary(rows, envs_id, envs_stress, GENERALIST_MODELS, f)

        if args.probes:
            f.write("\n## G16-Probes (event-AUROC)\n\n")
            f.write("Pooled AUROC across the 7 probe-eligible envs. See "
                    "results/aggregate/event_probes_table.md for per-(env, target) detail.\n\n")
            f.write("(populated in Stage 5)\n")

        if args.align:
            f.write("\n## G16-Align (event-align ρ)\n\n")
            f.write("Pearson correlation between obs event strength and latent first-difference. "
                    "See results/aggregate/event_align_table.md for per-(env, model) detail.\n\n")
            f.write("(populated in Stage 5)\n")

    out_json.write_text(json.dumps({
        "suite": args.suite,
        "seeds": seeds,
        "models": GENERALIST_MODELS,
        "envs_id": envs_id,
        "envs_stress": envs_stress,
        "rows": rows,
    }, indent=2))
    print(f"[aggregate_master] wrote {out_md}")
    print(f"[aggregate_master] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
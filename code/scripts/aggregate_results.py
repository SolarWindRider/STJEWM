"""Aggregate eval.json from all 50 (env, model) pairs into a single comparison table.

Output:
  /home/lx/snn/results/aggregate/STJEWM_vs_LeWM.json
  /home/lx/snn/results/aggregate/STJEWM_vs_LeWM.md

Usage:
    python -m code.scripts.aggregate_results --results-dir /home/lx/snn/results --out-dir /home/lx/snn/results/aggregate
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_eval(eval_path: Path) -> Optional[dict]:
    if not eval_path.exists():
        return None
    try:
        with open(eval_path) as f:
            return json.load(f)
    except Exception:
        return None


def aggregate(results_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip the output dir if it lives inside results_dir
    try:
        is_outside = not out_dir.resolve().is_relative_to(results_dir.resolve())
    except AttributeError:
        is_outside = str(out_dir.resolve()) not in str(results_dir.resolve())
    envs = sorted([d for d in results_dir.iterdir()
                   if d.is_dir() and (is_outside or d.resolve() != out_dir.resolve())])
    rows = []
    for env_dir in envs:
        env = env_dir.name
        row = {"env": env, "stjewm": {}, "lewm_baseline": {}}
        for model in ("stjewm", "lewm_baseline"):
            ckpt = env_dir / model / "final.pt"
            eval_p = env_dir / model / "eval.json"
            loss_p = env_dir / model / "loss_log.json"
            d = {"ckpt_exists": ckpt.exists(), "eval": None, "loss": None}
            ev = load_eval(eval_p)
            if ev is not None:
                # Extract per-seed aggregate
                per_seed = ev.get("per_seed", ev.get("per_seed_results", []))
                if per_seed:
                    lewm_srs = [p.get("success_rate_lewm", 0.0) for p in per_seed]
                    env_srs  = [p.get("success_rate_env", 0.0) for p in per_seed]
                    cos      = [p.get("mean_cos_dist", float("nan")) for p in per_seed]
                    phys     = [p.get("mean_phys_dist", float("nan")) for p in per_seed]
                    d["eval"] = {
                        "n_seeds": len(per_seed),
                        "n_eps":   per_seed[0].get("n", 0) if per_seed else 0,
                        "lewm_sr_mean":  float(np.mean(lewm_srs)) if lewm_srs else 0.0,
                        "lewm_sr_std":   float(np.std(lewm_srs))  if lewm_srs else 0.0,
                        "env_sr_mean":   float(np.mean(env_srs))  if env_srs  else 0.0,
                        "env_sr_std":    float(np.std(env_srs))   if env_srs  else 0.0,
                        "cos_dist_mean": float(np.mean(cos))      if cos      else 0.0,
                        "phys_dist_mean":float(np.mean(phys))     if phys     else 0.0,
                    }
            loss = load_eval(loss_p)
            if loss is not None and "losses" in loss:
                ls = loss["losses"]
                d["loss"] = {
                    "n_steps":  len(ls),
                    "final_loss": float(ls[-1].get("total", 0.0)) if ls else 0.0,
                    "first_loss":  float(ls[0].get("total", 0.0))  if ls else 0.0,
                }
            row[model] = d
        rows.append(row)

    # Save JSON
    with open(out_dir / "STJEWM_vs_LeWM.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Save markdown table
    md = ["# STJEWM vs LeWM-style comparison", ""]
    md.append(f"Results dir: {results_dir}")
    md.append("")
    md.append("## Per-environment table")
    md.append("")
    md.append("| Env | STJEWM LeWM SR | LeWM LeWM SR | STJEWM env SR | LeWM env SR | STJEWM cos | LeWM cos | STJEWM ckpt | LeWM ckpt |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    n_stj = 0
    n_lew = 0
    n_stj_w = 0
    n_lew_w = 0
    n_stj_e = 0
    n_lew_e = 0
    n_both = 0
    n_stj_w_better = 0
    for r in rows:
        env = r["env"]
        stj_ev = r["stjewm"].get("eval")
        lew_ev = r["lewm_baseline"].get("eval")
        stj_sr  = f"{stj_ev['lewm_sr_mean']*100:.1f}%" if stj_ev else "-"
        lew_sr  = f"{lew_ev['lewm_sr_mean']*100:.1f}%" if lew_ev else "-"
        stj_e   = f"{stj_ev['env_sr_mean']*100:.1f}%" if stj_ev else "-"
        lew_e   = f"{lew_ev['env_sr_mean']*100:.1f}%" if lew_ev else "-"
        stj_co  = f"{stj_ev['cos_dist_mean']:.3f}" if stj_ev else "-"
        lew_co  = f"{lew_ev['cos_dist_mean']:.3f}" if lew_ev else "-"
        stj_ck  = "✓" if r["stjewm"]["ckpt_exists"] else "✗"
        lew_ck  = "✓" if r["lewm_baseline"]["ckpt_exists"] else "✗"
        md.append(f"| {env} | {stj_sr} | {lew_sr} | {stj_e} | {lew_e} | {stj_co} | {lew_co} | {stj_ck} | {lew_ck} |")
        if stj_ev:
            n_stj += 1
            if stj_ev["lewm_sr_mean"] > 0:
                n_stj_w += 1
            if stj_ev["env_sr_mean"] > 0:
                n_stj_e += 1
        if lew_ev:
            n_lew += 1
            if lew_ev["lewm_sr_mean"] > 0:
                n_lew_w += 1
            if lew_ev["env_sr_mean"] > 0:
                n_lew_e += 1
        if stj_ev and lew_ev:
            n_both += 1
            if stj_ev["lewm_sr_mean"] > lew_ev["lewm_sr_mean"]:
                n_stj_w_better += 1

    md.append("")
    md.append("## Headline numbers")
    md.append("")
    md.append(f"- Envs with STJEWM ckpt: {n_stj}/{len(rows)}")
    md.append(f"- Envs with LeWM ckpt:   {n_lew}/{len(rows)}")
    md.append(f"- Envs with both ckpts:  {n_both}/{len(rows)}")
    md.append(f"- STJEWM envs with non-zero LeWM SR:  {n_stj_w}/{n_stj}")
    md.append(f"- LeWM envs with non-zero LeWM SR:    {n_lew_w}/{n_lew}")
    md.append(f"- STJEWM envs with non-zero env-native SR: {n_stj_e}/{n_stj}")
    md.append(f"- LeWM envs with non-zero env-native SR:   {n_lew_e}/{n_lew}")
    md.append(f"- STJEWM > LeWM on LeWM SR (per-env): {n_stj_w_better}/{n_both}")
    md.append("")
    with open(out_dir / "STJEWM_vs_LeWM.md", "w") as f:
        f.write("\n".join(md))

    print(f"Wrote {out_dir / 'STJEWM_vs_LeWM.json'}")
    print(f"Wrote {out_dir / 'STJEWM_vs_LeWM.md'}")
    print()
    print(f"Headline: {n_stj} STJEWM ckpts, {n_lew} LeWM ckpts, {n_both} both.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/home/lx/snn/results")
    ap.add_argument("--out-dir",      default="/home/lx/snn/results/aggregate")
    args = ap.parse_args()
    aggregate(Path(args.results_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()

"""Data-budget compression sweep (v0.7.8 utility experiment).

For each (model, data-budget) cell, train a new generalist checkpoint with
scaled per-entry `max_windows` (a half/double of the G16 baseline) and then
evaluate the four collapse-robust diagnostics:

- env-SR (env-native closed-loop success rate)
- div  (latent per-dim std, mean across dims)
- resp (mean |delta-lat| / mean |delta-obs|)
- event-align rho = corr(obs, latent) at first differences

The 1.0x baseline is the existing G16 ckpt at
results/generalist_G16/<model>/seed_0/final.pt. The 0.5x and 2.0x ckpts are
trained into results/generalist_G16_compression/<model>/<frac>/seed_0/.

Usage (per cell):
    python -m code.scripts.utility.compression_sweep \\
        --model stjewm_trace_only \\
        --frac 0.5 \\
        --out results/utility/compression_sweep/stjewm_trace_only/0.5.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, "/home/lx/snn")


# 6 DMC envs shared with measure_latent_stats / event_align so that
# 1.0x numbers can be re-used from results/generalist_G16/.
DMC_ENVS = [
    "cheetah",
    "walker",
    "cartpole_2d",
    "pendulum_2d",
    "finger",
    "ball_in_cup",
]

# Where the existing 1.0x G16 ckpts + their diagnostics live.
G16_BASELINE_DIR = "results/generalist_G16"

# Per-env max_windows used for the 1.0x G16 training (from
# configs/generalist_G16_train.json). We scale this for the sweep:
#   0.5x -> BASE_PER_ENV // 2
#   1.0x -> BASE_PER_ENV
#   2.0x -> BASE_PER_ENV * 2
BASE_PER_ENV = 10000


def write_spec(frac: float, out_path: Path) -> int:
    """Write a multi-env JSON spec with per-entry max_windows = frac * BASE_PER_ENV.

    Returns the per-entry max_windows used.
    """
    base = json.loads(Path("configs/generalist_G16_train.json").read_text())
    per_env = int(round(BASE_PER_ENV * frac))
    for entry in base:
        entry["max_windows"] = per_env
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(base, indent=2))
    return per_env


def train_ckpt(model: str, frac: float, frac_label: str, base_seed: int = 0) -> Path:
    """Train a single (model, frac) ckpt if it doesn't already exist.

    Re-uses code/scripts/generalist_v0_7_5/train_one.sh — only the spec file
    changes between cells. Writes to
        results/generalist_G16_compression/<model>/<frac_label>/seed_<seed>/
    """
    out_dir = Path(f"results/generalist_G16_compression/{model}/{frac_label}/seed_{base_seed}")
    ckpt = out_dir / "final.pt"
    if ckpt.exists():
        print(f"[train] {model}/frac={frac_label}: ckpt already exists, skipping")
        return ckpt

    spec_path = Path(f"configs/_compression_sweep_{model}_{frac_label}.json")
    per_env = write_spec(frac, spec_path)
    print(f"[train] {model}/frac={frac_label}: spec per-env max_windows={per_env}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/bin/bash",
        "code/scripts/generalist_v0_7_5/train_one.sh",
        model,
        str(spec_path),
        str(out_dir),
        str(base_seed),
    ]
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    if rc != 0 or not ckpt.exists():
        raise RuntimeError(f"train failed for {model}/{frac_label}: rc={rc} (no ckpt at {ckpt})")
    print(f"[train] {model}/{frac_label}: done in {dt/60:.1f} min -> {ckpt}")
    return ckpt


def run_one_env(model_dir: Path, env: str, base_seed: int = 0) -> Dict[str, float]:
    """Return {div, resp, rho, env_sr} for one (ckpt, env) cell.

    div/resp come from measure_latent_stats; rho comes from event_align;
    env_sr comes from the closed_loop eval JSON. Each sub-tool is invoked
    via subprocess so we mirror the existing run_*.sh conventions.
    """
    ckpt = model_dir / "final.pt"
    pad_obs = 128
    action_dim = 56

    # div / resp
    stats_out = model_dir / f"latent_stats_{env}.json"
    if not stats_out.exists():
        cmd = [
            "/home/lx/miniconda3/envs/snn/bin/python",
            "-m", "code.scripts.generalist_v0_7_5.measure_latent_stats",
            "--ckpt", str(ckpt),
            "--env", env,
            "--out", str(stats_out),
            "--n-steps", "200",
            "--seed", str(base_seed),
            "--device", "cpu",
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            return {"error": f"measure_latent_stats rc={rc}"}
    stats = json.loads(stats_out.read_text())
    div = float(stats.get("divergence", 0.0))
    resp = float(stats.get("responsiveness", 0.0))

    # event-align rho
    align_out = model_dir / f"align_{env}.json"
    if not align_out.exists():
        cmd = [
            "/home/lx/miniconda3/envs/snn/bin/python",
            "-m", "code.scripts.event_align",
            "--env", env,
            "--model", model_dir.parent.name,
            "--ckpt", str(ckpt),
            "--out", str(align_out),
            "--n-steps", "100",
            "--pad-obs-to", str(pad_obs),
            "--action-dim-eval", str(action_dim),
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            rho = float("nan")
        else:
            rho = float(json.loads(align_out.read_text()).get("corr_obs_latent", 0.0))
    else:
        rho = float(json.loads(align_out.read_text()).get("corr_obs_latent", 0.0))

    # env-SR (env-native success on a 3-episode CEM rollout)
    eval_out = model_dir / f"eval_{env}.json"
    if not eval_out.exists():
        # Map event-align env names to closed_loop env names.
        clo_env = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum"}.get(env, env)
        # Find the matching data path from the base spec.
        base_spec = json.loads(Path("configs/generalist_G16_train.json").read_text())
        data_path = None
        for entry in base_spec:
            if entry["env_id"] == env:
                data_path = entry["path"]
                break
        if data_path is None:
            return {"error": f"no data path for env={env}"}
        cmd = [
            "/home/lx/miniconda3/envs/snn/bin/python",
            "-m", "code.eval.closed_loop",
            "--env", clo_env,
            "--ckpt", str(ckpt),
            "--data", data_path,
            "--out", str(eval_out),
            "--n-episodes", "3",
            "--n-seeds", "1",
            "--horizon", "5",
            "--eval-budget", "50",
            "--history-size", "1",
            "--goal-offset", "25",
            "--pad-obs-eval", str(pad_obs),
            "--action-dim-eval", str(action_dim),
        ]
        rc = subprocess.call(cmd)
        if rc != 0 or not eval_out.exists():
            env_sr = float("nan")
        else:
            env_sr = float(json.loads(eval_out.read_text()).get("success_rate_env", 0.0))
    else:
        env_sr = float(json.loads(eval_out.read_text()).get("success_rate_env", 0.0))

    return {"env-SR": env_sr, "div": div, "resp": resp, "rho": rho}


def aggregate(per_env: List[Dict[str, float]]) -> Dict[str, float]:
    """Average the four diagnostics across the 6 DMC envs, ignoring NaNs."""
    import math
    out = {}
    for key in ("env-SR", "div", "resp", "rho"):
        vals = [float(v[key]) for v in per_env if key in v and not math.isnan(float(v[key]))]
        if vals:
            avg = sum(vals) / len(vals)
            # std across envs (unbiased)
            if len(vals) > 1:
                mu = avg
                var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
                std = var ** 0.5
            else:
                std = 0.0
            out[f"{key}_avg"] = avg
            out[f"{key}_std"] = std
        else:
            out[f"{key}_avg"] = float("nan")
            out[f"{key}_std"] = float("nan")
    out["n_envs"] = len(per_env)
    return out


def run_for_1x_baseline(model: str, out_path: Path) -> Dict[str, Any]:
    """Re-use the 1.0x G16 baselines — train_one is unnecessary.

    The 6-DMC latent_stats / event-align / closed_loop eval JSONs already exist
    under results/generalist_G16/<model>/seed_0/. We just aggregate them.
    """
    model_dir = Path(f"results/generalist_G16/{model}/seed_0")
    if not (model_dir / "final.pt").exists():
        return {"error": f"no 1.0x ckpt at {model_dir}"}

    per_env: List[Dict[str, float]] = []
    for env in DMC_ENVS:
        stats_p = model_dir / f"latent_stats_{env}.json"
        align_p = model_dir / f"align_{env}.json"
        eval_p = model_dir / f"eval_{env}.json"

        # For 1.0x, align JSONs live under results/generalist_G16/event_align/.
        if not align_p.exists():
            align_path = Path(f"results/generalist_G16/event_align/{env}_{model}_seed0.json")
            if align_path.exists():
                align_p = align_path
        row: Dict[str, float] = {}
        if stats_p.exists():
            stats = json.loads(stats_p.read_text())
            row["div"] = float(stats.get("divergence", 0.0))
            row["resp"] = float(stats.get("responsiveness", 0.0))
        if align_p.exists():
            row["rho"] = float(json.loads(align_p.read_text()).get("corr_obs_latent", 0.0))
        if eval_p.exists():
            row["env-SR"] = float(json.loads(eval_p.read_text()).get("success_rate_env", 0.0))
        per_env.append({"env": env, **row})

    summary = aggregate(per_env)
    summary["model"] = model
    summary["frac"] = 1.0
    summary["frac_label"] = "1.0"
    summary["per_env"] = per_env
    summary["source"] = "results/generalist_G16"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[1.0x] {model}: env-SR_avg={summary['env-SR_avg']:.3f} "
          f"div={summary['div_avg']:.4f} resp={summary['resp_avg']:.3f} "
          f"rho={summary['rho_avg']:.3f}")
    return summary


def run_for_new_ckpt(model: str, frac: float, frac_label: str,
                     out_path: Path, base_seed: int = 0) -> Dict[str, Any]:
    """Train (if needed), eval on the 6 DMC envs, aggregate, dump JSON."""
    ckpt = train_ckpt(model, frac, frac_label, base_seed)
    model_dir = ckpt.parent

    per_env: List[Dict[str, float]] = []
    for env in DMC_ENVS:
        row = run_one_env(model_dir, env, base_seed)
        per_env.append({"env": env, **row})
        print(f"  [eval] {model}/{frac_label}/{env}: {row}")

    summary = aggregate(per_env)
    summary["model"] = model
    summary["frac"] = frac
    summary["frac_label"] = frac_label
    summary["per_env"] = per_env
    summary["source"] = str(model_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--frac", type=float, required=True,
                   help="data-budget fraction relative to G16 baseline")
    p.add_argument("--frac-label", default=None,
                   help="directory/file label (e.g. '0.5'). Defaults to str(frac).")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", action="store_true",
                   help="Read existing 1.0x G16 outputs without training.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    frac_label = args.frac_label or str(args.frac)
    out_path = Path(args.out)

    t0 = time.time()
    if args.baseline or args.frac == 1.0:
        result = run_for_1x_baseline(args.model, out_path)
    else:
        result = run_for_new_ckpt(args.model, args.frac, frac_label, out_path, args.seed)
    print(f"[done] {args.model}/frac={frac_label} in {(time.time() - t0)/60:.1f} min -> {out_path}")
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())

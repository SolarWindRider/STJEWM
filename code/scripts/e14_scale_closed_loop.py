#!/usr/bin/env python
"""E14 — closed-loop env-SR on the task-diversity (scale) axis.

Question: does the control utility measured on the shared in-union environments
survive as training diversity and per-env data volume grow (G4 -> G8 -> G16)?
E5 measured only the latent-dynamical diagnostics (div/resp/rho, random-policy
rollout); env-SR was deliberately not part of that grid ("not evaluated by this
diagnostic grid"). This grid adds the closed-loop half under the exact E1
protocol, restricted to the common in-union env core so that cross-scale
differences are not confounded with out-of-union coverage:

    scales: G4 (4 envs, 2000 windows/env), G8 (8 envs, 2000), G16 (16 envs, 10000)
    models: the 12 scale-axis models
    envs:   cartpole_2d, pendulum_2d, cheetah, pusht  (in every training union)
    cells:  3 x 12 x 4 = 144
    protocol: closed_loop, CEM 300/30/x10 (pusht 30), horizon 25, budget 50,
              goal_offset 25, history 1, 5 episodes, 1 seed, pad 128/act 56,
              split in_dist — byte-identical recipe to the E1 final grid.

Checkpoints are the final-repair generation (training manifest 4376bb14...,
protocol causal_grad_sigreg_B_20260916); no retraining is needed.

Run:
    python -m code.scripts.e14_scale_closed_loop run --workers 4
    python -m code.scripts.e14_scale_closed_loop aggregate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/home/lx/snn")

SNN_ROOT = Path("/home/lx/snn")
RESULTS_ROOT = Path("/data/lx/tmp/results")
OUT_ROOT = RESULTS_ROOT / "scale_closed_loop_20260917"
CKPT_ROOT = RESULTS_ROOT
SIZES = ("G4", "G8", "G16")
MODELS = (
    "stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only", "stjewm_no_trace",
    "stjewm_hidden_leak", "stjewm_membrane_readout",
    "alif_timecell_baseline", "stacked_lif_trace", "stacked_lif_free",
    "gru_baseline", "lewm_baseline_v2", "mlp_baseline",
)
# Common in-union core: present in G4, G8 and G16 training configs.
ENVS = (
    # env_id, clo_env, data path, cem_iters
    ("cartpole_2d", "cartpole", "data/dm_control/cartpole_250k.npz", 10),
    ("pendulum_2d", "pendulum", "data/dm_control/pendulum_250k.npz", 10),
    ("cheetah", "cheetah", "data/dm_control/3d_rollouts_250k/cheetah_250k.npz", 10),
    ("pusht", "pusht", "/home/lx/LeWM/data/pusht_expert_train.h5", 30),
)
BUDGET = {
    "n_episodes": 5, "n_seeds": 1, "horizon": 25, "eval_budget": 50,
    "history_size": 1, "cem_samples": 300, "cem_elites": 30,
    "goal_offset": 25, "pad_obs_eval": 128, "action_dim_eval": 56,
}
SOURCE_FILES = (
    "code/eval/closed_loop.py", "code/core/cem.py", "code/core/encode.py",
    "code/core/envs/dmc_env.py", "code/core/envs/delayed_t_maze.py",
    "code/core/envs/reacher_env.py", "code/data/loaders.py", "code/data/base.py",
    "code/core/envs/swm_envs.py", "code/scripts/e14_scale_closed_loop.py",
)
NUMERIC_FIELDS = (
    "success_rate_lewm", "success_rate_lewm_005", "success_rate_env",
    "mean_cos_dist", "mean_phys_dist",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cells() -> list[dict]:
    cells = []
    for size in SIZES:
        for model in MODELS:
            checkpoint = CKPT_ROOT / f"generalist_{size}" / model / "seed_0" / "final.pt"
            if not checkpoint.exists():
                raise SystemExit(f"missing checkpoint: {checkpoint}")
            for env_id, clo_env, data, cem_iters in ENVS:
                output = OUT_ROOT / "E14" / size / model / "seed_0" / f"eval_{env_id}.json"
                command = [
                    sys.executable, "-u", "-m", "code.eval.closed_loop",
                    "--env", clo_env, "--ckpt", str(checkpoint), "--data", data,
                    "--out", str(output), "--device", "cuda:0", "--split", "in_dist",
                ]
                for key, value in dict(BUDGET, cem_iters=cem_iters).items():
                    command.extend(("--" + key.replace("_", "-"), str(value)))
                cells.append({
                    "id": f"E14/{size}/{model}/seed_0/{env_id}",
                    "scale": size, "model": model, "env": env_id, "clo_env": clo_env,
                    "checkpoint": str(checkpoint), "data": data,
                    "output": str(output),
                    "log": str(OUT_ROOT / "logs" / f"E14_{size}_{model}_{env_id}.log"),
                    "cem_iters": cem_iters, "command": command,
                })
    return cells


def valid_output(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    episodes = payload.get("per_episode")
    if not isinstance(episodes, list) or len(episodes) != 5:
        return False
    for key in NUMERIC_FIELDS:
        value = payload.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        if key.startswith("success_rate") and not 0 <= value <= 1:
            return False
    return payload.get("n_seeds") == 1 and payload.get("horizon") == 25


def run_cell(cell: dict, gpu: int) -> dict:
    output = Path(cell["output"])
    receipt = {
        "id": cell["id"], "command": cell["command"], "gpu": gpu,
        "resumed_valid": False, "returncode": None,
        "output_sha256": None, "started": time.time(),
    }
    if valid_output(output):
        receipt["resumed_valid"] = True
        receipt["output_sha256"] = sha256(output)
        receipt["returncode"] = 0
        receipt["finished"] = time.time()
        return receipt
    output.parent.mkdir(parents=True, exist_ok=True)
    log = Path(cell["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    env_vars = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(SNN_ROOT))
    with log.open("w") as handle:
        proc = subprocess.run(cell["command"], cwd=str(SNN_ROOT), env=env_vars,
                              stdout=handle, stderr=subprocess.STDOUT)
    receipt["returncode"] = proc.returncode
    if proc.returncode == 0 and valid_output(output):
        receipt["output_sha256"] = sha256(output)
    else:
        raise RuntimeError(f"E14 cell failed ({cell['id']}): rc={proc.returncode}, log={log}")
    receipt["finished"] = time.time()
    return receipt


def _worker(args: tuple) -> dict:
    cell, gpu = args
    return run_cell(cell, gpu)


def cmd_run(workers: int) -> int:
    cells = build_cells()
    done = sum(1 for c in cells if valid_output(Path(c["output"])))
    print(f"[e14] planned {len(cells)} cells; {done} already valid; workers={workers}", flush=True)
    receipts_path = OUT_ROOT / "receipts.jsonl"
    receipts_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with receipts_path.open("a") as sink:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, (cell, index % workers)): cell
                       for index, cell in enumerate(cells)}
            for count, future in enumerate(as_completed(futures), 1):
                cell = futures[future]
                try:
                    receipt = future.result()
                except Exception as error:  # noqa: BLE001 — record and continue the grid
                    failures += 1
                    receipt = {"id": cell["id"], "error": str(error)}
                sink.write(json.dumps(receipt) + "\n")
                sink.flush()
                mark = "resumed" if receipt.get("resumed_valid") else ("done" if "error" not in receipt else "FAILED")
                print(f"[e14] {count}/{len(cells)} {mark} {cell['id']}", flush=True)
    if failures:
        print(f"[e14] COMPLETE WITH {failures} FAILURES", flush=True)
        return 1
    print("[e14] COMPLETE 144/144", flush=True)
    return 0


def source_digests() -> dict:
    return {path: sha256(SNN_ROOT / path) for path in SOURCE_FILES}


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def cmd_aggregate() -> int:
    from code.scripts.audited_results import write_new_json

    cells = build_cells()
    missing = [c["id"] for c in cells if not valid_output(Path(c["output"]))]
    if missing:
        raise SystemExit(f"refusing to aggregate: {len(missing)} invalid/missing cells, e.g. {missing[:3]}")

    rows = []
    for cell in cells:
        payload = json.loads(Path(cell["output"]).read_text())
        rows.append({
            "scale": cell["scale"], "model": cell["model"], "env": cell["env"],
            "env_sr": float(payload["success_rate_env"]),
            "cos": float(payload["mean_cos_dist"]),
            "phys": float(payload["mean_phys_dist"]),
            "lewm_sr_005": float(payload["success_rate_lewm_005"]),
            "source": cell["output"],
            "source_sha256": sha256(Path(cell["output"])),
        })

    summary = {
        "status": "completed",
        "analysis": "E14_scale_closed_loop_v1",
        "question": "Does closed-loop control utility on the shared in-union env core survive increasing training diversity (G4/G8/G16)?",
        "protocol": {"budget": BUDGET, "cem_iters": "pusht 30, else 10",
                     "in_union_envs": [e[0] for e in ENVS],
                     "note": "G4/G8 trained 2000 windows/env, G16 10000 — scale mixes diversity and per-env data volume; env-SR is collapse-insensitive (E1/E7), interpret jointly with E5 dynamics."},
        "training_manifest_sha256": "4376bb14c50e30e12c3a1d47435b17489568f148b6f0abd57592efe5936e1f09",
        "source_sha256": source_digests(),
        "cells": rows,
        "pooled_per_scale_model": [],
    }
    for size in SIZES:
        for model in MODELS:
            subset = [r for r in rows if r["scale"] == size and r["model"] == model]
            summary["pooled_per_scale_model"].append({
                "scale": size, "model": model, "n_envs": len(subset),
                "env_sr": _mean([r["env_sr"] for r in subset]),
                "cos": _mean([r["cos"] for r in subset]),
                "phys": _mean([r["phys"] for r in subset]),
                "lewm_sr_005": _mean([r["lewm_sr_005"] for r in subset]),
            })

    out_dir = SNN_ROOT / "results" / "journal_prep" / "E14_scale_closed_loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_new_json(out_dir / "scale_closed_loop_summary.json", summary)

    lines = [
        "# E14 — closed-loop env-SR on the scale axis (G4/G8/G16, in-union core)",
        "",
        "Protocol: exact E1 recipe (CEM 300/30, horizon 25, budget 50, goal=t+25, 5 eps, pad 128/act 56).",
        "Envs restricted to the common in-union core (cartpole_2d, pendulum_2d, cheetah, pusht) so cross-scale",
        "differences are not confounded with out-of-union coverage. cos = mean_cos_dist (lower is better).",
        "",
        "## Pooled per scale x model (4 envs)",
        "",
        "| scale | model | env-SR | cos | phys | LeWM-SR@0.05 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["pooled_per_scale_model"]:
        lines.append(f"| {row['scale']} | {row['model']} | {row['env_sr']:.2f} | {row['cos']:.3f} | "
                     f"{row['phys']:.3f} | {row['lewm_sr_005']:.2f} |")
    lines += ["", "## Per-env env-SR (scale x model)", "",
              "| env | " + " | ".join(f"{s} {m}" for s in SIZES for m in MODELS) + " |",
              "|---|" + "---:|" * (len(SIZES) * len(MODELS))]
    for env, _, _, _ in ENVS:
        vals = []
        for size in SIZES:
            for model in MODELS:
                v = next(r["env_sr"] for r in rows if r["scale"] == size and r["model"] == model and r["env"] == env)
                vals.append(f"{v:.2f}")
        lines.append(f"| {env} | " + " | ".join(vals) + " |")
    write_new = out_dir / "scale_closed_loop_table.md"
    if write_new.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {write_new}")
    write_new.write_text("\n".join(lines) + "\n")
    print(f"[e14] wrote {out_dir}/scale_closed_loop_summary.json and scale_closed_loop_table.md", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "aggregate"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "run":
        return cmd_run(args.workers)
    return cmd_aggregate()


if __name__ == "__main__":
    raise SystemExit(main())

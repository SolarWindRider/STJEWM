#!/bin/bash
# Source-only 30-episode sensitivity grid; never the primary pixel publication grid.
set -euo pipefail
cd /home/lx/snn
: "${TRAINING_MANIFEST:?Set the final consolidated training repair manifest}"
: "${EVAL_ROOT:?Set a fresh output root}"
export TRAINING_MANIFEST EVAL_ROOT
export PIXEL_GPUS="${PIXEL_GPUS:-0,1,2,3}"
export MUJOCO_GL=egl
/home/lx/miniconda3/envs/snn/bin/python - <<'PY'
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import traceback

from code.scripts.audited_results import (
    ROOT, TrainingAudit, load_json, require, sha256, validate_pixel_summary, write_new_json,
)

audit = TrainingAudit(os.environ["TRAINING_MANIFEST"])
output_root = Path(os.environ["EVAL_ROOT"]).resolve()
audit.protect_output(output_root)
gpus = os.environ["PIXEL_GPUS"].split(",")
require(gpus and len(gpus) == len(set(gpus)) and all(gpu.isdigit() for gpu in gpus), "Invalid physical GPU list")
splits = ("cross_benchmark_F1", "cross_benchmark_F2", "cross_benchmark_F3", "oodc_F1", "oodc_F1F2",
          "oodc_F1F3", "oodc_F2", "oodc_F2F3", "oodc_F3", "generalist_16env")
models = ("stacked_lif_trace", "stjewm", "stjewm_spike_only")
envs = ("cartpole", "pendulum", "finger", "cheetah")
producer = ROOT / "code/scripts/generalist_v0_7_5_5m_pixel/eval_pixel_ckpt.py"
producer_hash = sha256(producer)
jobs = []
for split in splits:
    for model in models:
        relative = Path(split) / model / "seed_0"
        checkpoint = audit.results_root / "5m_pixel" / relative / "final.pt"
        jobs.append((split, model, checkpoint, output_root / relative, audit.checkpoint(checkpoint)["sha256"]))
output_root.mkdir(parents=True, exist_ok=False)

def work(gpu, queue):
    outcomes = []
    for split, model, checkpoint, output, checkpoint_hash in queue:
        record = {"split": split, "model": model, "checkpoint": str(checkpoint),
                  "checkpoint_sha256": checkpoint_hash, "output": str(output), "status": "failed"}
        try:
            require(sha256(producer) == producer_hash, "Canonical producer changed after planning")
            output.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, "-u", str(producer), "--ckpt", str(checkpoint),
                       "--training-manifest", str(audit.path), "--out_dir", str(output),
                       "--image_size", "84", "--n_episodes", "30", "--horizon", "5", "--samples", "300",
                       "--elites", "30", "--cem_iters", "10", "--device", "cuda:0", "--envs", ",".join(envs)]
            child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu)
            with (output / "eval.log").open("x") as log:
                process = subprocess.run(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
            record["returncode"] = process.returncode
            require(process.returncode == 0, f"Canonical evaluator exited {process.returncode}")
            require(audit.checkpoint(checkpoint)["sha256"] == checkpoint_hash, "Checkpoint changed during evaluation")
            require(sha256(producer) == producer_hash, "Canonical producer changed during evaluation")
            summary_path = output / "eval_summary.json"
            summary = load_json(summary_path)
            audit.validate_provenance(summary, checkpoint)
            validate_pixel_summary(summary, checkpoint, output, envs, episodes=30)
            record.update(status="completed", summary_sha256=sha256(summary_path))
        except Exception:
            record["error"] = traceback.format_exc()
        outcomes.append(record)
    return outcomes

with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
    futures = [pool.submit(work, gpu, jobs[index::len(gpus)]) for index, gpu in enumerate(gpus)]
    outcomes = [row for future in futures for row in future.result()]
completed = all(row["status"] == "completed" for row in outcomes)
write_new_json(output_root / "grid_status.json", {
    "protocol_version": 2, "status": "completed" if completed else "failed", "scope": "highpower_sensitivity_only",
    "training_manifest": str(audit.path), "training_manifest_sha256": audit.digest,
    "producer_sha256": producer_hash, "planned_checkpoints": len(jobs), "planned_cells": len(jobs) * len(envs),
    "planned_envs": list(envs), "n_episodes": 30, "outcomes": outcomes,
})
if not completed:
    raise SystemExit("Pixel sensitivity grid incomplete; inspect grid_status.json")
PY

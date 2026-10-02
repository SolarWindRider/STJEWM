#!/bin/bash
# Full final pixel grid: 13 models x 10 splits x 13 environments.
# Bounded independent checkpoint workers share physical GPU2 by default.
# CUDA and EGL use the same GPU. Native actions and corrected physical goals
# change the old protocol; old summaries are never used as completion markers.
# Usage: EVAL_ROOT=/data/lx/tmp/results/5m_pixel_final_RUN \
#        TRAINING_MANIFEST=/data/lx/tmp/results/_repair_archive/20260916T102154Z/training_final_repair_manifest.json bash "$0"
set -euo pipefail
cd /home/lx/snn
PY=/home/lx/miniconda3/envs/snn/bin/python
CKPT_ROOT=${CKPT_ROOT:-${OUT_ROOT:-/data/lx/tmp/results}/5m_pixel}
: "${EVAL_ROOT:?Set EVAL_ROOT to a fresh output root before launching}"
: "${TRAINING_MANIFEST:?Set TRAINING_MANIFEST to the final repair manifest before launching}"
GPU=${GPU:-2}
WORKERS=${WORKERS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_EGL_DEVICE_ID="$GPU"
export MUJOCO_GL=egl
export PYTHONPATH=/home/lx/snn

exec "$PY" - "$CKPT_ROOT" "$EVAL_ROOT" "$WORKERS" "$TRAINING_MANIFEST" <<'PY'
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from pathlib import Path
from code.scripts.generalist_v0_7_5_5m_pixel.eval_pixel_ckpt import DMC_ENVS
from code.scripts.audited_results import TrainingAudit, sha256, validate_pixel_summary

checkpoint_root = Path(sys.argv[1]).resolve()
output_root = Path(sys.argv[2]).resolve()
workers = int(sys.argv[3])
training_manifest_path = Path(sys.argv[4]).resolve()
audit = TrainingAudit(training_manifest_path)
training_manifest_sha256 = audit.digest
training_protocol_version = audit.payload["required_checkpoint_metadata"]["training_protocol_version"]
audit.protect_output(output_root)
if workers < 1:
    raise ValueError("WORKERS must be positive")
splits = (
    "oodc_F1", "oodc_F2", "oodc_F3", "oodc_F1F2", "oodc_F1F3", "oodc_F2F3",
    "cross_benchmark_F1", "cross_benchmark_F2", "cross_benchmark_F3", "generalist_16env",
)
models = (
    "stjewm", "stjewm_spike_only", "stjewm_rate_only", "stjewm_no_trace",
    "stjewm_hidden_leak", "stjewm_membrane_readout", "alif_timecell_baseline",
    "gru_baseline", "lewm_baseline", "stacked_lif_trace", "stacked_lif_free",
    "mlp_baseline", "lif_transformer_baseline",
)
jobs = []
for split in splits:
    for model in models:
        relative = Path(split) / model / "seed_0"
        checkpoint = checkpoint_root / relative / "final.pt"
        admitted = audit.checkpoint(checkpoint)
        jobs.append((checkpoint, output_root / relative, admitted["sha256"]))

# A unique root is mandatory: no raw checkpoint, result or log is replaced.
output_root.mkdir(parents=True, exist_ok=False)
planned_cells = len(jobs) * len(DMC_ENVS)
print(f"[pixel_eval] {len(jobs)} checkpoints / {planned_cells} planned cells", flush=True)
def evaluate(job):
    checkpoint, output_dir, checkpoint_hash = job
    record = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "output": str(output_dir),
        "returncode": None,
        "summary_sha256": None,
        "status": "failed",
        "error": None,
    }
    try:
        if audit.checkpoint(checkpoint)["sha256"] != checkpoint_hash:
            raise ValueError("Checkpoint changed after grid admission")
        output_dir.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable, "-u", "code/scripts/generalist_v0_7_5_5m_pixel/eval_pixel_ckpt.py",
            "--ckpt", str(checkpoint), "--out_dir", str(output_dir),
            "--training-manifest", str(audit.path),
            "--n_episodes", "5", "--samples", "300", "--elites", "30",
            "--cem_iters", "10", "--horizon", "5", "--device", "cuda:0",
        ]
        print(f"[pixel_eval] START {checkpoint}", flush=True)
        with (output_dir / "eval.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record["returncode"] = result.returncode
        summary_path = output_dir / "eval_summary.json"
        if summary_path.is_file():
            record["summary_sha256"] = sha256(summary_path)
        if result.returncode:
            raise RuntimeError(f"Evaluation exited {result.returncode}")
        if audit.checkpoint(checkpoint)["sha256"] != checkpoint_hash:
            raise ValueError("Checkpoint changed during evaluation")
        summary = json.loads(summary_path.read_text())
        if summary["training_protocol_version"] != training_protocol_version:
            raise ValueError("Summary lacks the final training protocol version")
        audit.validate_provenance(summary, checkpoint)
        validate_pixel_summary(summary, checkpoint, output_dir, DMC_ENVS)
        record["status"] = "completed"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    print(f"[pixel_eval] {record['status'].upper()} {output_dir}", flush=True)
    return record

outcomes = []
with ThreadPoolExecutor(max_workers=workers) as pool:
    for future in as_completed([pool.submit(evaluate, job) for job in jobs]):
        outcomes.append(future.result())
with (output_root / "grid_status.json").open("x") as handle:
    json.dump({"protocol_version": 2,
               "training_protocol_version": training_protocol_version,
               "training_manifest": str(training_manifest_path),
               "training_manifest_sha256": training_manifest_sha256,
               "planned_checkpoints": len(jobs),
               "planned_cells": planned_cells,
               "planned_envs": list(DMC_ENVS),
               "outcomes": outcomes},
              handle, indent=2, allow_nan=False)
failures = [row for row in outcomes if row["status"] != "completed"]
if failures:
    raise SystemExit(f"[pixel_eval] Failed checkpoint evaluations: {len(failures)}; see grid_status.json")
print(f"[pixel_eval] COMPLETE: {planned_cells} planned cells in {output_root}", flush=True)
PY

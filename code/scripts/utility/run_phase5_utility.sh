#!/bin/bash
# run_phase5_utility.sh — utility series rerun on the new G16 checkpoints.
# Waits for PHASE4_DONE + SEED12_ALL_DONE markers, then runs the 5 drivers with
# --results-dir pointing at /data/lx/tmp/results/generalist_G16.
# budget_scaling runs --skip-train first (frac=1.0 reuses G16 cells); the 0.5x/2.0x
# fresh trainings are queued separately when GPUs free up.
set -u
cd /home/lx/snn
LOG_DIR=/data/lx/tmp/logs
PY=/home/lx/miniconda3/envs/snn/bin/python
G16=/data/lx/tmp/results/generalist_G16
UTIL=/data/lx/tmp/results/utility
mkdir -p "$LOG_DIR" "$UTIL"

while [ ! -f "$LOG_DIR/PHASE4_DONE" ] || [ ! -f "$LOG_DIR/SEED12_ALL_DONE" ]; do sleep 300; done
echo "[phase5] start $(date)" | tee -a "$LOG_DIR/run_all.log"

$PY -m code.scripts.utility.run_latent_goal_mpc \
  --results-dir "$G16" --out-dir "$UTIL/latent_goal_mpc" 2>&1 | tail -3
echo "done latent_goal_mpc" >> "$LOG_DIR/phase5_progress.log"

$PY -m code.scripts.utility.run_latent_env_grad \
  --results-dir "$G16" --out-dir "$UTIL/latent_env_grad" 2>&1 | tail -3
echo "done latent_env_grad" >> "$LOG_DIR/phase5_progress.log"

$PY -m code.scripts.utility.run_sample_efficiency \
  --results-dir "$G16" --out-dir "$UTIL/sample_efficiency" 2>&1 | tail -3
echo "done sample_efficiency" >> "$LOG_DIR/phase5_progress.log"

$PY -m code.scripts.utility.run_cross_env_gen \
  --results-dir "$G16" --out-dir "$UTIL/cross_env_gen" 2>&1 | tail -3
echo "done cross_env_gen" >> "$LOG_DIR/phase5_progress.log"

$PY -m code.scripts.utility.run_budget_scaling \
  --results-dir "$G16" --out-dir "$UTIL/budget_scaling" --skip-train 2>&1 | tail -3
echo "done budget_scaling(frac1.0)" >> "$LOG_DIR/phase5_progress.log"

echo "[phase5] ALL DONE $(date)" | tee -a "$LOG_DIR/run_all.log"
touch "$LOG_DIR/PHASE5_DONE"

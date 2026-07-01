#!/usr/bin/env bash
# Sequential wrapper: run the 16-env retrain (membrane_readout mode) THEN
# the stress-env retrain (membrane_readout x 3 seeds), both on GPU 2.
#
# Why sequential: both contend for GPU 2; serializing avoids OOM and makes
# logs easier to read.
#
# Usage: ./membrane_pipeline.sh
set -e
cd /home/lx/snn

LOG_ROOT=${LOG_ROOT:-/home/lx/snn/logs}
mkdir -p "$LOG_ROOT"

echo "===== PHASE 1: retrain 16 envs x membrane_readout ====="
bash code/scripts/retrain_with_readout_modes.sh > "$LOG_ROOT/membrane_phase1.log" 2>&1

echo "===== PHASE 2: stress envs x membrane_readout x 3 seeds ====="
bash code/scripts/train_stress_membrane.sh > "$LOG_ROOT/membrane_phase2.log" 2>&1

echo "===== PHASE 3: eval v1 readout (membrane_readout only) ====="
MODES="membrane_readout" bash code/scripts/eval_v1_readout.sh > "$LOG_ROOT/membrane_phase3.log" 2>&1

echo "===== PHASE 4: eval stress (membrane_readout) ====="
# Eval each of 4 stress envs (the eval script picks any seed<n>/final.pt)
for env in pusht_ood tworoom_long cartpole_flicker cheetah_velhidden; do
    MODEL_FILTER=stjewm_membrane_readout bash code/scripts/eval_stress_suite.sh > "$LOG_ROOT/membrane_phase4_${env}.log" 2>&1
done

echo "===== MEMBRANE PIPELINE COMPLETE ====="

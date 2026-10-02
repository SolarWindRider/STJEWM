#!/bin/bash
# Logged utility execution gated by final checkpoint provenance, not stale
# PHASE4_DONE/SEED12_ALL_DONE markers from unrelated historical generations.
set -euo pipefail
set -o noclobber
cd /home/lx/snn
: "${TRAINING_MANIFEST:?Set the final consolidated training repair manifest}"
: "${UTILITY_OUT_ROOT:?Set a fresh utility output root}"
: "${CHECKPOINT_ROOT:?Set the results_root from the final training manifest}"
: "${PHASE5_LOG_PATH:?Set a fresh phase log path}"
export TRAINING_MANIFEST UTILITY_OUT_ROOT CHECKPOINT_ROOT
: > "$PHASE5_LOG_PATH"
bash code/scripts/utility/run_all_utilities.sh 2>&1 | tee -a "$PHASE5_LOG_PATH"
printf 'All utility drivers exited successfully. Manifest: %s\n' "$TRAINING_MANIFEST" > "$UTILITY_OUT_ROOT/PHASE5_DONE"

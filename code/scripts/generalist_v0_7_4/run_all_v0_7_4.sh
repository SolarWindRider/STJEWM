#!/bin/bash
# Full v0.7.4 pipeline runner.
#
# Usage:
#   bash code/scripts/generalist_v0_7_4/run_all_v0_7_4.sh [--seeds N] [--skip-g4] [--skip-g8] [--skip-g16]
#
# Trains 12 model variants on G4/G8/G16, evaluates on the G16_eval spec
# (which includes the 4 stress envs), then aggregates everything. Total
# wallclock at --seeds 1 on a single CPU is ~6 hours; at --seeds 3 it's
# ~18 hours.
set -e
cd /home/lx/snn

N_SEEDS=1
SKIP_G4=""
SKIP_G8=""
SKIP_G16=""
for arg in "$@"; do
  case "$arg" in
    --seeds=*) N_SEEDS="${arg#--seeds=}" ;;
    --skip-g4) SKIP_G4="yes" ;;
    --skip-g8) SKIP_G8="yes" ;;
    --skip-g16) SKIP_G16="yes" ;;
  esac
done

echo "[run_all] N_SEEDS=$N_SEEDS skip_g4=$SKIP_G4 skip_g8=$SKIP_G8 skip_g16=$SKIP_G16"

if [[ -z "$SKIP_G4" ]]; then
    echo ""
    echo "[run_all] === G4 ==="
    bash code/scripts/generalist_v0_7_4/run_suite.sh G4 \
        configs/generalist_G4_train.json \
        configs/generalist_G16_eval.json "$N_SEEDS"
fi

if [[ -z "$SKIP_G8" ]]; then
    echo ""
    echo "[run_all] === G8 ==="
    bash code/scripts/generalist_v0_7_4/run_suite.sh G8 \
        configs/generalist_G8_train.json \
        configs/generalist_G16_eval.json "$N_SEEDS"
fi

if [[ -z "$SKIP_G16" ]]; then
    echo ""
    echo "[run_all] === G16 ==="
    bash code/scripts/generalist_v0_7_4/run_suite.sh G16 \
        configs/generalist_G16_train.json \
        configs/generalist_G16_eval.json "$N_SEEDS"
fi

echo ""
echo "[run_all] === aggregating all suites ==="
for SUITE in G4 G8 G16; do
    /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.generalist_v0_7_4.aggregate_master --suite "$SUITE"
done

echo "[run_all] DONE"
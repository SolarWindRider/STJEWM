#!/bin/bash
# Full v0.7.5 pipeline runner.
#
# Usage:
#   bash code/scripts/generalist_v0_7_5/run_all.sh [--seeds N] [--skip-g4] [--skip-g8] [--skip-g16]
#
# Trains 12 model variants on G4/G8/G16, evaluates on the G16_eval spec
# (which includes the 4 stress envs). Historical training/evaluation only;
# publication requires separate completed final-generation manifests.
set -euo pipefail
cd /home/lx/snn

N_SEEDS=1
SKIP_G4=""
SKIP_G8=""
SKIP_G16=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds=*) N_SEEDS="${1#--seeds=}" ;;
    --seeds) N_SEEDS="${2:?--seeds requires a count}"; shift ;;
    --skip-g4) SKIP_G4="yes" ;;
    --skip-g8) SKIP_G8="yes" ;;
    --skip-g16) SKIP_G16="yes" ;;
    *) echo "[run_all] Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

echo "[run_all] N_SEEDS=$N_SEEDS skip_g4=$SKIP_G4 skip_g8=$SKIP_G8 skip_g16=$SKIP_G16"

if [[ -z "$SKIP_G4" ]]; then
    echo ""
    echo "[run_all] === G4 ==="
    bash code/scripts/generalist_v0_7_5/run_suite.sh G4 \
        configs/generalist_G4_train.json \
        configs/generalist_G16_eval.json "$N_SEEDS"
fi

if [[ -z "$SKIP_G8" ]]; then
    echo ""
    echo "[run_all] === G8 ==="
    bash code/scripts/generalist_v0_7_5/run_suite.sh G8 \
        configs/generalist_G8_train.json \
        configs/generalist_G16_eval.json "$N_SEEDS"
fi

if [[ -z "$SKIP_G16" ]]; then
    echo ""
    echo "[run_all] === G16 ==="
    bash code/scripts/generalist_v0_7_5/run_suite.sh G16 \
        configs/generalist_G16_train.json \
        configs/generalist_G16_eval.json "$N_SEEDS"
fi

echo ""
echo "[run_all] Historical raw outputs are not publication inputs."
echo "[run_all] Use master_aggregate.sh with --training-manifest, --state-run and --out."

echo "[run_all] DONE"
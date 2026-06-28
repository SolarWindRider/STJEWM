#!/bin/bash
# Train **LeWM first, then STJEWM** on all envs.
# Per user's directive: "先跑LeWM, 然后跑我们提出的方法".
# This script runs all 25 LeWM-style trainings, then all 25 STJEWM trainings.

set -e
cd /home/lx/snn
EPOCHS=${EPOCHS:-5}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "PHASE 1: LeWM-style baseline (all 25 envs)"
echo "============================================================"
MODEL_FILTER=lewm_baseline ./code/scripts/train_all.sh

echo ""
echo "============================================================"
echo "PHASE 2: STJEWM (all 25 envs)"
echo "============================================================"
MODEL_FILTER=stjewm ./code/scripts/train_all.sh

echo ""
echo "============================================================"
echo "ALL TRAININGS COMPLETE (LeWM first, then STJEWM)"
echo "============================================================"

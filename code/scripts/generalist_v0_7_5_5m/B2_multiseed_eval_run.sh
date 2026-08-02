#!/bin/bash
# B2 eval orchestrator runner (v2): wrap python in screen for true detachment.
LOG_DIR=/home/lx/snn/results/journal_prep/B2_multiseed/_eval_logs
mkdir -p "$LOG_DIR"

# Kill any prior eval orchestrator if rerun
pkill -f "B2_multiseed_eval_orchestrator" 2>/dev/null || true
sleep 1

screen -dmS b2_eval bash -c "/home/lx/miniconda3/envs/snn/bin/python /home/lx/snn/code/scripts/generalist_v0_7_5_5m/B2_multiseed_eval_orchestrator.py > $LOG_DIR/orchestrator_screen.log 2>&1"
echo "Launched"

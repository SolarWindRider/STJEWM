#!/bin/bash
# Watch for LeWM phase completion, then auto-launch STJEWM phase.
# Usage: ./watch_and_launch_next.sh
set -e
cd /home/lx/snn
LOG_DIR=/tmp/train_logs_gpu{0,1}
LEWM_LOG=/tmp/lewm_train_logs/launch3.log
echo "Waiting for LeWM phase to complete..."
while true; do
    # Check if both GPU logs have "ALL TRAININGS COMPLETE"
    GPU0_DONE=$(grep -c "ALL TRAININGS COMPLETE" /tmp/train_logs_gpu0/all.log 2>/dev/null || echo 0)
    GPU1_DONE=$(grep -c "ALL TRAININGS COMPLETE" /tmp/train_logs_gpu1/all.log 2>/dev/null || echo 0)
    if [ "$GPU0_DONE" -ge 1 ] && [ "$GPU1_DONE" -ge 1 ]; then
        break
    fi
    # Also check no training process is running
    RUNNING=$(ps aux | grep -c "code.train.train" | grep -v grep || echo 0)
    if [ "$RUNNING" -eq 0 ] && [ "$GPU0_DONE" -ge 1 ] && [ "$GPU1_DONE" -ge 1 ]; then
        break
    fi
    sleep 60
done
echo ""
echo "LeWM phase complete. Launching STJEWM phase..."
nohup bash /home/lx/snn/code/scripts/train_all_2gpu.sh stjewm > /tmp/stjewm_train_logs/launch.log 2>&1 &
disown
echo "STJEWM launched. Tail: /tmp/stjewm_train_logs/launch.log"

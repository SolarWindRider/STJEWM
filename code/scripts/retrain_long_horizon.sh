#!/usr/bin/env bash
# Faster retrain for long-horizon envs (pusht, tworoom) — uses smaller max_windows
# to keep training time bounded when goal_offset=100.
set -e
cd /home/lx/snn
export CUDA_VISIBLE_DEVICES=2
PYTHON=/home/lx/miniconda3/envs/snn/bin/python
EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-32}
LR=${LR:-3e-4}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
LOG_DIR=${LOG_DIR:-/home/lx/snn/logs/retrain_long_horizon}
mkdir -p "$LOG_DIR"

declare -A ENVS=(
    [pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 1 100 4000"
    [tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 1 100 4000"
)

for env in "${!ENVS[@]}"; do
    spec="${ENVS[$env]}"
    env_kind=$(echo "$spec" | awk '{print $1}')
    data_path=$(echo "$spec" | awk '{print $2}')
    hist=$(echo "$spec" | awk '{print $3}')
    goal=$(echo "$spec" | awk '{print $4}')
    maxw=$(echo "$spec" | awk '{print $5}')

    for mode in trace_only hidden_leak spike_only; do
        model="stjewm_${mode}"
        out_dir="$RESULTS_DIR/$env/$model"
        if [ -f "$out_dir/final.pt" ]; then
            echo "[skip] $env/$model"
            continue
        fi
        echo "[retrain] $env / $model  (ep=$EPOCHS, h=$hist, goal=$goal, maxw=$maxw)"
        mkdir -p "$out_dir"
        $PYTHON -m code.train.train \
            --model stjewm \
            --readout-mode "$mode" \
            --env-kind "$env_kind" \
            --data "$data_path" \
            --out "$out_dir" \
            --epochs "$EPOCHS" \
            --batch "$BATCH" \
            --lr "$LR" \
            --save-every 0 \
            --n-layers 4 \
            --history-size "$hist" \
            --goal-offset "$goal" \
            --max-windows "$maxw" \
            > "$out_dir/train.log" 2>&1
    done
done

echo "RETRAIN_LONG_HORIZON COMPLETE"

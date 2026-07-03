#!/bin/bash
# Train gru_baseline + mlp_baseline on the 4 stress envs (8 ckpts).
# The stress wrapper is applied at EVAL time only.
# Background-only. Skips existing final.pt.

set -u
cd /home/lx/snn
PYTHON=/home/lx/miniconda3/envs/snn/bin/python
LOG_ROOT=/home/lx/snn/logs/gru_mlp_stress_train
mkdir -p "$LOG_ROOT"

EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-32}
LR=${LR:-3e-4}
RESULTS_DIR=/home/lx/snn/results

# ENV -> env_kind, data, history_size, goal_offset
declare -A ENVS=(
    [pusht_ood]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 1 100"
    [tworoom_long]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 1 100"
    [cartpole_flicker]="dmc /home/lx/snn/data/dm_control/cartpole_250k.npz 1 25"
    [cheetah_velhidden]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 1 25"
)

for env_name in "${!ENVS[@]}"; do
    spec="${ENVS[$env_name]}"
    env_kind=$(echo "$spec" | awk '{print $1}')
    data_path=$(echo "$spec" | awk '{print $2}')
    hist=$(echo "$spec" | awk '{print $3}')
    goal=$(echo "$spec" | awk '{print $4}')

    for model in gru_baseline mlp_baseline; do
        out_dir="$RESULTS_DIR/$env_name/$model"
        if [ -f "$out_dir/final.pt" ]; then
            echo "[skip] $env_name/$model (exists)"
            continue
        fi
        log="$LOG_ROOT/${env_name}_${model}.log"
        echo "[train] $env_name / $model"
        nohup $PYTHON -m code.train.train \
            --model $model \
            --env-kind $env_kind \
            --data "$data_path" \
            --out "$out_dir" \
            --epochs $EPOCHS --batch $BATCH --lr $LR \
            --history-size $hist --goal-offset $goal \
            --save-every 0 --log-every 200 --num-workers 0 \
            --max-windows 800 \
            > $log 2>&1 &
        echo "  PID=$!"
    done
done

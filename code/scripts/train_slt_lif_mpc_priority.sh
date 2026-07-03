#!/bin/bash
# Train SLT-LIF-MPC trace + free variants on the 10 priority envs
# (6 standard + 4 stress). 1 epoch x max-windows=2000 per (env, variant).
set -e
cd /home/lx/snn

EPOCHS=1
MAX_WINDOWS=2000
BATCH=64
LR=3e-4
PY=/home/lx/miniconda3/envs/snn/bin/python
RESULTS_DIR=/home/lx/snn/results
LOG_DIR=/home/lx/snn/logs/slt_lif_mpc_train_v2
mkdir -p "$LOG_DIR"

# (env_name env_kind data_path goal_offset)
declare -A ENVS=(
    [ball_in_cup]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz 25"
    [cartpole_2d]="dmc /home/lx/snn/data/dm_control/cartpole_250k.npz 25"
    [cheetah]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 25"
    [finger]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz 25"
    [pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 100"
    [tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 100"
    [cartpole_flicker]="dmc /home/lx/snn/data/dm_control/cartpole_250k.npz 25"
    [cheetah_velhidden]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 25"
    [pusht_ood]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 100"
    [tworoom_long]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 100"
)

ENV_ORDER=(
    ball_in_cup cartpole_2d cheetah finger pusht tworoom
    cartpole_flicker cheetah_velhidden pusht_ood tworoom_long
)

train_one() {
    local env=$1
    local variant=$2  # trace | free
    local spec=${ENVS[$env]}
    local env_kind=$(echo "$spec" | awk '{print $1}')
    local data_path=$(echo "$spec" | awk '{print $2}')
    local goal=$(echo "$spec" | awk '{print $3}')
    local out_dir="$RESULTS_DIR/$env/slt_lif_mpc_${variant}"
    if [ -f "$out_dir/final.pt" ]; then
        echo "[skip] $env/$variant already trained"
        return
    fi
    mkdir -p "$out_dir"
    log="$LOG_DIR/${env}_${variant}.log"
    echo "=== [train-slt] $env / slt_lif_mpc_${variant} ==="
    CUDA_VISIBLE_DEVICES=0 $PY -m code.train.train \
        --model "slt_lif_mpc_${variant}" \
        --env-kind "$env_kind" \
        --data "$data_path" \
        --out "$out_dir" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --lr "$LR" \
        --save-every 0 \
        --n-layers 4 \
        --history-size 1 \
        --goal-offset "$goal" \
        --max-windows "$MAX_WINDOWS" \
        > "$log" 2>&1 && echo "ok: $env $variant" || echo "FAIL: $env $variant (see $log)"
}

for env in "${ENV_ORDER[@]}"; do
    train_one "$env" trace
    train_one "$env" free
done

echo "=== SLT-LIF-MPC PRIORITY TRAINING DONE ==="
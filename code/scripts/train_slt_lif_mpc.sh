#!/bin/bash
# Train both SLT-LIF-MPC variants on 16 standard + 4 stress envs.
# Distributes across 2 GPUs (one variant per GPU).
set -u
cd /home/lx/snn

EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
MAX_WINDOWS=${MAX_WINDOWS:-8000}

# (env_name env_kind data_path history_size goal_offset)
declare -A ENVS=(
    [pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 1 100"
    [tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 1 100"
    [reacher]="reacher_4d /home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz 1 25"
    [cartpole_2d]="dmc /home/lx/snn/data/dm_control/cartpole_250k.npz 1 25"
    [pendulum_2d]="dmc /home/lx/snn/data/dm_control/pendulum_250k.npz 1 25"
    [finger]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz 1 25"
    [ball_in_cup]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz 1 25"
    [cheetah]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 1 25"
    [walker]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz 1 25"
    [hopper]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz 1 25"
    [quadruped]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz 1 25"
    [humanoid]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz 1 25"
    [humanoid_CMU]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz 1 25"
    [dog]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz 1 25"
    [fish]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz 1 25"
    [stacker]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz 1 25"
)

ENV_ORDER=(
    pusht tworoom reacher cartpole_2d pendulum_2d finger ball_in_cup
    cheetah walker hopper quadruped humanoid humanoid_CMU dog fish stacker
)

# Train one (env, variant) on one GPU
train_one() {
    local env=$1
    local variant=$2  # trace | free
    local gpu=$3
    local spec=${ENVS[$env]}
    local env_kind=$(echo "$spec" | awk '{print $1}')
    local data_path=$(echo "$spec" | awk '{print $2}')
    local hist=$(echo "$spec" | awk '{print $3}')
    local goal=$(echo "$spec" | awk '{print $4}')
    local out_dir="$RESULTS_DIR/$env/slt_lif_mpc_${variant}"
    if [ -f "$out_dir/final.pt" ]; then
        echo "[skip] $env/$variant already trained"
        return
    fi
    mkdir -p "$out_dir"
    local log="$out_dir/train.log"
    echo "=== [train] $env / slt_lif_mpc_${variant} on GPU $gpu ==="
    CUDA_VISIBLE_DEVICES=$gpu /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
        --model "slt_lif_mpc_${variant}" \
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
        --max-windows "$MAX_WINDOWS" \
        > "$log" 2>&1 && echo "ok: $env $variant" || echo "FAIL: $env $variant (see $log)"
}

# Train all 16 envs for one variant sequentially on one GPU
train_variant() {
    local variant=$1
    local gpu=$2
    for env in "${ENV_ORDER[@]}"; do
        train_one "$env" "$variant" "$gpu"
    done
}

# Run both variants in parallel across 2 GPUs
echo "=== Training trace variant on GPU 0 ==="
train_variant trace 0
echo "=== Training free variant on GPU 1 ==="
train_variant free 1
echo "=== ALL TRAINING DONE ==="
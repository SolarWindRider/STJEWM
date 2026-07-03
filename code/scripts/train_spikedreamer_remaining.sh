#!/bin/bash
# Train SpikeDreamer on the 10 missing envs (dog, fish, hopper, humanoid,
# humanoid_CMU, pendulum_2d, quadruped, reacher, stacker, walker).
# 1 epoch x max-windows=2000 per env on a single RTX 4090.
set -e
cd /home/lx/snn

EPOCHS=1
MAX_WINDOWS=2000
BATCH=64
LR=3e-4
PY=/home/lx/miniconda3/envs/snn/bin/python
RESULTS_DIR=/home/lx/snn/results
LOG_DIR=/home/lx/snn/logs/spikedreamer_train_v2
mkdir -p "$LOG_DIR"

# (env_name data_path goal_offset)
declare -A ENVS=(
    [dog]="/home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz 25"
    [fish]="/home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz 25"
    [hopper]="/home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz 25"
    [humanoid]="/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz 25"
    [humanoid_CMU]="/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz 25"
    [pendulum_2d]="/home/lx/snn/data/dm_control/pendulum_250k.npz 25"
    [quadruped]="/home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz 25"
    [reacher]="/home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz 25"
    [stacker]="/home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz 25"
    [walker]="/home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz 25"
)

ENV_ORDER=(
    dog fish hopper humanoid humanoid_CMU pendulum_2d quadruped reacher stacker walker
)

for env in "${ENV_ORDER[@]}"; do
    spec=${ENVS[$env]}
    data=$(echo "$spec" | awk '{print $1}')
    goal=$(echo "$spec" | awk '{print $2}')
    out_dir="$RESULTS_DIR/$env/spikedreamer_baseline"
    if [ -f "$out_dir/final.pt" ]; then
        echo "[skip] $env already trained"
        continue
    fi
    mkdir -p "$out_dir"
    log="$LOG_DIR/$env.log"
    echo "=== [train-spike] $env ==="
    CUDA_VISIBLE_DEVICES=0 $PY -m code.train.train \
        --model spikedreamer_baseline \
        --env-kind dmc \
        --data "$data" \
        --out "$out_dir" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --lr "$LR" \
        --save-every 0 \
        --n-layers 4 \
        --history-size 1 \
        --goal-offset "$goal" \
        --max-windows "$MAX_WINDOWS" \
        > "$log" 2>&1 && echo "ok: $env" || echo "FAIL: $env (see $log)"
done

echo "=== SPIKEDREAMER REMAINING TRAINING DONE ==="
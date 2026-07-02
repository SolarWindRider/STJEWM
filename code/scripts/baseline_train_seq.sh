#!/bin/bash
# Sequential baseline trainer. Caps OMP threads to avoid CPU thrash.
# Loops through ALL 24 ckpts, skips ones where final.pt exists.
# Trains one at a time, then waits 30s.

set -u
cd /home/lx/snn
PYTHON=/home/lx/miniconda3/envs/snn/bin/python
LOG_ROOT=/home/lx/snn/logs/baseline_train
RESULTS_DIR=/home/lx/snn/results
mkdir -p "$LOG_ROOT"

EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-32}
MAX_WINDOWS=${MAX_WINDOWS:-5000}

declare -A ENVS=(
    [ball_in_cup]=/home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz
    [dog]=/home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz
    [finger]=/home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz
    [fish]=/home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz
    [hopper]=/home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz
    [humanoid]=/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz
    [humanoid_CMU]=/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz
    [pendulum_2d]=/home/lx/snn/data/dm_control/pendulum_250k.npz
    [quadruped]=/home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz
    [reacher]=/home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz
    [stacker]=/home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz
    [walker]=/home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz
)

declare -A ENV_KIND=(
    [reacher]=reacher_4d
)

ALL_ENVS=(ball_in_cup dog finger fish hopper humanoid humanoid_CMU pendulum_2d quadruped reacher stacker walker)

run_one() {
    local env="$1" model="$2"
    local out_dir="$RESULTS_DIR/$env/$model"
    if [ -f "$out_dir/final.pt" ]; then
        echo "[$(date +%H:%M:%S)] [skip] $env/$model already done"
        return 0
    fi
    local data="${ENVS[$env]}"
    local ekind="${ENV_KIND[$env]:-dmc}"
    mkdir -p "$out_dir"
    local logf="$LOG_ROOT/${env}_${model}_v2.log"
    echo "[$(date +%H:%M:%S)] START $env/$model (epochs=$EPOCHS batch=$BATCH maxw=$MAX_WINDOWS)"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES="" "$PYTHON" -m code.train.train \
        --model "$model" \
        --env-kind "$ekind" \
        --data "$data" \
        --out "$out_dir" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --lr 0.0003 \
        --history-size 1 \
        --goal-offset 25 \
        --seed 0 \
        --num-workers 0 \
        --save-every 0 \
        --log-every 200 \
        --max-windows "$MAX_WINDOWS" \
        > "$logf" 2>&1
    local rc=$?
    if [ -f "$out_dir/final.pt" ]; then
        echo "[$(date +%H:%M:%S)] OK   $env/$model (rc=$rc)"
    else
        echo "[$(date +%H:%M:%S)] FAIL $env/$model (rc=$rc) log=$logf"
        tail -10 "$logf" | head -8
    fi
}

# Order: smaller envs first (so we get quick wins)
# Order matters slightly: smaller state_dim/action_dim first
ALL_ENVS=(ball_in_cup finger pendulum_2d hopper walker quadruped dog fish reacher humanoid stacker humanoid_CMU)

for env in "${ALL_ENVS[@]}"; do
    for model in gru_baseline mlp_baseline; do
        run_one "$env" "$model"
        sleep 10
    done
done

echo "[$(date +%H:%M:%S)] ALL DONE"

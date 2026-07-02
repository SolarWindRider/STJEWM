#!/bin/bash
# Sequential-conservative retry: 1 at a time, 60s stagger, 5000 windows, 2 epochs.
# Triggered after main driver completes.
set -u
cd /home/lx/snn
PYTHON=/home/lx/miniconda3/envs/snn/bin/python
LOG_ROOT=/home/lx/snn/logs/baseline_train
RESULTS_DIR=/home/lx/snn/results

declare -A ENVS=(
    [ball_in_cup]=/home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz
    [finger]=/home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz
    [dog]=/home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz
    [pendulum_2d]=/home/lx/snn/data/dm_control/pendulum_250k.npz
    [quadruped]=/home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz
    [reacher]=/home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz
    [stacker]=/home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz
    [walker]=/home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz
    [fish]=/home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz
    [hopper]=/home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz
    [humanoid]=/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz
    [humanoid_CMU]=/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz
)

declare -A ENV_KIND=(
    [reacher]=reacher_4d
)

ALL_ENVS=(ball_in_cup dog finger fish hopper humanoid humanoid_CMU pendulum_2d quadruped reacher stacker walker)
TASKS=()
for env in "${ALL_ENVS[@]}"; do
    for model in gru_baseline mlp_baseline; do
        if [ ! -f "$RESULTS_DIR/$env/$model/final.pt" ]; then
            ekind="${ENV_KIND[$env]:-dmc}"
            TASKS+=("$env|$model|${ENVS[$env]}|$ekind")
        fi
    done
done

echo "[$(date +%H:%M:%S)] RETRY start, ${#TASKS[@]} pending tasks"
for t in "${TASKS[@]}"; do echo "  pending: $t"; done

run_one() {
    local spec="$1"
    IFS='|' read -r env model data ekind <<<"$spec"
    local out_dir="$RESULTS_DIR/$env/$model"
    if [ -f "$out_dir/final.pt" ]; then
        echo "[$(date +%H:%M:%S)] [skip] $env/$model already exists"
        return 0
    fi
    mkdir -p "$out_dir"
    local logf="$LOG_ROOT/${env}_${model}_retry.log"
    echo "[$(date +%H:%M:%S)] RETRY start $env/$model (epochs=2, maxw=5000)"
    CUDA_VISIBLE_DEVICES="" "$PYTHON" -m code.train.train \
        --model "$model" \
        --env-kind "$ekind" \
        --data "$data" \
        --out "$out_dir" \
        --epochs 2 \
        --batch 32 \
        --lr 0.0003 \
        --history-size 1 \
        --goal-offset 25 \
        --seed 0 \
        --num-workers 0 \
        --save-every 0 \
        --log-every 200 \
        --max-windows 5000 \
        > "$logf" 2>&1
    local rc=$?
    if [ -f "$out_dir/final.pt" ]; then
        echo "[$(date +%H:%M:%S)] RETRY OK  $env/$model (rc=$rc)"
    else
        echo "[$(date +%H:%M:%S)] RETRY FAIL $env/$model (rc=$rc) log=$logf"
    fi
}

# Sequentially with 60s stagger to dodge oomd
for spec in "${TASKS[@]}"; do
    run_one "$spec"
    sleep 60
done
echo "[$(date +%H:%M:%S)] RETRY done"

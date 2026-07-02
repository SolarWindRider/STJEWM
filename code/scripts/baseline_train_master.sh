#!/bin/bash
# Train gru_baseline + mlp_baseline on the 12 missing envs (24 ckpts).
# Background-only. Skips existing final.pt.
# Fan-out: 6 in parallel per wave to stay within 4h total.

set -u
cd /home/lx/snn
PYTHON=/home/lx/miniconda3/envs/snn/bin/python
LOG_ROOT=/home/lx/snn/logs/baseline_train
mkdir -p "$LOG_ROOT"

EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
RESULTS_DIR=/home/lx/snn/results
PARALLEL=${PARALLEL:-6}

# ENV -> data_path. env_kind is always "dmc" since all are DMC 250k npz except
# pusht (already done), tworoom (already done), cartpole_2d (already done),
# reacher (already done with reacher_4d). pendulum_2d uses pendulum_250k.npz.
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

# For reacher, env_kind="reacher_4d" (state_dim=4) so loader picks the right
# slicing; for the rest, "dmc" handles flat obs correctly.
declare -A ENV_KIND=(
    [reacher]=reacher_4d
)

# Build flat task list: "env|model|data|env_kind"
TASKS=()
for env in ball_in_cup dog finger fish hopper humanoid humanoid_CMU pendulum_2d quadruped reacher stacker walker; do
    data="${ENVS[$env]}"
    ekind="${ENV_KIND[$env]:-dmc}"
    for model in gru_baseline mlp_baseline; do
        TASKS+=("$env|$model|$data|$ekind")
    done
done

echo "Total tasks: ${#TASKS[@]}"
mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/master.log"
echo "=== baseline_train_master started $(date) ===" > "$MASTER_LOG"

run_one() {
    local spec="$1"
    IFS='|' read -r env model data ekind <<<"$spec"
    local out_dir="$RESULTS_DIR/$env/$model"
    if [ -f "$out_dir/final.pt" ]; then
        echo "[skip] $env/$model: final.pt exists"
        return 0
    fi
    mkdir -p "$out_dir"
    local logf="$LOG_ROOT/${env}_${model}.log"
    echo "[$(date +%H:%M:%S)] start $env/$model"
    cd /home/lx/snn
    CUDA_VISIBLE_DEVICES="" "$PYTHON" -m code.train.train \
        --model "$model" \
        --env-kind "$ekind" \
        --data "$data" \
        --out "$out_dir" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --lr "$LR" \
        --history-size 1 \
        --goal-offset 25 \
        --seed 0 \
        --num-workers 0 \
        --save-every 0 \
        --log-every 200 \
        --max-windows 10000 \
        > "$logf" 2>&1
    local rc=$?
    if [ -f "$out_dir/final.pt" ]; then
        echo "[$(date +%H:%M:%S)] OK  $env/$model (rc=$rc)"
        echo "OK $env $model rc=$rc" >> "$MASTER_LOG"
    else
        echo "[$(date +%H:%M:%S)] FAIL $env/$model (rc=$rc) -- see $logf"
        echo "FAIL $env $model rc=$rc log=$logf" >> "$MASTER_LOG"
    fi
}
export -f run_one
export EPOCHS BATCH LR RESULTS_DIR PYTHON LOG_ROOT MASTER_LOG

# Wave-based parallel runner
PIDS=()
i=0
for spec in "${TASKS[@]}"; do
    run_one "$spec" &
    PIDS+=($!)
    i=$((i+1))
    if [ $((i % PARALLEL)) -eq 0 ]; then
        # Wait for the wave to clear
        wait "${PIDS[@]}"
        PIDS=()
    fi
done
# Drain remaining
if [ ${#PIDS[@]} -gt 0 ]; then
    wait "${PIDS[@]}"
fi

echo "=== baseline_train_master done $(date) ===" >> "$MASTER_LOG"
echo "=== baseline_train_master done ==="

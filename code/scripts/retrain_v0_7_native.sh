#!/bin/bash
# v0.7 re-train: each SNN baseline uses its native loss (no longer ST-JEWM's 3-term).
# 4 model variants x 10 priority envs = 40 training runs.
# Train 1 epoch x max_windows=2000, batch=64, n_layers=4.
# Parallelize across 4 GPUs (one model variant per GPU).
set -e
cd /home/lx/snn

EPOCHS=1
MAX_WINDOWS=2000
BATCH=64
LR=3e-4
PY=/home/lx/miniconda3/envs/snn/bin/python
RESULTS_DIR=/home/lx/snn/results
LOG_DIR=/home/lx/snn/logs/v0_7_retrain
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

# Run a single (env, model) train task on a specific GPU.
train_one() {
    local env=$1
    local model=$2
    local gpu=$3
    local spec=${ENVS[$env]}
    local env_kind=$(echo "$spec" | awk '{print $1}')
    local data_path=$(echo "$spec" | awk '{print $2}')
    local goal=$(echo "$spec" | awk '{print $3}')
    local out_dir="$RESULTS_DIR/$env/$model"
    if [ -f "$out_dir/final.pt" ]; then
        echo "[skip] $env/$model already trained"
        return
    fi
    mkdir -p "$out_dir"
    log="$LOG_DIR/${env}_${model}.log"
    echo "=== [train-${model}] $env on GPU${gpu} ==="
    CUDA_VISIBLE_DEVICES=$gpu $PY -m code.train.train \
        --model "$model" \
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
        --lambda-sigreg 0.09 \
        --lambda-goal 0.5 \
        > "$log" 2>&1 && echo "ok: $env/$model" || echo "FAIL: $env/$model (see $log)"
}

# Launch all (env, model) pairs in parallel, one model per GPU:
#   GPU0: cubifae_baseline
#   GPU1: spikedreamer_baseline
#   GPU2: slt_lif_mpc_trace
#   GPU3: slt_lif_mpc_free
for env in "${ENV_ORDER[@]}"; do
    train_one "$env" cubifae_baseline 0 &
    train_one "$env" spikedreamer_baseline 1 &
    train_one "$env" slt_lif_mpc_trace 2 &
    train_one "$env" slt_lif_mpc_free 3 &
    # wait between envs to avoid OOM
    wait
done

echo "=== V0.7 NATIVE-LOSS RETRAIN DONE ==="

#!/bin/bash
# Train CubifAE on the 2 missing stress envs (cartpole_flicker, cheetah_velhidden).
# Use env_kind=dmc + underlying data (same pattern as SpikeDreamer stress training).
set -e
cd /home/lx/snn

EPOCHS=1
MAX_WINDOWS=2000
BATCH=64
LR=3e-4
PY=/home/lx/miniconda3/envs/snn/bin/python
RESULTS_DIR=/home/lx/snn/results
LOG_DIR=/home/lx/snn/logs/cubifae_train_v2
mkdir -p "$LOG_DIR"

declare -A ENVS=(
    [cartpole_flicker]="dmc /home/lx/snn/data/dm_control/cartpole_250k.npz 25"
    [cheetah_velhidden]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 25"
)

for env in "${!ENVS[@]}"; do
    spec=${ENVS[$env]}
    env_kind=$(echo "$spec" | awk '{print $1}')
    data=$(echo "$spec" | awk '{print $2}')
    goal=$(echo "$spec" | awk '{print $3}')
    out_dir="$RESULTS_DIR/$env/cubifae_baseline"
    if [ -f "$out_dir/final.pt" ]; then
        echo "[skip] $env already trained"
        continue
    fi
    mkdir -p "$out_dir"
    log="$LOG_DIR/$env.log"
    echo "=== [train-cubi] $env ==="
    CUDA_VISIBLE_DEVICES=0 $PY -m code.train.train \
        --model cubifae_baseline \
        --env-kind "$env_kind" \
        --data "$data" \
        --out "$out_dir" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --lr "$LR" \
        --lambda-sigreg 0.09 \
        --lambda-goal 0.5 \
        --save-every 0 \
        --n-layers 4 \
        --history-size 1 \
        --goal-offset "$goal" \
        --max-windows "$MAX_WINDOWS" \
        > "$log" 2>&1 && echo "ok: $env" || echo "FAIL: $env (see $log)"
done

echo "=== CUBIFAE STRESS TRAINING DONE ==="
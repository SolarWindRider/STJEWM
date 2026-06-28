#!/bin/bash
# Train BOTH models on ALL 15 envs, 5 epochs each.
# Per-env hyperparameters (history_size, goal_offset, max_windows) follow LeWM App. F.1.
# Datasets are capped at 250K windows to keep run time bounded.
#
# Layout:
#   /home/lx/snn/results/<env>/<model>/final.pt
#   /home/lx/snn/results/<env>/<model>/loss_log.json
#
# Usage:
#   ./train_all.sh                       # all 30 combos (15 envs × 2 models)
#   ./train_all.sh pusht tworoom          # only these envs
#   MODEL_FILTER=stjewm ./train_all.sh   # only STJEWM
#   EPOCHS=3 ./train_all.sh               # 3 epochs instead of 5

set -e
cd /home/lx/snn
EPOCHS=${EPOCHS:-5}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
mkdir -p "$RESULTS_DIR"

# ============== ENV -> DATA + hyperparams (per LeWM App. F.1) ==============
# Format: env_name env_kind data_path history_size goal_offset max_windows
#   max_windows caps dataset size to keep training time bounded.
declare -A ENVS=(
    [pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 1 100 10000"
    [tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 1 100 10000"
    [reacher]="reacher_4d /home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz 1 25 10000"
    [cartpole_2d]="dmc /home/lx/snn/data/dm_control/cartpole_250k.npz 1 25 10000"
    [pendulum_2d]="dmc /home/lx/snn/data/dm_control/pendulum_250k.npz 1 25 10000"
    [finger]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz 1 25 10000"
    [ball_in_cup]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz 1 25 10000"
    [cheetah]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 1 25 10000"
    [walker]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz 1 25 10000"
    [hopper]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz 1 25 10000"
    [quadruped]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz 1 25 10000"
    [humanoid]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz 1 25 10000"
    [humanoid_CMU]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz 1 25 10000"
    [dog]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz 1 25 10000"
    [fish]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz 1 25 10000"
    [stacker]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz 1 25 10000"
)

# Collect into a flat list
TMP=$(mktemp)
for k in "${!ENVS[@]}"; do
    echo "${k}=${ENVS[$k]}" >> "$TMP"
done

# Optional filter by env name(s) — space-separated list.
if [ -n "$1" ]; then
    FILTER="$*"
    REGEX=$(echo "$FILTER" | tr ' ' '|')
    grep -E "^($REGEX)=" "$TMP" > "$TMP.f" || { echo "No env matches $FILTER"; rm "$TMP"; exit 1; }
    mv "$TMP.f" "$TMP"
fi

# Each line: env_name env_kind data_path history_size goal_offset max_windows
while IFS='=' read -r name spec; do
    [ -z "$name" ] && continue
    env_kind=$(echo "$spec"   | awk '{print $1}')
    data_path=$(echo "$spec"  | awk '{print $2}')
    hist=$(echo "$spec"       | awk '{print $3}')
    goal=$(echo "$spec"       | awk '{print $4}')
    maxw=$(echo "$spec"       | awk '{print $5}')

    for model in stjewm lewm_baseline; do
        if [ -n "$MODEL_FILTER" ] && [ "$MODEL_FILTER" != "$model" ]; then
            continue
        fi
        out_dir="$RESULTS_DIR/$name/$model"
        if [ -f "$out_dir/final.pt" ]; then
            echo "[skip] $name/$model: $out_dir/final.pt already exists"
            continue
        fi
        echo ""
        echo "============================================="
        echo "[train] $name / $model  ($EPOCHS epochs, h=$hist, goal=$goal, maxw=$maxw)"
        echo "  data:   $data_path"
        echo "  out:    $out_dir"
        echo "============================================="

        mkdir -p "$out_dir"
        log="$out_dir/train.log"
        /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
            --model "$model" \
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
            2>&1 | tee "$log"
    done
done < "$TMP"
rm "$TMP"

echo ""
echo "============================================="
echo "ALL TRAININGS COMPLETE"
echo "============================================="

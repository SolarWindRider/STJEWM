#!/bin/bash
# Train two missing baselines on 4 envs:
#   1. mlp_baseline  — per-step FFN, no memory (the "no memory" floor)
#   2. stjewm_rate_only — STJEWM with RATE_ONLY readout (membrane forbidden)
#
# Layout:
#   /home/lx/snn/results/<env>/<model>/final.pt
#   /home/lx/snn/results/<env>/<model>/loss_log.json
#   /home/lx/snn/results/<env>/<model>/train.log
#
# Total: 4 envs x 2 models = 8 checkpoints.
# Hyperparameters mirror retrain_with_readout_modes.sh.
#
# Usage:
#   ./mlp_rate_master.sh                    # all 8
#   EPOCHS=3 BATCH=64 ./mlp_rate_master.sh  # defaults
#   ./mlp_rate_master.sh pusht              # filter envs

set -e
cd /home/lx/snn

# GPU: lock to device 2 to leave 0/1 free for zhangfa.
export CUDA_VISIBLE_DEVICES=2
PYTHON=/home/lx/miniconda3/envs/snn/bin/python

EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
LOG_ROOT=${LOG_ROOT:-/home/lx/snn/logs/mlp_rate}
mkdir -p "$RESULTS_DIR" "$LOG_ROOT"

# ============== ENV -> DATA + hyperparams (per LeWM App. F.1) ==============
# Format: env_name env_kind data_path history_size goal_offset max_windows
declare -A ENVS=(
    [pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 1 100 10000"
    [tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 1 100 10000"
    [cartpole_2d]="dmc /home/lx/snn/data/dm_control/cartpole_250k.npz 1 25 10000"
    [cheetah]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 1 25 10000"
)

# Optional env-name filter (space-separated list)
TMP=$(mktemp)
for k in "${!ENVS[@]}"; do
    echo "${k}=${ENVS[$k]}" >> "$TMP"
done
if [ -n "$1" ]; then
    FILTER="$*"
    REGEX=$(echo "$FILTER" | tr ' ' '|')
    grep -E "^($REGEX)=" "$TMP" > "$TMP.f" || { echo "No env matches $FILTER"; rm "$TMP" "$TMP.f" 2>/dev/null; exit 1; }
    mv "$TMP.f" "$TMP"
fi

# Models to train
MODELS=${MODELS:-"mlp_baseline stjewm_rate_only"}

# Each line: env_name env_kind data_path history_size goal_offset max_windows
while IFS='=' read -r name spec; do
    [ -z "$name" ] && continue
    env_kind=$(echo "$spec"  | awk '{print $1}')
    data_path=$(echo "$spec" | awk '{print $2}')
    hist=$(echo "$spec"      | awk '{print $3}')
    goal=$(echo "$spec"      | awk '{print $4}')
    maxw=$(echo "$spec"      | awk '{print $5}')

    for model in $MODELS; do
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
        echo "  cuda:   $CUDA_VISIBLE_DEVICES"
        echo "============================================="

        mkdir -p "$out_dir"
        log="$out_dir/train.log"
        if [ "$model" = "mlp_baseline" ]; then
            $PYTHON -m code.train.train \
                --model mlp_baseline \
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
        elif [ "$model" = "stjewm_rate_only" ]; then
            $PYTHON -m code.train.train \
                --model stjewm \
                --readout-mode rate_only \
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
        else
            echo "Unknown model: $model — skipping"
            continue
        fi
    done
done < "$TMP"
rm "$TMP"

echo ""
echo "============================================="
echo "MLP RATE TRAIN DONE"
echo "============================================="
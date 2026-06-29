#!/bin/bash
# Retrain STJEWM under 3 readout modes (trace_only / hidden_leak / spike_only)
# across all 16 LeWM-style envs.
#
# Layout (mirrors train_all.sh):
#   /home/lx/snn/results/<env>/stjewm_<mode>/final.pt
#   /home/lx/snn/results/<env>/stjewm_<mode>/loss_log.json
#   /home/lx/snn/results/<env>/stjewm_<mode>/train.log
#
# Total: 16 envs x 3 modes = 48 checkpoints.
#
# Usage:
#   ./retrain_with_readout_modes.sh                       # all 48 combos
#   MODES="trace_only spike_only" ./retrain_with_readout_modes.sh  # subset
#   EPOCHS=3 BATCH=64 ./retrain_with_readout_modes.sh     # defaults
#   ./retrain_with_readout_modes.sh pusht cheetah         # filter envs

set -e
cd /home/lx/snn

# GPU: lock to device 2 to leave 0/1 free for zhangfa.
export CUDA_VISIBLE_DEVICES=2
PYTHON=/home/lx/miniconda3/envs/snn/bin/python

EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
MODES=${MODES:-"trace_only hidden_leak spike_only"}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
LOG_ROOT=${LOG_ROOT:-/home/lx/snn/logs/retrain_readout}
mkdir -p "$RESULTS_DIR" "$LOG_ROOT"

# ============== ENV -> DATA + hyperparams (per LeWM App. F.1) ==============
# Format: env_name env_kind data_path history_size goal_offset max_windows
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

# Optional env-name filter (space-separated list)
TMP=$(mktemp)
for k in "${!ENVS[@]}"; do
    echo "${k}=${ENVS[$k]}" >> "$TMP"
done
if [ -n "$1" ]; then
    FILTER="$*"
    REGEX=$(echo "$FILTER" | tr ' ' '|')
    grep -E "^($REGEX)=" "$TMP" > "$TMP.f" || { echo "No env matches $FILTER"; rm "$TMP"; exit 1; }
    mv "$TMP.f" "$TMP"
fi

# Each line: env_name env_kind data_path history_size goal_offset max_windows
while IFS='=' read -r name spec; do
    [ -z "$name" ] && continue
    env_kind=$(echo "$spec"  | awk '{print $1}')
    data_path=$(echo "$spec" | awk '{print $2}')
    hist=$(echo "$spec"      | awk '{print $3}')
    goal=$(echo "$spec"      | awk '{print $4}')
    maxw=$(echo "$spec"      | awk '{print $5}')

    for mode in $MODES; do
        model="stjewm_${mode}"
        out_dir="$RESULTS_DIR/$name/$model"
        if [ -f "$out_dir/final.pt" ]; then
            echo "[skip] $name/$model: $out_dir/final.pt already exists"
            continue
        fi
        echo ""
        echo "============================================="
        echo "[retrain] $name / $model  ($EPOCHS epochs, h=$hist, goal=$goal, maxw=$maxw)"
        echo "  data:   $data_path"
        echo "  out:    $out_dir"
        echo "  cuda:   $CUDA_VISIBLE_DEVICES"
        echo "============================================="

        mkdir -p "$out_dir"
        log="$out_dir/train.log"
        $PYTHON -m code.train.train \
            --model stjewm \
            --readout-mode "$mode" \
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
echo "RETRAIN WITH READOUT MODES COMPLETE"
echo "============================================="

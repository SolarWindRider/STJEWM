#!/usr/bin/env bash
# Train the membrane_readout STJEWM on the 4 stress envs, 3 seeds each = 12 ckpts.
#
# This is the upper-bound ablation: the planner IS allowed to read the
# membrane potential (in addition to trace). If trace_only ≈
# membrane_readout, then the trace-only protocol is provably sufficient.
#
# Ckpt layout: /home/lx/snn/results/<env>/stjewm_membrane_readout_seed<s>/final.pt
#
# Usage:
#   ./train_stress_membrane.sh                # all 12 (4 envs x 3 seeds)
#   ./train_stress_membrane.sh pusht_ood       # one env, 3 seeds
#   EPOCHS=1 ./train_stress_membrane.sh       # fewer epochs
#   SEEDS="0 1" ./train_stress_membrane.sh    # only 2 seeds
#
# GPU: defaults to CUDA_VISIBLE_DEVICES=2 (use GPU 2 explicitly).
set -e
cd /home/lx/snn
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-32}
LR=${LR:-3e-4}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
MAX_WINDOWS=${MAX_WINDOWS:-800}
LOG_ROOT=${LOG_ROOT:-/home/lx/snn/logs/train_stress_membrane}
mkdir -p "$RESULTS_DIR" "$LOG_ROOT"

# GPU: lock to device 2 (the 0%-util one), per the assignment.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export PYTHONPATH=/home/lx/snn:${PYTHONPATH:-}

PYTHON=/home/lx/miniconda3/envs/snn/bin/python

# ============== ENV -> DATA ==============
# Stress env name, loader kind, data path, action_dim, history_size, goal_offset
declare -A ENVS=(
    [pusht_ood]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 2 1 100"
    [tworoom_long]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 2 1 100"
    [cartpole_flicker]="dmc /home/lx/snn/data/dm_control/cartpole_250k.npz 1 1 25"
    [cheetah_velhidden]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 6 1 25"
)

# ============== MODEL ==============
# Only membrane_readout — the upper bound.
MODEL_KIND="stjewm"
MODEL_DIRNAME="stjewm_membrane_readout"
READOUT_MODE="membrane_readout"

# ============== SEEDS ==============
SEEDS=${SEEDS:-"0 1 2"}

# Optional env-name filter (space-separated regex)
ENV_FILTER=""
if [ -n "${1:-}" ]; then
    ENV_FILTER="$*"
fi

# Iterate
for env_name in "${!ENVS[@]}"; do
    if [ -n "$ENV_FILTER" ]; then
        echo "$env_name" | grep -qE "^(${ENV_FILTER// /|})$" || continue
    fi

    spec="${ENVS[$env_name]}"
    env_kind=$(echo "$spec" | awk '{print $1}')
    data_path=$(echo "$spec" | awk '{print $2}')
    action_dim=$(echo "$spec" | awk '{print $3}')
    hist=$(echo "$spec" | awk '{print $4}')
    goal=$(echo "$spec" | awk '{print $5}')

    for seed in $SEEDS; do
        out_dir="$RESULTS_DIR/${env_name}/${MODEL_DIRNAME}_seed${seed}"
        if [ -f "$out_dir/final.pt" ]; then
            echo "[skip] $env_name/$MODEL_DIRNAME seed=$seed: $out_dir/final.pt already exists"
            continue
        fi
        mkdir -p "$out_dir"
        log="$out_dir/train.log"
        echo ""
        echo "============================================="
        echo "[train] $env_name / $MODEL_DIRNAME seed=$seed  ($EPOCHS ep, b=$BATCH)"
        echo "  data: $data_path"
        echo "  out:  $out_dir"
        echo "  mode: $READOUT_MODE (upper bound)"
        echo "============================================="

        $PYTHON -m code.train.train \
            --model "$MODEL_KIND" \
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
            --seed "$seed" \
            --readout-mode "$READOUT_MODE" \
            2>&1 | tee "$log"
    done
done

echo ""
echo "============================================="
echo "STRESS-MEMBRANE TRAINING COMPLETE"
echo "============================================="

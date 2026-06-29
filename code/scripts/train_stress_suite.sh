#!/usr/bin/env bash
# Train 60 stress-suite ckpts:
#   4 envs  (pusht_ood, tworoom_long, cartpole_flicker, cheetah_velhidden)
# × 5 models (stjewm_trace_only, stjewm_hidden_leak, stjewm_spike_only,
#             stjewm_no_trace, lewm_baseline_v2)
# × 3 seeds (0, 1, 2)
# = 60 ckpts total
#
# The stress env wrapper (FlickeringDMC, vel-hidden) is applied at EVAL
# time only — training uses the same raw data as the base env. The
# "tworoom_long" env is the same as tworoom but the eval forces a long
# goal_offset.
#
# Output: /home/lx/snn/results/<env>/<model>_seed<s>/final.pt
#
# Usage:
#   ./train_stress_suite.sh                   # all 60
#   ./train_stress_suite.sh 0 1               # only seeds 0 and 1
#   EPOCHS=1 ./train_stress_suite.sh          # fewer epochs
#   MODEL_FILTER=stjewm_trace_only ./train_stress_suite.sh  # only trace_only
#
# The script is CPU-only. Set CUDA_VISIBLE_DEVICES= before invoking if
# you want to force CPU even on a GPU node.
set -e
cd /home/lx/snn
EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-32}
LR=${LR:-3e-4}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
MAX_WINDOWS=${MAX_WINDOWS:-800}
mkdir -p "$RESULTS_DIR"

# Force CPU usage (GPU 2 is busy with Workstream A's retrain).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
export PYTHONPATH=/home/lx/snn:${PYTHONPATH:-}

# ============== ENV -> DATA ==============
# Stress env name, loader kind, data path, action_dim, history_size, goal_offset
declare -A ENVS=(
    [pusht_ood]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 2 1 100"
    [tworoom_long]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 2 1 100"
    [cartpole_flicker]="dmc /home/lx/snn/data/dm_control/cartpole_250k.npz 1 1 25"
    [cheetah_velhidden]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 6 1 25"
)

# ============== MODELS ==============
# 5 models: 4 STJEWM readout variants + 1 LeWM-style baseline
declare -a MODELS=(
    "stjewm stjewm_trace_only trace_only"
    "stjewm stjewm_hidden_leak hidden_leak"
    "stjewm stjewm_spike_only spike_only"
    "stjewm stjewm_no_trace no_trace"
    "lewm_baseline lewm_baseline_v2 none"
)

# ============== SEEDS ==============
SEEDS=(0 1 2)

# Optional env-name filter (positional)
FILTER=""
if [ -n "${1:-}" ]; then
    FILTER="$*"
fi

# Iterate
for env_name in "${!ENVS[@]}"; do
    # Apply env-name filter
    if [ -n "$FILTER" ]; then
        echo "$env_name" | grep -qE "^(${FILTER// /|})$" || continue
    fi

    spec="${ENVS[$env_name]}"
    env_kind=$(echo "$spec" | awk '{print $1}')
    data_path=$(echo "$spec" | awk '{print $2}')
    action_dim=$(echo "$spec" | awk '{print $3}')
    hist=$(echo "$spec" | awk '{print $4}')
    goal=$(echo "$spec" | awk '{print $5}')

    for model_spec in "${MODELS[@]}"; do
        # MODEL_FILTER override
        model_kind=$(echo "$model_spec" | awk '{print $1}')
        model_dirname=$(echo "$model_spec" | awk '{print $2}')
        readout_mode=$(echo "$model_spec" | awk '{print $3}')
        if [ -n "${MODEL_FILTER:-}" ] && [ "$model_dirname" != "$MODEL_FILTER" ]; then
            continue
        fi

        for seed in "${SEEDS[@]}"; do
            out_dir="$RESULTS_DIR/${env_name}/${model_dirname}_seed${seed}"
            if [ -f "$out_dir/final.pt" ]; then
                echo "[skip] $env_name/$model_dirname seed=$seed: $out_dir/final.pt already exists"
                continue
            fi
            mkdir -p "$out_dir"
            log="$out_dir/train.log"
            echo ""
            echo "============================================="
            echo "[train] $env_name / $model_dirname seed=$seed  ($EPOCHS ep, b=$BATCH)"
            echo "  data: $data_path"
            echo "  out:  $out_dir"
            echo "  mode: $readout_mode"
            echo "============================================="

            /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
                --model "$model_kind" \
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
                --readout-mode "$readout_mode" \
                2>&1 | tee "$log"
        done
    done
done

echo ""
echo "============================================="
echo "STRESS-SUITE TRAINING COMPLETE"
echo "============================================="

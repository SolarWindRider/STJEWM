#!/usr/bin/env bash
# Quick eval script for the "lewm_trace_only_like" ablation: re-evaluate the
# existing LeWM baseline (which has no trace) under the trace-only contract
# (i.e. pretend trace_proj exists and is zero). Since LeWM has no trace, its
# "trace_only" output is the result of trace_proj(zero_trace) = 0, which
# collapses the prediction. This is the right control: it shows that the
# membrane-forbidden protocol breaks LeWM because it has nothing to read.
#
# Output: results/aggregate/lewm_no_trace_eval/<env>.json
set -e
cd /home/lx/snn
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
OUT_DIR=${OUT_DIR:-/home/lx/snn/results/aggregate/lewm_no_trace_eval}
N_EPISODES=${N_EPISODES:-25}
N_SEEDS=${N_SEEDS:-2}
LOG_DIR=${LOG_DIR:-/home/lx/snn/logs/lewm_no_trace_eval}

mkdir -p "$OUT_DIR" "$LOG_DIR"

declare -A ENVS=(
    [pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 1 100"
    [tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 1 100"
    [cheetah]="cheetah /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 1 25"
    [cartpole_2d]="cartpole /home/lx/snn/data/dm_control/cartpole_250k.npz 1 25"
)

for env in "${!ENVS[@]}"; do
    spec="${ENVS[$env]}"
    eval_env=$(echo "$spec" | awk '{print $1}')
    data_path=$(echo "$spec" | awk '{print $2}')
    hist=$(echo "$spec" | awk '{print $3}')
    goal=$(echo "$spec" | awk '{print $4}')
    ckpt="$RESULTS_DIR/$env/lewm_baseline_v2/final.pt"
    if [ ! -f "$ckpt" ]; then
        echo "[skip] $env: no ckpt"
        continue
    fi
    out="$OUT_DIR/${env}.json"
    log="$LOG_DIR/${env}.log"
    if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[skip-eval] $env: $out exists"
        continue
    fi
    echo "============================================="
    echo "[eval-lewm-no-trace] $env"
    echo "============================================="
    /home/lx/miniconda3/envs/snn/bin/python -m code.eval.closed_loop \
        --env "$eval_env" \
        --ckpt "$ckpt" \
        --data "$data_path" \
        --out "$out" \
        --n-episodes "$N_EPISODES" \
        --n-seeds "$N_SEEDS" \
        --horizon 5 \
        --eval-budget 50 \
        --history-size "$hist" \
        --goal-offset "$goal" \
        2>&1 | tee "$log"
done

echo ""
echo "LEWM_NO_TRACE_EVAL COMPLETE"

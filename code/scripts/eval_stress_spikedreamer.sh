#!/bin/bash
# Eval SpikeDreamer on the 4 stress envs.
set -e
cd /home/lx/snn

PY=/home/lx/miniconda3/envs/snn/bin/python
RESULTS_DIR=/home/lx/snn/results
OUT_DIR=/home/lx/snn/results/aggregate/eval_stress_spikedreamer
LOG_DIR=/home/lx/snn/logs/eval_stress_spikedreamer_v2
mkdir -p "$OUT_DIR" "$LOG_DIR"

declare -A ENVS=(
    [cartpole_flicker]="cartpole /home/lx/snn/data/dm_control/cartpole_250k.npz 25"
    [cheetah_velhidden]="cheetah /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 25"
    [pusht_ood]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 100"
    [tworoom_long]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 100"
)

for env in "${!ENVS[@]}"; do
    spec=${ENVS[$env]}
    eval_env=$(echo "$spec" | awk '{print $1}')
    data=$(echo "$spec" | awk '{print $2}')
    goal=$(echo "$spec" | awk '{print $3}')
    ckpt="$RESULTS_DIR/$env/spikedreamer_baseline/final.pt"
    if [ ! -f "$ckpt" ]; then
        echo "[skip-eval] $env: no ckpt"
        continue
    fi
    out="$OUT_DIR/$env.json"
    if [ -f "$out" ]; then
        echo "[skip-eval] $env: $out exists"
        continue
    fi
    log="$LOG_DIR/$env.log"
    echo "=== [eval-spike-stress] $env ==="
    CUDA_VISIBLE_DEVICES=0 $PY -m code.eval.closed_loop \
        --env "$eval_env" \
        --ckpt "$ckpt" \
        --data "$data" \
        --out "$out" \
        --n-episodes 10 \
        --n-seeds 1 \
        --history-size 1 \
        --goal-offset "$goal" \
        --horizon 5 \
        --eval-budget 30 \
        > "$log" 2>&1 && echo "ok: $env" || echo "FAIL: $env (see $log)"
done

echo "=== SPIKE STRESS EVAL DONE ==="

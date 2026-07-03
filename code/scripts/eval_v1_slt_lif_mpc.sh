#!/bin/bash
# Eval SLT-LIF-MPC trace + free variants on the 10 priority envs (6 standard + 4 stress).
set -e
cd /home/lx/snn

PY=/home/lx/miniconda3/envs/snn/bin/python
RESULTS_DIR=/home/lx/snn/results
OUT_DIR=/home/lx/snn/results/aggregate/eval_v1_slt_lif_mpc
LOG_DIR=/home/lx/snn/logs/eval_v1_slt_lif_mpc
mkdir -p "$OUT_DIR" "$LOG_DIR"

declare -A ENVS=(
    [ball_in_cup]="ball_in_cup /home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz 25"
    [cartpole_2d]="cartpole /home/lx/snn/data/dm_control/cartpole_250k.npz 25"
    [cheetah]="cheetah /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 25"
    [finger]="finger /home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz 25"
    [pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 100"
    [tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 100"
    [cartpole_flicker]="cartpole /home/lx/snn/data/dm_control/cartpole_250k.npz 25"
    [cheetah_velhidden]="cheetah /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 25"
    [pusht_ood]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 100"
    [tworoom_long]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 100"
)

ENV_ORDER=(
    ball_in_cup cartpole_2d cheetah finger pusht tworoom
    cartpole_flicker cheetah_velhidden pusht_ood tworoom_long
)

for variant in trace free; do
    for env in "${ENV_ORDER[@]}"; do
        spec=${ENVS[$env]}
        eval_env=$(echo "$spec" | awk '{print $1}')
        data=$(echo "$spec" | awk '{print $2}')
        goal=$(echo "$spec" | awk '{print $3}')
        ckpt="$RESULTS_DIR/$env/slt_lif_mpc_${variant}/final.pt"
        if [ ! -f "$ckpt" ]; then
            echo "[skip-eval] $env/$variant: no ckpt"
            continue
        fi
        out="$OUT_DIR/${env}_${variant}.json"
        if [ -f "$out" ]; then
            echo "[skip-eval] $env/$variant: $out exists"
            continue
        fi
        log="$LOG_DIR/${env}_${variant}.log"
        echo "=== [eval-slt] $env / $variant ==="
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
            > "$log" 2>&1 && echo "ok: $env $variant" || echo "FAIL: $env $variant (see $log)"
    done
done

echo "=== SLT EVAL DONE ==="
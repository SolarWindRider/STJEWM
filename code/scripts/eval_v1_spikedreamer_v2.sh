#!/bin/bash
# Eval SpikeDreamer on the 16-env standard suite (10 trained envs use ckpts,
# missing envs are skipped).
set -e
cd /home/lx/snn

PY=/home/lx/miniconda3/envs/snn/bin/python
RESULTS_DIR=/home/lx/snn/results
OUT_DIR=/home/lx/snn/results/aggregate/eval_v1_spikedreamer
LOG_DIR=/home/lx/snn/logs/eval_v1_spikedreamer_v2
mkdir -p "$OUT_DIR" "$LOG_DIR"

declare -A ENVS=(
    [pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 100"
    [tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 100"
    [reacher]="reacher /home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz 25"
    [cartpole_2d]="cartpole /home/lx/snn/data/dm_control/cartpole_250k.npz 25"
    [pendulum_2d]="pendulum /home/lx/snn/data/dm_control/pendulum_250k.npz 25"
    [finger]="finger /home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz 25"
    [ball_in_cup]="ball_in_cup /home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz 25"
    [cheetah]="cheetah /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 25"
    [walker]="walker /home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz 25"
    [hopper]="hopper /home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz 25"
    [quadruped]="quadruped /home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz 25"
    [humanoid]="humanoid /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz 25"
    [humanoid_CMU]="humanoid_cmu /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz 25"
    [dog]="dog /home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz 25"
    [fish]="fish /home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz 25"
    [stacker]="stacker /home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz 25"
)

ENV_ORDER=(
    pusht tworoom reacher cartpole_2d pendulum_2d finger ball_in_cup cheetah walker
    hopper quadruped humanoid humanoid_CMU dog fish stacker
)

for env in "${ENV_ORDER[@]}"; do
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
    echo "=== [eval-spike] $env ==="
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

echo "=== SPIKE 16-ENV EVAL DONE ==="
#!/bin/bash
# Train SLT-LIF-MPC trace+free on the 10 missing standard envs.
# Background-only. Skips existing final.pt.

set -u
cd /home/lx/snn
PYTHON=/home/lx/miniconda3/envs/snn/bin/python
LOG_ROOT=/home/lx/snn/logs/slt_stress_missing_train
mkdir -p "$LOG_ROOT"

EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-32}
LR=${LR:-3e-4}
RESULTS_DIR=/home/lx/snn/results

# (env, data_path)
declare -A ENVS=(
    [cheetah_velhidden]=/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz
    [dog]=/home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz
    [fish]=/home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz
    [hopper]=/home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz
    [humanoid]=/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz
    [humanoid_CMU]=/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz
    [pendulum_2d]=/home/lx/snn/data/dm_control/pendulum_250k.npz
    [quadruped]=/home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz
    [reacher]=/home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz
    [stacker]=/home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz
    [walker]=/home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz
)

for env in "${!ENVS[@]}"; do
    data="${ENVS[$env]}"
    case $env in
        pendulum_2d) envarg=pendulum; goal=25 ;;
        cheetah_velhidden) envarg=cheetah; goal=25 ;;
        *) envarg=$env; goal=25 ;;
    esac
    for variant in trace free; do
        out="$RESULTS_DIR/$env/slt_lif_mpc_$variant"
        if [ -f "$out/final.pt" ]; then
            echo "[skip] $env/slt_lif_mpc_$variant (exists)"
            continue
        fi
        log="$LOG_ROOT/${env}_${variant}.log"
        echo "[train] $env / slt_lif_mpc_$variant"
        nohup $PYTHON -m code.train.train \
            --model "slt_lif_mpc_$variant" \
            --env-kind dmc \
            --data "$data" \
            --out "$out" \
            --epochs $EPOCHS --batch $BATCH --lr $LR \
            --history-size 1 --goal-offset $goal \
            --save-every 0 --log-every 200 --num-workers 0 \
            --max-windows 800 \
            > $log 2>&1 &
        echo "  PID=$!"
    done
done

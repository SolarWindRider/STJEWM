#!/bin/bash
# Retrain spikedreamer on 15 envs with native loss (v0.7 fix).
# Skips ckpts newer than 2026-07-04.
set -u
cd /home/lx/snn
PYTHON=/home/lx/miniconda3/envs/snn/bin/python
LOG=/home/lx/snn/logs/spikedreamer_native_retrain
mkdir -p "$LOG"

ENVS=(
    "ball_in_cup:/home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz:25"
    "cartpole_2d:/home/lx/snn/data/dm_control/cartpole_250k.npz:25"
    "cheetah:/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz:25"
    "cheetah_velhidden:/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz:25"
    "dog:/home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz:25"
    "finger:/home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz:25"
    "fish:/home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz:25"
    "hopper:/home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz:25"
    "humanoid:/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz:25"
    "humanoid_CMU:/home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz:25"
    "pendulum_2d:/home/lx/snn/data/dm_control/pendulum_250k.npz:25"
    "quadruped:/home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz:25"
    "reacher:/home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz:25"
    "stacker:/home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz:25"
    "tworoom:/home/lx/LeWM/data/tworoom_extract/tworoom.h5:100"
    "walker:/home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz:25"
    "pusht:/home/lx/LeWM/data/pusht_expert_train.h5:100"
    "cartpole_flicker:/home/lx/snn/data/dm_control/cartpole_250k.npz:25"
    "pusht_ood:/home/lx/LeWM/data/pusht_expert_train.h5:100"
    "tworoom_long:/home/lx/LeWM/data/tworoom_extract/tworoom.h5:100"
)

EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-32}
LR=${LR:-3e-4}

# Skip ckpts newer than 2026-07-04 (they were already retrained)
for spec in "${ENVS[@]}"; do
    env=$(echo "$spec" | cut -d: -f1)
    data=$(echo "$spec" | cut -d: -f2)
    goal=$(echo "$spec" | cut -d: -f3)
    out="/home/lx/snn/results/$env/spikedreamer_baseline"
    if [ -f "$out/final.pt" ]; then
        # Check timestamp
        ts=$(stat -c %Y "$out/final.pt" 2>/dev/null)
        if [ -n "$ts" ] && [ "$ts" -ge 1751606400 ]; then
            echo "[skip] $env/spikedreamer (recent: $(date -d @$ts '+%Y-%m-%d %H:%M'))"
            continue
        fi
    fi
    log="$LOG/${env}.log"
    nohup $PYTHON -m code.train.train \
        --model spikedreamer_baseline \
        --env-kind dmc \
        --data "$data" \
        --out "$out" \
        --epochs $EPOCHS --batch $BATCH --lr $LR \
        --history-size 1 --goal-offset $goal \
        --save-every 0 --log-every 200 --num-workers 0 \
        --max-windows 800 \
        > $log 2>&1 &
    echo "[train] $env/spikedreamer PID=$!"
done

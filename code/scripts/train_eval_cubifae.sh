#!/usr/bin/env bash
# Train CubifAE on the 16-env standard suite + 4 stress envs, then eval.
# Per-env hyperparameters mirror train_all.sh.
#
# Usage:
#   ./train_eval_cubifae.sh
set -e
cd /home/lx/snn
EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
PY=/home/lx/miniconda3/envs/snn/bin/python
mkdir -p "$RESULTS_DIR"

# Standard 16-env suite (per LeWM App. F.1)
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

# Stress envs (4): cartpole_flicker, cheetah_velhidden, pusht_ood, tworoom_long
declare -A STRESS=(
    [cartpole_flicker]="cartpole /home/lx/snn/data/dm_control/cartpole_250k.npz 1 25 10000"
    [cheetah_velhidden]="cheetah /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 1 25 10000"
    [pusht_ood]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 1 100 10000"
    [tworoom_long]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 1 100 10000"
)

train_env() {
    local name="$1" env_kind="$2" data="$3" hist="$4" goal="$5" maxw="$6"
    local out_dir="$RESULTS_DIR/$name/cubifae_baseline"
    if [ -f "$out_dir/final.pt" ]; then
        echo "[skip-train] $name/cubifae_baseline: $out_dir/final.pt exists"
        return
    fi
    echo ""
    echo "============================================="
    echo "[train] $name / cubifae_baseline  (ep=$EPOCHS, h=$hist, goal=$goal, maxw=$maxw)"
    echo "  data: $data"
    echo "  out:  $out_dir"
    echo "============================================="
    mkdir -p "$out_dir"
    log="$out_dir/train.log"
    $PY -m code.train.train \
        --model cubifae_baseline \
        --env-kind "$env_kind" \
        --data "$data" \
        --out "$out_dir" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --lr "$LR" \
        --lambda-sigreg 0.09 \
        --lambda-goal 0.5 \
        --save-every 0 \
        --n-layers 4 \
        --history-size "$hist" \
        --goal-offset "$goal" \
        --max-windows "$maxw" \
        2>&1 | tee "$log"
}

train_all() {
    local -n assoc=$1
    for name in "${!assoc[@]}"; do
        IFS=' ' read -r env_kind data hist goal maxw <<< "${assoc[$name]}"
        train_env "$name" "$env_kind" "$data" "$hist" "$goal" "$maxw"
    done
}

echo "=========== TRAINING 16-ENV STANDARD SUITE ==========="
train_all ENVS

echo "=========== TRAINING 4 STRESS ENVS ==========="
train_all STRESS

echo ""
echo "============================================="
echo "TRAINING COMPLETE"
echo "============================================="

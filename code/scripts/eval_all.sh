#!/bin/bash
# Evaluate ALL trained (env, model) pairs with per-env hyperparams.
#
# Output: /home/lx/snn/results/<env>/<model>/eval.json
#
# Usage:
#   ./eval_all.sh                    # all 50 combos
#   ./eval_all.sh pusht tworoom       # only these envs
#   MODEL_FILTER=stjewm ./eval_all.sh
#   N_EPISODES=20 ./eval_all.sh      # override eval budget

set -e
cd /home/lx/snn
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
N_EPISODES=${N_EPISODES:-50}
N_SEEDS=${N_SEEDS:-3}
HORIZON=${HORIZON:-5}
EVAL_BUDGET=${EVAL_BUDGET:-50}

# (env_name, eval_env_string, env_kind_in_train, train_data_path, action_dim, history_size, goal_offset)
# Per LeWM App. F.1 + per-env goal_offset.
declare -A ENVS=(
    [pusht]="pusht pusht /home/lx/LeWM/data/pusht_expert_train.h5 10 1 100"
    [tworoom]="tworoom tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 2 1 100"
    [reacher]="reacher reacher_4d /home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz 2 1 25"
    [cartpole_2d]="cartpole dmc /home/lx/snn/data/dm_control/cartpole_250k.npz 1 1 25"
    [pendulum_2d]="pendulum dmc /home/lx/snn/data/dm_control/pendulum_250k.npz 1 1 25"
    [finger]="finger dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz 2 1 25"
    [ball_in_cup]="ball_in_cup dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz 2 1 25"
    [cheetah]="cheetah dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 6 1 25"
    [walker]="walker dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz 6 1 25"
    [hopper]="hopper dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz 4 1 25"
    [quadruped]="quadruped dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz 12 1 25"
    [humanoid]="humanoid dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz 21 1 25"
    [humanoid_CMU]="humanoid_cmu dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz 56 1 25"
    [dog]="dog dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz 38 1 25"
    [fish]="fish dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz 5 1 25"
    [stacker]="stacker dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz 5 1 25"
)

# Collect into a flat list
TMP=$(mktemp)
for k in "${!ENVS[@]}"; do
    echo "${k}=${ENVS[$k]}" >> "$TMP"
done

# Optional env filter
if [ -n "$1" ]; then
    FILTER="$1"
    REGEX=$(echo "$FILTER" | tr ' ' '|')
    grep -E "^($REGEX)=" "$TMP" > "$TMP.f" || { echo "No env matches $FILTER"; rm "$TMP"; exit 1; }
    mv "$TMP.f" "$TMP"
fi

while IFS='=' read -r name spec; do
    [ -z "$name" ] && continue
    eval_env=$(echo "$spec"   | awk '{print $1}')
    env_kind=$(echo "$spec"   | awk '{print $2}')
    data_path=$(echo "$spec"  | awk '{print $3}')
    act_dim=$(echo "$spec"    | awk '{print $4}')
    hist=$(echo "$spec"       | awk '{print $5}')
    goal=$(echo "$spec"       | awk '{print $6}')

    for model in stjewm lewm_baseline; do
        if [ -n "$MODEL_FILTER" ] && [ "$MODEL_FILTER" != "$model" ]; then
            continue
        fi
        ckpt="$RESULTS_DIR/$name/$model/final.pt"
        if [ ! -f "$ckpt" ]; then
            echo "[skip] $name/$model: no ckpt at $ckpt"
            continue
        fi
        out="$RESULTS_DIR/$name/$model/eval.json"
        log="$RESULTS_DIR/$name/$model/eval.log"
        echo ""
        echo "============================================="
        echo "[eval] $name / $model  (h=$hist, goal=$goal)"
        echo "  ckpt:    $ckpt"
        echo "  out:     $out"
        echo "============================================="

        /home/lx/miniconda3/envs/snn/bin/python -m code.eval.closed_loop \
            --env "$eval_env" \
            --ckpt "$ckpt" \
            --data "$data_path" \
            --out "$out" \
            --n-episodes "$N_EPISODES" \
            --n-seeds "$N_SEEDS" \
            --horizon "$HORIZON" \
            --eval-budget "$EVAL_BUDGET" \
            --history-size "$hist" \
            --goal-offset "$goal" \
            2>&1 | tee "$log"
    done
done < "$TMP"
rm "$TMP"

echo ""
echo "============================================="
echo "ALL EVALS COMPLETE"
echo "============================================="

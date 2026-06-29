#!/usr/bin/env bash
# Eval the 4 stress-suite envs.
#
# Each env_kind in ENVS_4X points to a stress env that exercises a different
# property of the world model:
#   pusht_ood         — held-out 20% of pusht dataset (B4: OOD goal split)
#   tworoom_long      — same env, goal_offset forced to 200 (B2: long horizon)
#   cartpole_flicker  — obs randomly zeroed with prob 0.5 (B1: flicker)
#   cheetah_velhidden — velocity components zeroed at every step (B3: vel hidden)
#
# Usage:
#   ./eval_stress_suite.sh                    # all 4 stress envs, model dir stjewm_v2
#   ./eval_stress_suite.sh stjewm_trace_only  # override model dir
#   MODEL_FILTER=stjewm_trace_only ./eval_stress_suite.sh
#   N_EPISODES=10 ./eval_stress_suite.sh      # smaller eval
set -e
cd /home/lx/snn
MODEL=${1:-${MODEL_FILTER:-stjewm_v2}}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
N_EPISODES=${N_EPISODES:-25}
N_SEEDS=${N_SEEDS:-2}
HORIZON=${HORIZON:-5}
EVAL_BUDGET=${EVAL_BUDGET:-50}
HIST=${HIST:-1}
# goal_offset is the DEFAULT used by the script; for tworoom_long the
# eval_closed_loop main() forces goal_offset_override=200 regardless.
GOAL=${GOAL:-100}
SPLIT=${SPLIT:-in_dist}

# (eval_env, train_data_path, split_arg)
declare -A STRESS_ENVS=(
    [pusht_ood]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 unseen_goal"
    [tworoom_long]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 in_dist"
    [cartpole_flicker]="cartpole_flicker /home/lx/snn/data/dm_control/cartpole_250k.npz in_dist"
    [cheetah_velhidden]="cheetah_velhidden /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz in_dist"
)

# Collect into a flat list
TMP=$(mktemp)
for k in "${!STRESS_ENVS[@]}"; do
    echo "${k}=${STRESS_ENVS[$k]}" >> "$TMP"
done

mkdir -p "$RESULTS_DIR/aggregate/stress_logs"
while IFS='=' read -r name spec; do
    [ -z "$name" ] && continue
    eval_env=$(echo "$spec" | awk '{print $1}')
    data_path=$(echo "$spec" | awk '{print $2}')
    split_arg=$(echo "$spec" | awk '{print $3}')

    ckpt="$RESULTS_DIR/$name/$MODEL/final.pt"
    # Stress envs store ckpts at <env>/<model>_seed<s>/final.pt
    # Try the direct path first; if missing, look for any _seed0 ckpt
    if [ ! -f "$ckpt" ]; then
        seed_ckpt=$(ls -1 "$RESULTS_DIR/$name"/${MODEL}_seed*/final.pt 2>/dev/null | head -1)
        if [ -n "$seed_ckpt" ]; then
            ckpt="$seed_ckpt"
        else
            echo "[skip] $name/$MODEL: no ckpt at $ckpt"
            continue
        fi
    fi
    out="$RESULTS_DIR/$name/$MODEL/eval.json"
    log="$RESULTS_DIR/aggregate/stress_logs/eval_${MODEL}_${name}.log"
    echo ""
    echo "============================================="
    echo "[eval] $name / $MODEL  (h=$HIST, goal=$GOAL, split=$split_arg)"
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
        --history-size "$HIST" \
        --goal-offset "$GOAL" \
        --split "$split_arg" \
        2>&1 | tee "$log"
done < "$TMP"
rm "$TMP"

echo ""
echo "============================================="
echo "STRESS-SUITE EVALS COMPLETE"
echo "============================================="

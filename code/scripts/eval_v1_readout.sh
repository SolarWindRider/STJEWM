#!/usr/bin/env bash
# Evaluate the new STJEWM readout-mode checkpoints (trace_only, hidden_leak, spike_only)
# across all 16 LeWM-style envs.
#
# Output: /home/lx/snn/results/aggregate/eval_v1_readout/<env>_<mode>.json
#
# Usage:
#   ./eval_v1_readout.sh                          # all 48 combos
#   ./eval_v1_readout.sh trace_only                # one mode, all envs
#   ./eval_v1_readout.sh trace_only cheetah dog    # one mode, specific envs
set -e
cd /home/lx/snn

MODES=${1:-${MODES:-"trace_only hidden_leak spike_only"}}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
OUT_DIR=${OUT_DIR:-/home/lx/snn/results/aggregate/eval_v1_readout}
N_EPISODES=${N_EPISODES:-25}
N_SEEDS=${N_SEEDS:-2}
HORIZON=${HORIZON:-5}
EVAL_BUDGET=${EVAL_BUDGET:-50}
LOG_DIR=${LOG_DIR:-/home/lx/snn/logs/eval_v1_readout}

mkdir -p "$OUT_DIR" "$LOG_DIR"

# Per-env hyperparams (must mirror train_all.sh).
declare -A ENVS=(
    [pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 1 100"
    [tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 1 100"
    [reacher]="reacher /home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz 1 25"
    [cartpole_2d]="cartpole /home/lx/snn/data/dm_control/cartpole_250k.npz 1 25"
    [pendulum_2d]="pendulum /home/lx/snn/data/dm_control/pendulum_250k.npz 1 25"
    [finger]="finger /home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz 1 25"
    [ball_in_cup]="ball_in_cup /home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz 1 25"
    [cheetah]="cheetah /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 1 25"
    [walker]="walker /home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz 1 25"
    [hopper]="hopper /home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz 1 25"
    [quadruped]="quadruped /home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz 1 25"
    [humanoid]="humanoid /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz 1 25"
    [humanoid_CMU]="humanoid_cmu /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz 1 25"
    [dog]="dog /home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz 1 25"
    [fish]="fish /home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz 1 25"
    [stacker]="stacker /home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz 1 25"
)

# Optional positional env filter
ENV_FILTER="${@:2}"

# Collect env list to a flat file
TMP=$(mktemp)
for k in "${!ENVS[@]}"; do
    echo "${k}=${ENVS[$k]}" >> "$TMP"
done
if [ -n "$ENV_FILTER" ]; then
    REGEX=$(echo "$ENV_FILTER" | tr ' ' '|')
    grep -E "^($REGEX)=" "$TMP" > "$TMP.f" || { echo "No env matches $ENV_FILTER"; rm "$TMP"; exit 1; }
    mv "$TMP.f" "$TMP"
fi

# Eval each (env, mode) combo. The model is selected by the path.
while IFS='=' read -r name spec; do
    [ -z "$name" ] && continue
    eval_env=$(echo "$spec" | awk '{print $1}')
    data_path=$(echo "$spec" | awk '{print $2}')
    hist=$(echo "$spec" | awk '{print $3}')
    goal=$(echo "$spec" | awk '{print $4}')

    for mode in $MODES; do
        ckpt="$RESULTS_DIR/$name/stjewm_${mode}/final.pt"
        if [ ! -f "$ckpt" ]; then
            echo "[skip] $name/stjewm_${mode}: no ckpt at $ckpt"
            continue
        fi
        out="$OUT_DIR/${name}_${mode}.json"
        log="$LOG_DIR/${name}_${mode}.log"
        # Skip if already evaluated
        if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "[skip-eval] $name/stjewm_${mode}: $out exists"
            continue
        fi
        echo ""
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
echo "EVAL_V1_READOUT COMPLETE"
echo "============================================="

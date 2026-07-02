#!/usr/bin/env bash
# ============================================================
# Stress difficulty sweep (Experiment 1).
#
# Runs closed-loop eval across a multi-level difficulty axis per stress env
# and 5 STJEWM modes (× N seeds), reusing the existing /home/lx/snn/results
# checkpoints. NEVER retrains.
#
# Difficulty axes:
#   cartpole_flicker    : flicker mask ratio {0.25, 0.50, 0.75}
#   pusht_ood           : goal_offset {50, 100, 200} (split=unseen_goal)
#   tworoom_long        : goal_offset {50, 100, 200} (split=in_dist)
#   cheetah_velhidden   : in_dist baseline (no sweep; kept as a single row)
#
# Models: 5 STJEWM modes from /home/lx/snn/results/<env>/<model>_seed<s>/final.pt
# Seeds: 0, 1, 2 by default. Override with SEEDS="0" to drop to seed 0 only
#        (use this if wall-time is tight).
#
# Eval budget per cell:
#   --n-episodes 20 --n-seeds 2 --horizon 5 --eval-budget 50 --history-size 1
#
# Total cells (5 models × 1 seed × 9 difficulties) ≈ 45 ≈ ~45-60 min.
#
# Usage:
#   ./run_stress_sweep.sh                            # default: all 5 modes × 1 seed
#   SEEDS="0 1 2" ./run_stress_sweep.sh              # full 3 seeds (2x time)
#   SEEDS="0" DIFFS_OVERRIDE="cartpole_flicker:0.25" ./run_stress_sweep.sh
# ============================================================
set -u

ROOT="/home/lx/snn"
RESULTS_DIR="$ROOT/results"
LOG_DIR="$ROOT/logs/stress_sweep"
PY="/home/lx/miniconda3/envs/snn/bin/python"

mkdir -p "$LOG_DIR" "$RESULTS_DIR/aggregate"

cd "$ROOT" || exit 1

# ---------- knobs ----------
SEEDS=${SEEDS:-"0"}                            # default: 1 seed (45 cells)
N_EPISODES=${N_EPISODES:-20}
N_SEEDS=${N_SEEDS:-2}
HORIZON=${HORIZON:-5}
EVAL_BUDGET=${EVAL_BUDGET:-50}
HIST=${HIST:-1}

# 5 STJEWM modes
MODELS=(
    stjewm_trace_only
    stjewm_hidden_leak
    stjewm_spike_only
    stjewm_no_trace
    stjewm_membrane_readout
)

# Difficulty axes. Format: env_kind|out_dir|data_path|split|knob:value,knob:value,knob:value
# knob is one of: goal_offset | flicker_mask_ratio | vel_hidden_mask_obs_ratio
DIFFICULTIES=(
    "cartpole_flicker|cartpole_flicker|$ROOT/data/dm_control/cartpole_250k.npz|in_dist|flicker_mask_ratio:0.25"
    "cartpole_flicker|cartpole_flicker|$ROOT/data/dm_control/cartpole_250k.npz|in_dist|flicker_mask_ratio:0.50"
    "cartpole_flicker|cartpole_flicker|$ROOT/data/dm_control/cartpole_250k.npz|in_dist|flicker_mask_ratio:0.75"
    "pusht_ood|pusht_ood|/home/lx/LeWM/data/pusht_expert_train.h5|unseen_goal|goal_offset:50"
    "pusht_ood|pusht_ood|/home/lx/LeWM/data/pusht_expert_train.h5|unseen_goal|goal_offset:100"
    "pusht_ood|pusht_ood|/home/lx/LeWM/data/pusht_expert_train.h5|unseen_goal|goal_offset:200"
    "tworoom_long|tworoom_long|/home/lx/LeWM/data/tworoom_extract/tworoom.h5|in_dist|goal_offset:50"
    "tworoom_long|tworoom_long|/home/lx/LeWM/data/tworoom_extract/tworoom.h5|in_dist|goal_offset:100"
    "tworoom_long|tworoom_long|/home/lx/LeWM/data/tworoom_extract/tworoom.h5|in_dist|goal_offset:200"
    "cheetah_velhidden|cheetah_velhidden|$ROOT/data/dm_control/3d_rollouts_250k/cheetah_250k.npz|in_dist|vel_hidden_mask_obs_ratio:0.0"
)

# Difficulty label per knob: "knob=value" -> short tag for filename
short_tag() {
    local knob_val="$1"
    local knob="${knob_val%%:*}"
    local val="${knob_val##*:}"
    case "$knob" in
        flicker_mask_ratio)            printf "f%s" "$(echo "$val" | tr -d '.')" ;;
        goal_offset)                   printf "g%s" "$val" ;;
        vel_hidden_mask_obs_ratio)     printf "vh%s" "$(echo "$val" | tr -d '.')" ;;
        *)                             printf "%s_%s" "$knob" "$val" ;;
    esac
}

total=0
ok=0
fail=0
skipped=0
failed_cells=()

run_cell() {
    local model="$1" seed="$2" env_kind="$3" out_dir="$4" data_path="$5" split="$6" knob_val="$7"

    local knob="${knob_val%%:*}"
    local val="${knob_val##*:}"
    local tag; tag="$(short_tag "$knob_val")"
    local diff_label="${env_kind}_${tag}"

    local ckpt="$RESULTS_DIR/$out_dir/${model}_seed${seed}/final.pt"
    local out="$RESULTS_DIR/$out_dir/${model}_seed${seed}/eval_${tag}.json"
    local log="$LOG_DIR/${diff_label}_${model}_seed${seed}.log"

    if [ ! -f "$ckpt" ]; then
        echo "[skip] no ckpt: $ckpt" | tee -a "$LOG_DIR/_missing_ckpts.log"
        skipped=$((skipped + 1))
        return
    fi

    local flicker_arg=""
    local vhmr_arg=""
    case "$knob" in
        flicker_mask_ratio)        flicker_arg="--flicker-mask-ratio $val" ;;
        vel_hidden_mask_obs_ratio) vhmr_arg="--vel-hidden-mask-obs-ratio $val" ;;
        goal_offset)               ;;  # passed below
    esac

    local goal_arg=""
    if [ "$knob" = "goal_offset" ]; then
        goal_arg="--goal-offset $val"
    else
        # default goal_offset for non-goal sweeps: 100 (matches the existing
        # stress eval defaults)
        goal_arg="--goal-offset 100"
    fi

    total=$((total + 1))
    echo ""
    echo "============================================="
    echo "[$total] $diff_label / $model (seed $seed)"
    echo "  ckpt=$ckpt"
    echo "  knob=$knob_val split=$split"
    echo "  out=$out"
    echo "============================================="

    # Run with `|| true` so a single failure does not abort the whole sweep.
    if "$PY" -m code.eval.closed_loop \
        --env "$env_kind" \
        --ckpt "$ckpt" \
        --data "$data_path" \
        --out "$out" \
        --n-episodes "$N_EPISODES" \
        --n-seeds "$N_SEEDS" \
        --horizon "$HORIZON" \
        --eval-budget "$EVAL_BUDGET" \
        --history-size "$HIST" \
        --split "$split" \
        $flicker_arg \
        $vhmr_arg \
        $goal_arg \
        2>&1 | tee "$log"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        failed_cells+=("$diff_label/$model/seed${seed}")
        echo "[FAIL] $diff_label/$model/seed${seed}" >> "$LOG_DIR/_failures.log"
    fi
}

# main loop
for diff_spec in "${DIFFICULTIES[@]}"; do
    IFS='|' read -r env_kind out_dir data_path split knob_val <<< "$diff_spec"
    for model in "${MODELS[@]}"; do
        for seed in $SEEDS; do
            run_cell "$model" "$seed" "$env_kind" "$out_dir" "$data_path" "$split" "$knob_val"
        done
    done
done

echo ""
echo "============================================="
echo "STRESS SWEEP COMPLETE"
echo "  total cells: $total"
echo "  ok:          $ok"
echo "  fail:        $fail"
echo "  skipped:     $skipped"
echo "============================================="
if [ "$fail" -gt 0 ]; then
    echo "Failed cells (see $LOG_DIR/_failures.log):"
    for c in "${failed_cells[@]}"; do echo "  - $c"; done
fi

# Aggregate results
echo ""
echo "[aggregate] running aggregate_stress_sweep.py"
"$PY" /home/lx/snn/code/scripts/aggregate_stress_sweep.py
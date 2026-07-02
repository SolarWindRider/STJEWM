#!/usr/bin/env bash
# Run the event-window causal ablation sweep across multiple envs and models.
#
# For each (env, model) cell, runs 5 ablation modes (baseline, event_window,
# non_event_window, random_window, ablate_all_sanity) on the closed-loop eval.
#
# Usage:
#   ./run_event_window_ablation.sh                 # default sweep
#   MODELS="stjewm_v2 stjewm_trace_only" ./run_event_window_ablation.sh
#   N_EPISODES=10 ./run_event_window_ablation.sh
set -e
cd /home/lx/snn

OUT_DIR=${OUT_DIR:-/home/lx/snn/results/aggregate/event_window_ablation}
LOG_DIR=${LOG_DIR:-/home/lx/snn/logs/event_window_ablation}
N_EPISODES=${N_EPISODES:-15}
N_SEEDS=${N_SEEDS:-2}
HORIZON=${HORIZON:-5}
EVAL_BUDGET=${EVAL_BUDGET:-50}
PROBE_STEPS=${PROBE_STEPS:-99}
MAX_WINDOWS=${MAX_WINDOWS:-12}
HALF_W=${HALF_W:-2}

# Envs to test
ENVS=${ENVS:-"ball_in_cup cartpole_2d cheetah pusht"}
# Models to test
MODELS=${MODELS:-"stjewm_v2"}

mkdir -p "$OUT_DIR" "$LOG_DIR"

for env in $ENVS; do
    for model in $MODELS; do
        ckpt="/home/lx/snn/results/${env}/${model}/final.pt"
        if [ ! -f "$ckpt" ]; then
            # try the seedX directory layout
            seed_ckpt=$(ls -1 "/home/lx/snn/results/${env}"/${model}_seed*/final.pt 2>/dev/null | head -1)
            if [ -n "$seed_ckpt" ]; then
                ckpt="$seed_ckpt"
            else
                echo "[skip] $env/$model: no ckpt"
                continue
            fi
        fi
        out="$OUT_DIR/${env}_${model}.json"
        log="$LOG_DIR/${env}_${model}.log"
        echo "============================================="
        echo "[ablation] $env / $model"
        echo "  ckpt: $ckpt"
        echo "  out:  $out"
        echo "============================================="
        /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.event_window_ablation \
            --env "$env" --model "$model" --ckpt "$ckpt" \
            --out "$out" \
            --n-episodes "$N_EPISODES" --n-seeds "$N_SEEDS" \
            --horizon "$HORIZON" --eval-budget "$EVAL_BUDGET" \
            --probe-steps "$PROBE_STEPS" \
            --max-windows "$MAX_WINDOWS" --half-w "$HALF_W" \
            2>&1 | tee "$log"
    done
done

echo ""
echo "============================================="
echo "EVENT-WINDOW ABLATION SWEEP COMPLETE"
echo "============================================="

# Aggregate results into a markdown table
/home/lx/miniconda3/envs/snn/bin/python -m code.scripts.aggregate_event_window_ablation \
    --in-dir "$OUT_DIR" \
    --out-md "$OUT_DIR/../event_window_ablation_table.md"
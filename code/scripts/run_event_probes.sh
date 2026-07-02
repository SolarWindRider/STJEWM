#!/bin/bash
# Event-type probe sweep across STJEWM modes.
#
# For each (env, model, target) cell:
#   python -m code.scripts.probe --env <env> --model <model> \\
#           --probe-target <target> --out results/aggregate/event_probes/<...>.json \\
#           --max-windows 2000 --epochs 5 --device cpu
#
# Cells: 6 envs * 5 STJEWM modes * 3 event targets = 90 cells.
# With max_windows=2000 + cpu, each cell runs in ~5-20 s; total ~20 min.
set -e
cd /home/lx/snn

PY=/home/lx/miniconda3/envs/snn/bin/python
OUT_DIR=/home/lx/snn/results/aggregate/event_probes
mkdir -p "$OUT_DIR"
MODELS=(
    stjewm_trace_only
    stjewm_hidden_leak
    stjewm_spike_only
    stjewm_no_trace
    stjewm_membrane_readout
    lewm_baseline_v2
    gru_baseline
    mlp_baseline
)

ENVS=(
    ball_in_cup
    cartpole_2d
    cheetah
    finger
    pusht
    tworoom
    delayed_t_maze
)

# Per-env event targets (matches EVENT_PROBES_PER_ENV in probe.py).
declare -A TARGETS
TARGETS[ball_in_cup]="event_contact event_high_motion event_future_k5"
TARGETS[cartpole_2d]="event_contact event_high_motion event_future_k5"
TARGETS[cheetah]="event_high_motion event_low_motion event_future_k10"
TARGETS[finger]="event_contact event_high_motion event_future_k5"
TARGETS[pusht]="event_contact event_block_near_target event_future_k10"
TARGETS[tworoom]="event_room_entered event_high_motion event_future_k5"
TARGETS[delayed_t_maze]="event_cue_state event_future_k5 event_high_motion"

count=0
for env in "${ENVS[@]}"; do
    for model in "${MODELS[@]}"; do
        for tgt in ${TARGETS[$env]}; do
            out="$OUT_DIR/${env}_${model}_${tgt}.json"
            if [ -f "$out" ]; then
                echo "[skip] $env $model $tgt (already done)"
                count=$((count+1))
                continue
            fi
            ckpt="/home/lx/snn/results/${env}/${model}/final.pt"
            if [ ! -f "$ckpt" ]; then
                echo "[skip] $env $model (no ckpt)"
                continue
            fi
            echo "=== [$count] $env / $model / $tgt ==="
            $PY -m code.scripts.probe \
                --env "$env" --model "$model" --probe-target "$tgt" \
                --out "$out" --max-windows 2000 --epochs 5 --device cpu \
                2>&1 | tail -1
            count=$((count+1))
        done
    done
done
echo "=== event-probe sweep done: $count cells ==="
#!/bin/bash
# Run event probes for cubifae_baseline, spikedreamer_baseline, slt_lif_mpc_trace, slt_lif_mpc_free.
# Same as run_event_probes.sh but limited to the 3 baseline models (and runs on CPU).
set -e
cd /home/lx/snn

PY=/home/lx/miniconda3/envs/snn/bin/python
OUT_DIR=/home/lx/snn/results/aggregate/event_probes
mkdir -p "$OUT_DIR"
LOG_DIR=/home/lx/snn/logs/event_probes_v2
mkdir -p "$LOG_DIR"

MODELS=(
    cubifae_baseline
    spikedreamer_baseline
    slt_lif_mpc_trace
    slt_lif_mpc_free
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
            log="$LOG_DIR/${env}_${model}_${tgt}.log"
            echo "=== [$count] $env / $model / $tgt ==="
            $PY -m code.scripts.probe \
                --env "$env" --model "$model" --probe-target "$tgt" \
                --out "$out" --max-windows 2000 --epochs 5 --device cpu \
                > "$log" 2>&1 && echo "ok: $env $model $tgt" || echo "FAIL: $env $model $tgt (see $log)"
            count=$((count+1))
        done
    done
done
echo "=== event-probe sweep done: $count cells ==="
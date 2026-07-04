#!/bin/bash
# Run event_align ρ for all 12 models × 6 DMC envs.
# Each cell is a 99-step random rollout. Takes ~10s on CPU per cell.
# Total = 72 cells, expected ~12 min wall clock when run in serial.

set -u
cd /home/lx/snn
PYTHON=/home/lx/miniconda3/envs/snn/bin/python
OUT=/home/lx/snn/results/aggregate/event_align_v2
LOG=/home/lx/snn/logs/event_align_v2
mkdir -p "$OUT" "$LOG"

MODELS=(
    stjewm_trace_only
    stjewm_hidden_leak
    stjewm_spike_only
    stjewm_no_trace
    stjewm_membrane_readout
    stjewm_rate_only
    cubifae_baseline
    spikedreamer_baseline
    slt_lif_mpc_trace
    slt_lif_mpc_free
    lewm_baseline_v2
    gru_baseline
    mlp_baseline
)

ENVS=(cheetah walker cartpole_2d pendulum_2d finger ball_in_cup)

n=0
for env in "${ENVS[@]}"; do
    for model in "${MODELS[@]}"; do
        out="$OUT/${env}_${model}.json"
        if [ -f "$out" ]; then
            continue
        fi
        ckpt=/home/lx/snn/results/$env/$model/final.pt
        if [ ! -f "$ckpt" ]; then
            echo "[skip] $env/$model (no ckpt)"
            continue
        fi
        log="$LOG/${env}_${model}.log"
        $PYTHON -m code.scripts.event_align \
            --env $env --model $model --out "$out" \
            > $log 2>&1 &
        n=$((n+1))
    done
done
echo "Launched $n event_align jobs"

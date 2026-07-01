#!/usr/bin/env bash
# Evaluate all 5 STJEWM readout modes on delayed_t_maze
set -e
cd /home/lx/snn
OUT=${OUT:-/home/lx/snn/results/aggregate/dt_modes}
LOG=${LOG:-/home/lx/snn/logs/dt_modes}
mkdir -p "$OUT" "$LOG"
DATA=/home/lx/snn/data/delayed_t_maze_30k.npz

for mode in trace_only hidden_leak spike_only no_trace rate_only; do
  out="$OUT/dt_stjewm_${mode}.json"
  /home/lx/miniconda3/envs/snn/bin/python -m code.eval.closed_loop --env delayed_t_maze --ckpt /home/lx/snn/results/delayed_t_maze/stjewm_${mode}/final.pt --data $DATA --out $out --n-episodes 25 --n-seeds 2 --horizon 5 --eval-budget 50 --history-size 1 --goal-offset 25 > $LOG/dt_${mode}.log 2>&1 && echo "ok: dt $mode" || echo "FAIL: dt $mode"
done

#!/bin/bash
# Eval the fair-state STJEWM ckpts (results/5m_5mpar) with the standard state
# protocol: CEM 300x30x10, H=5, budget 50, goal_offset=25, history=1, 5 eps x 1 seed.
# Uses eval_one.sh (which reads the split's eval spec) for each (split, readout).
set -e
cd /home/lx/snn

SPLITS="cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 oodc_F1 oodc_F1F2 oodc_F1F3 oodc_F2 oodc_F2F3 oodc_F3 generalist_16env"
READOUTS="trace_only spike_only rate_only no_trace hidden_leak membrane_readout"

i=0
declare -a PIDS
for split in $SPLITS; do
  for ro in $READOUTS; do
    ckpt=results/5m_5mpar/$split/stjewm_${ro}/seed_0/final.pt
    [ -f "$ckpt" ] || { echo "[skip] no ckpt $ckpt"; continue; }
    spec=configs/oodc_5m/$split.json
    GPU=$((i % 4))
    echo "[eval] $split stjewm_$ro GPU=$GPU"
    CUDA_VISIBLE_DEVICES=$GPU OUT_PARENT=results/5m_5mpar \
      bash code/scripts/generalist_v0_7_5_5m/eval_one.sh \
      stjewm_${ro} $ckpt $spec 0 \
      > results/5m_5mpar/_logs/eval_${split}_${ro}.log 2>&1 &
    PIDS+=($!)
    i=$((i + 1))
  done
done

echo "[wait] launched $i eval jobs"
fail=0
for pid in "${PIDS[@]}"; do
  wait $pid || { echo "[fail] pid $pid"; fail=$((fail+1)); }
done
echo "[done] all eval finished, failures=$fail"

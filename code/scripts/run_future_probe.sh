#!/bin/bash
# Future k-step probing: k = 1, 5, 10, 25, 50
# Run on existing trace_only + lewm_baseline ckpts for 4 envs
set -e
cd /home/lx/snn
mkdir -p results/probe
PYTHON=/home/lx/miniconda3/envs/snn/bin/python

ENVS=("cheetah" "cartpole_2d" "pusht" "tworoom")
KS=(1 5 10 25 50)

for env in "${ENVS[@]}"; do
  for model in stjewm_v2 lewm_baseline_v2; do
    ckpt="/home/lx/snn/results/$env/$model/final.pt"
    [ -f "$ckpt" ] || continue
    for k in "${KS[@]}"; do
      out="results/probe/${env}_${model}_future_k_${k}.json"
      [ -f "$out" ] && continue
      echo "[probe future_k=$k] $env/$model"
      $PYTHON -m code.scripts.probe \
        --env "$env" --model "$model" --probe-target future_k \
        --future-k "$k" --out "$out" \
        --max-windows 2000 --epochs 3 --device cpu 2>&1 | tail -1
    done
  done
done
echo "FUTURE_K_PROBE DONE"

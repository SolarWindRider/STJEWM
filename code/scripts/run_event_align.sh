#!/bin/bash
# Continue event_align for remaining envs
set -e
cd /home/lx/snn
for env in cartpole_2d pendulum_2d finger ball_in_cup; do
  for model in stjewm_v2 lewm_baseline_v2; do
    ckpt="/home/lx/snn/results/$env/$model/final.pt"
    out="/home/lx/snn/results/event_align/${env}_${model}.json"
    [ -f "$ckpt" ] || continue
    [ -f "$out" ] && continue
    echo "=== $env/$model ==="
    /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.event_align --env "$env" --model "$model" --out "$out" --n-steps 100
  done
done
echo "event_align done"

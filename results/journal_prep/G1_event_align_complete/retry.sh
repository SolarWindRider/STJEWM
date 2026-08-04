#!/usr/bin/env bash
set -u
PY=/home/lx/miniconda3/envs/snn/bin/python
export PYTHONPATH=/home/lx/snn
OUT=/home/lx/snn/results/journal_prep/G1_event_align_complete/raw
SHARD=${1:?shard}; GPU=${2:?gpu}; i=0
for split in cross_benchmark_F1 generalist_16env; do
 for env in cheetah ball_in_cup pendulum_2d finger; do
  for model in cubifae_baseline spikedreamer_baseline; do
   if (( i % 4 == SHARD )); then
    out="$OUT/$split/$env/$model.json"; mkdir -p "$(dirname "$out")"
    "$PY" -m code.scripts.event_align --env "$env" --model "$model" --ckpt "/home/lx/snn/results/5m/$split/$model/seed_0/final.pt" --pad-obs-to 128 --action-dim-eval 56 --n-steps 200 --n-resets 2 --device "cuda:$GPU" --out "$out"
   fi
   ((i+=1))
  done
 done
done

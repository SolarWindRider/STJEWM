#!/bin/bash
# Worker process: pulls from /tmp/probe5m_gpu${gpu}.tsv, runs probe for each line.
# Args: $1 = gpu id
set -u
gpu=$1
WL=/tmp/probe5m_gpu${gpu}.tsv
LOG=/tmp/probe5m_gpu${gpu}.log
: > "$LOG"
echo "[probe-gpu${gpu}] start $(date -Iseconds) | jobs=$(wc -l < $WL)" >> "$LOG"
while IFS=$'\t' read -r env model target ckpt out; do
  echo "[probe-gpu${gpu}] $env $model $target  $(date +%H:%M:%S)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$gpu timeout 200 /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.probe \
    --env "$env" --model "$model" --ckpt "$ckpt" \
    --probe-target "$target" --pad-obs-to 128 --action-dim-eval 56 \
    --max-windows 200 --out "$out" >> "$LOG" 2>&1
  rc=$?
  echo "[probe-gpu${gpu}]   -> rc=$rc ($(date +%H:%M:%S))" >> "$LOG"
done < "$WL"
echo "[probe-gpu${gpu}] done $(date -Iseconds)" >> "$LOG"

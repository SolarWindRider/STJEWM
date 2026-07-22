#!/bin/bash
# Run linear-probe event-AUROC for one 5M ckpt.
#
# Usage:
#   ./probe_one.sh <env_id> <model_name> <ckpt_path>
#
# Output goes to results/probe_5m/<env>_<model>_<target>.json (and a copy
# under results/aggregate/event_probes_5m/ so the aggregator picks them up).
set -e
cd /home/lx/snn

ENV=${1:?usage: probe_one.sh <env_id> <model_name> <ckpt_path>}
MODEL=${2:?usage: probe_one.sh <env_id> <model_name> <ckpt_path>}
CKPT=${3:?usage: probe_one.sh <env_id> <model_name> <ckpt_path>}

OUT_DIR=results/probe_5m
AGG_DIR=results/aggregate/event_probes_5m
mkdir -p "$OUT_DIR" "$AGG_DIR"

# Target list (mirrors the existing run_probes.sh)
TARGETS="event_contact event_high_motion event_low_motion event_block_near_target event_room_entered event_future_k5 event_future_k10"

for target in $TARGETS; do
  out="${OUT_DIR}/${ENV}_${MODEL}_${target}.json"
  agg_out="${AGG_DIR}/${ENV}_${MODEL}_${target}.json"
  if [[ -f "$out" ]] || [[ -f "$agg_out" ]]; then
    echo "[skip] $ENV $MODEL $target"
    continue
  fi
  echo "[probe] $ENV $MODEL $target"
  /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.probe \
    --env "$ENV" --model "$CKPT" --probe-target "$target" \
    --out "$out" 2>&1 | tail -3
  # Mirror to aggregator dir
  if [[ -f "$out" ]]; then
    cp "$out" "$agg_out"
  fi
done

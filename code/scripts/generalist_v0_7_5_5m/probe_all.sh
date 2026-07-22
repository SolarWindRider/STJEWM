#!/bin/bash
# Parallel probe runner for 5M ckpts. Uses 4 GPU workers.
#
# Usage:
#   bash code/scripts/generalist_v0_7_5_5m/probe_all.sh

set -e
cd /home/lx/snn
PROBE_BASE=results/probe_5m
AGG_BASE=results/aggregate/event_probes_5m
mkdir -p "$PROBE_BASE" "$AGG_BASE"

# Per-env target list
ENV_TARGETS=(
  "ball_in_cup:event_contact event_high_motion event_low_motion event_block_near_target event_room_entered"
  "cartpole_2d:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "cheetah:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "delayed_t_maze:event_cue_state"
  "dog:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "finger:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "fish:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "hopper:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "humanoid:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "humanoid_CMU:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "pendulum_2d:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "pusht:event_contact event_block_near_target event_future_k5"
  "quadruped:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "reacher:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "stacker:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
  "tworoom:event_room_entered event_contact event_high_motion"
  "walker:event_contact event_high_motion event_low_motion event_future_k5 event_future_k10"
)

MODELS=(
  stjewm_trace_only stjewm_spike_only stjewm_rate_only stjewm_no_trace
  stjewm_hidden_leak stjewm_membrane_readout
  mlp_baseline lewm_baseline_v2 gru_baseline cubifae_baseline
  slt_lif_mpc_trace slt_lif_mpc_free spikedreamer_baseline
)

# Build worklist: every (env, model, target) where ckpt exists and output missing
WORKLIST=/tmp/probe5m_worklist.tsv
> "$WORKLIST"
for entry in "${ENV_TARGETS[@]}"; do
  env="${entry%%:*}"
  targets="${entry#*:}"
  for model in "${MODELS[@]}"; do
    ckpt=$(find results/5m/ -name "final.pt" -path "*${model}*" 2>/dev/null | head -1)
    [[ -z "$ckpt" ]] && continue
    for target in $targets; do
      out="${PROBE_BASE}/${env}_${model}_${target}.json"
      [[ -f "$out" ]] && continue
      echo -e "${env}\t${model}\t${target}\t${ckpt}\t${out}" >> "$WORKLIST"
    done
  done
done
total=$(wc -l < "$WORKLIST")
echo "[probe_all] $total probe jobs queued"

# Split across 4 GPU workers
for gpu in 0 1 2 3; do
  awk -v gpu=$gpu 'NR % 4 == gpu' "$WORKLIST" > "/tmp/probe5m_gpu${gpu}.tsv"
  c=$(wc -l < "/tmp/probe5m_gpu${gpu}.tsv")
  echo "  GPU $gpu: $c jobs"
done

# Launch workers
for gpu in 0 1 2 3; do
  (
    while IFS=$'\t' read -r env model target ckpt out; do
      log="/tmp/probe5m_gpu${gpu}.log"
      echo "[probe-gpu${gpu}] $env $model $target" >> "$log"
      CUDA_VISIBLE_DEVICES=$gpu timeout 200 /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.probe \
        --env "$env" --model "$model" --ckpt "$ckpt" \
        --probe-target "$target" --pad-obs-to 128 --action-dim-eval 56 \
        --max-windows 200 --out "$out" >> "$log" 2>&1
      # Mirror to aggregator dir
      if [[ -f "$out" ]] && [[ ! -s "${out}.skip" ]]; then
        mkdir -p "$AGG_BASE"
        cp "$out" "$AGG_BASE/$(basename $out)"
      fi
    done < "/tmp/probe5m_gpu${gpu}.tsv"
  ) &
done

wait
echo "[probe_all] done: $(date -Iseconds)"

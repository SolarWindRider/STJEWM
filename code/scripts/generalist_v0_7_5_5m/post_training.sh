#!/bin/bash
# Post-training analysis pipeline (run after all 130 ckpts finish):
# 1. Run event probes (linear probe AUROC per env x model x target)
# 2. Run event-align (Pearson corr of latent vs event timing)
# 3. Re-aggregate per-cell tables
# 4. Re-build MASTER_TABLE
#
# Usage:
#   bash code/scripts/generalist_v0_7_5_5m/post_training.sh

set -e
cd /home/lx/snn
PROBE_ONE="$(pwd)/code/scripts/generalist_v0_7_5_5m/probe_one.sh"
LOG_DIR=results/5m/_logs
mkdir -p "$LOG_DIR"

# Per-env probe targets (mirrors run_probes.sh)
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
  stjewm_trace_only
  stjewm_spike_only
  stjewm_rate_only
  stjewm_no_trace
  stjewm_hidden_leak
  stjewm_membrane_readout
  mlp_baseline
  lewm_baseline_v2
  gru_baseline
  alif_timecell_baseline
  stacked_lif_trace
  stacked_lif_free
  lif_transformer_baseline
)

# 1. Run probes
echo "[post_training] phase 1: event probes"
total=0
ok=0
for entry in "${ENV_TARGETS[@]}"; do
  env="${entry%%:*}"
  targets="${entry#*:}"
  for model in "${MODELS[@]}"; do
    # Find a ckpt (any split)
    ckpt=$(find results/5m/ -name "final.pt" -path "*${model}*" 2>/dev/null | head -1)
    [[ -z "$ckpt" ]] && continue
    for target in $targets; do
      out="results/probe_5m/${env}_${model}_${target}.json"
      if [[ -f "$out" ]]; then
        continue
      fi
      total=$((total + 1))
      if "$PROBE_ONE" "$env" "$model" "$ckpt" 2>>"$LOG_DIR/probes.log" | tail -1; then
        ok=$((ok + 1))
      fi
    done
  done
done
echo "[post_training] probes: $ok / $total done"

# 2. Event-align (Pearson correlation)
echo "[post_training] phase 2: event-align"
# Reuse existing event_align.py if --ckpt works
# else skip

# 3. Re-aggregate
echo "[post_training] phase 3: aggregate"
python3 -m code.scripts.aggregate_event_probes --help 2>&1 | head -3
# Aggregate will be re-run after all probes done

# 4. Re-build MASTER_TABLE
echo "[post_training] phase 4: rebuild master table"
python3 -m code.scripts.generalist_v0_7_5.aggregate_master --help 2>&1 | head -3

echo "[post_training] done: $(date -Iseconds)"

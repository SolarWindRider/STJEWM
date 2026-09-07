#!/bin/bash
# run_phase3_diagnostics.sh — G1 event-alignment + div/resp latent stats on the NEW
# checkpoints (post-backward-fix). Runs after run_phase2 (pixel) finishes.
set -u
cd /home/lx/snn
export OUT_ROOT=${OUT_ROOT:-/data/lx/tmp/results}
LOG_DIR=/data/lx/tmp/logs
PY=/home/lx/miniconda3/envs/snn/bin/python
echo "[phase3] start $(date)" | tee -a "$LOG_DIR/run_all.log"

MODELS_STJ="stjewm_trace_only stjewm_spike_only stjewm_rate_only stjewm_no_trace stjewm_hidden_leak stjewm_membrane_readout"
MODELS_BASE="alif_timecell_baseline gru_baseline lewm_baseline_v2 stacked_lif_trace stacked_lif_free mlp_baseline lif_transformer_baseline"

# ---- G1: event-alignment rho (2 envs x 2 splits, 200 random-policy steps) ----
g1_job () { # model ckpt_root gpu
  local model=$1 root=$2 gpu=$3
  local out="$OUT_ROOT/g1/$root/$model"
  mkdir -p "$out"
  for env in cartpole_2d cheetah; do
    for split in cross_benchmark_F1 oodc_F2; do
      local o="$out/${env}_${split}.json"
      [ -f "$o" ] && continue
      local ck="$OUT_ROOT/$root/$split/$model/seed_0/final.pt"
      [ -f "$ck" ] || continue
      CUDA_VISIBLE_DEVICES=$gpu $PY -m code.scripts.event_align \
        --env "$env" --model "$model" --ckpt "$ck" --out "$o" \
        --n-steps 200 --pad-obs-to 128 --action-dim-eval 56 \
        > /dev/null 2>&1
    done
  done
  echo "done g1 $root $model" >> "$LOG_DIR/phase3_progress.log"
}
i=0
for m in $MODELS_STJ; do g1_job "$m" 5m_5mpar $((i % 4)) & i=$((i + 1)); done
for m in $MODELS_BASE; do g1_job "$m" 5m $((i % 4)) & i=$((i + 1)); done
wait
echo "[phase3] G1 done $(date)" | tee -a "$LOG_DIR/run_all.log"

# ---- div/resp latent stats (5m_stats style) ----
$PY code/scripts/generalist_v0_7_5_5m/measure_latent_stats_5m.py \
  --results "$OUT_ROOT/5m" --out "$OUT_ROOT/5m_stats" \
  > "$LOG_DIR/phase3_stats.log" 2>&1
echo "done 5m_stats rc=$?" >> "$LOG_DIR/phase3_progress.log"
$PY code/scripts/generalist_v0_7_5_5m/measure_latent_stats_5m.py \
  --results "$OUT_ROOT/5m_5mpar" --out "$OUT_ROOT/5m_stats_fair" \
  > "$LOG_DIR/phase3_stats_fair.log" 2>&1
echo "done 5m_stats_fair rc=$?" >> "$LOG_DIR/phase3_progress.log"
echo "[phase3] ALL DONE $(date)" | tee -a "$LOG_DIR/run_all.log"
touch "$LOG_DIR/PHASE3_DONE"

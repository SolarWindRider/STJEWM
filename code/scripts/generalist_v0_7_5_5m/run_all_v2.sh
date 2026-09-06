#!/bin/bash
# run_all_v2.sh — full retrain + eval pipeline after the 2026-09 data-generation reset.
#
# Protocol (aligned with LeWM arXiv:2603.19312 App. D/F.1):
#   goal_offset=25, eval_budget=50, horizon=25 env steps (LeWM: H5 x frame-skip 5),
#   CEM 300 samples / 30 elites, PushT 30 iters, others 10 iters,
#   env-native success (PushT 20px+pi/9, TwoRoom 16px, DMC L2/sqrt(nq)<=0.1).
#
# All artifacts go to OUT_ROOT (default /data/lx/tmp/results) — nothing in-repo.
#
# Usage: nohup bash code/scripts/generalist_v0_7_5_5m/run_all_v2.sh > /data/lx/tmp/logs/run_all.log 2>&1 &
set -u
cd /home/lx/snn

export OUT_ROOT=${OUT_ROOT:-/data/lx/tmp/results}
LOG_DIR=/data/lx/tmp/logs
mkdir -p "$LOG_DIR" "$OUT_ROOT"
DISPATCH=code/scripts/generalist_v0_7_5_5m

echo "[run_all] OUT_ROOT=$OUT_ROOT start=$(date)" | tee -a "$LOG_DIR/run_all.log"

# ---------------- Phase T: train state 5M (13 models x 10 splits + seeds 1/2) ----------------
echo "[run_all] Phase T: training" | tee -a "$LOG_DIR/run_all.log"
python3 "$DISPATCH/train_dispatcher.py" 2>&1 | tee -a "$LOG_DIR/train_dispatcher.log"

# ---------------- Phase E: closed-loop eval, 4 GPU round-robin ----------------
echo "[run_all] Phase E: evaluation" | tee -a "$LOG_DIR/run_all.log"
MODELS_STJ="stjewm_trace_only stjewm_spike_only stjewm_rate_only stjewm_no_trace stjewm_hidden_leak stjewm_membrane_readout"
MODELS_BASE="alif_timecell_baseline gru_baseline lewm_baseline_v2 stacked_lif_trace stacked_lif_free mlp_baseline lif_transformer_baseline"
SPLITS="oodc_F1 oodc_F2 oodc_F3 oodc_F1F2 oodc_F1F3 oodc_F2F3 cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 generalist_16env"

JOB_FILE="$LOG_DIR/eval_jobs.tsv"
> "$JOB_FILE"
add_jobs () { # $1=ckpt_root  $2=models...  $3=seed
  local root="$1"; shift
  local models="$1"; shift
  local seed="$1"; shift
  for sp in $SPLITS; do
    for m in $models; do
      ckpt="$OUT_ROOT/$root/$sp/$m/seed_$seed/final.pt"
      [ -f "$ckpt" ] || continue
      echo -e "$m\t$ckpt\tconfigs/oodc_5m/$sp.json\t$sp\t$seed" >> "$JOB_FILE"
    done
  done
}
add_jobs 5m_5mpar "$MODELS_STJ" 0
add_jobs 5m "$MODELS_BASE" 0
add_jobs 5m_seed1 "$MODELS_STJ $MODELS_BASE" 1
add_jobs 5m_seed2 "$MODELS_STJ $MODELS_BASE" 2

total=$(wc -l < "$JOB_FILE")
echo "[run_all] $total eval jobs" | tee -a "$LOG_DIR/run_all.log"

# 4-way GPU split
rm -f "$LOG_DIR"/gpu{0,1,2,3}_eval.tsv
i=0
while IFS=$'\t' read -r model ckpt spec sp seed; do
  echo -e "$model\t$ckpt\t$spec\t$sp\t$seed" >> "$LOG_DIR/gpu$((i % 4))_eval.tsv"
  i=$((i + 1))
done < "$JOB_FILE"

worker () {
  local gpu=$1
  local tag="gpu$gpu"
  while IFS=$'\t' read -r model ckpt spec sp seed; do
    out_parent="$OUT_ROOT/$(basename "$(dirname "$(dirname "$(dirname "$ckpt")")")")"
    CUDA_VISIBLE_DEVICES=$gpu OUT_PARENT=$out_parent N_SEEDS=1 \
      bash "$DISPATCH/eval_one.sh" "$model" "$ckpt" "$spec" "$seed" \
      >> "$LOG_DIR/eval_$tag.log" 2>&1
    echo "done $tag $sp $model rc=$?" >> "$LOG_DIR/eval_progress.log"
  done < "$LOG_DIR/${tag}_eval.tsv"
  echo "WORKER_${tag}_DONE" >> "$LOG_DIR/eval_progress.log"
}
for gpu in 0 1 2 3; do worker $gpu & done
wait
echo "[run_all] ALL DONE $(date)" | tee -a "$LOG_DIR/run_all.log"

#!/bin/bash
# run_phase2_pixel_sigreg.sh — wave 2 of the 2026-09 retrain: sigreg sweep (8 ckpts)
# + pixel 130 (13 models x 10 splits). Runs AFTER run_all_v2.sh (state wave) finishes.
# All artifacts under OUT_ROOT (/data/lx/tmp/results).
set -u
cd /home/lx/snn
export OUT_ROOT=${OUT_ROOT:-/data/lx/tmp/results}
LOG_DIR=/data/lx/tmp/logs
mkdir -p "$LOG_DIR"
PY=/home/lx/miniconda3/envs/snn/bin/python
echo "[phase2] start $(date)" | tee -a "$LOG_DIR/run_all.log"

# ---------------- sigreg sweep: STJEWM-trace x {0.09,0.01,0.001,0.0} x 2 splits ----------------
sigreg_job () {
  local sig=$1 split=$2 gpu=$3
  local out="$OUT_ROOT/5m_sigreg_sweep/$split/stjewm_trace_only_sig${sig}/seed_0"
  [ -f "$out/final.pt" ] && { echo "[skip sigreg] $split sig=$sig"; return; }
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m code.train.train \
    --model stjewm --multi-env-spec "configs/oodc_5m/$split.json" \
    --pad-obs-to 128 --action-dim 56 --embed-dim 192 --image-size 0 \
    --n-layers 4 --readout-mode trace_only \
    --lambda-sigreg "$sig" \
    --epochs 1 --batch 32 --lr 3e-4 --history-size 1 --goal-offset 25 \
    --seed 0 --no-amp --out "$out" > "$out/train.log" 2>&1
  echo "done sigreg $split sig=$sig rc=$?" >> "$LOG_DIR/phase2_progress.log"
}

SPLITS_STATE="cross_benchmark_F1 oodc_F2"
i=0
for sig in 0.09 0.01 0.001 0.0; do
  for split in $SPLITS_STATE; do
    sigreg_job "$sig" "$split" $((i % 4)) &
    i=$((i + 1))
  done
done
wait
echo "[phase2] sigreg done $(date)" | tee -a "$LOG_DIR/run_all.log"

# ---------------- pixel 130: 13 models x 10 splits, image 84, pad 21168 ----------------
PIX_SPLITS="oodc_F1 oodc_F2 oodc_F3 oodc_F1F2 oodc_F1F3 oodc_F2F3 cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 generalist_16env"
MODEL_LAYERS="stjewm:4 lewm_baseline:3 gru_baseline:2 mlp_baseline:12 stacked_lif_trace:8 stacked_lif_free:8 lif_transformer_baseline:3 alif_timecell_baseline:2"
MODEL_RO="stjewm:trace_only"   # pixel 主线的 STJEWM 用 trace_only(与旧 5m_pixel 一致)

pix_job () {
  local model=$1 split=$2 nl=$3 gpu=$4
  local out="$OUT_ROOT/5m_pixel/$split/$model/seed_0"
  [ -f "$out/final.pt" ] && return
  mkdir -p "$out"
  local ro_args=""
  if [ "$model" = "stjewm" ]; then ro_args="--readout-mode trace_only"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY -m code.train.train \
    --model "$model" --multi-env-spec "configs/oodc_5m_pixel/$split.json" \
    --pad-obs-to 21168 --action-dim 56 --embed-dim 192 --image-size 84 \
    --n-layers "$nl" $ro_args \
    --epochs 1 --batch 32 --lr 3e-4 --history-size 1 --goal-offset 25 \
    --seed 0 --no-amp --out "$out" > "$out/train.log" 2>&1
  echo "done pixel $split $model rc=$?" >> "$LOG_DIR/phase2_progress.log"
}

i=0
for split in $PIX_SPLITS; do
  for entry in $MODEL_LAYERS; do
    model=${entry%%:*}; nl=${entry##*:}
    pix_job "$model" "$split" "$nl" $((i % 4)) &
    i=$((i + 1))
    if (( i % 16 == 0 )); then wait; fi
  done
done
wait
echo "[phase2] pixel train done $(date)" | tee -a "$LOG_DIR/run_all.log"
echo "[phase2] ALL DONE $(date)" | tee -a "$LOG_DIR/run_all.log"

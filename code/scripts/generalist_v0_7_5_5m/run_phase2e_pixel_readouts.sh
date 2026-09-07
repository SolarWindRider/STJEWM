#!/bin/bash
# run_phase2e_pixel_readouts.sh — remaining 5 STJEWM readouts x 10 splits for pixel
# (spike/rate/no_trace/hidden_leak/membrane). 45GB each -> ONE per GPU.
set -u
cd /home/lx/snn
export OUT_ROOT=${OUT_ROOT:-/data/lx/tmp/results}
LOG_DIR=/data/lx/tmp/logs
PY=/home/lx/miniconda3/envs/snn/bin/python
READOUTS="spike_only rate_only no_trace hidden_leak membrane_readout"
PIX_SPLITS="oodc_F1 oodc_F2 oodc_F3 oodc_F1F2 oodc_F1F3 oodc_F2F3 cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 generalist_16env"

echo "[phase2e] start $(date)" | tee -a "$LOG_DIR/run_all.log"
job () { # ro split gpu
  local ro=$1 split=$2 gpu=$3
  local model="stjewm_${ro}"
  local out="$OUT_ROOT/5m_pixel/$split/$model/seed_0"
  [ -f "$out/final.pt" ] && return
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m code.train.train \
    --model stjewm --multi-env-spec "configs/oodc_5m_pixel/$split.json" \
    --pad-obs-to 21168 --action-dim 56 --embed-dim 192 --image-size 84 \
    --n-layers 4 --readout-mode "$ro" \
    --epochs 1 --batch 32 --lr 3e-4 --history-size 1 --goal-offset 25 \
    --seed 0 --no-amp --out "$out" > "$out/train.log" 2>&1
  echo "done pixel $split $model rc=$?" >> "$LOG_DIR/phase2e_progress.log"
}
i=0
for split in $PIX_SPLITS; do
  for ro in $READOUTS; do
    job "$ro" "$split" $((i % 4)) &
    i=$((i + 1))
    if (( i % 4 == 0 )); then wait; fi
  done
done
wait
echo "[phase2e] ALL DONE $(date)" | tee -a "$LOG_DIR/run_all.log"

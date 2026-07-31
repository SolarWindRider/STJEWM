#!/bin/bash
# Train ONE stjewm pixel ckpt with a specific readout.
# Usage: bash train_one_stjewm_pixel.sh <split> <readout> <seed> <image_size>
set -e
SPLIT=${1:-cross_benchmark_F1}
READOUT=${2:-trace_only}
SEED=${3:-0}
IMAGE_SIZE=${4:-84}
OUT_DIR=results/5m_pixel/${SPLIT}/stjewm_${READOUT}
LOG=results/_logs/5m_pixel_${SPLIT}_stjewm_${READOUT}_seed${SEED}.log
mkdir -p $OUT_DIR $(dirname $LOG)
echo "[train_one_stjewm_pixel] ${SPLIT} readout=${READOUT} seed=${SEED} image_size=${IMAGE_SIZE}"
PYTHONPATH=/home/lx/snn /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
  --model stjewm --readout-mode $READOUT \
  --multi-env-spec configs/oodc_5m_pixel/${SPLIT}.json \
  --pad-obs-to 21168 --action-dim 56 --embed-dim 192 \
  --image-size $IMAGE_SIZE --n-layers 4 \
  --epochs 1 --batch 32 --lr 3e-4 \
  --history-size 1 --goal-offset 25 \
  --seed $SEED \
  --out $OUT_DIR/seed_${SEED} \
  > $LOG 2>&1
echo "[done] $OUT_DIR/seed_${SEED}"

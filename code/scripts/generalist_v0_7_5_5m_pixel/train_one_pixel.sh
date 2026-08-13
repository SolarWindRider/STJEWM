#!/bin/bash
# Train ONE pixel ckpt (1 model, 1 split, 1 seed) using v0.7.14 5M-aligned settings.
# Usage: bash train_one_pixel.sh <model> <split> <seed> <image_size>
set -e
MODEL=${1:-stjewm}
SPLIT=${2:-cross_benchmark_F1}
SEED=${3:-0}
IMAGE_SIZE=${4:-84}
case "$MODEL" in
  stjewm) N_LAYERS=4 ;;
  lewm_baseline) N_LAYERS=3 ;;
  gru_baseline) N_LAYERS=2 ;;
  mlp_baseline) N_LAYERS=12 ;;
  stacked_lif_trace|stacked_lif_free) N_LAYERS=8 ;;
  lif_transformer_baseline) N_LAYERS=3 ;;
  alif_timecell_baseline) N_LAYERS=2 ;;
  *) echo "Unknown model: $MODEL" && exit 1 ;;
esac

OUT=results/5m_pixel/${SPLIT}/${MODEL}/seed_${SEED}
LOG=results/_logs/5m_pixel_${SPLIT}_${MODEL}_seed${SEED}.log
mkdir -p $(dirname $OUT) $(dirname $LOG)
echo "[train_one_pixel] ${MODEL} ${SPLIT} seed=${SEED} image_size=${IMAGE_SIZE}"

PYTHONPATH=/home/lx/snn /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
  --model "$MODEL" \
  --multi-env-spec configs/oodc_5m_pixel/${SPLIT}.json \
  --pad-obs-to 21168 \
  --action-dim 56 \
  --embed-dim 192 \
  --image-size $IMAGE_SIZE \
  --n-layers $N_LAYERS \
  --epochs 1 --batch 32 --lr 3e-4 \
  --history-size 1 --goal-offset 25 \
  --seed $SEED \
  --out $OUT \
  > $LOG 2>&1
echo "[done] $OUT"

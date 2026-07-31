#!/bin/bash
# Evaluate ONE pixel ckpt closed-loop.
set -e
MODEL=${1:-stjewm}
SPLIT=${2:-cross_benchmark_F1}
IMAGE_SIZE=${3:-84}
SEED=${4:-0}
CKPT=results/5m_pixel/${SPLIT}/${MODEL}/seed_${SEED}/final.pt
OUT=results/5m_pixel/${SPLIT}/${MODEL}/seed_${SEED}
mkdir -p $OUT
echo "[eval_one_pixel] ${MODEL} ${SPLIT} ckpt=${CKPT}"
PYTHONPATH=/home/lx/snn /home/lx/miniconda3/envs/snn/bin/python -m code.eval.closed_loop \
  --ckpt $CKPT \
  --env-kind dmc_pixel \
  --image-size $IMAGE_SIZE \
  --episodes 5 --horizon 5 --samples 300 --elites 30 \
  --out-dir $OUT \
  > $OUT/eval.log 2>&1
echo "[done] $OUT"

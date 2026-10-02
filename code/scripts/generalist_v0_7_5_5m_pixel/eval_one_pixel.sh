#!/bin/bash
# Evaluate ONE pixel checkpoint with the same protocol as run_pixel_eval_all.sh.
# EVAL_ROOT must be a separate output tree; existing evaluation files are kept.
set -euo pipefail
cd /home/lx/snn
MODEL=${1:-stjewm}
SPLIT=${2:-cross_benchmark_F1}
IMAGE_SIZE=${3:-84}
SEED=${4:-0}
CKPT_ROOT=${CKPT_ROOT:-${OUT_ROOT:-/data/lx/tmp/results}/5m_pixel}
: "${EVAL_ROOT:?Set EVAL_ROOT to a fresh output root before launching}"
: "${TRAINING_MANIFEST:?Set TRAINING_MANIFEST to the final repair manifest before launching}"
GPU=${GPU:-2}
CKPT="$CKPT_ROOT/$SPLIT/$MODEL/seed_$SEED/final.pt"
OUT="$EVAL_ROOT/$SPLIT/$MODEL/seed_$SEED"
test -f "$CKPT"
mkdir -p "$EVAL_ROOT/$SPLIT/$MODEL"
mkdir "$OUT"
echo "[eval_one_pixel] $MODEL $SPLIT ckpt=$CKPT"
CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU" MUJOCO_GL=egl \
  PYTHONPATH=/home/lx/snn /home/lx/miniconda3/envs/snn/bin/python -u \
  code/scripts/generalist_v0_7_5_5m_pixel/eval_pixel_ckpt.py \
  --ckpt "$CKPT" --out_dir "$OUT" --image_size "$IMAGE_SIZE" \
  --training-manifest "$TRAINING_MANIFEST" \
  --n_episodes 5 --horizon 5 --samples 300 --elites 30 --cem_iters 10 \
  --device cuda:0 > "$OUT/eval.log" 2>&1
echo "[done] $OUT"

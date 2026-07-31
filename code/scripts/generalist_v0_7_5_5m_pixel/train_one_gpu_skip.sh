#!/bin/bash
# Skip-aware wrapper around train_gpuN.sh: skips already-trained items.
# Usage: train_one_gpu_skip.sh <gpu_id> <train_gpu_script_path>
set -u
GPU=$1
TRAIN_SH=$2
SEED=0
IMAGE_SIZE=84
LOGDIR=/home/lx/snn/results/_logs
RESULTS=/home/lx/snn/results/5m_pixel

cd /home/lx/snn

# Read OUT paths from the script and run them with skip logic
N_DONE=0
N_SKIP=0
N_FAIL=0
while IFS= read -r OUT; do
  if [ -z "$OUT" ]; then continue; fi
  if [[ -f $OUT/final.pt && -f $OUT/loss_log.json ]]; then
    echo "[$(date '+%T')] [GPU $GPU] [skip] $OUT"
    N_SKIP=$((N_SKIP+1))
    continue
  fi
  # Read the rest of the command from the original script to get all flags
  # The OUT line identifies which item; we'll re-extract the full command via grep
  echo "[$(date '+%T')] [GPU $GPU] === ${OUT} ==="
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/home/lx/snn \
    /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
      $(grep -B10000 -A0 "\-\-out $OUT" $TRAIN_SH | tail -200 | grep -E "^\s*--[a-z]" | tr '\n' ' ' | sed 's/--/ --/g' | sed 's/^ //') \
      --out $OUT \
      > $LOGDIR/skip_g${GPU}_$(basename $(dirname $(dirname $OUT)))_$(basename $(dirname $OUT)).log 2>&1
  if [[ -f $OUT/final.pt && -f $OUT/loss_log.json ]]; then
    N_DONE=$((N_DONE+1))
  else
    N_FAIL=$((N_FAIL+1))
  fi
done < <(grep -E '^\s*--out' $TRAIN_SH | sed 's|.*--out\s*||' | awk '{print $1}')

echo "[$(date '+%T')] [GPU $GPU] DONE: $N_DONE new, $N_SKIP skipped, $N_FAIL failed"

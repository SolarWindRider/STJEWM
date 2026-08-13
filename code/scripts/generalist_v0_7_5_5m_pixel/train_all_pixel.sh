#!/bin/bash
# Train ALL 130 pixel ckpts sequentially.
# Estimated runtime on RTX 4090: ~24-48h. On CPU: ~1 week (NOT recommended).
set -e
SPLITS="cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 oodc_F1 oodc_F1F2 oodc_F1F3 oodc_F2 oodc_F2F3 oodc_F3 generalist_16env"
# 6 STJEWM readouts (need special handling) + 7 baselines
STJEWM_READOUTS="trace_only hidden_leak spike_only rate_only no_trace membrane_readout"
BASELINES="alif_timecell_baseline gru_baseline lewm_baseline_v2 stacked_lif_trace stacked_lif_free lif_transformer_baseline mlp_baseline"
SEED=${1:-0}
IMAGE_SIZE=${2:-84}

for SPLIT in $SPLITS; do
  for READOUT in $STJEWM_READOUTS; do
    echo "[$(date)] === stjewm_${READOUT} ${SPLIT} ==="
    OUT_DIR=results/5m_pixel/${SPLIT}/stjewm_${READOUT}
    LOG=results/_logs/5m_pixel_${SPLIT}_stjewm_${READOUT}_seed${SEED}.log
    mkdir -p $OUT_DIR $(dirname $LOG)
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
  done
  for BASELINE in $BASELINES; do
    echo "[$(date)] === ${BASELINE} ${SPLIT} ==="
    bash code/scripts/generalist_v0_7_5_5m_pixel/train_one_pixel.sh $BASELINE $SPLIT $SEED $IMAGE_SIZE
  done
done
echo "[train_all_pixel] DONE: 130 ckpts"

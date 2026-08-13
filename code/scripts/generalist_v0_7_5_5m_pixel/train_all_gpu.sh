#!/bin/bash
set -e
cd /home/lx/snn
SPLITS="cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 oodc_F1 oodc_F1F2 oodc_F1F3 oodc_F2 oodc_F2F3 oodc_F3 generalist_16env"
STJEWM_READOUTS="trace_only hidden_leak spike_only rate_only no_trace membrane_readout"
BASELINES="alif_timecell_baseline gru_baseline lewm_baseline_v2 stacked_lif_trace stacked_lif_free lif_transformer_baseline mlp_baseline"
SEED=0
IMAGE_SIZE=84
LOG_DIR=results/_logs
mkdir -p $LOG_DIR

count=0
total=130
for SPLIT in $SPLITS; do
  for READOUT in $STJEWM_READOUTS; do
    count=$((count+1))
    OUT_DIR=results/5m_pixel/${SPLIT}/stjewm_${READOUT}
    LOG=$LOG_DIR/5m_pixel_${SPLIT}_stjewm_${READOUT}_seed${SEED}.log
    if [ -f "$OUT_DIR/seed_${SEED}/final.pt" ]; then
      echo "[$(date)] [$count/$total] SKIP ${SPLIT} stjewm_${READOUT} (already done)"
      continue
    fi
    echo "[$(date)] [$count/$total] stjewm_${READOUT} ${SPLIT}"
    PYTHONPATH=/home/lx/snn /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
      --model stjewm --readout-mode $READOUT \
      --multi-env-spec configs/oodc_5m_pixel/${SPLIT}.json \
      --pad-obs-to 21168 --action-dim 56 --embed-dim 192 \
      --image-size $IMAGE_SIZE --n-layers 4 \
      --epochs 1 --batch 32 --lr 3e-4 \
      --history-size 1 --goal-offset 25 \
      --seed $SEED \
      --out $OUT_DIR/seed_${SEED} \
      > $LOG 2>&1 || echo "FAILED: $LOG"
  done
  for BASELINE in $BASELINES; do
    count=$((count+1))
    OUT_DIR=results/5m_pixel/${SPLIT}/${BASELINE}
    LOG=$LOG_DIR/5m_pixel_${SPLIT}_${BASELINE}_seed${SEED}.log
    if [ -f "$OUT_DIR/seed_${SEED}/final.pt" ]; then
      echo "[$(date)] [$count/$total] SKIP ${SPLIT} ${BASELINE} (already done)"
      continue
    fi
    echo "[$(date)] [$count/$total] ${BASELINE} ${SPLIT}"
    PYTHONPATH=/home/lx/snn /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
      --model "$BASELINE" \
      --multi-env-spec configs/oodc_5m_pixel/${SPLIT}.json \
      --pad-obs-to 21168 --action-dim 56 \
      --image-size $IMAGE_SIZE \
      --epochs 1 --batch 32 --lr 3e-4 \
      --history-size 1 --goal-offset 25 \
      --seed $SEED \
      --out $OUT_DIR/seed_${SEED} \
      > $LOG 2>&1 || echo "FAILED: $LOG"
  done
done
echo "[$(date)] DONE: 130 ckpts"

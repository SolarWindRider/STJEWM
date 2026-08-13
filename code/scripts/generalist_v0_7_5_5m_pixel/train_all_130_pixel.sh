#!/bin/bash
# Launch all 130 v0.7.15 pixel ckpts (13 models × 10 splits × 1 seed).
# Targets GPU 3 only (single-GPU scheduling).
# Skips ckpts that already have final.pt AND loss_log.json.
set -u
SEED=${1:-0}
IMAGE_SIZE=${2:-84}
GPU=3
LOGDIR=/home/lx/snn/results/_logs
RESULTS=/home/lx/snn/results/5m_pixel
FAIL_LOG=$LOGDIR/pixel_failures.log
mkdir -p $LOGDIR

cd /home/lx/snn

SPLITS="cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 oodc_F1 oodc_F1F2 oodc_F1F3 oodc_F2 oodc_F2F3 oodc_F3 generalist_16env"
STJEWM_READOUTS="trace_only hidden_leak spike_only rate_only no_trace membrane_readout"
BASELINES="alif_timecell_baseline gru_baseline lewm_baseline_v2 stacked_lif_trace stacked_lif_free lif_transformer_baseline mlp_baseline"

START_TS=$(date +%s)

run_stjewm() {
  local SPLIT=$1
  local READOUT=$2
  local OUT=$RESULTS/${SPLIT}/stjewm_${READOUT}/seed_${SEED}
  local LOG=$LOGDIR/5m_pixel_${SPLIT}_stjewm_${READOUT}_seed${SEED}.log
  if [[ -f $OUT/final.pt && -f $OUT/loss_log.json ]]; then
    echo "[skip] stjewm_${READOUT} ${SPLIT} (already done)"
    return 0
  fi
  echo "[$(date '+%T')] === stjewm_${READOUT} ${SPLIT} ==="
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/home/lx/snn \
    /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
      --model stjewm --readout-mode $READOUT \
      --multi-env-spec configs/oodc_5m_pixel/${SPLIT}.json \
      --pad-obs-to 21168 --action-dim 56 --embed-dim 192 \
      --image-size $IMAGE_SIZE --n-layers 4 \
      --epochs 1 --batch 32 --lr 3e-4 \
      --history-size 1 --goal-offset 25 \
      --seed $SEED \
      --out $OUT \
      > $LOG 2>&1
  if [[ ! -f $OUT/final.pt || ! -f $OUT/loss_log.json ]]; then
    echo "[fail] stjewm_${READOUT} ${SPLIT}" >> $FAIL_LOG
    return 1
  fi
  echo "[$(date '+%T')] OK stjewm_${READOUT} ${SPLIT}"
}

run_baseline() {
  local MODEL=$1
  local SPLIT=$2
  case "$MODEL" in
    lewm_baseline_v2) N_LAYERS=3 ;;
    mlp_baseline) N_LAYERS=12 ;;
    stacked_lif_trace|stacked_lif_free) N_LAYERS=8 ;;
    lif_transformer_baseline) N_LAYERS=3 ;;
    alif_timecell_baseline) N_LAYERS=2 ;;
    gru_baseline) N_LAYERS=2 ;;
    stjewm) N_LAYERS=4 ;;
    *) echo "Unknown MODEL: $MODEL"; return 1 ;;
  esac
  local OUT=$RESULTS/${SPLIT}/${MODEL}/seed_${SEED}
  local LOG=$LOGDIR/5m_pixel_${SPLIT}_${MODEL}_seed${SEED}.log
  if [[ -f $OUT/final.pt && -f $OUT/loss_log.json ]]; then
    echo "[skip] ${MODEL} ${SPLIT} (already done)"
    return 0
  fi
  echo "[$(date '+%T')] === ${MODEL} ${SPLIT} ==="
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/home/lx/snn \
    /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
      --model $MODEL \
      --multi-env-spec configs/oodc_5m_pixel/${SPLIT}.json \
      --pad-obs-to 21168 --action-dim 56 --embed-dim 192 \
      --image-size $IMAGE_SIZE --n-layers $N_LAYERS \
      --epochs 1 --batch 32 --lr 3e-4 \
      --history-size 1 --goal-offset 25 \
      --seed $SEED \
      --out $OUT \
      > $LOG 2>&1
  if [[ ! -f $OUT/final.pt || ! -f $OUT/loss_log.json ]]; then
    echo "[fail] ${MODEL} ${SPLIT}" >> $FAIL_LOG
    return 1
  fi
  echo "[$(date '+%T')] OK ${MODEL} ${SPLIT}"
}

# Process in priority order: full grid for each model, then retry once on failures
declare -a JOBS
for SPLIT in $SPLITS; do
  for READOUT in $STJEWM_READOUTS; do
    JOBS+=("stjewm $READOUT $SPLIT")
  done
done
for SPLIT in $SPLITS; do
  for BASELINE in $BASELINES; do
    JOBS+=("base $BASELINE $SPLIT")
  done
done

for J in "${JOBS[@]}"; do
  read -r KIND A B <<< "$J"
  if [[ $KIND == "stjewm" ]]; then
    if ! run_stjewm "$B" "$A"; then
      echo "[retry] stjewm_${A} ${B}"
      run_stjewm "$B" "$A"
    fi
  else
    if ! run_baseline "$A" "$B"; then
      echo "[retry] ${A} ${B}"
      run_baseline "$A" "$B"
    fi
  fi
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
H=$((ELAPSED/3600)); M=$(((ELAPSED%3600)/60))
echo "[$(date '+%T')] DONE all 130 pixel ckpts (wall=${H}h${M}m)"
echo "[$(date '+%T')] summary: $(find $RESULTS -name final.pt | wc -l) ckpts, $(find $RESULTS -name loss_log.json | wc -l) loss logs"

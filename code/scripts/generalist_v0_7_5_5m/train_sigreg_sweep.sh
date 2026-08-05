#!/bin/bash
# sigreg sweep: does lowering lambda_sigreg improve STJEWM calibration/AUROC vs SLT?
# STJEWM-trace x {0.09 (orig), 0.01, 0.001, 0.0} x {cross_benchmark_F1, oodc_F2}
set -e
cd /home/lx/snn

OUT_BASE=results/5m_sigreg_sweep
mkdir -p $OUT_BASE/_logs

SIGREGS="0.09 0.01 0.001 0.0"
SPLITS="cross_benchmark_F1 oodc_F2"

i=0
declare -a PIDS
for sig in $SIGREGS; do
  for split in $SPLITS; do
    tag="sig${sig}"
    out=$OUT_BASE/$split/stjewm_trace_only_${tag}/seed_0
    log=$OUT_BASE/_logs/${split}_sig${sig}.log
    if [ -f "$out/final.pt" ] && [ -f "$out/loss_log.json" ]; then
      echo "[skip] $split sig=$sig"
      continue
    fi
    GPU=3
    echo "[train] $split sigreg=$sig GPU=$GPU"
    CUDA_VISIBLE_DEVICES=$GPU /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
      --model stjewm --multi-env-spec configs/oodc_5m/${split}.json \
      --pad-obs-to 128 --action-dim 56 --embed-dim 192 --image-size 0 \
      --n-layers 4 --epochs 1 --batch 32 --lr 3e-4 \
      --history-size 1 --goal-offset 25 --seed 0 \
      --readout-mode trace_only \
      --lambda-sigreg $sig \
      --out $out > $log 2>&1 &
    PIDS+=($!)
    i=$((i + 1))
  done
done

echo "[wait] launched $i jobs"
fail=0
for pid in "${PIDS[@]}"; do
  wait $pid || { echo "[fail] pid $pid"; fail=$((fail+1)); }
done
echo "[done] training finished, failures=$fail"

#!/bin/bash
# Fair-state rerun: STJEWM 6 readouts at n_layers=4 (trainable 5.06M, matching
# the ~5M baselines) on all 10 state splits. Output to results/5m_5mpar/.
# Protocol identical to the original 5m run except n_layers 2 -> 4:
#   --pad-obs-to 128 --action-dim 56 --embed-dim 192 --image-size 0
#   --epochs 1 --batch 32 --lr 3e-4 --history-size 1 --goal-offset 25 --seed 0
set -e
cd /home/lx/snn

SPLITS="cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 oodc_F1 oodc_F1F2 oodc_F1F3 oodc_F2 oodc_F2F3 oodc_F3 generalist_16env"
READOUTS="trace_only spike_only rate_only no_trace hidden_leak membrane_readout"
OUT_BASE=results/5m_5mpar
mkdir -p $OUT_BASE/_logs

i=0
declare -a PIDS
for split in $SPLITS; do
  for ro in $READOUTS; do
    out=$OUT_BASE/$split/stjewm_${ro}/seed_0
    log=$OUT_BASE/_logs/${split}_stjewm_${ro}.log
    if [ -f "$out/final.pt" ] && [ -f "$out/loss_log.json" ]; then
      echo "[skip] $split $ro"
      continue
    fi
    GPU=$((i % 4))
    echo "[train] $split stjewm_$ro GPU=$GPU"
    CUDA_VISIBLE_DEVICES=$GPU /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
      --model stjewm --multi-env-spec configs/oodc_5m/${split}.json \
      --pad-obs-to 128 --action-dim 56 --embed-dim 192 --image-size 0 \
      --n-layers 4 --epochs 1 --batch 32 --lr 3e-4 \
      --history-size 1 --goal-offset 25 --seed 0 \
      --readout-mode $ro \
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
echo "[done] all jobs finished, failures=$fail"

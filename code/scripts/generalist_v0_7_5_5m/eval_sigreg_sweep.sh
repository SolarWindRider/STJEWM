#!/bin/bash
# Eval sigreg sweep ckpts: closed_loop CEM (same protocol), all envs of each split
set -e
cd /home/lx/snn

OUT_BASE=results/5m_sigreg_sweep
SIGS="0.09 0.01 0.001 0.0"
SPLITS="cross_benchmark_F1 oodc_F2"

for sig in $SIGS; do
  for split in $SPLITS; do
    ckpt=$OUT_BASE/$split/stjewm_trace_only_sig${sig}/seed_0/final.pt
    [ -f "$ckpt" ] || { echo "[skip] no ckpt $split sig=$sig"; continue; }
    # count existing evals
    n=$(ls $OUT_BASE/$split/stjewm_trace_only_sig${sig}/seed_0/eval_*.json 2>/dev/null | wc -l)
    if [ "$n" -ge 14 ] && [ "$split" = "cross_benchmark_F1" ]; then echo "[skip-eval] $split sig=$sig ($n evals)"; continue; fi
    if [ "$n" -ge 5 ] && [ "$split" = "oodc_F2" ]; then echo "[skip-eval] $split sig=$sig ($n evals)"; continue; fi
    echo "[eval] $split sig=$sig on GPU 3"
    CUDA_VISIBLE_DEVICES=3 OUT_PARENT=$OUT_BASE \
      bash code/scripts/generalist_v0_7_5_5m/eval_one.sh \
      stjewm_trace_only_sig${sig} $ckpt configs/oodc_5m/$split.json 0 \
      > $OUT_BASE/_logs/eval_${split}_sig${sig}.log 2>&1 || echo "[eval-FAIL] $split sig=$sig"
  done
done
echo "[done] sigreg eval"

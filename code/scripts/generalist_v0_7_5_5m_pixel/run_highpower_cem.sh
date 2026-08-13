#!/bin/bash
# High-power rerun: Stacked-LIF-trace vs STJEWM-trace vs STJEWM-spike
# 10 splits x 3 models, 4 competitive envs (cartpole,p pendulum,finger,cheetah),
# n_episodes=30, FULL CEM (300x10, budget 50, horizon 5).
# 4 GPUs round-robin. Each job ~4-8 min (30 eps x 4 envs).
set -e
cd /home/lx/snn

SPLITS="cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 oodc_F1 oodc_F1F2 oodc_F1F3 oodc_F2 oodc_F2F3 oodc_F3 generalist_16env"
MODELS="stacked_lif_trace stjewm_trace_only stjewm_spike_only"
ENVS="cartpole,pendulum,finger,cheetah"
OUTDIR=results/5m_pixel_highpower

mkdir -p $OUTDIR/logs

i=0
declare -a PIDS
for split in $SPLITS; do
  for model in $MODELS; do
    ckpt=results/5m_pixel/${split}/${model}/seed_0/final.pt
    out_dir=${OUTDIR}/${split}/${model}/seed_0
    log=${OUTDIR}/logs/${split}_${model}.log
    GPU=$((i % 4))
    echo "[launch] ${split}/${model} GPU=$GPU -> $log"
    CUDA_VISIBLE_DEVICES=$GPU /home/lx/miniconda3/envs/snn/bin/python \
      code/scripts/generalist_v0_7_5_5m_pixel/eval_pixel_ckpt_cem.py \
      --ckpt $ckpt \
      --out_dir $out_dir \
      --image_size 84 \
      --n_episodes 30 \
      --eval_budget 50 \
      --horizon 5 \
      --cem_samples 300 \
      --cem_elites 30 \
      --cem_iters 10 \
      --device cuda:0 \
      --envs $ENVS \
      > $log 2>&1 &
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

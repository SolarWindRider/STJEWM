#!/bin/bash
# High-concurrency retrain v3: 4 GPUs, 5-6 parallel jobs per GPU (state
# training uses ~3GB each; 24GB cards fit 5-6). Tasks: seed1 stjewm (14
# remaining) + sigreg (8). All --image-size 224.
set -u
cd /home/lx/snn
export PYTHONPATH=/home/lx/snn
PY=/home/lx/miniconda3/envs/snn/bin/python
readouts="trace_only hidden_leak spike_only rate_only no_trace membrane_readout"
splits_state="cross_benchmark_F1 generalist_16env oodc_F2"

# Build task list (unfinished only)
declare -a TASKS=()
for s in $splits_state; do
  for m in $readouts; do
    out="results/5m_seed1/${s}/stjewm_${m}/seed_0"
    [ -f "$out/final.pt" ] && continue
    TASKS+=("seed1|$m|$s|$out")
  done
done
for sig in 0.09 0.01 0.001 0.0; do
  for sp in cross_benchmark_F1 oodc_F2; do
    out="results/5m_sigreg_sweep/${sp}/stjewm_trace_only_sig${sig}/seed_0"
    [ -f "$out/final.pt" ] && continue
    TASKS+=("sigreg|$sig|$sp|$out")
  done
done
echo "total tasks: ${#TASKS[@]}"

run_one() {  # gpu kind a b out
  local gpu=$1 kind=$2 a=$3 b=$4 out=$5
  export CUDA_VISIBLE_DEVICES=$gpu
  rm -f "$out/final.pt" "$out/step2000.pt" "$out/step4000.pt"
  mkdir -p "$out"
  if [ "$kind" = seed1 ]; then
    $PY -m code.train.train \
      --model stjewm --readout-mode "$a" \
      --multi-env-spec "configs/oodc_5m/${b}.json" \
      --pad-obs-to 128 --action-dim 56 --embed-dim 192 --image-size 224 \
      --n-layers 4 --epochs 1 --batch 32 --lr 3e-4 \
      --history-size 1 --goal-offset 25 --seed 1 \
      --out "$out" > "$out/train.log" 2>&1
  else
    $PY -m code.train.train \
      --model stjewm --multi-env-spec "configs/oodc_5m/${b}.json" \
      --pad-obs-to 128 --action-dim 56 --embed-dim 192 --image-size 224 \
      --n-layers 4 --epochs 1 --batch 32 --lr 3e-4 \
      --history-size 1 --goal-offset 25 --seed 0 \
      --readout-mode trace_only --lambda-sigreg "$a" \
      --out "$out" > "$out/train.log" 2>&1
  fi
  echo "[DONE gpu$gpu] $kind $a $b rc=$? $(date +%T)" >> /tmp/retrain_v3_progress.log
}

run_gpu_group() {  # gpu then task strings
  local gpu=$1; shift
  export CUDA_VISIBLE_DEVICES=$gpu
  local pids=()
  for spec in "$@"; do
    IFS="|" read -r kind a b out <<< "$spec"
    run_one "$gpu" "$kind" "$a" "$b" "$out" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
}

# Distribute round-robin to 4 groups
g0=() g1=() g2=() g3=()
i=0
for t in "${TASKS[@]}"; do
  case $((i % 4)) in 0) g0+=("$t");; 1) g1+=("$t");; 2) g2+=("$t");; 3) g3+=("$t");; esac
  i=$((i+1))
done
run_gpu_group 0 "${g0[@]}" > /tmp/retrain_v3_g0.log 2>&1 &
run_gpu_group 1 "${g1[@]}" > /tmp/retrain_v3_g1.log 2>&1 &
run_gpu_group 2 "${g2[@]}" > /tmp/retrain_v3_g2.log 2>&1 &
run_gpu_group 3 "${g3[@]}" > /tmp/retrain_v3_g3.log 2>&1 &
wait
echo "ALL V3 DONE $(date +%T)"

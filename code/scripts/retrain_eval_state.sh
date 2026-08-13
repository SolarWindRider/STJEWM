#!/bin/bash
# Phase 1: re-evaluate retrained state checkpoints (5m_seed1 x39, sigreg x8).
# 4-GPU round-robin queues.
set -u
cd /home/lx/snn
eval_one() {  # gpu model ckpt spec out_parent
  local gpu=$1 model=$2 ckpt=$3 spec=$4 outpar=$5
  local envs
  CUDA_VISIBLE_DEVICES=$gpu OUT_PARENT=$outpar bash code/scripts/generalist_v0_7_5_5m/eval_one.sh \
    "$model" "$ckpt" "$spec" 0 > /tmp/state_eval_g${gpu}.log 2>&1
  echo "[DONE gpu$gpu] $model $ckpt rc=$?" >> /tmp/state_eval_progress.log
}

# build task list
declare -a TASKS=()
for split in cross_benchmark_F1 generalist_16env oodc_F2; do
  for m in $(ls results/5m_seed1/$split/); do
    ckpt="results/5m_seed1/$split/$m/seed_0/final.pt"
    [ -f "$ckpt" ] || continue
    TASKS+=("1|$m|$ckpt|configs/oodc_5m/$split.json|results/5m_seed1")
  done
done
for split in cross_benchmark_F1 oodc_F2; do
  for m in $(ls results/5m_sigreg_sweep/$split/ 2>/dev/null); do
    ckpt="results/5m_sigreg_sweep/$split/$m/seed_0/final.pt"
    [ -f "$ckpt" ] || continue
    TASKS+=("1|$m|$ckpt|configs/oodc_5m/$split.json|results/5m_sigreg_sweep")
  done
done
echo "total state eval tasks: ${#TASKS[@]}" >> /tmp/state_eval_progress.log

q0=() q1=() q2=() q3=()
i=0
for t in "${TASKS[@]}"; do
  case $((i % 4)) in 0) q0+=("$t");; 1) q1+=("$t");; 2) q2+=("$t");; 3) q3+=("$t");; esac
  i=$((i+1))
done
run_q() {  # gpu tasks...
  local gpu=$1; shift
  local pids=()
  for spec in "$@"; do
    IFS="|" read -r s model ckpt specp outpar <<< "$spec"
    eval_one "$gpu" "$model" "$ckpt" "$specp" "$outpar" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
}
run_q 0 "${q0[@]}" > /dev/null 2>&1 &
run_q 1 "${q1[@]}" > /dev/null 2>&1 &
run_q 2 "${q2[@]}" > /dev/null 2>&1 &
run_q 3 "${q3[@]}" > /dev/null 2>&1 &
wait
echo "ALL STATE EVAL DONE $(date +%T)" >> /tmp/state_eval_progress.log

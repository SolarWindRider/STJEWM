#!/bin/bash
# 5M-aligned re-training orchestrator (v0.7.14).
# 12 model kinds × 10 splits = 120 ckpts.
# Schedules: OODC (5 splits, 2-11 envs each, small) first → cross-bench (3 splits, 15 envs)
# → generalist G16 (1 split, 16 envs, biggest) last. Within each tier, jobs
# run sequentially on the chosen GPU (default GPU 0). Multiple scripts can run
# in parallel on different GPUs by setting CUDA_VISIBLE_DEVICES before invoking.
#
# Usage:
#   cd /home/lx/snn
#   CUDA_VISIBLE_DEVICES=0 nohup bash code/scripts/generalist_v0_7_5_5m/launch_120.sh > /tmp/launch_120.log 2>&1 &
#   tail -f /tmp/launch_120.log

set -e
cd /home/lx/snn
TRAIN_ONE="$(pwd)/code/scripts/generalist_v0_7_5_5m/train_one_5m.sh"
LOG_DIR=results/5m/_logs
mkdir -p "$LOG_DIR"

# 12 model kinds
MODELS=(
  stjewm_trace_only
  stjewm_spike_only
  stjewm_rate_only
  stjewm_no_trace
  stjewm_hidden_leak
  stjewm_membrane_readout
  mlp_baseline
  lewm_baseline_v2
  gru_baseline
  alif_timecell_baseline
  stacked_lif_trace
  stacked_lif_free
  lif_transformer_baseline
)

# Tier A: OODC (2-11 envs, fast)
# Tier B: cross-bench (15 envs, medium)
# Tier C: G16 generalist (16 envs, slowest)
TIER_A=(
  "configs/oodc/oodc_F1.json"
  "configs/oodc/oodc_F2.json"
  "configs/oodc/oodc_F3.json"
  "configs/oodc/oodc_F1F2.json"
  "configs/oodc/oodc_F1F3.json"
  "configs/oodc/oodc_F2F3.json"
)
TIER_B=(
  "configs/oodc/cross_benchmark_F1.json"
  "configs/oodc/cross_benchmark_F2.json"
  "configs/oodc/cross_benchmark_F3.json"
)
TIER_C=(
  "configs/generalist_16env.json"
)

echo "[launch_120] start: $(date -Iseconds)"
echo "[launch_120] tiers: A=${#TIER_A[@]} B=${#TIER_B[@]} C=${#TIER_C[@]} models=${#MODELS[@]}"
total_jobs=0
ok_jobs=0
fail_jobs=0
tier=A
for spec in "${TIER_A[@]}" "${TIER_B[@]}" "${TIER_C[@]}"; do
  split_name=$(python3 -c "import json; d=json.load(open('${spec}')); print(d.get('split_name', '${spec##*/}'))")
  for model in "${MODELS[@]}"; do
    out_dir="results/5m/${split_name}/${model}/seed_0"
    log_path="${LOG_DIR}/${split_name}_${model}.log"
    if [[ -f "${out_dir}/final.pt" ]]; then
      echo "[launch_120] skip: ${split_name}/${model} (already trained)"
      continue
    fi
    total_jobs=$((total_jobs + 1))
    echo "[launch_120] [$total_jobs] tier=$tier ${split_name}/${model} @ $(date +%H:%M:%S)"
    if "$TRAIN_ONE" "$model" "$spec" "$out_dir" 0 > "$log_path" 2>&1; then
      ok_jobs=$((ok_jobs + 1))
      echo "[launch_120]   OK ${split_name}/${model}"
    else
      fail_jobs=$((fail_jobs + 1))
      echo "[launch_120]   FAIL ${split_name}/${model} (see $log_path)"
    fi
  done
  # advance tier label
  if [[ "$spec" == "${TIER_A[$((${#TIER_A[@]}-1))]}" ]]; then tier=B; fi
  if [[ "$spec" == "${TIER_B[$((${#TIER_B[@]}-1))]}" ]]; then tier=C; fi
done
echo "[launch_120] done: $(date -Iseconds) | total=$total_jobs ok=$ok_jobs fail=$fail_jobs"

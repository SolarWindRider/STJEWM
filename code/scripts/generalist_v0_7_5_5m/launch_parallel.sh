#!/bin/bash
# Parallel launcher: 120 ckpts across 4 GPUs.
# Strategy:
#   - GPU 0: all 6 STJEWM variants (slowest, ~10 min each, no contention)
#   - GPU 1-3: 7 baseline models, round-robin
# Wall time: ~2-3 hours.

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

SPLITS=(
  "configs/oodc_5m/oodc_F1.json"
  "configs/oodc_5m/oodc_F2.json"
  "configs/oodc_5m/oodc_F3.json"
  "configs/oodc_5m/oodc_F1F2.json"
  "configs/oodc_5m/oodc_F1F3.json"
  "configs/oodc_5m/oodc_F2F3.json"
  "configs/oodc_5m/cross_benchmark_F1.json"
  "configs/oodc_5m/cross_benchmark_F2.json"
  "configs/oodc_5m/cross_benchmark_F3.json"
  "configs/oodc_5m/generalist_16env.json"
)

# Build flat job list with split_name for output dir
WORK_FILE="$LOG_DIR/jobs.tsv"
> "$WORK_FILE"
for spec in "${SPLITS[@]}"; do
  # Extract split_name: try _split_name key, then split_name key, then spec filename
  split_name=$(python3 -c "
import json
d = json.load(open('${spec}'))
if isinstance(d, list):
    print('${spec##*/}'.replace('.json',''))
else:
    print(d.get('_split_name') or d.get('split_name') or '${spec##*/}'.replace('.json',''))
" 2>/dev/null)
  for model in "${MODELS[@]}"; do
    out_dir="results/5m/${split_name}/${model}/seed_0"
    if [[ -f "${out_dir}/final.pt" ]]; then
      continue
    fi
    echo -e "${model}\t${spec}\t${split_name}\t${out_dir}" >> "$WORK_FILE"
  done
done
total=$(wc -l < "$WORK_FILE")
echo "[parallel] total jobs to run: $total"

# Split jobs by GPU
GPU0_JOBS="$LOG_DIR/gpu0_jobs.tsv"
GPU123_JOBS="$LOG_DIR/gpu123_jobs.tsv"
grep -E "^(stjewm_)" "$WORK_FILE" > "$GPU0_JOBS" || true
grep -vE "^(stjewm_)" "$WORK_FILE" > "$GPU123_JOBS" || true
gpu0_count=$(wc -l < "$GPU0_JOBS")
gpu123_count=$(wc -l < "$GPU123_JOBS")
echo "[parallel] GPU 0 (STJEWM only): $gpu0_count jobs"
echo "[parallel] GPUs 1-3 (baselines): $gpu123_count jobs, round-robin"

# Split GPU123 jobs across 3 GPUs
> "$LOG_DIR/gpu1_jobs.tsv"
> "$LOG_DIR/gpu2_jobs.tsv"
> "$LOG_DIR/gpu3_jobs.tsv"
i=0
while IFS=$'\t' read -r model spec split_name out_dir; do
  gpu=$((1 + (i % 3)))
  echo -e "${model}\t${spec}\t${split_name}\t${out_dir}" >> "$LOG_DIR/gpu${gpu}_jobs.tsv"
  i=$((i+1))
done < "$GPU123_JOBS"
for gpu in 1 2 3; do
  c=$(wc -l < "$LOG_DIR/gpu${gpu}_jobs.tsv")
  echo "[parallel] GPU $gpu: $c jobs"
done

# Launch 4 workers
for gpu in 0 1 2 3; do
  job_file="$LOG_DIR/gpu${gpu}_jobs.tsv"
  if [[ ! -s "$job_file" ]]; then
    echo "[parallel] GPU $gpu: 0 jobs, skipping"
    continue
  fi
  worker_log="$LOG_DIR/gpu${gpu}.worker.log"
  (
    while IFS=$'\t' read -r model spec split_name out_dir; do
      log_path="${LOG_DIR}/${split_name}_${model}.log"
      echo "[gpu${gpu}@$(date +%H:%M:%S)] $model $split_name"
      if CUDA_VISIBLE_DEVICES=$gpu "$TRAIN_ONE" "$model" "$spec" "$out_dir" 0 > "$log_path" 2>&1; then
        echo "[gpu${gpu}@$(date +%H:%M:%S)]   OK $model $split_name" >> "$worker_log"
      else
        echo "[gpu${gpu}@$(date +%H:%M:%S)]   FAIL $model $split_name" >> "$worker_log"
      fi
    done < "$job_file"
  ) &
done

wait
echo "[parallel] all GPUs done: $(date -Iseconds)"

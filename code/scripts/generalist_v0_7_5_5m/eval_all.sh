#!/bin/bash
# Eval all 5M ckpts (one ev per (split, model)). Uses 4 GPUs in parallel.
# Each GPU processes a queue of eval jobs in priority order.
#
# Usage:
#   nohup bash code/scripts/generalist_v0_7_5_5m/eval_all.sh > /tmp/launch/eval_all.log 2>&1 &

set -e
cd /home/lx/snn
EVAL_ONE="$(pwd)/code/scripts/generalist_v0_7_5_5m/eval_one.sh"
LOG_DIR=results/5m/_logs
mkdir -p "$LOG_DIR"

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

# Build flat eval-job list (only if ckpt exists and at least one env needs eval)
JOB_FILE="$LOG_DIR/eval_jobs.tsv"
> "$JOB_FILE"
for spec in "${SPLITS[@]}"; do
  split_name=$(python3 -c "
import json
d = json.load(open('${spec}'))
if isinstance(d, list):
    print('${spec##*/}'.replace('.json',''))
else:
    print(d.get('_split_name') or d.get('split_name') or '${spec##*/}'.replace('.json',''))
" 2>/dev/null)
  for model in "${MODELS[@]}"; do
    ckpt="results/5m/${split_name}/${model}/seed_0/final.pt"
    if [[ ! -f "$ckpt" ]]; then
      continue
    fi
    # Quick check: does any env need eval?
    eval_spec="$spec"
    echo -e "${model}\t${ckpt}\t${eval_spec}\t${split_name}" >> "$JOB_FILE"
  done
done
total=$(wc -l < "$JOB_FILE")
echo "[eval_all] $total eval jobs queued"

# Split across 4 GPUs round-robin
> "$LOG_DIR/gpu0_eval.tsv"
> "$LOG_DIR/gpu1_eval.tsv"
> "$LOG_DIR/gpu2_eval.tsv"
> "$LOG_DIR/gpu3_eval.tsv"
i=0
while IFS=$'\t' read -r model ckpt eval_spec split_name; do
  gpu=$((i % 4))
  echo -e "${model}\t${ckpt}\t${eval_spec}\t${split_name}" >> "$LOG_DIR/gpu${gpu}_eval.tsv"
  i=$((i + 1))
done < "$JOB_FILE"
for gpu in 0 1 2 3; do
  c=$(wc -l < "$LOG_DIR/gpu${gpu}_eval.tsv" 2>/dev/null || echo 0)
  echo "[eval_all] GPU $gpu: $c eval jobs"
done

# Launch 4 workers
for gpu in 0 1 2 3; do
  job_file="$LOG_DIR/gpu${gpu}_eval.tsv"
  if [[ ! -s "$job_file" ]]; then
    continue
  fi
  worker_log="$LOG_DIR/gpu${gpu}_eval.worker.log"
  (
    while IFS=$'\t' read -r model ckpt eval_spec split_name; do
      log_path="${LOG_DIR}/eval_${split_name}_${model}.log"
      echo "[eval-gpu${gpu}@$(date +%H:%M:%S)] $model $split_name"
      if CUDA_VISIBLE_DEVICES=$gpu OUT_PARENT=/home/lx/snn/results/5m STRESS_OUT_PARENT=/home/lx/snn/results/5m_stress \
           "$EVAL_ONE" "$model" "$ckpt" "$eval_spec" 0 > "$log_path" 2>&1; then
        echo "[eval-gpu${gpu}@$(date +%H:%M:%S)]   OK $model $split_name" >> "$worker_log"
      else
        echo "[eval-gpu${gpu}@$(date +%H:%M:%S)]   FAIL $model $split_name" >> "$worker_log"
      fi
    done < "$job_file"
  ) &
done

wait
echo "[eval_all] all done: $(date -Iseconds)"

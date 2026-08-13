#!/bin/bash
# B2 recovery launcher: train the remaining 2 missing ckpts.
#   seed=2 generalist_16env stjewm_spike_only
#   seed=2 generalist_16env stacked_lif_trace
# Uses single GPU (0 or 1) to avoid contention; non-screen-detached via nohup + setsid.

set -e
cd /home/lx/snn

LOG_DIR=/home/lx/snn/results/journal_prep/B2_multiseed/_recover_logs
mkdir -p "$LOG_DIR"

MODELS=(
  "generalist_16env stjewm_spike_only   stjewm  spike_only   /home/lx/snn/results/5m_seed2/generalist_16env/stjewm_spike_only/seed_0   2"
  "generalist_16env stacked_lif_trace   stacked_lif_trace  ''   /home/lx/snn/results/5m_seed2/generalist_16env/stacked_lif_trace/seed_0   2"
)

# Run sequentially to avoid GPU contention
for m in "${MODELS[@]}"; do
  set -- $m
  split=$1; model_kind=$2; model=$3; readout=$4; out_dir=$5; seed=$6
  log="${LOG_DIR}/${split}_${model_kind}_seed${seed}.train.log"
  echo "[$(date +%H:%M:%S)] training $model_kind on $split seed=$seed"
  read_flag=()
  if [[ -n "$readout" ]]; then
    read_flag=(--readout-mode "$readout")
  fi
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/lx/snn \
    /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
      --model "$model" \
      --multi-env-spec "configs/oodc_5m/${split}.json" \
      --pad-obs-to 128 --action-dim 56 --embed-dim 192 --image-size 0 --n-layers 2 \
      --epochs 1 --batch 32 --lr 3e-4 --history-size 1 --goal-offset 25 \
      --seed "$seed" \
      --out "$out_dir" \
      "${read_flag[@]}" \
    > "$log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] finished rc=$rc"
done
echo "all done $(date -Iseconds)"

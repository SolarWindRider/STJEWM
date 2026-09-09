#!/bin/bash
# run_phase4_generalist.sh — G4/G8/G16 scale-axis retrain (12 models x 3 scales = 36 ckpts)
# on the fixed pipeline. Outputs under OUT_ROOT/generalist_G{4,8,16}.
set -u
cd /home/lx/snn
export OUT_ROOT=${OUT_ROOT:-/data/lx/tmp/results}
LOG_DIR=/data/lx/tmp/logs
PY=/home/lx/miniconda3/envs/snn/bin/python
echo "[phase4] start $(date)" | tee -a "$LOG_DIR/run_all.log"

SCALES="4 8 16"
declare -A LAYERS=( [stjewm_trace_only]=4 [stjewm_spike_only]=4 [stjewm_rate_only]=4 [stjewm_no_trace]=4 [stjewm_hidden_leak]=4 [stjewm_membrane_readout]=4 [alif_timecell_baseline]=2 [gru_baseline]=2 [lewm_baseline_v2]=3 [stacked_lif_trace]=8 [stacked_lif_free]=8 [mlp_baseline]=12 )
declare -A KIND=( [stjewm_trace_only]=stjewm [stjewm_spike_only]=stjewm [stjewm_rate_only]=stjewm [stjewm_no_trace]=stjewm [stjewm_hidden_leak]=stjewm [stjewm_membrane_readout]=stjewm [alif_timecell_baseline]=alif_timecell_baseline [gru_baseline]=gru_baseline [lewm_baseline_v2]=lewm_baseline [stacked_lif_trace]=stacked_lif_trace [stacked_lif_free]=stacked_lif_free [mlp_baseline]=mlp_baseline )
declare -A RO=( [stjewm_trace_only]=trace_only [stjewm_spike_only]=spike_only [stjewm_rate_only]=rate_only [stjewm_no_trace]=no_trace [stjewm_hidden_leak]=hidden_leak [stjewm_membrane_readout]=membrane_readout )

wait_gpu () { # gpu min_free_mb — shared-GPU guard: block until enough free memory
  local gpu=$1 need=$2
  while true; do
    local free
    free=$(nvidia-smi --id=$gpu --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${free%% *}" -ge "$need" ] && return 0
    sleep 120
  done
}
g_job () { # scale model gpu
  local scale=$1 model=$2 gpu=$3
  local kind=${KIND[$model]} nl=${LAYERS[$model]}
  local ro_args=""
  [ -n "${RO[$model]:-}" ] && ro_args="--readout-mode ${RO[$model]}"
  local out="$OUT_ROOT/generalist_G$scale/$model/seed_0"
  [ -f "$out/final.pt" ] && return
  wait_gpu "$gpu" 9000
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m code.train.train \
    --model "$kind" --multi-env-spec "configs/generalist_G${scale}_train.json" \
    --pad-obs-to 128 --action-dim 56 --embed-dim 192 --image-size 0 \
    --n-layers "$nl" $ro_args \
    --epochs 1 --batch 32 --lr 3e-4 --history-size 1 --goal-offset 25 \
    --seed 0 --no-amp --out "$out" > "$out/train.log" 2>&1
  echo "done G$scale $model rc=$?" >> "$LOG_DIR/phase4_progress.log"
}
i=0
for scale in $SCALES; do
  for model in "${!LAYERS[@]}"; do
    g_job "$scale" "$model" $((i % 4)) &
    i=$((i + 1))
    if (( i % 8 == 0 )); then wait; fi
  done
done
wait
echo "[phase4] ALL DONE $(date)" | tee -a "$LOG_DIR/run_all.log"
touch "$LOG_DIR/PHASE4_DONE"

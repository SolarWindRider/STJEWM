#!/bin/bash
# Train ONE generalist checkpoint at 5M-aligned params on a multi-env spec.
# v0.7.14: all 8 baselines at 4.97-5.13M, STJEWM 5.06M trainable.
#
# Usage:
#   ./train_one_5m.sh <model_kind> <train_spec.json> <out_dir> [seed]
#
# Example:
#   ./train_one_5m.sh mlp_baseline configs/oodc/cross_benchmark_F1.json \
#       results/5m/cross_benchmark_F1/mlp_baseline/seed_0 0

set -e
cd /home/lx/snn

MODEL_KIND=${1:?usage: train_one_5m.sh <model_kind> <spec.json> <out_dir> [seed]}
SPEC=${2:?usage: train_one_5m.sh <model_kind> <spec.json> <out_dir> [seed]}
OUT_DIR=${3:?usage: train_one_5m.sh <model_kind> <spec.json> <out_dir> [seed]}
SEED=${4:-0}

# Per-model 5M-aligned param table.
# Format: model_kind|model|readout|extra_args...
#   - STJEWM variants differ only in --readout-mode
#   - Baselines have hardcoded 5M sizes in code/train/train.py build_model
MODEL_FOR=$(cat <<'EOF'
stjewm_trace_only|stjewm|trace_only|
stjewm_spike_only|stjewm|spike_only|
stjewm_rate_only|stjewm|rate_only|
stjewm_no_trace|stjewm|no_trace|
stjewm_hidden_leak|stjewm|hidden_leak|
stjewm_membrane_readout|stjewm|membrane_readout|
stjewm_raw_spike|stjewm|raw_spike|
alif_timecell_baseline|alif_timecell_baseline||
gru_baseline|gru_baseline||
lewm_baseline_v2|lewm_baseline|hidden_leak|
stacked_lif_trace|stacked_lif_trace||
stacked_lif_free|stacked_lif_free||
mlp_baseline|mlp_baseline||
lif_transformer_baseline|lif_transformer_baseline||
EOF
)
ENTRY=$(echo "$MODEL_FOR" | grep -E "^${MODEL_KIND}\|")
if [[ -z "$ENTRY" ]]; then
  echo "[train_one_5m] unknown model_kind=$MODEL_KIND" >&2
  exit 2
fi
MODEL=$(echo "$ENTRY" | awk -F'|' '{print $2}')
READOUT=$(echo "$ENTRY" | awk -F'|' '{print $3}')

mkdir -p "$OUT_DIR"
CMD=(
  /home/lx/miniconda3/envs/snn/bin/python -m code.train.train
  --model "$MODEL"
  --multi-env-spec "$SPEC"
  --pad-obs-to 128
  --action-dim 56
  --n-layers 2
  --epochs 1
  --batch 32
  --lr 3e-4
  --seed "$SEED"
  --save-every 0
  --log-every 200
  --out "$OUT_DIR"
)
if [[ -n "$READOUT" ]]; then
  CMD+=(--readout-mode "$READOUT")
fi
echo "[train_one_5m] $MODEL_KIND ($MODEL, readout=$READOUT) seed=$SEED spec=$SPEC out=$OUT_DIR"
exec "${CMD[@]}"

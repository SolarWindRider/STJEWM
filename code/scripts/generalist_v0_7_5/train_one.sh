#!/bin/bash
# Train ONE generalist checkpoint on a multi-env spec.
#
# Usage:
#   ./train_one.sh <model_kind> <train_spec.json> <out_dir> [seed]
#
# Example:
#   ./train_one.sh stjewm_trace_only configs/generalist_G4_train.json results/generalist/stjewm_trace_only/seed_0 0
#
# This script bakes the v0.7.3/v0.7.5 budget so the 12 model variants are
# reproducible: --pad-obs-to 128 --action-dim 56 --embed-dim 192 --n-layers 2
# --epochs 1 --batch 32 --lr 3e-4. It does NOT call any new code; it just
# invokes the existing trainer with the right CLI.
set -e
cd /home/lx/snn

MODEL_KIND=${1:?usage: train_one.sh <model_kind> <spec.json> <out_dir> [seed]}
SPEC=${2:?usage: train_one.sh <model_kind> <spec.json> <out_dir> [seed]}
OUT_DIR=${3:?usage: train_one.sh <model_kind> <spec.json> <out_dir> [seed]}
SEED=${4:-0}

# Per-model CLI table. Edit here to add or change variants.
# NOTE: lewm_baseline_v2 -> --model lewm_baseline (the v2 lives in code, not the CLI flag).
# NOTE: cubifae / slt variants are "supplementary SNN baselines" — see plan §3.
MODEL_FOR=$(cat <<'EOF'
stjewm_trace_only|stjewm|trace_only
stjewm_spike_only|stjewm|spike_only
stjewm_rate_only|stjewm|rate_only
stjewm_no_trace|stjewm|no_trace
stjewm_hidden_leak|stjewm|hidden_leak
stjewm_membrane_readout|stjewm|membrane_readout
cubifae_baseline|cubifae_baseline|hidden_leak
gru_baseline|gru_baseline|
lewm_baseline_v2|lewm_baseline|hidden_leak
slt_lif_mpc_trace|slt_lif_mpc_trace|trace_only
slt_lif_mpc_free|slt_lif_mpc_free|
mlp_baseline|mlp_baseline|
EOF
)

ENTRY=$(echo "$MODEL_FOR" | grep -E "^${MODEL_KIND}\|")
if [[ -z "$ENTRY" ]]; then
    echo "[train_one] unknown model_kind=$MODEL_KIND" >&2
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
  --embed-dim 192
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

echo "[train_one] $MODEL_KIND ($MODEL, readout=$READOUT) seed=$SEED spec=$SPEC out=$OUT_DIR"
exec "${CMD[@]}"
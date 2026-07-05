#!/bin/bash
# Train ONE generalist ckpt on the union of N envs' training sets.
# Produces results/generalist/<MODEL>/final.pt
#
# Usage:
#   ./train_generalist.sh                                    # all 4 models, 16-env spec
#   ./train_generalist.sh stjewm_trace_only                  # one model
#   ./train_generalist.sh stjewm_trace_only configs/generalist_20env.json
#   EPOCHS=1 BATCH=32 ./train_generalist.sh gru_baseline    # quick smoke
set -e
cd /home/lx/snn

EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
SPEC=${SPEC:-configs/generalist_16env.json}
PAD=${PAD:-128}
ACTION_DIM=${ACTION_DIM:-56}
EMBED_DIM=${EMBED_DIM:-192}
N_LAYERS=${N_LAYERS:-4}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results/generalist}
mkdir -p "$RESULTS_DIR"

# Map of (logical model name) -> (model flag, readout-mode)
# stjewm_trace_only uses --readout-mode trace_only (the membrane-forbidden protocol)
# stjewm_hidden_leak uses --readout-mode hidden_leak
# lewm_baseline_v2 and gru_baseline ignore readout-mode
declare -A MODELS=(
    [stjewm_trace_only]="stjewm|trace_only"
    [stjewm_hidden_leak]="stjewm|hidden_leak"
    [lewm_baseline_v2]="lewm_baseline|hidden_leak"
    [gru_baseline]="gru_baseline|hidden_leak"
)

# Optional: filter by model name
FILTER="$*"

for model_name in "${!MODELS[@]}"; do
    if [ -n "$FILTER" ] && ! echo "$FILTER" | tr ' ' '\n' | grep -qx "$model_name"; then
        continue
    fi
    spec_str="${MODELS[$model_name]}"
    model_kind="${spec_str%%|*}"
    readout_mode="${spec_str##*|}"

    out_dir="$RESULTS_DIR/$model_name"
    if [ -f "$out_dir/final.pt" ]; then
        echo "[skip] $model_name: $out_dir/final.pt already exists"
        continue
    fi
    echo ""
    echo "============================================="
    echo "[train_generalist] $model_name  ($EPOCHS epochs)"
    echo "  model:   $model_kind (readout=$readout_mode)"
    echo "  spec:    $SPEC"
    echo "  out:     $out_dir"
    echo "============================================="

    mkdir -p "$out_dir"
    log="$out_dir/train.log"
    /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
        --model "$model_kind" \
        --multi-env-spec "$SPEC" \
        --pad-obs-to "$PAD" \
        --action-dim "$ACTION_DIM" \
        --embed-dim "$EMBED_DIM" \
        --n-layers "$N_LAYERS" \
        --out "$out_dir" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" \
        --lr "$LR" \
        --save-every 0 \
        --seed 3072 \
        --readout-mode "$readout_mode" \
        2>&1 | tee "$log"
done

echo ""
echo "============================================="
echo "GENERALIST TRAINING COMPLETE"
echo "============================================="

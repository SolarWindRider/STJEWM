#!/bin/bash
# Re-evaluate every generalist ckpt on the 4 stress envs.
#
# Usage:
#   ./eval_stress.sh <suite_name> [n_seeds]
#
# Iterates the 12 model variants and re-evaluates them against
# configs/generalist_G4_stress.json, writing to
# results/generalist_<suite>_stress/<model>/seed_<s>/eval_<env>.json.
#
# The SUITE arg selects which suite's training ckpt is loaded:
#   G4  -> results/generalist/<model>/seed_0/final.pt
#   G8  -> results/generalist_G8/<model>/seed_0/final.pt
#   G16 -> results/generalist_G16/<model>/seed_0/final.pt
set -e
cd /home/lx/snn

SUITE=${1:-G16}
N_SEEDS=${2:-3}
MODELS=(
    stjewm_trace_only
    stjewm_spike_only
    stjewm_rate_only
    stjewm_no_trace
    stjewm_hidden_leak
    stjewm_membrane_readout
    cubifae_baseline
    gru_baseline
    lewm_baseline_v2
    slt_lif_mpc_trace
    slt_lif_mpc_free
    mlp_baseline
)

# Map SUITE -> ckpt dir (under results/) and to stress-output dir.
# The stress output dir is keyed by suite because G4/G8/G16-trained
# ckpts may have meaningfully different stress behaviour.
case "$SUITE" in
    G4)  SUITE_DIR="generalist";        STRESS_DIR="generalist_stress" ;;
    G8)  SUITE_DIR="generalist_G8";      STRESS_DIR="generalist_G8_stress" ;;
    G16) SUITE_DIR="generalist_G16";     STRESS_DIR="generalist_G16_stress" ;;
    *) echo "Usage: $0 <G4|G8|G16> [n_seeds]"; exit 2 ;;
esac

CKPT_BASE=/home/lx/snn/results/$SUITE_DIR
OUT_BASE=/home/lx/snn/results/$STRESS_DIR
mkdir -p "$OUT_BASE"

for MODEL in "${MODELS[@]}"; do
    for ((SEED=0; SEED<N_SEEDS; SEED++)); do
        CKPT="$CKPT_BASE/$MODEL/seed_$SEED/final.pt"
        if [[ ! -f "$CKPT" ]]; then
            echo "[eval_stress] skip $MODEL seed=$SEED (no ckpt at $CKPT)"
            continue
        fi
        echo ""
        echo "============================================="
        echo "[eval_stress] $SUITE / $MODEL / seed=$SEED  (ckpt=$CKPT)"
        echo "============================================="
        bash code/scripts/generalist_v0.7.5/eval_closed_loop_one.sh \
            "$MODEL" "$CKPT" configs/generalist_G4_stress.json "$SEED"
    done
done

/home/lx/miniconda3/envs/snn/bin/python -m code.scripts.generalist_v0.7.5.aggregate_master \
    --suite "$SUITE-stress" \
    --results-dir "$SUITE_DIR" \
    --stress-dir "$STRESS_DIR"

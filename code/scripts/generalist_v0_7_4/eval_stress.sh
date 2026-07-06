#!/bin/bash
# Re-evaluate every generalist ckpt on the 4 stress envs.
#
# Usage:
#   ./eval_stress.sh <suite_name> [n_seeds]
#
# Iterates the 12 model variants and re-evaluates them against
# configs/generalist_G4_stress.json, writing to
# results/generalist_stress/<model>/seed_<s>/eval_<env>.json.
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

OUT_BASE=/home/lx/snn/results/generalist

for MODEL in "${MODELS[@]}"; do
    for ((SEED=0; SEED<N_SEEDS; SEED++)); do
        CKPT="$OUT_BASE/$MODEL/seed_$SEED/final.pt"
        if [[ ! -f "$CKPT" ]]; then
            echo "[eval_stress] skip $MODEL seed=$SEED (no ckpt)"
            continue
        fi
        echo ""
        echo "============================================="
        echo "[eval_stress] $SUITE / $MODEL / seed=$SEED"
        echo "============================================="
        bash code/scripts/generalist_v0_7_4/eval_closed_loop_one.sh \
            "$MODEL" "$CKPT" configs/generalist_G4_stress.json "$SEED"
    done
done

/home/lx/miniconda3/envs/snn/bin/python -m code.scripts.generalist_v0_7_4.aggregate_master \
    --suite "$SUITE-stress"
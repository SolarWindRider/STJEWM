#!/bin/bash
# Top-level orchestrator for one generalist suite (G4/G8/G16).
#
# Usage:
#   ./run_suite.sh <suite_name> <train_spec.json> <eval_spec.json> <n_seeds>
#
# Example:
#   ./run_suite.sh G4 configs/generalist_G4_train.json configs/generalist_G16_eval.json 1
#
# For each of the 12 model variants, trains n_seeds checkpoints and runs
# the per-env closed_loop eval against eval_spec. ID envs go to
# results/generalist/<model>/seed_<s>/, stress envs go to
# results/generalist_stress/<model>/seed_<s>/.
set -e
cd /home/lx/snn

SUITE=${1:?usage: run_suite.sh <suite> <train_spec> <eval_spec> <n_seeds>}
TRAIN_SPEC=${2:?usage: run_suite.sh <suite> <train_spec> <eval_spec> <n_seeds>}
EVAL_SPEC=${3:?usage: run_suite.sh <suite> <train_spec> <eval_spec> <n_seeds>}
N_SEEDS=${4:-1}

MODELS=(
    stjewm_trace_only
    stjewm_spike_only
    stjewm_rate_only
    stjewm_no_trace
    stjewm_hidden_leak
    stjewm_membrane_readout
    alif_timecell_baseline
    gru_baseline
    lewm_baseline_v2
    stacked_lif_trace
    stacked_lif_free
    mlp_baseline
)

OUT_BASE=/home/lx/snn/results/generalist
mkdir -p "$OUT_BASE"

# Smoke mode: N_SEEDS=0 → only run aggregate on whatever exists.
if [[ "$N_SEEDS" == "0" ]]; then
    echo "[run_suite] smoke mode: skipping train+eval, only aggregate"
    /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.generalist_v0.7.5.aggregate_master \
        --suite "$SUITE"
    exit $?
fi

for MODEL in "${MODELS[@]}"; do
    for ((SEED=0; SEED<N_SEEDS; SEED++)); do
        OUT_DIR="$OUT_BASE/$MODEL/seed_$SEED"
        CKPT="$OUT_DIR/final.pt"
        echo ""
        echo "============================================="
        echo "[run_suite] $SUITE / $MODEL / seed=$SEED"
        echo "============================================="
        if [[ ! -f "$CKPT" ]]; then
            bash code/scripts/generalist_v0.7.5/train_one.sh \
                "$MODEL" "$TRAIN_SPEC" "$OUT_DIR" "$SEED"
        else
            echo "[run_suite] ckpt exists, skipping train: $CKPT"
        fi
        bash code/scripts/generalist_v0.7.5/eval_closed_loop_one.sh \
            "$MODEL" "$CKPT" "$EVAL_SPEC" "$SEED"
    done
done

echo ""
echo "============================================="
echo "[run_suite] $SUITE done — aggregating"
echo "============================================="
/home/lx/miniconda3/envs/snn/bin/python -m code.scripts.generalist_v0.7.5.aggregate_master \
    --suite "$SUITE"
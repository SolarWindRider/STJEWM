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
# results/generalist/<suite>/<model>/seed_<s>/, stress envs go to
# results/generalist_stress/<suite>/<model>/seed_<s>/.
set -euo pipefail
cd /home/lx/snn

SUITE=${1:?usage: run_suite.sh <suite> <train_spec> <eval_spec> <n_seeds>}
TRAIN_SPEC=${2:?usage: run_suite.sh <suite> <train_spec> <eval_spec> <n_seeds>}
EVAL_SPEC=${3:?usage: run_suite.sh <suite> <train_spec> <eval_spec> <n_seeds>}
N_SEEDS=${4:-1}
case "$SUITE" in
    G4|G8|G16) ;;
    *) echo "[run_suite] Unknown suite: $SUITE" >&2; exit 2 ;;
esac
if [[ ! "$N_SEEDS" =~ ^[0-9]+$ ]]; then
    echo "[run_suite] n_seeds must be a nonnegative integer" >&2
    exit 2
fi

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

export OUT_BASE="${RESULTS_ROOT:-/home/lx/snn/results}/generalist/$SUITE"
export STRESS_OUT_BASE="${RESULTS_ROOT:-/home/lx/snn/results}/generalist_stress/$SUITE"

# Historical unversioned files may not be published by directory discovery.
if [[ "$N_SEEDS" == "0" ]]; then
    echo "[run_suite] Aggregate-only mode was removed. Use master_aggregate.sh with --training-manifest, --state-run and --out." >&2
    exit 2
fi
mkdir -p "$OUT_BASE" "$STRESS_OUT_BASE"

for MODEL in "${MODELS[@]}"; do
    for ((SEED=0; SEED<N_SEEDS; SEED++)); do
        OUT_DIR="$OUT_BASE/$MODEL/seed_$SEED"
        CKPT="$OUT_DIR/final.pt"
        echo ""
        echo "============================================="
        echo "[run_suite] $SUITE / $MODEL / seed=$SEED"
        echo "============================================="
        if [[ ! -f "$CKPT" ]]; then
            bash code/scripts/generalist_v0_7_5/train_one.sh \
                "$MODEL" "$TRAIN_SPEC" "$OUT_DIR" "$SEED"
        else
            echo "[run_suite] ckpt exists, skipping train: $CKPT"
        fi
        bash code/scripts/generalist_v0_7_5/eval_closed_loop_one.sh \
            "$MODEL" "$CKPT" "$EVAL_SPEC" "$SEED"
    done
done

echo ""
echo "============================================="
echo "[run_suite] Historical raw outputs only; automatic publication was removed."
echo "[run_suite] Use master_aggregate.sh with an explicit completed final state manifest."
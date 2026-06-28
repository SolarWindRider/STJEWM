#!/bin/bash
# MASTER pipeline: collect 250K → train LeWM → train STJEWM → eval all → aggregate
# Per user's directive: "先跑LeWM, 然后跑我们提出的方法" (LeWM first, then STJEWM).
#
# Usage: ./run_all.sh           # full pipeline
#        ./run_all.sh collect   # just collect 250K data
#        ./run_all.sh lewm       # just train LeWM
#        ./run_all.sh stjewm     # just train STJEWM
#        ./run_all.sh eval       # just eval all
#        ./run_all.sh aggregate  # just aggregate

set -e
cd /home/lx/snn
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}
DATA_250K_DIR=/home/lx/snn/data/dm_control/3d_rollouts_250k

phase=${1:-all}

phase_collect() {
    echo "============================================================"
    echo "PHASE 1: Collect 250K rollouts (12 DMC envs)"
    echo "============================================================"
    mkdir -p "$DATA_250K_DIR"
    CUDA_VISIBLE_DEVICES=1 /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.collect_250k --all --n-episodes 1250
    # cartpole, pendulum go to dm_control/ (2D paths)
    CUDA_VISIBLE_DEVICES=1 /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.collect_250k --env cartpole --n-episodes 1250
    CUDA_VISIBLE_DEVICES=1 /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.collect_250k --env pendulum --n-episodes 1250
}

phase_lewm() {
    echo "============================================================"
    echo "PHASE 2: Train LeWM-style baseline (all envs, 2 GPUs in parallel)"
    echo "============================================================"
    bash code/scripts/train_all_2gpu.sh lewm_baseline
}

phase_stjewm() {
    echo "============================================================"
    echo "PHASE 3: Train STJEWM (all envs, 2 GPUs in parallel)"
    echo "============================================================"
    bash code/scripts/train_all_2gpu.sh stjewm
}

phase_eval() {
    echo "============================================================"
    echo "PHASE 4: Eval all 50 (env, model) pairs"
    echo "============================================================"
    # Run eval in parallel on 2 GPUs
    for model in stjewm lewm_baseline; do
        CUDA_VISIBLE_DEVICES=0 setsid nohup bash -c "MODEL_FILTER=$model /home/lx/snn/code/scripts/eval_all.sh" > /tmp/eval_logs/${model}_gpu0.log 2>&1 &
        disown
    done
    wait
}

phase_aggregate() {
    echo "============================================================"
    echo "PHASE 5: Aggregate results"
    echo "============================================================"
    /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.aggregate_results \
        --results-dir "$RESULTS_DIR" --out-dir "$RESULTS_DIR/aggregate"
}

case "$phase" in
    collect)   phase_collect ;;
    lewm)      phase_lewm ;;
    stjewm)    phase_stjewm ;;
    eval)      phase_eval ;;
    aggregate) phase_aggregate ;;
    all)
        phase_collect
        phase_lewm
        phase_stjewm
        phase_eval
        phase_aggregate
        echo ""
        echo "============================================================"
        echo "PIPELINE COMPLETE"
        echo "============================================================"
        ;;
    *)
        echo "Unknown phase: $phase"
        exit 1
        ;;
esac

#!/bin/bash
# Train LeWM-style on all 15 envs, splitting between 2 GPUs in parallel.
# Each GPU runs its envs sequentially.
# Usage: ./train_all_2gpu.sh stjewm   or   ./train_all_2gpu.sh lewm_baseline

set -e
cd /home/lx/snn
MODEL="${1:-lewm_baseline}"
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results}

# Partition: 8 envs on GPU 0, 7 envs on GPU 1
GPU0_ENVS=(cheetah walker hopper quadruped humanoid humanoid_CMU dog fish)
GPU1_ENVS=(stacker reacher finger ball_in_cup cartpole_2d pendulum_2d pusht tworoom)

mkdir -p /tmp/train_logs_gpu0 /tmp/train_logs_gpu1

# Build a space-separated env list (train_all.sh supports space-separated names)
GPU0_STR="${GPU0_ENVS[*]}"
GPU1_STR="${GPU1_ENVS[*]}"

echo "=== GPU 0: $GPU0_STR ==="
echo "=== GPU 1: $GPU1_STR ==="

# Use a function so we can background with the right shell
launch_gpu0() {
    cd /home/lx/snn
    CUDA_VISIBLE_DEVICES=0 MODEL_FILTER="$MODEL" EPOCHS=5 /home/lx/snn/code/scripts/train_all.sh $GPU0_STR
}
launch_gpu1() {
    cd /home/lx/snn
    CUDA_VISIBLE_DEVICES=1 MODEL_FILTER="$MODEL" EPOCHS=5 /home/lx/snn/code/scripts/train_all.sh $GPU1_STR
}

# Run them in background
( launch_gpu0 ) > /tmp/train_logs_gpu0/all.log 2>&1 &
GPU0_PID=$!
( launch_gpu1 ) > /tmp/train_logs_gpu1/all.log 2>&1 &
GPU1_PID=$!

echo "GPU 0 PID: $GPU0_PID"
echo "GPU 1 PID: $GPU1_PID"
echo ""
echo "Tail logs at /tmp/train_logs_gpu0/all.log and /tmp/train_logs_gpu1/all.log"

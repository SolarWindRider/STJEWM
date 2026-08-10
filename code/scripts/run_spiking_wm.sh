#!/bin/bash
# Train Spiking-WM (Brain-Cog-Lab, real external baseline) on DMC tasks.
# Uses dmc_proprio config (state input) to match ST-JEWM's state-based protocol.
# Usage: bash code/scripts/run_spiking_wm.sh <task> <gpu>
set -euo pipefail

TASK="${1:?task e.g. cartpole_swingup}"
GPU="${2:?gpu id}"
ROOT=/home/lx/snn
LOGDIR=${ROOT}/results/spiking_wm/logs_${TASK}

mkdir -p "${LOGDIR}"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL=egl
nohup /home/lx/miniconda3/envs/snn/bin/python code/scripts/run_spiking_wm.py \
    --configs dmc_proprio \
    --task "dmc_${TASK}" \
    --spike_times 5 \
    --envs 8 \
    --train_ratio 1024 \
    --eval_every 20000 \
    --eval_episode_num 5 \
    --seed 0 \
    --steps 5e5 \
    --logdir "${LOGDIR}" \
    > "${LOGDIR}/train.log" 2>&1 &
echo "started ${TASK} on gpu ${GPU} -> ${LOGDIR}/train.log (pid $!)"

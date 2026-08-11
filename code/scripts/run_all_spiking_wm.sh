#!/bin/bash
# Full-coverage Spiking-WM training pipeline: all 12 DMC tasks used by the
# ST-JEWM 10-split setting that Spiking-WM can run (dm_control suite).
# 4 GPUs; each GPU runs its queue sequentially. Budgets by difficulty:
#   S 2e5: pendulum_swingup, cup_catch (ball_in_cup), reacher_easy
#   M 3e5: hopper_hop, fish_swim
#   L 5e5: cheetah_run (done), walker_walk (done), quadruped_walk, dog_walk, humanoid_run
# ST-JEWM custom envs not runnable by Spiking-WM (noted in report):
#   pusht, tworoom, reacher_4d, stacker, delayed_t_maze, humanoid_CMU
# Usage: bash code/scripts/run_all_spiking_wm.sh
set -u
ROOT=/home/lx/snn
cd "${ROOT}"

declare -A STEPS=(
  [pendulum_swingup]=2e5
  [cup_catch]=2e5
  [reacher_easy]=2e5
  [hopper_hop]=3e5
  [fish_swim]=3e5
  [quadruped_walk]=5e5
  [dog_walk]=5e5
  [humanoid_run]=5e5
)

# GPU queues (already-done tasks skipped)
QUEUE0="humanoid_run"
QUEUE1="quadruped_walk dog_walk"
QUEUE2="hopper_hop fish_swim"
QUEUE3="pendulum_swingup cup_catch reacher_easy"

run_queue() {
  local gpu=$1
  shift
  for task in $@; do
    if [ -f "${ROOT}/results/spiking_wm/logs_${task}/latest_model.pt" ] && \
       grep -q "eval_return" "${ROOT}/results/spiking_wm/logs_${task}/train.log"; then
      echo "[gpu ${gpu}] SKIP ${task} (already trained)"
      continue
    fi
    rm -rf "${ROOT}/results/spiking_wm/logs_${task}"
    mkdir -p "${ROOT}/results/spiking_wm/logs_${task}"
    echo "[gpu ${gpu}] START ${task} (${STEPS[$task]} steps) $(date '+%F %T')"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export MUJOCO_GL=egl
    /home/lx/miniconda3/envs/snn/bin/python code/scripts/run_spiking_wm.py \
      --configs dmc_proprio \
      --task "dmc_${task}" \
      --spike_times 5 \
      --envs 8 \
      --train_ratio 64 \
      --eval_every 20000 \
      --eval_episode_num 5 \
      --seed 0 \
      --steps "${STEPS[$task]}" \
      --logdir "${ROOT}/results/spiking_wm/logs_${task}" \
      > "${ROOT}/results/spiking_wm/logs_${task}/train.log" 2>&1
    echo "[gpu ${gpu}] DONE ${task} $(date '+%F %T') rc=$?"
  done
}

run_queue 0 ${QUEUE0} &
run_queue 1 ${QUEUE1} &
run_queue 2 ${QUEUE2} &
run_queue 3 ${QUEUE3} &
wait
echo "ALL QUEUES DONE $(date '+%F %T')"

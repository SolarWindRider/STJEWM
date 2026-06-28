#!/usr/bin/env bash
# Retrain STJEWM and LeWM-baseline with FIXED goal loss on all 16 envs.
# Reuses the same configs as the previous training.
set -e
cd /home/lx/snn

# Env configs: name -> "env_kind data_path goal_offset"
declare -A ENVS
ENVS[ball_in_cup]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz 25"
ENVS[cartpole_2d]="dmc /home/lx/snn/data/dm_control/cartpole_250k.npz 25"
ENVS[cheetah]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 25"
ENVS[dog]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz 25"
ENVS[finger]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz 25"
ENVS[fish]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz 25"
ENVS[hopper]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz 25"
ENVS[humanoid]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz 25"
ENVS[humanoid_CMU]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz 25"
ENVS[pendulum_2d]="dmc /home/lx/snn/data/dm_control/pendulum_250k.npz 25"
ENVS[pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 100"
ENVS[quadruped]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz 25"
ENVS[reacher]="reacher_4d /home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz 25"
ENVS[stacker]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz 25"
ENVS[tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 100"
ENVS[walker]="dmc /home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz 25"

MODEL=${1:-stjewm}    # stjewm or lewm_baseline
GPU=${2:-0}
LAMBDAGOAL=${3:-0.5}
EPOCHS=${4:-5}

mkdir -p /home/lx/snn/results/${GPU}_logs
for env in ball_in_cup cartpole_2d cheetah dog finger fish hopper humanoid humanoid_CMU pendulum_2d pusht quadruped reacher stacker tworoom walker; do
  cfg="${ENVS[$env]}"
  kind=$(echo $cfg | cut -d' ' -f1)
  data=$(echo $cfg | cut -d' ' -f2)
  goal_off=$(echo $cfg | cut -d' ' -f3)
  # Use suffix based on lambda_goal to keep all 3 variants separate
  if [ "${LAMBDAGOAL}" = "0" ]; then
    SUFFIX="nogoal"
  else
    SUFFIX="v2"
  fi
  out=/home/lx/snn/results/${env}/${MODEL}_${SUFFIX}
  log=/home/lx/snn/results/${GPU}_logs/${MODEL}_${SUFFIX}_${env}.log
  if [ -f "${out}/final.pt" ]; then
    echo "Skipping ${env} ${MODEL} (final.pt already exists)"
    continue
  fi
  rm -rf $out
  # Use 62500 windows for dmc envs, 10K for pusht/tworoom/reacher (matching original training)
  if [[ "$env" == "pusht" || "$env" == "tworoom" ]]; then
    MW=10000
  elif [[ "$env" == "reacher" ]]; then
    MW=62500
  else
    MW=62500
  fi
  echo "Training ${MODEL} on ${env} (kind=${kind}, data=${data}, goal_offset=${goal_off}, max_windows=${MW}) -> $out"
  CUDA_VISIBLE_DEVICES=${GPU} /home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
      --model ${MODEL} --env-kind ${kind} \
      --data ${data} \
      --out ${out} \
      --epochs ${EPOCHS} --batch 64 --lr 3e-4 --save-every 0 --n-layers 4 \
      --history-size 1 --goal-offset ${goal_off} --t-pred 3 \
      --max-windows ${MW} --lambda-goal ${LAMBDAGOAL} > ${log} 2>&1
  echo "Done ${env} ${MODEL} -> $out"
done

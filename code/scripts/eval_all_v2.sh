#!/usr/bin/env bash
# Run all evals (STJEWM v2, LeWM v2, optionally LeWM-no-goal).
# Usage: ./eval_all_v2.sh <model_dirname>
set -e
cd /home/lx/snn
MODEL=${1:-stjewm_v2}
# Env: env_kind, data_path, goal_offset
declare -A EVALS
EVALS[ball_in_cup]="ball_in_cup /home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz 25"
EVALS[cartpole_2d]="cartpole /home/lx/snn/data/dm_control/cartpole_250k.npz 25"
EVALS[cheetah]="cheetah /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 25"
EVALS[dog]="dog /home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz 25"
EVALS[finger]="finger /home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz 25"
EVALS[fish]="fish /home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz 25"
EVALS[hopper]="hopper /home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz 25"
EVALS[humanoid]="humanoid /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz 25"
EVALS[humanoid_CMU]="humanoid_cmu /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz 25"
EVALS[pendulum_2d]="pendulum /home/lx/snn/data/dm_control/pendulum_250k.npz 25"
EVALS[pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 100"
EVALS[quadruped]="quadruped /home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz 25"
EVALS[reacher]="reacher /home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz 25"
EVALS[stacker]="stacker /home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz 25"
EVALS[tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 100"
EVALS[walker]="walker /home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz 25"

mkdir -p /home/lx/snn/results/aggregate/eval_logs
for env in ball_in_cup cartpole_2d cheetah dog finger fish hopper humanoid humanoid_CMU pendulum_2d pusht quadruped reacher stacker tworoom walker; do
  cfg="${EVALS[$env]}"
  env_kind=$(echo $cfg | cut -d' ' -f1)
  data=$(echo $cfg | cut -d' ' -f2)
  goal=$(echo $cfg | cut -d' ' -f3)
  ckpt=/home/lx/snn/results/${env}/${MODEL}/final.pt
  out=/home/lx/snn/results/${env}/${MODEL}/eval.json
  if [ ! -f "$ckpt" ]; then echo "skip $env (no ckpt)"; continue; fi
  if [ -f "$out" ]; then echo "skip $env (already done)"; continue; fi
  log=/home/lx/snn/results/aggregate/eval_logs/eval_${MODEL}_${env}.log
  /home/lx/miniconda3/envs/snn/bin/python -m code.eval.closed_loop \
      --env ${env_kind} --ckpt ${ckpt} --data ${data} \
      --out ${out} --n-episodes 25 --n-seeds 2 --horizon 5 --eval-budget 50 \
      --history-size 1 --goal-offset ${goal} > ${log} 2>&1
  echo "done $env $MODEL"
done

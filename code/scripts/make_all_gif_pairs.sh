#!/usr/bin/env bash
# Make success/failure gif pairs for STJEWM v2 and LeWM v2 on all 16 envs.
# Outputs to /home/lx/snn/results/aggregate/gifs/{env}/{model}/{success,failure}.gif
set -e
cd /home/lx/snn

OUT_BASE="/home/lx/snn/results/aggregate/gifs"
mkdir -p "$OUT_BASE"

# Env configs: name -> "env_kind data_path goal_offset history_size"
declare -A ENVS
ENVS[ball_in_cup]="ball_in_cup /home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz 25 1"
ENVS[cartpole_2d]="cartpole /home/lx/snn/data/dm_control/cartpole_250k.npz 25 1"
ENVS[cheetah]="cheetah /home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz 25 1"
ENVS[dog]="dog /home/lx/snn/data/dm_control/3d_rollouts_250k/dog_250k.npz 25 1"
ENVS[finger]="finger /home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz 25 1"
ENVS[fish]="fish /home/lx/snn/data/dm_control/3d_rollouts_250k/fish_250k.npz 25 1"
ENVS[hopper]="hopper /home/lx/snn/data/dm_control/3d_rollouts_250k/hopper_250k.npz 25 1"
ENVS[humanoid]="humanoid /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz 25 1"
ENVS[humanoid_CMU]="humanoid_cmu /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz 25 1"
ENVS[pendulum_2d]="pendulum /home/lx/snn/data/dm_control/pendulum_250k.npz 25 1"
ENVS[pusht]="pusht /home/lx/LeWM/data/pusht_expert_train.h5 100 1"
ENVS[quadruped]="quadruped /home/lx/snn/data/dm_control/3d_rollouts_250k/quadruped_250k.npz 25 1"
ENVS[reacher]="reacher /home/lx/snn/data/dm_control/3d_rollouts_250k/reacher_250k.npz 25 1"
ENVS[stacker]="stacker /home/lx/snn/data/dm_control/3d_rollouts_250k/stacker_250k.npz 25 1"
ENVS[tworoom]="tworoom /home/lx/LeWM/data/tworoom_extract/tworoom.h5 100 1"
ENVS[walker]="walker /home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz 25 1"

# Models
declare -A CKPT_DIRS
CKPT_DIRS[stjewm_v2]="stjewm_v2"
CKPT_DIRS[lewm_v2]="lewm_baseline_v2"

for model in stjewm_v2 lewm_v2; do
  for env in ball_in_cup cartpole_2d cheetah dog finger fish hopper humanoid humanoid_CMU pendulum_2d pusht quadruped reacher stacker tworoom walker; do
    cfg="${ENVS[$env]}"
    env_kind=$(echo $cfg | cut -d' ' -f1)
    data=$(echo $cfg | cut -d' ' -f2)
    goal_off=$(echo $cfg | cut -d' ' -f3)
    hist_size=$(echo $cfg | cut -d' ' -f4)
    dir="${CKPT_DIRS[$model]}"
    ckpt="/home/lx/snn/results/${env}/${dir}/final.pt"
    eval_json="/home/lx/snn/results/${env}/${dir}/eval.json"
    out_dir="${OUT_BASE}/${env}/${dir}"
    if [ ! -f "$ckpt" ]; then
      echo "skip $env $model (no ckpt)"
      continue
    fi
    if [ ! -f "$eval_json" ]; then
      echo "skip $env $model (no eval)"
      continue
    fi
    # Skip if already done
    if [ -f "${out_dir}/${env}_${dir}_success.gif" ] && [ -f "${out_dir}/${env}_${dir}_failure.gif" ]; then
      echo "skip $env $model (already done)"
      continue
    fi
    echo "=== $env $model ==="
    /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.make_gif_pairs \
        --env ${env_kind} \
        --ckpt ${ckpt} \
        --data ${data} \
        --goal-offset ${goal_off} --history-size ${hist_size} \
        --eval-json ${eval_json} \
        --out-dir ${out_dir} \
        --name ${env}_${dir} \
        --criterion lewm_success 2>&1 | tail -5
  done
done

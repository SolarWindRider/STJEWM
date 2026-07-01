#!/usr/bin/env bash
# Evaluate rate_only, no_trace, gru_baseline, mlp_baseline on 16 envs
# Builds the full 5-condition (5-way) comparison table.
set -e
cd /home/lx/snn
OUT=${OUT:-/home/lx/snn/results/aggregate/eval_v2_5way}
LOG=${LOG:-/home/lx/snn/logs/eval_v2_5way}
mkdir -p "$OUT" "$LOG"

ENVS="ball_in_cup cartpole_2d cheetah dog finger fish hopper humanoid humanoid_CMU pendulum_2d pusht quadruped reacher stacker tworoom walker"

for env in $ENVS; do
  case $env in
    pusht)       DATA=/home/lx/LeWM/data/pusht_expert_train.h5; H=1; GO=100; ENV=pusht ;;
    tworoom)     DATA=/home/lx/LeWM/data/tworoom_extract/tworoom.h5; H=1; GO=100; ENV=tworoom ;;
    *)           DATA=/home/lx/snn/data/dm_control/3d_rollouts_250k/${env}_250k.npz; H=1; GO=25; ENV=$env ;;
  esac
  for mode in rate_only no_trace; do
    out="$OUT/${env}_stjewm_${mode}.json"
    if [ -f "$out" ] && [ "$out" -nt /home/lx/snn/results/$env/stjewm_${mode}/final.pt ]; then continue; fi
    /home/lx/miniconda3/envs/snn/bin/python -m code.eval.closed_loop --env $ENV --ckpt /home/lx/snn/results/$env/stjewm_${mode}/final.pt --data $DATA --out $out --n-episodes 25 --n-seeds 2 --horizon 5 --eval-budget 50 --history-size $H --goal-offset $GO > $LOG/${env}_stjewm_${mode}.log 2>&1 && echo "ok: $env $mode" || echo "FAIL: $env $mode"
  done
  for model in gru_baseline mlp_baseline; do
    out="$OUT/${env}_${model}.json"
    if [ -f "$out" ] && [ "$out" -nt /home/lx/snn/results/$env/${model}/final.pt ]; then continue; fi
    /home/lx/miniconda3/envs/snn/bin/python -m code.eval.closed_loop --env $ENV --ckpt /home/lx/snn/results/$env/${model}/final.pt --data $DATA --out $out --n-episodes 25 --n-seeds 2 --horizon 5 --eval-budget 50 --history-size $H --goal-offset $GO > $LOG/${env}_${model}.log 2>&1 && echo "ok: $env $model" || echo "FAIL: $env $model"
  done
done

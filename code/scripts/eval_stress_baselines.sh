#!/usr/bin/env bash
# Evaluate GRU + MLP on the 4 unsaturated stress tasks
set -e
cd /home/lx/snn

OUT=${OUT:-/home/lx/snn/results/aggregate/stress_baselines}
LOG=${LOG:-/home/lx/snn/logs/stress_baselines}
mkdir -p "$OUT" "$LOG"

# (env, ckpt_env, data, history, goal_offset)
ENVS=(
  "cartpole_flicker|cartpole_2d|/home/lx/snn/data/dm_control/cartpole_250k.npz|1|25"
  "cheetah_velhidden|cheetah|/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz|1|25"
  "pusht_ood|pusht|/home/lx/LeWM/data/pusht_expert_train.h5|1|100"
  "tworoom_long|tworoom|/home/lx/LeWM/data/tworoom_extract/tworoom.h5|1|100"
)

for spec in "${ENVS[@]}"; do
  IFS='|' read -r env ckpt_env data hs go <<< "$spec"
  for model in gru_baseline mlp_baseline; do
    out="$OUT/${env}_${model}.json"
    if [ -f "$out" ]; then continue; fi
    ckpt="/home/lx/snn/results/$ckpt_env/$model/final.pt"
    if [ ! -f "$ckpt" ]; then echo "no ckpt $env $model"; continue; fi
    log="$LOG/${env}_${model}.log"
    /home/lx/miniconda3/envs/snn/bin/python -m code.eval.closed_loop --env $env --ckpt $ckpt --data $data --out $out --n-episodes 25 --n-seeds 2 --horizon 5 --eval-budget 50 --history-size $hs --goal-offset $go > $log 2>&1 && echo "ok: $env $model" || echo "FAIL: $env $model"
  done
done

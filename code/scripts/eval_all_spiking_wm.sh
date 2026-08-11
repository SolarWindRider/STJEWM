#!/bin/bash
# After full Spiking-WM training: protocol evals on all 12 DMC tasks,
# then final eval-return summary. Run on 4 GPUs in parallel.
set -u
cd /home/lx/snn
export MUJOCO_GL=egl
PY=/home/lx/miniconda3/envs/snn/bin/python

TASKS="cartpole_swingup cheetah_run walker_walk finger_spin pendulum_swingup cup_catch reacher_easy hopper_hop quadruped_walk dog_walk fish_swim humanoid_run"

run_eval() {
  local gpu=$1; shift
  for t in $@; do
    CKPT=results/spiking_wm/logs_${t}/latest_model.pt
    if [ ! -f "$CKPT" ]; then echo "[gpu ${gpu}] SKIP $t (no ckpt)"; continue; fi
    echo "[gpu ${gpu}] eval $t $(date '+%T')"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    $PY code/scripts/eval_spiking_wm_protocol.py \
      --task "$t" --ckpt "$CKPT" \
      --out results/spiking_wm/protocol_${t}.json \
      --n-steps 2000 --device cuda:0
  done
}

run_eval 0 cartpole_swingup cheetah_run walker_walk &
run_eval 1 finger_spin pendulum_swingup cup_catch &
run_eval 2 reacher_easy hopper_hop quadruped_walk &
run_eval 3 dog_walk fish_swim humanoid_run &
wait

echo "== FINAL EVAL RETURNS =="
for t in $TASKS; do
  echo -n "$t: "
  grep "eval_return" results/spiking_wm/logs_${t}/train.log 2>/dev/null | tail -1 || echo "MISSING"
done
echo "== PROTOCOL METRICS =="
for t in $TASKS; do
  if [ -f results/spiking_wm/protocol_${t}.json ]; then
    $PY -c "
import json; d=json.load(open('results/spiking_wm/protocol_${t}.json'))
print('${t}', '| embed_rho', round(d['corr_obs_embed'],4), '| stoch_rho', round(d['corr_obs_latent'],4), '| spike_rho', round(d['event_rho'],4), '| rate', round(d['mean_spike_rate'],4), '| stoch_std', round(d['stoch_std'],4))"
  fi
done

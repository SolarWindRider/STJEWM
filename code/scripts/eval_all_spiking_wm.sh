#!/bin/bash
# After Spiking-WM training finishes: run protocol evals on all 4 tasks
# and extract final eval returns from training logs.
set -euo pipefail
cd /home/lx/snn
export MUJOCO_GL=egl
PY=/home/lx/miniconda3/envs/snn/bin/python

for t in cartpole_swingup cheetah_run walker_walk finger_spin; do
  CKPT=results/spiking_wm/logs_${t}/latest_model.pt
  if [ ! -f "$CKPT" ]; then echo "SKIP $t (no ckpt)"; continue; fi
  echo "== protocol eval: $t"
  $PY code/scripts/eval_spiking_wm_protocol.py \
    --task "$t" --ckpt "$CKPT" \
    --out results/spiking_wm/protocol_${t}.json \
    --n-steps 2000 --device cuda:0
done
echo "== final eval returns (last per task):"
for t in cartpole_swingup cheetah_run walker_walk finger_spin; do
  echo -n "$t: "
  grep "eval_return" results/spiking_wm/logs_${t}/train.log | tail -1
done

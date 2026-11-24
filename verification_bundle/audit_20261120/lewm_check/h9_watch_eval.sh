#!/usr/bin/env bash
# Watch the two H9 training runs; every time a checkpoint file changes, snapshot it and evaluate the
# snapshot with the corrected evaluator (H10: our env loop, official CEMSolver, budget 50, any-time).
PY=/home/lx/miniconda3/envs/snn/bin/python
cd /tmp/lewm_check
declare -A LAST
while true; do
  alive=0
  for tag in lr5e-5 lr2e-4; do
    f=/tmp/lewm_check/h9_state_lewm_${tag}.pt
    if pgrep -f "h9_faithful_state_lewm.py .* $tag" > /dev/null; then alive=1; fi
    if [ -f "$f" ]; then
      m=$(stat -c %Y "$f"); s1=$(stat -c %s "$f"); sleep 10; s2=$(stat -c %s "$f" 2>/dev/null || echo 0)
      if [ "${LAST[$tag]:-0}" != "$m" ] && [ "$s1" = "$s2" ] && [ "$s1" -gt 1000 ]; then
        step=$($PY -c "import torch;print(torch.load('$f',map_location='cpu',weights_only=False)['step'])" 2>/dev/null)
        cp "$f" "/tmp/lewm_check/h9_ck_${tag}_step${step}.pt"; LAST[$tag]=$m
        CUDA_VISIBLE_DEVICES=2 $PY /tmp/lewm_check/h10_expert_replay.py "/tmp/lewm_check/h9_ck_${tag}_step${step}.pt" 2>&1 \
          | grep -E "^RESULT" | grep "H9 checkpoint" | sed "s/^RESULT/RESULT[$tag]/" | cut -c1-700
      fi
    fi
  done
  [ "$alive" = "0" ] && { echo "DONE H9 both runs finished $(date +%H:%M:%S)"; exit 0; }
  sleep 45
done

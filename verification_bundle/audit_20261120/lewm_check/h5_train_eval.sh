#!/usr/bin/env bash
# H5: positive control. Train OUR LeWM-baseline architecture in-distribution on PushT only with a
# real budget, then evaluate each checkpoint in OUR unmodified harness (closed_loop + our CEM).
#   $1 = tag (A: paper-like history-3 / 6 layers / 192-d;  B: benchmark config history-1 / 3 layers / 288-d)
#   $2 = CUDA device index
set -u
TAG=$1; GPU=$2
if [ "$TAG" = "A" ]; then H=3; L=6; E=192; else H=1; L=3; E=288; fi
PY=/home/lx/miniconda3/envs/snn/bin/python
OUT=/data/lx/tmp/results/lewm_check_20261124/h5_${TAG}
H5=/home/lx/LeWM/data/pusht_expert_train.h5
mkdir -p "$OUT"
cd /home/lx/snn
echo "START H5 $TAG hist=$H layers=$L embed=$E $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=$GPU $PY -m code.train.train --model lewm_baseline --env-kind pusht --data $H5 \
  --out $OUT --epochs 5 --batch 128 --lr 1e-4 --n-layers $L --embed-dim $E --history-size $H --t-pred 3 \
  --goal-offset 25 --lambda-sigreg 0.09 --lambda-goal 0.5 --save-every 5000 --log-every 1000 --no-amp --seed 0 \
  > $OUT/train.log 2>&1 &
TRAIN_PID=$!
echo "train pid $TRAIN_PID"
for STEP in 5000 10000 20000 40000 60000 80000; do
  CK=step${STEP}.pt
  # wait for a fully written checkpoint (size stable across 15 s)
  while true; do
    if [ -f "$OUT/$CK" ]; then
      S1=$(stat -c %s "$OUT/$CK"); sleep 15; S2=$(stat -c %s "$OUT/$CK" 2>/dev/null || echo 0)
      [ "$S1" = "$S2" ] && [ "$S1" -gt 1000 ] && break
    else
      kill -0 $TRAIN_PID 2>/dev/null || break 2
      sleep 20
    fi
  done
  CUDA_VISIBLE_DEVICES=$GPU $PY -m code.eval.closed_loop --env pusht --ckpt $OUT/$CK --data $H5 \
    --out $OUT/eval_$CK.json --device cuda:0 --split in_dist --n-episodes 50 --n-seeds 1 --horizon 25 \
    --eval-budget 50 --history-size $H --cem-samples 300 --cem-elites 30 --goal-offset 25 --cem-iters 30 \
    > $OUT/eval_$CK.log 2>&1
  $PY - <<EOF
import json
d = json.load(open("$OUT/eval_$CK.json"))
print("RESULT", json.dumps(dict(tag="$TAG", ckpt="$CK", hist=$H, envSR=d["success_rate_env"], cos=round(d["mean_cos_dist"],4), phys=round(d["mean_phys_dist"],1))), flush=True)
EOF
done
wait $TRAIN_PID
echo "TRAIN DONE rc=$? $(date +%H:%M:%S)"; grep -E "step (10000|30000|60000|90000) " $OUT/train.log | tail -5 | cut -c1-150
CUDA_VISIBLE_DEVICES=$GPU $PY -m code.eval.closed_loop --env pusht --ckpt $OUT/final.pt --data $H5 \
  --out $OUT/eval_final.json --device cuda:0 --split in_dist --n-episodes 50 --n-seeds 1 --horizon 25 \
  --eval-budget 50 --history-size $H --cem-samples 300 --cem-elites 30 --goal-offset 25 --cem-iters 30 \
  > $OUT/eval_final.log 2>&1
$PY - <<EOF
import json
d = json.load(open("$OUT/eval_final.json"))
print("RESULT", json.dumps(dict(tag="$TAG", ckpt="final", hist=$H, envSR=d["success_rate_env"], cos=round(d["mean_cos_dist"],4), phys=round(d["mean_phys_dist"],1))), flush=True)
EOF
echo "DONE H5 $TAG $(date +%H:%M:%S)"

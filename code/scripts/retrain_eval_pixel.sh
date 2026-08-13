#!/bin/bash
# Phase 2: re-evaluate retrained pixel checkpoints (5m_pixel retrained 65 + post-hoc 4).
set -u
cd /home/lx/snn
PY=/home/lx/miniconda3/envs/snn/bin/python
EV=code/scripts/generalist_v0_7_5_5m_pixel/eval_pixel_ckpt_cem.py

# Collect retrained pixel ckpts (mtime >= 2026-08-11 16:00 = all retrain/post-hoc)
declare -a CKPTS=()
while IFS= read -r p; do CKPTS+=("$p"); done < <(/home/lx/miniconda3/envs/snn/bin/python -c "
import glob, os, datetime
for p in sorted(glob.glob('results/5m_pixel/**/final.pt', recursive=True)):
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(p))
    if mt >= datetime.datetime(2026,8,11,16,0):
        print(p)
")
echo "pixel retrained ckpts: ${#CKPTS[@]}" >> /tmp/pixel_eval_progress.log

# delete old eval files for these ckpts
/home/lx/miniconda3/envs/snn/bin/python -c "
import glob, os
n=0
for p in glob.glob('results/5m_pixel/**/final.pt', recursive=True):
    import datetime
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(p))
    if mt >= datetime.datetime(2026,8,11,16,0):
        d = os.path.dirname(p)
        for e in glob.glob(d + '/eval_*'):
            os.remove(e); n+=1
print('removed', n, 'old pixel eval files')
" >> /tmp/pixel_eval_progress.log

run_one() {  # gpu ckpt
  local gpu=$1 ckpt=$2
  local out_dir=$(dirname "$ckpt")
  CUDA_VISIBLE_DEVICES=$gpu $PY $EV --ckpt "$ckpt" --out_dir "$out_dir" \
     --image_size 84 --n_episodes 5 --device cuda \
     > "$out_dir/eval_cem_full.log" 2>&1
  echo "[DONE gpu$gpu] $ckpt rc=$?" >> /tmp/pixel_eval_progress.log
}
q0=() q1=() q2=() q3=()
i=0
for c in "${CKPTS[@]}"; do
  case $((i % 4)) in 0) q0+=("$c");; 1) q1+=("$c");; 2) q2+=("$c");; 3) q3+=("$c");; esac
  i=$((i+1))
done
run_q() {
  local gpu=$1; shift
  local pids=()
  for c in "$@"; do
    run_one "$gpu" "$c" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
}
run_q 0 "${q0[@]}" &
run_q 1 "${q1[@]}" &
run_q 2 "${q2[@]}" &
run_q 3 "${q3[@]}" &
wait
echo "ALL PIXEL EVAL DONE $(date +%T)" >> /tmp/pixel_eval_progress.log

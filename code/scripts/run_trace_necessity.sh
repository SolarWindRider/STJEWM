#!/bin/bash
# Run all trace necessity experiments: lesion + decay + shuffle
# across 4 envs (cheetah, cartpole_2d, pusht, tworoom)
# Total: 3 scripts × 4 envs × 5-6 conditions ≈ 72 runs × 2 min ≈ 2.4h
set -e
cd /home/lx/snn
mkdir -p results/trace_necessity
PYTHON=/home/lx/miniconda3/envs/snn/bin/python
export CUDA_VISIBLE_DEVICES=2

echo "=== TRACE NECESSITY EXPERIMENTS ==="
echo "start: $(date)"

# === E1: trace lesion ===
echo "--- E1: trace lesion ---"
for env in cheetah cartpole_2d pusht tworoom; do
  for ratio in 0.0 0.1 0.25 0.5 0.75 0.9; do
    out="results/trace_necessity/lesion_${env}_r${ratio}.json"
    [ -f "$out" ] && continue
    echo "[lesion] $env ratio=$ratio"
    $PYTHON -m code.scripts.trace_lesion \
      --env "$env" --lesion-ratio "$ratio" --out "$out" \
      --n-episodes 10 --n-seeds 2 2>&1 | tail -1
  done
done

# === E2: trace decay sweep ===
echo "--- E2: decay sweep ---"
for env in cheetah cartpole_2d pusht tworoom; do
  for decay in 0.0 0.3 0.5 0.7 0.9 0.99; do
    out="results/trace_necessity/decay_${env}_d${decay}.json"
    [ -f "$out" ] && continue
    echo "[decay] $env decay=$decay"
    $PYTHON -m code.scripts.trace_decay_sweep \
      --env "$env" --decay "$decay" --out "$out" \
      --n-episodes 10 --n-seeds 2 2>&1 | tail -1
  done
done

# === E3: spike timing shuffle ===
echo "--- E3: timing shuffle ---"
for env in cheetah cartpole_2d pusht tworoom; do
  for shuffle in none window5 window10 global; do
    out="results/trace_necessity/shuffle_${env}_${shuffle}.json"
    [ -f "$out" ] && continue
    echo "[shuffle] $env $shuffle"
    $PYTHON -m code.scripts.timing_shuffle \
      --env "$env" --shuffle "$shuffle" --out "$out" \
      --n-episodes 10 --n-seeds 2 2>&1 | tail -1
  done
done

echo "=== TRACE NECESSITY COMPLETE ==="
echo "end: $(date)"

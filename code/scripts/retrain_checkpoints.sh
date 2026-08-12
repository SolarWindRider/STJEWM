#!/bin/bash
# Retrain corrupted auxiliary checkpoints (4-GPU queues).
# Recovered: 5m_seed1 (39 state) + sigreg (8) + 5m_pixel (130).
# Not recoverable (no command record): 16 single-env v0.7.x ckpts.
set -u
cd /home/lx/snn
ROOT=/home/lx/snn
export PYTHONPATH=$ROOT
PY=/home/lx/miniconda3/envs/snn/bin/python

splits_state="cross_benchmark_F1 generalist_16env oodc_F2"
readouts="trace_only hidden_leak spike_only rate_only no_trace membrane_readout"
baselines="alif_timecell_baseline gru_baseline lewm_baseline_v2 stacked_lif_trace stacked_lif_free lif_transformer_baseline mlp_baseline"

train_pixel_stjewm() {  # split seed imgsz
  local split=$1 seed=$2 imgsz=$3 ro
  for ro in $readouts; do
    OUT=results/5m_pixel/${split}/stjewm_${ro}/seed_${seed}
    [ -f "$OUT/final.pt" ] && { echo "[skip] pix stjewm_$ro $split"; continue; }
    echo "[pix] START stjewm_$ro $split $(date +%T)"
    mkdir -p "$OUT"
    $PY -m code.train.train \
      --model stjewm --readout-mode "$ro" \
      --multi-env-spec "configs/oodc_5m_pixel/${split}.json" \
      --pad-obs-to 21168 --action-dim 56 --embed-dim 192 \
      --image-size "$imgsz" --n-layers 4 \
      --epochs 1 --batch 32 --lr 3e-4 \
      --history-size 1 --goal-offset 25 \
      --seed "$seed" --out "$OUT" > "$OUT/train.log" 2>&1
    echo "[pix] DONE stjewm_$ro $split $(date +%T)"
  done
}
train_pixel_base() {  # split seed imgsz
  local split=$1 seed=$2 imgsz=$3 m
  for m in $baselines; do
    OUT=results/5m_pixel/${split}/$m/seed_${seed}
    [ -f "$OUT/final.pt" ] && { echo "[skip] pix $m $split"; continue; }
    echo "[pix] START $m $split $(date +%T)"
    mkdir -p "$OUT"
    $PY -m code.train.train \
      --model "$m" --multi-env-spec "configs/oodc_5m_pixel/${split}.json" \
      --pad-obs-to 21168 --action-dim 56 --embed-dim 192 \
      --image-size "$imgsz" --n-layers 4 \
      --epochs 1 --batch 32 --lr 3e-4 \
      --history-size 1 --goal-offset 25 \
      --seed "$seed" --out "$OUT" > "$OUT/train.log" 2>&1
    echo "[pix] DONE $m $split $(date +%T)"
  done
}
run_pixel_split() {  # gpu split
  export CUDA_VISIBLE_DEVICES=$1
  train_pixel_stjewm "$2" 0 84
  train_pixel_base "$2" 0 84
}
run_pixel_queue() {  # gpu split_list...
  export CUDA_VISIBLE_DEVICES=$1
  shift
  for s in "$@"; do
    run_pixel_split ${CUDA_VISIBLE_DEVICES} "$s"
  done
}

run_seed1() {  # gpu
  export CUDA_VISIBLE_DEVICES=$1
  for s in $splits_state; do
    for ro in $readouts; do
      OUT=results/5m_seed1/${s}/stjewm_${ro}/seed_0
      [ -f "$OUT/final.pt" ] && { echo "[skip] s1 stjewm_$ro $s"; continue; }
      echo "[s1] START stjewm_$ro $s $(date +%T)"
      bash code/scripts/generalist_v0_7_5_5m/train_one_5m.sh "stjewm_${ro}" "configs/oodc_5m/${s}.json" "$OUT" 1 > /dev/null 2>&1
      echo "[s1] DONE stjewm_$ro $s $(date +%T)"
    done
    for b in $baselines; do
      OUT=results/5m_seed1/${s}/$b/seed_0
      [ -f "$OUT/final.pt" ] && { echo "[skip] s1 $b $s"; continue; }
      echo "[s1] START $b $s $(date +%T)"
      bash code/scripts/generalist_v0_7_5_5m/train_one_5m.sh "$b" "configs/oodc_5m/${s}.json" "$OUT" 1 > /dev/null 2>&1
      echo "[s1] DONE $b $s $(date +%T)"
    done
  done
  # sigreg sweep after seed1 on same GPU
  for sig in 0.09 0.01 0.001 0.0; do
    for sp in cross_benchmark_F1 oodc_F2; do
      OUT=results/5m_sigreg_sweep/${sp}/stjewm_trace_only_sig${sig}/seed_0
      [ -f "$OUT/final.pt" ] && { echo "[skip] sig $sp $sig"; continue; }
      echo "[s1] START sigreg $sp $sig $(date +%T)"
      mkdir -p "$OUT"
      $PY -m code.train.train \
        --model stjewm --multi-env-spec "configs/oodc_5m/${sp}.json" \
        --pad-obs-to 128 --action-dim 56 --embed-dim 192 --image-size 0 \
        --n-layers 4 --epochs 1 --batch 32 --lr 3e-4 \
        --history-size 1 --goal-offset 25 --seed 0 \
        --readout-mode trace_only --lambda-sigreg $sig \
        --out "$OUT" > "$OUT/train.log" 2>&1
      echo "[s1] DONE sigreg $sp $sig $(date +%T)"
    done
  done
}

# GPU0: seed1 (39) + sigreg (8); GPU1-3: pixel (130 split 3+3+4 splits)
run_seed1 0 > /tmp/retrain_s1.log 2>&1 &
run_pixel_queue 1 cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 > /tmp/retrain_p1.log 2>&1 &
run_pixel_queue 2 oodc_F1 oodc_F1F2 oodc_F1F3 > /tmp/retrain_p2.log 2>&1 &
run_pixel_queue 3 oodc_F2 oodc_F2F3 oodc_F3 generalist_16env > /tmp/retrain_p3.log 2>&1 &
wait
echo "ALL RETRAIN QUEUES DONE $(date +%T)"

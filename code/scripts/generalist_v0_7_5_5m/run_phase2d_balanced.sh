#!/bin/bash
# run_phase2d_balanced.sh — remaining pixel trainings, longest-first greedy split
# across 4 GPUs (no batch-barrier tail latency). stjewm pixel = ~45GB = one per GPU.
set -u
cd /home/lx/snn
export OUT_ROOT=${OUT_ROOT:-/data/lx/tmp/results}
LOG_DIR=/data/lx/tmp/logs
PY=/home/lx/miniconda3/envs/snn/bin/python
PIX_SPLITS="oodc_F1 oodc_F2 oodc_F3 oodc_F1F2 oodc_F1F3 oodc_F2F3 cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 generalist_16env"
MODEL_LAYERS="stjewm:4 lewm_baseline:3 gru_baseline:2 mlp_baseline:12 stacked_lif_trace:8 stacked_lif_free:8 lif_transformer_baseline:3 alif_timecell_baseline:2"
COST="stjewm:45 stacked_lif_trace:30 stacked_lif_free:30 lewm_baseline:30 lif_transformer_baseline:25 gru_baseline:15 alif_timecell_baseline:15 mlp_baseline:12"

# ---- collect missing tasks with cost ----
TASKS=()
for split in $PIX_SPLITS; do
  for entry in $MODEL_LAYERS; do
    model=${entry%%:*}; nl=${entry##*:}
    [ -f "$OUT_ROOT/5m_pixel/$split/$model/seed_0/final.pt" ] && continue
    cost=15; for c in ${COST//:/ }; do :; done
    case $model in
      stjewm) cost=45;; stacked_lif_trace|stacked_lif_free|lewm_baseline) cost=30;;
      lif_transformer_baseline) cost=25;; gru_baseline|alif_timecell_baseline) cost=15;;
      mlp_baseline) cost=12;;
    esac
    TASKS+=("$cost $model $split $nl")
  done
done
echo "[phase2d] missing tasks: ${#TASKS[@]}"

# ---- greedy: sort desc by cost, assign to lightest queue ----
mapfile -t SORTED < <(printf '%s\n' "${TASKS[@]}" | sort -rn)
SUM=(0 0 0 0)
Q=(/tmp/p2d_q0.tsv /tmp/p2d_q1.tsv /tmp/p2d_q2.tsv /tmp/p2d_q3.tsv)
: > "${Q[0]}"; : > "${Q[1]}"; : > "${Q[2]}"; : > "${Q[3]}"
for t in "${SORTED[@]}"; do
  cost=${t%% *}
  min=0
  for g in 1 2 3; do [ "${SUM[$g]}" -lt "${SUM[$min]}" ] && min=$g; done
  echo "$t" >> "${Q[$min]}"
  SUM[$min]=$((SUM[$min] + cost))
done
echo "queue sums: ${SUM[*]}"

run_one () { # gpu model split nl
  local gpu=$1 model=$2 split=$3 nl=$4
  local out="$OUT_ROOT/5m_pixel/$split/$model/seed_0"
  mkdir -p "$out"
  local ro_args=""
  [ "$model" = "stjewm" ] && ro_args="--readout-mode trace_only"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m code.train.train \
    --model "$model" --multi-env-spec "configs/oodc_5m_pixel/$split.json" \
    --pad-obs-to 21168 --action-dim 56 --embed-dim 192 --image-size 84 \
    --n-layers "$nl" $ro_args \
    --epochs 1 --batch 32 --lr 3e-4 --history-size 1 --goal-offset 25 \
    --seed 0 --no-amp --out "$out" > "$out/train.log" 2>&1
  echo "done pixel $split $model rc=$?" >> "$LOG_DIR/phase2d_progress.log"
}

worker () { # gpu queuefile
  local gpu=$1 q=$2
  while IFS=' ' read -r cost model split nl; do
    run_one "$gpu" "$model" "$split" "$nl"
  done < "$q"
  echo "WORKER_gpu${gpu}_DONE" >> "$LOG_DIR/phase2d_progress.log"
}
for g in 0 1 2 3; do worker $g "${Q[$g]}" & done
wait

# sigreg fill (idempotent) + handoff marker for phase3
sig_job () {
  local sig=$1 split=$2 gpu=$3
  local out="$OUT_ROOT/5m_sigreg_sweep/$split/stjewm_trace_only_sig${sig}/seed_0"
  [ -f "$out/final.pt" ] && return
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m code.train.train \
    --model stjewm --multi-env-spec "configs/oodc_5m/$split.json" \
    --pad-obs-to 128 --action-dim 56 --embed-dim 192 --image-size 0 \
    --n-layers 4 --readout-mode trace_only --lambda-sigreg "$sig" \
    --epochs 1 --batch 32 --lr 3e-4 --history-size 1 --goal-offset 25 \
    --seed 0 --no-amp --out "$out" > "$out/train.log" 2>&1
}
i=0
for sig in 0.09 0.01 0.001 0.0; do
  for split in cross_benchmark_F1 oodc_F2; do
    sig_job "$sig" "$split" $((i % 4)) & i=$((i + 1))
  done
done
wait
echo "[phase2d] ALL DONE $(date)" | tee -a "$LOG_DIR/run_all.log"
touch "$LOG_DIR/PHASE2_DONE_MARK"

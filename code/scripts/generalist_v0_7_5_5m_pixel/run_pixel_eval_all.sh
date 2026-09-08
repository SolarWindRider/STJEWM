#!/bin/bash
# run_pixel_eval_all.sh — closed-loop eval for ALL 130 pixel ckpts (13 models x 10 splits),
# 3 jobs per GPU (eval uses <1GB). Protocol matches the legacy pixel main table:
# static-qpos goal, CEM 300x30x10, horizon 5, budget 50, 5 episodes.
set -u
cd /home/lx/snn
export OUT_ROOT=${OUT_ROOT:-/data/lx/tmp/results}
LOG_DIR=/data/lx/tmp/logs
PY=/home/lx/miniconda3/envs/snn/bin/python
EV=code/scripts/generalist_v0_7_5_5m_pixel/eval_pixel_ckpt.py
PIX_SPLITS="oodc_F1 oodc_F2 oodc_F3 oodc_F1F2 oodc_F1F3 oodc_F2F3 cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 generalist_16env"
MODELS="stjewm stjewm_spike_only stjewm_rate_only stjewm_no_trace stjewm_hidden_leak stjewm_membrane_readout alif_timecell_baseline gru_baseline lewm_baseline stacked_lif_trace stacked_lif_free mlp_baseline lif_transformer_baseline"

echo "[pixel_eval] start $(date)" | tee -a "$LOG_DIR/run_all.log"

# ---- build job list (skip finished) ----
JOBS=()
for split in $PIX_SPLITS; do
  for m in $MODELS; do
    out="$OUT_ROOT/5m_pixel/$split/$m/seed_0"
    [ -f "$out/eval_summary.json" ] && continue
    [ -f "$out/final.pt" ] || continue
    JOBS+=("$m $split")
  done
done
echo "[pixel_eval] jobs: ${#JOBS[@]}"

Q=(/tmp/pev_q0.tsv /tmp/pev_q1.tsv /tmp/pev_q2.tsv /tmp/pev_q3.tsv)
: > "${Q[0]}"; : > "${Q[1]}"; : > "${Q[2]}"; : > "${Q[3]}"
i=0
for t in "${JOBS[@]}"; do
  echo "$t" >> "${Q[$((i % 4))]}"
  i=$((i + 1))
done

worker () { # gpu queuefile slot
  local gpu=$1 q=$2
  while IFS=' ' read -r m split; do
    out="$OUT_ROOT/5m_pixel/$split/$m/seed_0"
    CUDA_VISIBLE_DEVICES=$gpu $PY "$EV" \
      --ckpt "$out/final.pt" --out_dir "$out" \
      --n_episodes 5 --samples 300 --elites 30 --cem_iters 10 --horizon 5 \
      --device cuda:0 > "$out/eval.log" 2>&1
    echo "done pixel-eval $split $m rc=$?" >> "$LOG_DIR/pixel_eval_progress.log"
  done < "$q"
  echo "WORKER_gpu${gpu}_DONE" >> "$LOG_DIR/pixel_eval_progress.log"
}
# 12 workers: GPUs 0-3, 3 queues each — round-robin job pulling via fd-locked lines
for g in 0 1 2 3; do
  for s in 0 1 2; do
    # split each GPU queue into 3 slice files
    :
  done
done
# simpler: re-split round-robin into 12 slice files
rm -f /tmp/pev_w*.tsv
i=0
while IFS= read -r line; do
  echo "$line" >> "/tmp/pev_w$((i % 12)).tsv"
  i=$((i + 1))
done < <(cat "${Q[@]}")
for w in 0 1 2 3 4 5 6 7 8 9 10 11; do
  worker $((w % 4)) "/tmp/pev_w${w}.tsv" &
done
wait
echo "[pixel_eval] ALL DONE $(date)" | tee -a "$LOG_DIR/run_all.log"
touch "$LOG_DIR/PIXEL_EVAL_DONE"

#!/bin/bash
# run_phase7_g2_trace.sh — Extended-Data reruns on new ckpts:
#  (a) G2 AUROC probe: 13 models x 7 DMC envs, 1-epoch linear probe
#  (b) trace causality window ablation: STJEWM-trace x 3 envs
# Outputs under /data/lx/tmp/results/{g2_probe,trace_ablation}.
set -u
cd /home/lx/snn
LOG_DIR=/data/lx/tmp/logs
PY=/home/lx/miniconda3/envs/snn/bin/python
echo "[phase7] start $(date)" | tee -a "$LOG_DIR/run_all.log"

STJ="stjewm_trace_only stjewm_spike_only stjewm_rate_only stjewm_no_trace stjewm_hidden_leak stjewm_membrane_readout"
BASE="alif_timecell_baseline gru_baseline lewm_baseline_v2 stacked_lif_trace stacked_lif_free mlp_baseline lif_transformer_baseline"
ENVS7="cartpole_2d pendulum_2d finger ball_in_cup cheetah walker hopper"

# ---- (a) G2 probe ----
i=0
for m in $STJ $BASE; do
  root=5m_5mpar
  case $m in alif*|gru*|lewm*|stacked*|mlp*|lif*) root=5m;; esac
  for env in $ENVS7; do
    out=/data/lx/tmp/results/g2_probe/$root/$m/${env}.json
    [ -f "$out" ] && continue
    mkdir -p "$(dirname "$out")"
    CUDA_VISIBLE_DEVICES=$((i % 4)) $PY -m code.scripts.probe \
      --env "$env" --model "$m" \
      --ckpt "/data/lx/tmp/results/$root/$( [ "$root" = 5m_5mpar ] && echo oodc_F1 || echo oodc_F1)/$m/seed_0/final.pt" \
      --probe-target position --out "$out" --epochs 1 --pad-obs-to 128 --action-dim-eval 56 \
      > /dev/null 2>&1
    echo "done g2 $root $m $env rc=$?" >> "$LOG_DIR/phase7_progress.log"
    i=$((i + 1))
  done
done
echo "[phase7] G2 done $(date)" | tee -a "$LOG_DIR/run_all.log"

# ---- (b) trace causality (STJEWM-trace, 3 envs) ----
i=0
for env in cartpole_2d cheetah ball_in_cup; do
  out=/data/lx/tmp/results/trace_ablation/${env}.json
  [ -f "$out" ] && continue
  data="data/dm_control/cartpole_250k.npz"
  [ "$env" = "cheetah" ] && data="data/dm_control/3d_rollouts_250k/cheetah_250k.npz"
  [ "$env" = "ball_in_cup" ] && data="data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz"
  CUDA_VISIBLE_DEVICES=$((i % 4)) $PY -m code.scripts.event_window_ablation \
    --env "$env" --model stjewm_trace_only \
    --ckpt "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt" \
    --out "$out" --n-episodes 10 --horizon 5 --eval-budget 50 --cem-samples 300 \
    > /dev/null 2>&1
  echo "done trace-ablation $env rc=$?" >> "$LOG_DIR/phase7_progress.log"
  i=$((i + 1))
done
echo "[phase7] ALL DONE $(date)" | tee -a "$LOG_DIR/run_all.log"
touch "$LOG_DIR/PHASE7_DONE"

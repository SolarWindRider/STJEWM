#!/bin/bash
# run_phase6_gstats.sh — div/resp latent stats for the 3 scale-axis generalist runs
# (G4/G8/G16), providing the scale-invariance evidence for Results 3.3.
# Waits for PHASE4_DONE, then measures each scale with the G-series model set.
set -u
cd /home/lx/snn
LOG_DIR=/data/lx/tmp/logs
PY=/home/lx/miniconda3/envs/snn/bin/python
while [ ! -f "$LOG_DIR/PHASE4_DONE" ]; do sleep 300; done
echo "[phase6] start $(date)" | tee -a "$LOG_DIR/run_all.log"

MODELS="stjewm_trace_only stjewm_spike_only stjewm_rate_only stjewm_no_trace stjewm_hidden_leak stjewm_membrane_readout alif_timecell_baseline gru_baseline lewm_baseline_v2 stacked_lif_trace stacked_lif_free mlp_baseline"

for scale in 4 8 16; do
  $PY -m code.scripts.generalist_v0_7_5_5m.measure_latent_stats_5m \
    --results "/data/lx/tmp/results" --out "/data/lx/tmp/results/generalist_G${scale}_stats" \
    --splits "generalist_G${scale}" --models $MODELS --device cuda:0 \
    > "$LOG_DIR/phase6_stats_G${scale}.log" 2>&1
  echo "done stats G$scale rc=$?" >> "$LOG_DIR/phase6_progress.log"
done
echo "[phase6] ALL DONE $(date)" | tee -a "$LOG_DIR/run_all.log"
touch "$LOG_DIR/PHASE6_DONE"

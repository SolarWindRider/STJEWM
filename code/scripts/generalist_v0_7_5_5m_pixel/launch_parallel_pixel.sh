#!/bin/bash
# Launch 130 pixel training in batches.
# Usage: bash launch_parallel_pixel.sh [seed] [image_size] [max_concurrent]
set -e
SEED=${1:-0}
IMAGE_SIZE=${2:-84}
MAX_CONCURRENT=${3:-4}
SPLITS="cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3 oodc_F1 oodc_F1F2 oodc_F1F3 oodc_F2 oodc_F2F3 oodc_F3 generalist_16env"
STJEWM_READOUTS="trace_only hidden_leak spike_only rate_only no_trace membrane_readout"
BASELINES="alif_timecell_baseline gru_baseline lewm_baseline_v2 stacked_lif_trace stacked_lif_free lif_transformer_baseline mlp_baseline"
> jobs_pixel.txt
for SPLIT in $SPLITS; do
  for READOUT in $STJEWM_READOUTS; do
    echo "bash code/scripts/generalist_v0_7_5_5m_pixel/train_one_stjewm_pixel.sh $SPLIT $READOUT $SEED $IMAGE_SIZE" >> jobs_pixel.txt
  done
  for BASELINE in $BASELINES; do
    echo "bash code/scripts/generalist_v0_7_5_5m_pixel/train_one_pixel.sh $BASELINE $SPLIT $SEED $IMAGE_SIZE" >> jobs_pixel.txt
  done
done
cat jobs_pixel.txt | xargs -I{} -P $MAX_CONCURRENT bash -c "{}"
echo "[parallel_pixel] DONE"

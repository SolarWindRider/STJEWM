#!/bin/bash
# Re-evaluate the audited pixel grid through the sole canonical producer.
# No timestamp heuristic, checkpoint-tree output, deletion, or historical reuse.
set -euo pipefail
cd /home/lx/snn
: "${TRAINING_MANIFEST:?Set the final consolidated training repair manifest}"
: "${EVAL_ROOT:?Set a fresh output root separate from all checkpoint/archive roots}"
export TRAINING_MANIFEST EVAL_ROOT
exec bash code/scripts/generalist_v0_7_5_5m_pixel/run_pixel_eval_all.sh

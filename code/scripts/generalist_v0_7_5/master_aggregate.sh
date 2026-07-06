#!/bin/bash
# Top-level aggregator: call all per-cell aggregators + master table writer.
#
# Usage:
#   ./master_aggregate.sh [--probes] [--align]
set -e
cd /home/lx/snn

PROBES=""
ALIGN=""
SUITE="G16"
for arg in "$@"; do
  case "$arg" in
    --probes) PROBES="--probes" ;;
    --align)  ALIGN="--align" ;;
    --suite=*) SUITE="${arg#--suite=}" ;;
    *) echo "[master_aggregate] unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Per-cell aggregators first (reads results/{probe,event_align}/<file>.json).
/home/lx/miniconda3/envs/snn/bin/python -m code.scripts.aggregate_event_probes \
    || echo "[master_aggregate] aggregate_event_probes skipped (no probe data yet)"
/home/lx/miniconda3/envs/snn/bin/python -m code.scripts.aggregate_analysis \
    || echo "[master_aggregate] aggregate_analysis skipped (no probe/align data yet)"

/home/lx/miniconda3/envs/snn/bin/python -m code.scripts.generalist_v0.7.5.aggregate_master \
    --suite "$SUITE" $PROBES $ALIGN

echo "[master_aggregate] done"
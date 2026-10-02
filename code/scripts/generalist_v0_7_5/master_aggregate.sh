#!/bin/bash
# Explicit final-manifest state aggregation only. Diagnostics use their own
# completed auxiliary manifest via aggregate_align.py / scaling_table.py.
# Usage: ./master_aggregate.sh --training-manifest <final.json> \
#          --state-run <complete-state-run> --out <fresh-report.md>
set -euo pipefail
cd /home/lx/snn
exec /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.generalist_v0_7_5.aggregate_master "$@"

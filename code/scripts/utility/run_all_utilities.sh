#!/bin/bash
# Run all v0.7.7 + v0.7.8 utility experiments end-to-end.
#
# Usage:  bash code/scripts/utility/run_all_utilities.sh
#
# Re-runs the per-cell drivers and re-aggregates the tables from the
# per-cell JSONs in results/utility/.

set -euo pipefail
cd "$(dirname "$0")/../../.."

PY=/home/lx/miniconda3/envs/snn/bin/python
OUT=/home/lx/snn/results/utility

echo "============================================================"
echo "1) v0.7.7 utility package"
echo "============================================================"
$PY -m code.scripts.utility.run_latent_goal_mpc     2>&1 | tail -5
$PY -m code.scripts.utility.run_latent_env_grad     2>&1 | tail -5
$PY -m code.scripts.utility.run_sample_efficiency   2>&1 | tail -5

echo ""
echo "============================================================"
echo "2) v0.7.8 cross-environment generalisation"
echo "============================================================"
$PY -m code.scripts.utility.run_cross_env_gen --aggregate-only 2>&1 | tail -5

echo ""
echo "============================================================"
echo "3) v0.7.8 data-budget compression"
echo "============================================================"
$PY -m code.scripts.utility.run_compression_sweep --aggregate-only 2>&1 | tail -5

echo ""
echo "============================================================"
echo "All v0.7.7 + v0.7.8 tables:"
echo "============================================================"
ls -la $OUT/*_table.md

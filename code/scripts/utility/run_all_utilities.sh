#!/bin/bash
# Source-only utility reproducer. The final manifest owns checkpoint training;
# these commands only consume its audited checkpoints into a fresh output root.
set -euo pipefail
cd /home/lx/snn
PY=/home/lx/miniconda3/envs/snn/bin/python
: "${TRAINING_MANIFEST:?Set the final consolidated training repair manifest}"
: "${UTILITY_OUT_ROOT:?Set a fresh utility output root}"
: "${CHECKPOINT_ROOT:?Set the results_root from the final training manifest}"
UTILITY_DEVICE=${UTILITY_DEVICE:-cpu}
export TRAINING_MANIFEST CHECKPOINT_ROOT

# The horizon driver has no manifest flag, so gate its entire checkpoint set here.
"$PY" -c 'import os; from pathlib import Path; from code.scripts.audited_results import TrainingAudit, require; from code.scripts.utility.run_sample_efficiency import G16_CKPTS; a=TrainingAudit(os.environ["TRAINING_MANIFEST"]); require(Path(os.environ["CHECKPOINT_ROOT"]).resolve() == a.results_root, "Wrong checkpoint root"); [a.checkpoint(a.results_root / "generalist_G16" / m / "seed_0" / "final.pt") for m in G16_CKPTS]'
mkdir "$UTILITY_OUT_ROOT"
"$PY" -m code.scripts.utility.run_latent_goal_mpc \
  --results-dir "$CHECKPOINT_ROOT/generalist_G16" --out-dir "$UTILITY_OUT_ROOT/latent_goal_mpc" --device "$UTILITY_DEVICE"
"$PY" -m code.scripts.utility.run_latent_env_grad \
  --training-manifest "$TRAINING_MANIFEST" --out-root "$UTILITY_OUT_ROOT/latent_env_grad" \
  --table-path "$UTILITY_OUT_ROOT/latent_env_grad_table.md" --json-path "$UTILITY_OUT_ROOT/latent_env_grad.json" \
  --device "$UTILITY_DEVICE"
"$PY" -m code.scripts.utility.run_sample_efficiency \
  --training-manifest "$TRAINING_MANIFEST" --out-root "$UTILITY_OUT_ROOT/sample_efficiency" \
  --table-path "$UTILITY_OUT_ROOT/sample_efficiency_table.md" --json-path "$UTILITY_OUT_ROOT/sample_efficiency.json" \
  --device "$UTILITY_DEVICE"
"$PY" -m code.scripts.utility.run_cross_env_gen \
  --training-manifest "$TRAINING_MANIFEST" --out-root "$UTILITY_OUT_ROOT/cross_env_gen" \
  --table-path "$UTILITY_OUT_ROOT/cross_env_gen_table.md" --device "$UTILITY_DEVICE"
"$PY" -m code.scripts.utility.run_budget_scaling \
  --training-manifest "$TRAINING_MANIFEST" --out-root "$UTILITY_OUT_ROOT/budget_scaling" \
  --table-path "$UTILITY_OUT_ROOT/budget_scaling_table.md" --device "$UTILITY_DEVICE"
printf 'All source-only utility grids completed in %s\n' "$UTILITY_OUT_ROOT"

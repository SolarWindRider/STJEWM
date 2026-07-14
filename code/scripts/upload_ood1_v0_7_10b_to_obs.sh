#!/usr/bin/env bash
# Upload the v0.7.10b OOD path-C artifacts to the OBS bucket.
#
# Artifacts (under /home/lx/snn/results/utility/):
#   ood1_table.md       - per-cell + per-(split,model) + per-(split,family) tables
#   cross_env_gen_table.md - v0.7.8 within-suite pilot (held for reference)
#   budget_scaling_table.md
#   sample_efficiency_table.md
#   latent_goal_mpc_table.md
#   latent_env_grad_table.md
#   generalist_scaling_table.md
#   MASTER_TABLE.md      - the v0.7.5 baseline master table
#
# The OOD path-C results sit under results/oodc/<split>/<split>/<model>/seed_0/:
# 6 splits x 12 models x 39 held-out envs = 468 cells of per-cell JSONs.
# We upload just the aggregate table (44K) and a single example per-(model, env)
# JSON for the OOD path-C results, to keep the upload size bounded.
set -e
cd /home/lx/snn
OBSUTIL=/home/lx/obsutil_linux_amd64_5.8.3/obsutil
BUCKET=obs://lixiang01/STJEWM_NMI/aggregate

# All utility experiment tables (v0.7.7 + v0.7.8 + v0.7.10b)
for fname in \
    ood1_table.md \
    cross_env_gen_table.md \
    budget_scaling_table.md \
    sample_efficiency_table.md \
    latent_goal_mpc_table.md \
    latent_env_grad_table.md \
    generalist_scaling_table.md; do
    local_path="results/utility/$fname"
    if [ ! -f "$local_path" ]; then
        echo "[skip] $local_path not found"
        continue
    fi
    obs_path="$BUCKET/$fname"
    echo "Uploading: $local_path -> $obs_path"
    $OBSUTIL cp "$local_path" "$obs_path" 2>&1 | tail -3
done

# Also push the v0.7.10b paper-context OOD results to a dedicated prefix
OODC_BUCKET=obs://lixiang01/STJEWM_NMI/ood1_v0_7_10b
echo ""
echo "Uploading per-cell JSONs (one example per (model, split)) to $OODC_BUCKET/"
# Upload the aggregate table at the top of the OODC_BUCKET prefix
$OBSUTIL cp results/utility/ood1_table.md "$OODC_BUCKET/ood1_table.md" 2>&1 | tail -3

# Upload 1 example cell per (split, model) — these prove the path is alive
for split in oodc_F1 oodc_F1F2 oodc_F1F3 oodc_F2 oodc_F2F3 oodc_F3; do
    for model in cubifae_baseline stjewm_trace_only gru_baseline lewm_baseline_v2 mlp_baseline; do
        # pick first env cell under each (split, model)
        cell=$(ls results/oodc/${split}/${split}/${model}/seed_0/*.json 2>/dev/null \
              | grep -v "_position\|_velocity\|_future_k\|loss_log\|^.*eval_" \
              | head -1)
        if [ -z "$cell" ]; then continue; fi
        env=$(basename "$cell" .json)
        # Use the per-cell path as the OBS key
        rel="results/oodc/${split}/${split}/${model}/seed_0/${env}.json"
        obs_path="$OODC_BUCKET/${split}/${model}/${env}.json"
        $OBSUTIL cp "$cell" "$obs_path" 2>&1 | tail -1
    done
done

# Status row
echo ""
echo "OBS upload complete."

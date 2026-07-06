#!/usr/bin/env bash
# Upload the master table + consolidated generalist tables to the OBS bucket.
set -e
cd /home/lx/snn
OBSUTIL=/home/lx/obsutil_linux_amd64_5.8.3/obsutil
BUCKET=obs://lixiang01/STJEWM_NMI/aggregate
LOCAL_DIR=/home/lx/snn/results/aggregate

# Each per-suite v0.7.5 result is consolidated into a single file:
#   - MASTER_TABLE.md            (main §1-§11)
#   - generalist_table.md / .json (v0.7.3 4-model pilot, kept for reference)
#   - generalist_master_table.md / .json  (G4+G8+G16 env-SR / collapse-gap /
#                                         responsiveness / divergence, 5-column)
#   - generalist_align_table.md           (G4+G8+G16 event-align ρ)
#   - event_probes_table.md              (G4+G8+G16 event-AUROC)
for fname in \
    MASTER_TABLE.md \
    generalist_table.md generalist_table.json \
    generalist_master_table.md generalist_master_table.json \
    generalist_align_table.md \
    event_probes_table.md; do
    local_path="$LOCAL_DIR/$fname"
    if [ ! -f "$local_path" ]; then
        echo "[skip] $local_path not found"
        continue
    fi
    obs_path="$BUCKET/$fname"
    echo "Uploading: $local_path -> $obs_path"
    $OBSUTIL cp "$local_path" "$obs_path" 2>&1 | tail -3
done
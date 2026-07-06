#!/usr/bin/env bash
# Upload the MASTER_TABLE.md (and the generalist table) to the OBS bucket.
set -e
cd /home/lx/snn
OBSUTIL=/home/lx/obsutil_linux_amd64_5.8.3/obsutil
BUCKET=obs://lixiang01/STJEWM_NMI/aggregate
LOCAL_DIR=/home/lx/snn/results/aggregate

for fname in MASTER_TABLE.md \
             generalist_table.md generalist_table.json \
             generalist_master_table_G4.md generalist_master_table_G4.json \
             generalist_master_table_G8.md generalist_master_table_G8.json \
             generalist_master_table_G16.md generalist_master_table_G16.json \
             generalist_align_table_G4.md \
             generalist_align_table_G8.md \
             generalist_align_table_G16.md \
             event_probes_table_G4.md \
             event_probes_table_G8.md \
             event_probes_table_G16.md; do
    local_path="$LOCAL_DIR/$fname"
    if [ ! -f "$local_path" ]; then
        echo "[skip] $local_path not found"
        continue
    fi
    obs_path="$BUCKET/$fname"
    echo "Uploading: $local_path -> $obs_path"
    $OBSUTIL cp "$local_path" "$obs_path" 2>&1 | tail -3
done

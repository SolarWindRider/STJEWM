#!/usr/bin/env bash
# Upload the MASTER_TABLE.md (and the generalist table) to the OBS bucket.
set -e
cd /home/lx/snn
OBSUTIL=/home/lx/obsutil_linux_amd64_5.8.3/obsutil
BUCKET=obs://lixiang01/STJEWM_NMI/aggregate
LOCAL_DIR=/home/lx/snn/results/aggregate

for fname in MASTER_TABLE.md generalist_table.md generalist_table.json; do
    local_path="$LOCAL_DIR/$fname"
    if [ ! -f "$local_path" ]; then
        echo "[skip] $local_path not found"
        continue
    fi
    obs_path="$BUCKET/$fname"
    echo "Uploading: $local_path -> $obs_path"
    $OBSUTIL cp "$local_path" "$obs_path" 2>&1 | tail -3
done

#!/usr/bin/env bash
# Upload all gifs to OBS bucket
set -e
cd /home/lx/snn
OBSUTIL=/home/lx/obsutil_linux_amd64_5.8.3/obsutil
BUCKET=obs://lixiang01/STJEWM_NMI/gifs
LOCAL_BASE=/home/lx/snn/results/aggregate/gifs

# First, recursive upload of all gifs
find $LOCAL_BASE -name "*.gif" | while read f; do
  rel="${f#$LOCAL_BASE/}"
  obs_path="$BUCKET/$rel"
  echo "Uploading: $f -> $obs_path"
  $OBSUTIL cp "$f" "$obs_path" 2>&1 | tail -1
done

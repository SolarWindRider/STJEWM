#!/bin/bash
# Generic train script. Usage:
#   ./train.sh <model> <env-kind> <data-path> <out-dir> [extra args]
# Example:
#   ./train.sh stjewm reacher_4d /path/to/reacher.npz /path/to/out --epochs 5

set -e

MODEL="${1:-stjewm}"
ENV_KIND="${2:-reacher_4d}"
DATA="${3:?Usage: $0 <model> <env-kind> <data-path> <out-dir>}"
OUT="${4:?Usage: $0 <model> <env-kind> <data-path> <out-dir>}"
shift 4

cd /home/lx/snn
/home/lx/miniconda3/envs/snn/bin/python -m code.train.train \
    --model "$MODEL" \
    --env-kind "$ENV_KIND" \
    --data "$DATA" \
    --out "$OUT" \
    "$@"

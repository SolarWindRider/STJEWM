#!/bin/bash
# Generic eval script. Usage:
#   ./eval.sh <env> <ckpt> <data> <out-json> [extra args]
# Example:
#   ./eval.sh reacher /path/to/ckpt.pt /path/to/data.npz /path/to/eval.json --n-episodes 50

set -e

ENV="${1:?Usage: $0 <env> <ckpt> <data> <out-json>}"
CKPT="${2:?Usage: $0 <env> <ckpt> <data> <out-json>}"
DATA="${3:?Usage: $0 <env> <ckpt> <data> <out-json>}"
OUT="${4:?Usage: $0 <env> <ckpt> <data> <out-json>}"
shift 4

cd /home/lx/snn
/home/lx/miniconda3/envs/snn/bin/python -m code.eval.closed_loop \
    --env "$ENV" \
    --ckpt "$CKPT" \
    --data "$DATA" \
    --out "$OUT" \
    "$@"

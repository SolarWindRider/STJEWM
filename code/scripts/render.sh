#!/bin/bash
# Generic render script. Usage:
#   ./render.sh <env> <ckpt> <data> <out-gif> [extra args]
# Example:
#   ./render.sh reacher /path/to/ckpt.pt /path/to/data.npz /path/to/out.gif

set -e

ENV="${1:?Usage: $0 <env> <ckpt> <data> <out-gif>}"
CKPT="${2:?Usage: $0 <env> <ckpt> <data> <out-gif>}"
DATA="${3:?Usage: $0 <env> <ckpt> <data> <out-gif>}"
OUT="${4:?Usage: $0 <env> <ckpt> <data> <out-gif>}"
shift 4

cd /home/lx/snn
/home/lx/miniconda3/envs/snn/bin/python -m code.eval.plan_then_render \
    --env "$ENV" \
    --ckpt "$CKPT" \
    --data "$DATA" \
    --out "$OUT" \
    "$@"

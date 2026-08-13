#!/bin/bash
# Parallel version of run_probes.sh — splits (env, model, target) work units
# across N parallel workers. Faster on multi-core hosts; same semantics
# otherwise.
#
# Usage:
#   ./run_probes_parallel.sh <suite_name> <n_workers> [n_seeds]

set -e
cd /home/lx/snn

SUITE=${1:-G16}
WORKERS=${2:-2}
N_SEEDS=${3:-1}
case "$SUITE" in
    G4)  SUITE_DIR="generalist" ;;
    G8)  SUITE_DIR="generalist_G8" ;;
    G16) SUITE_DIR="generalist_G16" ;;
    *) echo "Usage: $0 <G4|G8|G16> [n_workers] [n_seeds]"; exit 2 ;;
esac

PROBE_BASE=/home/lx/snn/results/${SUITE_DIR}/probe
ALT_BASE=/home/lx/snn/results/aggregate/event_probes
CKPT_BASE=/home/lx/snn/results/$SUITE_DIR
mkdir -p "$PROBE_BASE" "$ALT_BASE"

echo "[build worklist NUL-delimited]"
/home/lx/miniconda3/envs/snn/bin/python <<PY
import json, os
spec = json.load(open("configs/generalist_probe_eval.json"))
models = ["stjewm_trace_only","stjewm_spike_only","stjewm_rate_only","stjewm_no_trace",
          "stjewm_hidden_leak","stjewm_membrane_readout","alif_timecell_baseline",
          "gru_baseline","lewm_baseline_v2","stacked_lif_trace","stacked_lif_free",
          "mlp_baseline"]
probe_base = "$PROBE_BASE"
ckpt_base = "$CKPT_BASE"
suite = "$SUITE"

with open(f"/tmp/worklist_{suite}.txt", "wb") as f:
    for model in models:
        ckpt = os.path.join(ckpt_base, model, "seed_0", "final.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] {model} ckpt missing", file=__import__("sys").stderr)
            continue
        for entry in spec:
            env_id = entry["env_id"]
            clo_env = entry.get("clo_env") or env_id
            for target in entry["probe_targets"]:
                base = f"{env_id}_{model}_{target}"
                out = os.path.join(probe_base, base + ".json")
                if os.path.exists(out):
                    continue
                line = f"{env_id}|{model}|{target}|{clo_env}|{out}|{ckpt}\n".encode()
                f.write(line)
PY
echo "[worklist size]"
wc -l "/tmp/worklist_${SUITE}.txt"

echo "[parallel probe x $WORKERS]"
xargs -d '\n' -P "$WORKERS" -I {} bash -c '
    IFS="|" read -r env_id model target clo_env out ckpt <<< "{}"
    echo "[probe] $env_id $model $target"
    timeout 300 /home/lx/miniconda3/envs/snn/bin/python -m code.scripts.probe \
        --env "$clo_env" --model "$model" \
        --ckpt "$ckpt" \
        --probe-target "$target" \
        --pad-obs-to 128 --action-dim-eval 56 \
        --max-windows 200 \
        --out "$out" 2>&1
    if [ -f "$out" ]; then
        mkdir -p "'"$ALT_BASE"'"
        cp "$out" "'"$ALT_BASE"'/$(basename "$out")"
    fi
' < "/tmp/worklist_${SUITE}.txt"

echo "[run_probes_parallel] done for $SUITE"

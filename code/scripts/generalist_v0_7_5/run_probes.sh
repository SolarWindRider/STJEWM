#!/bin/bash
# Run linear-probe event-AUROC for every (model, env, target) cell.
#
# Usage:
#   ./run_probes.sh <suite_name> [n_seeds]
#
# Reads configs/generalist_probe_eval.json for the env list and per-env
# probe targets. SUITE selects which trained ckpts to load:
#   G4  -> results/generalist/<model>/seed_<s>/final.pt
#   G8  -> results/generalist_G8/<model>/seed_<s>/final.pt
#   G16 -> results/generalist_G16/<model>/seed_<s>/final.pt
#
# Output goes to results/probe/<env>_<model>_<target>.json plus a
# mirror at results/aggregate/event_probes/ so both aggregators pick
# them up.
set -e
cd /home/lx/snn

SUITE=${1:-G16}
N_SEEDS=${2:-3}
case "$SUITE" in
    G4)  SUITE_DIR="generalist" ;;
    G8)  SUITE_DIR="generalist_G8" ;;
    G16) SUITE_DIR="generalist_G16" ;;
    *) echo "Usage: $0 <G4|G8|G16> [n_seeds]"; exit 2 ;;
esac

MODELS=(
    stjewm_trace_only
    stjewm_spike_only
    stjewm_rate_only
    stjewm_no_trace
    stjewm_hidden_leak
    stjewm_membrane_readout
    alif_timecell_baseline
    gru_baseline
    lewm_baseline_v2
    stacked_lif_trace
    stacked_lif_free
    mlp_baseline
)

PROBE_BASE=/home/lx/snn/results/${SUITE_DIR}/probe
ALT_BASE=/home/lx/snn/results/aggregate/event_probes
CKPT_BASE=/home/lx/snn/results/$SUITE_DIR
mkdir -p "$PROBE_BASE" "$ALT_BASE"

/home/lx/miniconda3/envs/snn/bin/python - <<PY
import json, os, subprocess, sys
spec = json.loads(open("configs/generalist_probe_eval.json").read())
models = ["stjewm_trace_only","stjewm_spike_only","stjewm_rate_only","stjewm_no_trace",
          "stjewm_hidden_leak","stjewm_membrane_readout","alif_timecell_baseline",
          "gru_baseline","lewm_baseline_v2","stacked_lif_trace","stacked_lif_free",
          "mlp_baseline"]
n_seeds = int(os.environ.get("N_SEEDS", "3"))
pad = 128
action_dim = 56
SUITE = "$SUITE"   # mirror shell var into python scope
suite = "$SUITE"
probe_base = "$PROBE_BASE"
alt_base = "$ALT_BASE"
ckpt_base = "$CKPT_BASE"

for model in models:
    for seed in range(n_seeds):
        ckpt = os.path.join(ckpt_base, model, f"seed_{seed}", "final.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] {model}/seed_{seed} (no ckpt)", flush=True)
            continue
        for entry in spec:
            env_id = entry["env_id"]
            clo_env = entry.get("clo_env") or env_id
            for target in entry["probe_targets"]:
                base = f"{env_id}_{model}_{target}"
                out = os.path.join(probe_base, base + ".json")
                # Path-aware skip: skip if the suite-specific output already exists
                if os.path.exists(out):
                    print(f"[skip] {SUITE} {base}", flush=True)
                    continue
                cmd = [
                    "/home/lx/miniconda3/envs/snn/bin/python",
                    "-m", "code.scripts.probe",
                    "--env", clo_env,
                    "--model", model,
                    "--ckpt", ckpt,
                    "--probe-target", target,
                    "--pad-obs-to", str(pad),
                    "--action-dim-eval", str(action_dim),
                    "--out", out,
                    "--max-windows", "200",
                ]
                print(f"[probe] {SUITE} {model} seed={seed} {env_id} {target}", flush=True)
                # pathological (model, env) pair can spin indefinitely
                # (e.g. stjewm_rate_only × tworoom_long × event_high_motion
                # in v0.7.5). We hard-cap each cell to keep the
                # suite-level pipeline making forward progress.
                try:
                    rc = subprocess.call(cmd, timeout=300)
                except subprocess.TimeoutExpired:
                    print(f"[WARN] probe {SUITE}/{model}/{env_id}/{target} "
                          f"timed out at 5min; writing skip-stub", file=sys.stderr, flush=True)
                    with open(out, "w") as f:
                        f.write('{"skipped": true, "reason": "timeout 5min"}')
                    rc = -1
                if rc != 0:
                    print(f"[WARN] probe for {SUITE}/{model}/{env_id}/{target} exited rc={rc}", file=sys.stderr, flush=True)
                if os.path.exists(out):
                    os.makedirs(alt_base, exist_ok=True)
                    with open(out) as src, open(os.path.join(alt_base, base + ".json"), "w") as dst:
                        dst.write(src.read())
PY

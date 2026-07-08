#!/bin/bash
# Run linear-probe event-AUROC on the 4 stress envs (pusht_ood /
# tworoom_long / cartpole_flicker / cheetah_velhidden) for every
# (model, env, target) cell, per generalist suite.
#
# Usage:
#   ./run_stress_probes.sh <suite_name> [n_seeds]
#
# Reads configs/generalist_stress_probe_eval.json for env list and
# per-env probe targets. SUITE selects which trained ckpts to load
# (G4/G8/G16). Outputs land in
# results/<suite>/probe/<env>_<model>_<target>.json and mirror at
# results/aggregate/event_probes/ for the aggregator.
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
    cubifae_baseline
    gru_baseline
    lewm_baseline_v2
    slt_lif_mpc_trace
    slt_lif_mpc_free
    mlp_baseline
)

PROBE_BASE=/home/lx/snn/results/${SUITE_DIR}/stress_probe
ALT_BASE=/home/lx/snn/results/aggregate/event_probes
CKPT_BASE=/home/lx/snn/results/$SUITE_DIR
mkdir -p "$PROBE_BASE" "$ALT_BASE"

/home/lx/miniconda3/envs/snn/bin/python - <<PY
import json, os, subprocess, sys
spec = json.loads(open("configs/generalist_stress_probe_eval.json").read())
models = ["stjewm_trace_only","stjewm_spike_only","stjewm_rate_only","stjewm_no_trace",
          "stjewm_hidden_leak","stjewm_membrane_readout","cubifae_baseline",
          "gru_baseline","lewm_baseline_v2","slt_lif_mpc_trace","slt_lif_mpc_free",
          "mlp_baseline"]
n_seeds = int(os.environ.get("N_SEEDS", "3"))
pad = 128
action_dim = 56
probe_base = "$PROBE_BASE"
alt_base = "$ALT_BASE"
ckpt_base = "$CKPT_BASE"
suite = "$SUITE"

for model in models:
    for seed in range(n_seeds):
        ckpt = os.path.join(ckpt_base, model, f"seed_{seed}", "final.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] {model}/seed_{seed} (no ckpt)", flush=True)
            continue
        for entry in spec:
            env_id = entry["env_id"]
            clo_env = entry.get("clo_env") or env_id
            extra = entry.get("extra_flags") or []
            for target in entry["probe_targets"]:
                base = f"{env_id}_{model}_{target}"
                out = os.path.join(probe_base, base + ".json")
                if os.path.exists(out):
                    print(f"[skip] {suite} {base}", flush=True)
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
                    "--max-windows", "1000",
                    *extra,
                ]
                print(f"[probe] {suite} {model} seed={seed} {env_id} {target}", flush=True)
                rc = subprocess.call(cmd)
                if rc != 0:
                    print(f"[WARN] probe for {suite}/{model}/{env_id}/{target} exited rc={rc}", file=sys.stderr, flush=True)
                if os.path.exists(out):
                    os.makedirs(alt_base, exist_ok=True)
                    with open(out) as src, open(os.path.join(alt_base, base + ".json"), "w") as dst:
                        dst.write(src.read())
PY

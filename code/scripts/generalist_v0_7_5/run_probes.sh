#!/bin/bash
# Run linear-probe event-AUROC for every (model, env, target) cell.
#
# Usage:
#   ./run_probes.sh [n_seeds]
#
# Reads configs/generalist_probe_eval.json for the env list and per-env
# probe targets. For each generalist model and seed, writes one JSON
# per (env, target) to results/probe/<env>_<model>_<target>.json AND
# results/aggregate/event_probes/<env>_<model>_<target>.json (mirrored
# so both aggregators pick them up).
set -e
cd /home/lx/snn

N_SEEDS=${1:-3}
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

PROBE_BASE=/home/lx/snn/results/probe
ALT_BASE=/home/lx/snn/results/aggregate/event_probes
CKPT_BASE=/home/lx/snn/results/generalist
mkdir -p "$PROBE_BASE" "$ALT_BASE"

/home/lx/miniconda3/envs/snn/bin/python - <<'PY'
import json, os, subprocess, sys
spec = json.loads(open("configs/generalist_probe_eval.json").read())
models = ["stjewm_trace_only","stjewm_spike_only","stjewm_rate_only","stjewm_no_trace",
          "stjewm_hidden_leak","stjewm_membrane_readout","cubifae_baseline",
          "gru_baseline","lewm_baseline_v2","slt_lif_mpc_trace","slt_lif_mpc_free",
          "mlp_baseline"]
n_seeds = int(os.environ.get("N_SEEDS", "3"))
pad = 128
action_dim = 56
probe_base = "/home/lx/snn/results/probe"
alt_base = "/home/lx/snn/results/aggregate/event_probes"
ckpt_base = "/home/lx/snn/results/generalist"

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
                if os.path.exists(out):
                    print(f"[skip] {base}", flush=True)
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
                ]
                print(f"[probe] {model} seed={seed} {env_id} {target}", flush=True)
                rc = subprocess.call(cmd)
                if rc != 0:
                    print(f"[WARN] probe for {model}/{env_id}/{target} exited rc={rc}", file=sys.stderr, flush=True)
                # Mirror to the alt dir so aggregate_event_probes.py picks it up.
                if os.path.exists(out):
                    os.makedirs(alt_base, exist_ok=True)
                    with open(out) as src, open(os.path.join(alt_base, base + ".json"), "w") as dst:
                        dst.write(src.read())
PY
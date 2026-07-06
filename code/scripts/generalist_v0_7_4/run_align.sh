#!/bin/bash
# Run event-align ρ for every (model, dmc_env) cell.
#
# Usage:
#   ./run_align.sh [n_seeds]
#
# 6 DMC envs × 12 models × N_SEEDS = up to 216 runs. Writes
# results/event_align/<env>_<model>.json (Stage 5 will append
# seed_<s> disambiguation).
set -e
cd /home/lx/snn

N_SEEDS=${1:-3}
DMC_ENVS=(cheetah walker cartpole_2d pendulum_2d finger ball_in_cup)
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

ALIGN_BASE=/home/lx/snn/results/event_align
CKPT_BASE=/home/lx/snn/results/generalist
mkdir -p "$ALIGN_BASE"

/home/lx/miniconda3/envs/snn/bin/python - <<'PY'
import json, os, subprocess, sys
dmc_envs = ["cheetah", "walker", "cartpole_2d", "pendulum_2d", "finger", "ball_in_cup"]
models = ["stjewm_trace_only","stjewm_spike_only","stjewm_rate_only","stjewm_no_trace",
          "stjewm_hidden_leak","stjewm_membrane_readout","cubifae_baseline",
          "gru_baseline","lewm_baseline_v2","slt_lif_mpc_trace","slt_lif_mpc_free",
          "mlp_baseline"]
n_seeds = int(os.environ.get("N_SEEDS", "3"))
pad = 128
action_dim = 56
align_base = "/home/lx/snn/results/event_align"
ckpt_base = "/home/lx/snn/results/generalist"

# Map dmc env names to closed_loop arg names (event_align.py uses the same
# mapping internally).
clo_env_map = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum"}

for model in models:
    for seed in range(n_seeds):
        ckpt = os.path.join(ckpt_base, model, f"seed_{seed}", "final.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] {model}/seed_{seed} (no ckpt)", flush=True)
            continue
        for env in dmc_envs:
            base = f"{env}_{model}_seed{seed}"
            out = os.path.join(align_base, base + ".json")
            if os.path.exists(out):
                print(f"[skip] {base}", flush=True)
                continue
            cmd = [
                "/home/lx/miniconda3/envs/snn/bin/python",
                "-m", "code.scripts.event_align",
                "--env", env,
                "--model", model,
                "--ckpt", ckpt,
                "--out", out,
                "--n-steps", "100",
                "--pad-obs-to", str(pad),
                "--action-dim-eval", str(action_dim),
            ]
            print(f"[align] {model} seed={seed} {env}", flush=True)
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"[WARN] align for {model}/{env}/seed={seed} exited rc={rc}", file=sys.stderr, flush=True)
PY
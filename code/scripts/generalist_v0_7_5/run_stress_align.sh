#!/bin/bash
# Run event-align ρ on the 4 stress envs (pusht_ood / tworoom_long
# / cartpole_flicker / cheetah_velhidden) for every generalist
# (model, env) cell.
#
# Usage:
#   ./run_stress_align.sh <suite_name> [n_seeds]
#
# SUITE selects which trained ckpts to load:
#   G4  -> results/generalist/<model>/seed_<s>/final.pt
#   G8  -> results/generalist_G8/<model>/seed_<s>/final.pt
#   G16 -> results/generalist_G16/<model>/seed_<s>/final.pt
#
# Output goes to results/<suite>/stress_align/<env>_<model>_seed<s>.json.
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

STRESS_ENVS=(pusht_ood tworoom_long cartpole_flicker cheetah_velhidden)
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

ALIGN_BASE=/home/lx/snn/results/${SUITE_DIR}/stress_align
CKPT_BASE=/home/lx/snn/results/$SUITE_DIR
mkdir -p "$ALIGN_BASE"

/home/lx/miniconda3/envs/snn/bin/python - <<PY
import os, subprocess, sys
stress_envs = ["pusht_ood", "tworoom_long", "cartpole_flicker", "cheetah_velhidden"]
models = ["stjewm_trace_only","stjewm_spike_only","stjewm_rate_only","stjewm_no_trace",
          "stjewm_hidden_leak","stjewm_membrane_readout","alif_timecell_baseline",
          "gru_baseline","lewm_baseline_v2","stacked_lif_trace","stacked_lif_free",
          "mlp_baseline"]
n_seeds = int(os.environ.get("N_SEEDS", "3"))
pad = 128
action_dim = 56
align_base = "$ALIGN_BASE"
ckpt_base = "$CKPT_BASE"
suite = "$SUITE"

for model in models:
    for seed in range(n_seeds):
        ckpt = os.path.join(ckpt_base, model, f"seed_{seed}", "final.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] {model}/seed_{seed} (no ckpt)", flush=True)
            continue
        for env in stress_envs:
            base = f"{env}_{model}_seed{seed}"
            out = os.path.join(align_base, base + ".json")
            if os.path.exists(out):
                print(f"[skip] {suite} {base}", flush=True)
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
            print(f"[align] {suite} {model} seed={seed} {env}", flush=True)
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"[WARN] align for {suite}/{model}/{env}/seed={seed} exited rc={rc}", file=sys.stderr, flush=True)
PY

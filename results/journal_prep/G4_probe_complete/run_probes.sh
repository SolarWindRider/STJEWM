#!/bin/bash
# G4: Fill the gap from B3 — re-run the B3-fixed probe for the 9 missing
# models on the same env set, so we can produce a complete 13-model R² table.
#
# B3 covered: stjewm_trace_only, stjewm_spike_only, lewm_baseline_v2,
#             mlp_baseline  (4 models)
# G4 adds:    stjewm_rate_only, stjewm_no_trace, stjewm_hidden_leak,
#             stjewm_membrane_readout, cubifae_baseline, slt_lif_mpc_trace,
#             slt_lif_mpc_free, gru_baseline, spikedreamer_baseline
#             (9 models)
# Total: 13 models.
#
# Pattern mirrors B3's run_probes.sh exactly:
#   --pad-obs-to 128 --action-dim-eval 56 --max-windows 200
#
# Env list mirrors B3 exactly:
#   finger fish stacker humanoid cartpole_2d pendulum_2d cheetah walker
#   hopper quadruped
set -uo pipefail
cd /home/lx/snn

OUT_DIR=results/journal_prep/G4_probe_complete/probes
LOG_DIR=results/journal_prep/G4_probe_complete/logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

ENVS=(finger fish stacker humanoid cartpole_2d pendulum_2d cheetah walker hopper quadruped)
MODELS=(stjewm_rate_only stjewm_no_trace stjewm_hidden_leak stjewm_membrane_readout \
        cubifae_baseline slt_lif_mpc_trace slt_lif_mpc_free \
        gru_baseline spikedreamer_baseline)
TARGETS=(position velocity future_k goal_direction contact)

for env in "${ENVS[@]}"; do
  for model in "${MODELS[@]}"; do
    ckpt="results/5m/cross_benchmark_F1/${model}/seed_0/final.pt"
    [[ ! -f "$ckpt" ]] && { echo "[skip] no ckpt: $ckpt"; continue; }
    for tgt in "${TARGETS[@]}"; do
      out="${OUT_DIR}/${env}_${model}_${tgt}.json"
      log="${LOG_DIR}/${env}_${model}_${tgt}.log"
      [[ -f "$out" ]] && { echo "[cached] $out"; continue; }
      echo "[probe] $env / $model / $tgt"
      /home/lx/miniconda3/envs/snn/bin/python -u -m code.scripts.probe \
        --env "$env" --model "$model" --ckpt "$ckpt" \
        --probe-target "$tgt" \
        --pad-obs-to 128 --action-dim-eval 56 \
        --max-windows 200 \
        --out "$out" >"$log" 2>&1
    done
  done
done
echo "done."

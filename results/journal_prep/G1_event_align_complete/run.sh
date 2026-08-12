#!/usr/bin/env bash
set -u
PY=/home/lx/miniconda3/envs/snn/bin/python
export PYTHONPATH=/home/lx/snn
OUT=/home/lx/snn/results/journal_prep/G1_event_align_complete/raw
SHARD=${1:?shard}
GPU=${2:?gpu}
MODELS=(stjewm_no_trace stjewm_hidden_leak alif_timecell_baseline stacked_lif_free lif_transformer_baseline gru_baseline mlp_baseline)
SPLITS=(cross_benchmark_F1 generalist_16env)
ENVS=(cheetah ball_in_cup pendulum_2d finger)
i=0
for split in "${SPLITS[@]}"; do
  for env in "${ENVS[@]}"; do
    for model in "${MODELS[@]}"; do
      if [[ "$model" == gru_baseline || "$model" == mlp_baseline ]]; then
        [[ "$env" == cheetah ]] && continue
      fi
      if (( i % 4 == SHARD )); then
        out="$OUT/$split/$env/$model.json"
        mkdir -p "$(dirname "$out")"
        echo "[run] gpu=$GPU $split/$env/$model"
        "$PY" -m code.scripts.event_align --env "$env" --model "$model" \
          --ckpt "/home/lx/snn/results/5m/$split/$model/seed_0/final.pt" \
          --pad-obs-to 128 --action-dim-eval 56 --n-steps 200 --n-resets 2 \
          --device "cuda:$GPU" --out "$out"
      fi
      ((i+=1))
    done
  done
done

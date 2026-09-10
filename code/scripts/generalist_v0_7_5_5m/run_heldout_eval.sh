#!/bin/bash
# run_heldout_eval.sh — held-out evals (§3.1/3.2): 9 heldout_eval specs x 13 models.
# STJEWM<-5m_5mpar, baselines<-5m. OUT: /data/lx/tmp/results/heldout
set -u
cd /home/lx/snn
PY=/home/lx/miniconda3/envs/snn/bin/python
MODELS_STJ="stjewm_trace_only stjewm_spike_only stjewm_rate_only stjewm_no_trace stjewm_hidden_leak stjewm_membrane_readout"
MODELS_BASE="alif_timecell_baseline gru_baseline lewm_baseline_v2 stacked_lif_trace stacked_lif_free mlp_baseline lif_transformer_baseline"
SPECS="oodc_F1 oodc_F2 oodc_F3 oodc_F1F2 oodc_F1F3 oodc_F2F3 cross_benchmark_F1 cross_benchmark_F2 cross_benchmark_F3"

JOBS=/tmp/heldout_jobs.tsv
> $JOBS
for sp in $SPECS; do
  for m in $MODELS_STJ; do
    ck=/data/lx/tmp/results/5m_5mpar/$sp/$m/seed_0/final.pt
    out=/data/lx/tmp/results/heldout/$sp/$m/seed_0
    mkdir -p "$out"
    ls "$out"/eval_*.json >/dev/null 2>&1 && n=$(ls "$out"/eval_*.json | wc -l) || n=0
    spec_n=$($PY -c "import json;d=json.load(open('configs/heldout_eval/$sp.json'));print(len(d['specs']) if isinstance(d,dict) else len(d))")
    [ "$n" -ge "$spec_n" ] && continue
    echo -e "$m\t$ck\tconfigs/heldout_eval/$sp.json\t$sp" >> $JOBS
  done
  for m in $MODELS_BASE; do
    ck=/data/lx/tmp/results/5m/$sp/$m/seed_0/final.pt
    out=/data/lx/tmp/results/heldout/$sp/$m/seed_0
    mkdir -p "$out"
    ls "$out"/eval_*.json >/dev/null 2>&1 && n=$(ls "$out"/eval_*.json | wc -l) || n=0
    spec_n=$($PY -c "import json;d=json.load(open('configs/heldout_eval/$sp.json'));print(len(d['specs']) if isinstance(d,dict) else len(d))")
    [ "$n" -ge "$spec_n" ] && continue
    echo -e "$m\t$ck\tconfigs/heldout_eval/$sp.json\t$sp" >> $JOBS
  done
done
echo "heldout jobs: $(wc -l < $JOBS)"

rm -f /tmp/heldout_w{0,1,2,3}.tsv
i=0
while IFS=$'\t' read -r line; do
  echo "$line" >> /tmp/heldout_w$((i % 4)).tsv; i=$((i + 1))
done < $JOBS

worker () {
  local gpu=$1
  while IFS=$'\t' read -r model ckpt spec sp; do
    CUDA_VISIBLE_DEVICES=$gpu OUT_PARENT=/data/lx/tmp/results/heldout N_SEEDS=1 \
      bash code/scripts/generalist_v0_7_5_5m/eval_one.sh "$model" "$ckpt" "$spec" 0 \
      >> /data/lx/tmp/logs/heldout_eval.log 2>&1
    echo "done heldout $sp $model rc=$?" >> /data/lx/tmp/logs/heldout_progress.log
  done < /tmp/heldout_w$gpu.tsv
  echo "W$gpu DONE" >> /data/lx/tmp/logs/heldout_progress.log
}
for g in 0 1 2 3; do worker $g & done
wait
echo HELDOUT_ALL_DONE >> /data/lx/tmp/logs/heldout_progress.log

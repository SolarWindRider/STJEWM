#!/bin/bash
# G5 multi-seed launcher: extend B2's 5-model 3-seed coverage to the full 13 models.
# Trains 8 missing models (stjewm_rate_only, stjewm_no_trace, stjewm_hidden_leak,
# stjewm_membrane_readout, alif_timecell_baseline, stacked_lif_free, gru_baseline,
# lif_transformer_baseline) at seeds {1, 2} across 3 splits (cross_benchmark_F1,
# oodc_F2, generalist_16env). 8 models x 3 splits x 2 seeds = 48 ckpts.
#
# Same protocol as B2_multiseed_launcher.sh (state 5M, --n-layers 2:
# only stjewm uses it; other models have fixed internal n_layers).
#
# Usage:
#   bash code/scripts/generalist_v0_7_5_5m/G5_multiseed_launcher.sh
#
# Output:
#   - results/5m_seed1/<split>/<model>/seed_0/final.pt
#   - results/5m_seed2/<split>/<model>/seed_0/final.pt
#   - per-job logs in results/journal_prep/G5_multiseed/_logs/

set -e
cd /home/lx/snn

LOG_DIR=/home/lx/snn/results/journal_prep/G5_multiseed/_logs
mkdir -p "$LOG_DIR"

# Detach orchestrator from parent shell so it survives interactive exit
if [[ "${G5_NOSETSID:-0}" != "1" ]]; then
  exec setsid /home/lx/miniconda3/envs/snn/bin/python - <<'PYEOF'
import os, sys, json, time, subprocess, traceback, multiprocessing
from pathlib import Path

ROOT = Path("/home/lx/snn")
LOG_DIR = ROOT / "results/journal_prep/G5_multiseed/_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# The 8 missing models (B2 already covered the other 5).
MODELS = [
    "stjewm_rate_only",
    "stjewm_no_trace",
    "stjewm_hidden_leak",
    "stjewm_membrane_readout",
    "alif_timecell_baseline",
    "stacked_lif_free",
    "gru_baseline",
    "lif_transformer_baseline",
]
MAP = {
    "stjewm_trace_only":       ("stjewm",            "trace_only"),
    "stjewm_spike_only":       ("stjewm",            "spike_only"),
    "stjewm_rate_only":        ("stjewm",            "rate_only"),
    "stjewm_no_trace":         ("stjewm",            "no_trace"),
    "stjewm_hidden_leak":      ("stjewm",            "hidden_leak"),
    "stjewm_membrane_readout": ("stjewm",            "membrane_readout"),
    "stjewm_raw_spike":        ("stjewm",            "raw_spike"),
    "alif_timecell_baseline":        ("alif_timecell_baseline",  ""),
    "gru_baseline":            ("gru_baseline",      ""),
    "lewm_baseline_v2":        ("lewm_baseline",     "hidden_leak"),
    "stacked_lif_trace":       ("stacked_lif_trace", ""),
    "stacked_lif_free":        ("stacked_lif_free",  ""),
    "mlp_baseline":            ("mlp_baseline",      ""),
    "lif_transformer_baseline":   ("lif_transformer_baseline", ""),
}
SPLITS = ["configs/oodc_5m/cross_benchmark_F1.json",
          "configs/oodc_5m/oodc_F2.json",
          "configs/oodc_5m/generalist_16env.json"]
SEEDS = [1, 2]

jobs = []
for spec in SPLITS:
    sp = ROOT / spec
    d = json.load(open(sp))
    sn = sp.stem if isinstance(d, list) else (d.get("_split_name") or d.get("split_name") or sp.stem)
    for seed in SEEDS:
        for mk in MODELS:
            od = ROOT / f"results/5m_seed{seed}/{sn}/{mk}/seed_0"
            ck = od / "final.pt"
            if ck.exists():
                continue
            m, r = MAP[mk]
            jobs.append({
                "model_kind": mk, "model": m, "readout": r,
                "spec": spec, "split_name": sn, "seed": seed,
                "out_dir": str(od),
            })

print(f"[G5] jobs to train: {len(jobs)}", flush=True)
if not jobs:
    sys.exit(0)

gpu_jobs = {0: [], 1: [], 2: [], 3: []}
for i, j in enumerate(jobs):
    gpu_jobs[i % 4].append(j)
print("[G5] per-gpu: " + ", ".join(f"gpu{g}={len(gpu_jobs[g])}" for g in range(4)), flush=True)

# Per-job completion timestamps (for retryability)
for g in range(4):
    for j in gpu_jobs[g]:
        Path(j["out_dir"]).mkdir(parents=True, exist_ok=True)

def run_one(gpu, job):
    log = LOG_DIR / f"{job['split_name']}_{job['model_kind']}_seed{job['seed']}.train.log"
    worker_log = LOG_DIR / f"gpu{gpu}.worker.log"
    cmd = [
        "/home/lx/miniconda3/envs/snn/bin/python",
        "-m", "code.train.train",
        "--model", job["model"],
        "--multi-env-spec", job["spec"],
        "--pad-obs-to", "128",
        "--action-dim", "56",
        "--embed-dim", "192",
        "--image-size", "0",
        "--n-layers", "2",
        "--epochs", "1",
        "--batch", "32",
        "--lr", "3e-4",
        "--history-size", "1",
        "--goal-offset", "25",
        "--seed", str(job["seed"]),
        "--out", job["out_dir"],
    ]
    if job["readout"]:
        cmd += ["--readout-mode", job["readout"]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = "/home/lx/snn"
    with worker_log.open("a") as wf:
        ts = time.strftime("%H:%M:%S")
        wf.write(f"[{ts}] [gpu{gpu}] START {job['model_kind']}({job['model']},{job['readout']}) {job['split_name']} seed={job['seed']}\n")
    print(f"[gpu{gpu}@{time.strftime('%H:%M:%S')}] START {job['model_kind']}({job['model']},{job['readout']}) {job['split_name']} seed={job['seed']}", flush=True)
    t0 = time.time()
    with open(log, "wb") as outf:
        p = subprocess.Popen(cmd, stdout=outf, stderr=subprocess.STDOUT, env=env, cwd="/home/lx/snn")
        rc = p.wait()
    dt = time.time() - t0
    with worker_log.open("a") as wf:
        ts = time.strftime("%H:%M:%S")
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        wf.write(f"[{ts}] [gpu{gpu}] {status} {job['model_kind']} {job['split_name']} seed={job['seed']} ({dt:.0f}s)\n")
    print(f"[gpu{gpu}@{ts}] {status} {job['model_kind']} {job['split_name']} seed={job['seed']} ({dt:.0f}s)", flush=True)
    return rc

def gpu_worker(gpu, queue):
    for j in queue:
        try:
            run_one(gpu, j)
        except Exception:
            traceback.print_exc()

procs = []
for gpu in range(4):
    p = multiprocessing.Process(target=gpu_worker, args=(gpu, gpu_jobs[gpu]))
    p.start()
    procs.append(p)

while any(p.is_alive() for p in procs):
    time.sleep(60)
    done_count = sum(1 for g in range(4) for j in gpu_jobs[g] if (Path(j["out_dir"]) / "final.pt").exists())
    print(f"[G5] {done_count}/{len(jobs)} ckpts on disk", flush=True)

for p in procs:
    p.join()
print(f"[G5] orchestrator done at {time.strftime('%H:%M:%S')}", flush=True)
PYEOF
fi
echo "[G5] launcher script done: $(date -Iseconds)"

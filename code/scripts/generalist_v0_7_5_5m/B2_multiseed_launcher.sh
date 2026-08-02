#!/bin/bash
# B2 multi-seed launcher (v7): pure-python orchestrator, launched via setsid
# so the orchestrator survives parent-shell exit. Per-GPU serial python workers.
# Usage:
#   bash code/scripts/generalist_v0_7_5_5m/B2_multiseed_launcher.sh
#
# Output: jobs.tsv, gpu{N}.worker.log entries, training logs in
#         results/journal_prep/B2_multiseed/_logs/

set -e
cd /home/lx/snn

LOG_DIR=/home/lx/snn/results/journal_prep/B2_multiseed/_logs
mkdir -p "$LOG_DIR"

/home/lx/miniconda3/envs/snn/bin/python - <<'PYEOF'
import os, sys, json, time, subprocess, traceback, threading, atexit, signal
from pathlib import Path

ROOT = Path("/home/lx/snn")
LOG_DIR = ROOT / "results/journal_prep/B2_multiseed/_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["stjewm_trace_only","stjewm_spike_only","slt_lif_mpc_trace","lewm_baseline_v2","mlp_baseline"]
MAP = {
    "stjewm_trace_only":       ("stjewm",            "trace_only"),
    "stjewm_spike_only":       ("stjewm",            "spike_only"),
    "stjewm_rate_only":        ("stjewm",            "rate_only"),
    "stjewm_no_trace":         ("stjewm",            "no_trace"),
    "stjewm_hidden_leak":      ("stjewm",            "hidden_leak"),
    "stjewm_membrane_readout": ("stjewm",            "membrane_readout"),
    "stjewm_raw_spike":        ("stjewm",            "raw_spike"),
    "cubifae_baseline":        ("cubifae_baseline",  ""),
    "gru_baseline":            ("gru_baseline",      ""),
    "lewm_baseline_v2":        ("lewm_baseline",     "hidden_leak"),
    "slt_lif_mpc_trace":       ("slt_lif_mpc_trace", ""),
    "slt_lif_mpc_free":        ("slt_lif_mpc_free",  ""),
    "mlp_baseline":            ("mlp_baseline",      ""),
    "spikedreamer_baseline":   ("spikedreamer_baseline", ""),
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

print(f"[B2] jobs to train: {len(jobs)}", flush=True)
if not jobs:
    sys.exit(0)

gpu_jobs = {0: [], 1: [], 2: [], 3: []}
for i, j in enumerate(jobs):
    gpu_jobs[i % 4].append(j)
print("[B2] per-gpu: " + ", ".join(f"gpu{g}={len(gpu_jobs[g])}" for g in range(4)), flush=True)

# --- Worker process management with explicit proc tracking on disk ---
# Use atexit + signals to clean up child PIDs on interpreter exit.
active_procs = []
def cleanup():
    for p in active_procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
atexit.register(cleanup)

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
        active_procs.append(p)
        rc = p.wait()
        try: active_procs.remove(p)
        except ValueError: pass
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

# Use multiprocessing so a signal in one GPU doesn't kill others.
import multiprocessing
procs = []
for gpu in range(4):
    p = multiprocessing.Process(target=gpu_worker, args=(gpu, gpu_jobs[gpu]))
    p.start()
    procs.append(p)

# Wait for all (no parent killing issue: multiprocessing children are detached)
while any(p.is_alive() for p in procs):
    time.sleep(30)
    done_count = sum(1 for g in range(4) for j in gpu_jobs[g] if (Path(j["out_dir"]) / "final.pt").exists())
    print(f"[B2] {done_count}/{len(jobs)} ckpts on disk", flush=True)

for p in procs:
    p.join()
print(f"[B2] orchestrator done at {time.strftime('%H:%M:%S')}", flush=True)
PYEOF
echo "[B2] launcher script done: $(date -Iseconds)"

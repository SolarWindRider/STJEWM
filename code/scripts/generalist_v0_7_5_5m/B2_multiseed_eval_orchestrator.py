#!/usr/bin/env python3
"""B2 eval orchestrator: per-(split, model, seed) eval_one run on 4 GPUs (round-robin).
Launches eval_one.sh which iterates per env in the spec.

Uses multiprocessing for 4 GPU workers (each runs its assigned queue serially).
"""
import os, sys, json, time, subprocess, multiprocessing, traceback
from pathlib import Path

ROOT = Path("/home/lx/snn")
LOG_DIR = ROOT / "results/journal_prep/B2_multiseed/_eval_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["stjewm_trace_only","stjewm_spike_only","slt_lif_mpc_trace","lewm_baseline_v2","mlp_baseline"]
SPLITS = ["configs/oodc_5m/cross_benchmark_F1.json",
          "configs/oodc_5m/oodc_F2.json",
          "configs/oodc_5m/generalist_16env.json"]
SEEDS = [0, 1, 2]
SEED_DIRS = {0: "results/5m", 1: "results/5m_seed1", 2: "results/5m_seed2"}

eval_jobs = []
for spec in SPLITS:
    sp = ROOT / spec
    d = json.load(open(sp))
    sn = sp.stem if isinstance(d, list) else (d.get("_split_name") or d.get("split_name") or sp.stem)
    for seed in SEEDS:
        for mk in MODELS:
            ckpt_dir = ROOT / SEED_DIRS[seed] / sn / mk / "seed_0"
            ckpt = ckpt_dir / "final.pt"
            if not ckpt.exists():
                print(f"[eval] SKIP: ckpt missing {ckpt}", flush=True)
                continue
            eval_jobs.append({
                "model_kind": mk, "seed": seed, "seed_dir": SEED_DIRS[seed],
                "spec": spec, "split_name": sn, "ckpt": str(ckpt),
                "out_dir": str(ckpt_dir),
            })

print(f"[B2-eval] jobs to evaluate: {len(eval_jobs)}", flush=True)
if not eval_jobs:
    sys.exit(0)

# Exclude seed=0 (already evaluated) - we only need to evaluate seed=1,2
# But for completeness include seed=0 too if requested (skip if eval JSON exists)
# Always skip if all expected eval JSONs exist
def all_evals_done(out_dir, spec_path):
    sp = Path(spec_path)
    d = json.load(open(sp))
    spec = d.get("specs") if isinstance(d, dict) else d
    sn = sp.stem if isinstance(d, list) else (d.get("_split_name") or d.get("split_name") or sp.stem)
    expected = {f"eval_{e['env_id']}.json" for e in spec}
    existing = set(os.listdir(out_dir)) if Path(out_dir).exists() else set()
    return expected.issubset(existing)

new_jobs = []
for j in eval_jobs:
    if all_evals_done(j["out_dir"], j["spec"]):
        continue
    new_jobs.append(j)

print(f"[B2-eval] jobs needing eval: {len(new_jobs)}", flush=True)

gpu_jobs = {0: [], 1: [], 2: [], 3: []}
for i, j in enumerate(new_jobs):
    gpu_jobs[i % 4].append(j)
print("[B2-eval] per-gpu: " + ", ".join(f"gpu{g}={len(gpu_jobs[g])}" for g in range(4)), flush=True)

def run_eval(gpu, job):
    log = LOG_DIR / f"{job['split_name']}_{job['model_kind']}_seed{job['seed']}.eval.log"
    worker_log = LOG_DIR / f"gpu{gpu}.worker.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    # B2_multiseed_eval.sh uses env OUT_PARENT; we set seed_dir via parameter
    cmd = [
        "bash", "/home/lx/snn/code/scripts/generalist_v0_7_5_5m/B2_multiseed_eval.sh",
        job["model_kind"], job["ckpt"], job["spec"], str(job["seed"]),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = "/home/lx/snn"
    env["OUT_PARENT"] = f"/home/lx/snn/{job['seed_dir']}"
    with worker_log.open("a") as wf:
        ts = time.strftime("%H:%M:%S")
        wf.write(f"[{ts}] [gpu{gpu}] START eval {job['model_kind']} {job['split_name']} seed={job['seed']}\n")
    print(f"[eval-gpu{gpu}@{time.strftime('%H:%M:%S')}] START {job['model_kind']} {job['split_name']} seed={job['seed']}", flush=True)
    t0 = time.time()
    with open(log, "wb") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd="/home/lx/snn")
    dt = time.time() - t0
    with worker_log.open("a") as wf:
        ts = time.strftime("%H:%M:%S")
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        wf.write(f"[{ts}] [gpu{gpu}] {status} {job['model_kind']} {job['split_name']} seed={job['seed']} ({dt:.0f}s)\n")
    print(f"[eval-gpu{gpu}@{ts}] {status} {job['model_kind']} {job['split_name']} seed={job['seed']} ({dt:.0f}s)", flush=True)
    return rc

def gpu_worker(gpu, queue):
    for j in queue:
        try:
            run_eval(gpu, j)
        except Exception:
            traceback.print_exc()

procs = []
for gpu in range(4):
    p = multiprocessing.Process(target=gpu_worker, args=(gpu, gpu_jobs[gpu]))
    p.start()
    procs.append(p)

while any(p.is_alive() for p in procs):
    time.sleep(20)
    n_evals = 0
    for g in range(4):
        for j in gpu_jobs[g]:
            od = Path(j["out_dir"])
            if od.exists():
                n_evals += sum(1 for f in od.glob("eval_*.json"))
    print(f"[B2-eval] {n_evals} eval JSONs on disk across all (split, model, seed)", flush=True)

for p in procs:
    p.join()
print(f"[B2-eval] orchestrator done at {time.strftime('%H:%M:%S')}", flush=True)

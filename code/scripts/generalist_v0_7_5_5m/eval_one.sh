#!/bin/bash
# Eval one 5M-aligned generalist ckpt.
#
# Usage:
#   ./eval_one.sh <model_name> <ckpt_path> <eval_spec.json> [seed]
#
# For ID envs: results/5m/<split>/<model>/seed_<s>/eval_<env>.json
# For stress envs: results/5m_stress/<split>/<model>/seed_<s>/eval_<env>.json
#
# Reads each entry of the eval spec, applies clo_env mapping, writes eval JSON.
set -e
cd /home/lx/snn

MODEL=${1:?usage: eval_one.sh <model_name> <ckpt_path> <eval_spec.json> [seed]}
CKPT=${2:?usage: eval_one.sh <model_name> <ckpt_path> <eval_spec.json> [seed]}
SPEC=${3:?usage: eval_one.sh <model_name> <ckpt_path> <eval_spec.json> [seed]}
SEED=${4:-0}

PAD=${PAD:-128}
ACTION_DIM=${ACTION_DIM:-56}
N_EPISODES=${N_EPISODES:-5}
N_SEEDS=${N_SEEDS:-1}
# LeWM App. D/F.1 protocol: planning horizon covers goal_offset (frame-skip 5 x H5 = 25 env
# steps), the whole optimized sequence is executed before replanning, budget=50, goal=t+25.
HORIZON=${HORIZON:-25}
EVAL_BUDGET=${EVAL_BUDGET:-50}
HISTORY_SIZE=${HISTORY_SIZE:-1}
# CEM: 300 samples / 30 elites / 30 iters for PushT, 10 iters otherwise (LeWM App. D)
CEM_ITERS=${CEM_ITERS:-0}   # 0 = auto per-env (pusht:30, others:10)
# Out-of-repo scratch dir for all retrain/eval artifacts (data-generation reset 2026-09-06)
OUT_PARENT=${OUT_PARENT:-/data/lx/tmp/results/5m}
STRESS_OUT_PARENT=${STRESS_OUT_PARENT:-/data/lx/tmp/results/5m_stress}

/home/lx/miniconda3/envs/snn/bin/python - <<PY
import json, os, subprocess, sys
spec_path = "$SPEC"
spec_raw = json.load(open(spec_path))
if isinstance(spec_raw, dict):
    split_name = spec_raw.get('_split_name') or spec_raw.get('split_name') or spec_path.split('/')[-1].replace('.json','')
    spec = spec_raw.get('specs', [])
else:
    split_name = spec_path.split('/')[-1].replace('.json','')
    spec = spec_raw

out_dir = os.path.join("$OUT_PARENT", split_name, "$MODEL", "seed_$SEED")
os.makedirs(out_dir, exist_ok=True)
ckpt = "$CKPT"
n_episodes = $N_EPISODES
n_seeds = $N_SEEDS
horizon = $HORIZON
eval_budget = $EVAL_BUDGET
action_dim = $ACTION_DIM
cem_iters_cfg = $CEM_ITERS
history_size = $HISTORY_SIZE
pad = $PAD
action_dim = $ACTION_DIM

clo_env_map = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum", "humanoid_CMU": "humanoid_cmu", "humanoid_cmu": "humanoid_cmu"}
stress_out_dir = os.path.join("$STRESS_OUT_PARENT", split_name, "$MODEL", "seed_$SEED")
os.makedirs(stress_out_dir, exist_ok=True)

n_total = len(spec)
for i, entry in enumerate(spec):
    env_id = entry["env_id"]
    data_path = entry["path"]
    goal_offset = entry.get("goal_offset", 25)
    clo_env = entry.get("clo_env") or clo_env_map.get(env_id, env_id)
    extra = entry.get("extra_flags", [])
    is_stress = bool(extra)
    this_out_dir = stress_out_dir if is_stress else out_dir
    out_json = os.path.join(this_out_dir, f"eval_{env_id}.json")
    if os.path.exists(out_json):
        print(f"[skip] {env_id} (already exists)", flush=True)
        continue
    cmd = [
        "/home/lx/miniconda3/envs/snn/bin/python", "-m", "code.eval.closed_loop",
        "--env", clo_env, "--ckpt", ckpt, "--data", data_path, "--out", out_json,
        "--n-episodes", str(n_episodes), "--n-seeds", str(n_seeds),
        "--horizon", str(horizon), "--eval-budget", str(eval_budget),
        "--history-size", str(history_size), "--goal-offset", str(goal_offset),
        "--pad-obs-eval", str(pad), "--action-dim-eval", str(action_dim),
        "--cem-iters", str(30 if (cem_iters_cfg == 0 and clo_env.startswith("pusht")) or cem_iters_cfg == 30 else (cem_iters_cfg or 10)),
    ]
    cmd.extend(extra)
    print(f"[eval {i+1}/{n_total}] {env_id} -> {out_json}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[WARN] eval for {env_id} exited rc={rc}", file=sys.stderr, flush=True)
PY

#!/bin/bash
# Per-env closed_loop eval of a generalist ckpt.
#
# Usage:
#   ./eval_closed_loop_one.sh <model_name> <ckpt_path> <eval_spec.json> <seed>
#
# Reads each entry of the eval spec, applies clo_env mapping and stress
# extra_flags, and writes:
#   results/generalist/<model>/seed_<s>/eval_<env>.json
set -e
cd /home/lx/snn

MODEL=${1:?usage: eval_closed_loop_one.sh <model_name> <ckpt_path> <eval_spec.json> <seed>}
CKPT=${2:?usage: eval_closed_loop_one.sh <model_name> <ckpt_path> <eval_spec.json> <seed>}
SPEC=${3:?usage: eval_closed_loop_one.sh <model_name> <ckpt_path> <eval_spec.json> <seed>}
SEED=${4:-0}

PAD=${PAD:-128}
ACTION_DIM=${ACTION_DIM:-56}
N_EPISODES=${N_EPISODES:-5}
N_SEEDS=${N_SEEDS:-1}
HORIZON=${HORIZON:-5}
EVAL_BUDGET=${EVAL_BUDGET:-50}
HISTORY_SIZE=${HISTORY_SIZE:-1}
OUT_BASE=${OUT_BASE:-/home/lx/snn/results/generalist}
STRESS_OUT_BASE=${STRESS_OUT_BASE:-/home/lx/snn/results/generalist_stress}

OUT_DIR="$OUT_BASE/$MODEL/seed_$SEED"
mkdir -p "$OUT_DIR"

/home/lx/miniconda3/envs/snn/bin/python - <<PY
import json, os, subprocess, sys
spec = json.loads(open("$SPEC").read())
out_dir = "$OUT_DIR"
ckpt = "$CKPT"
n_episodes = $N_EPISODES
n_seeds = $N_SEEDS
horizon = $HORIZON
eval_budget = $EVAL_BUDGET
history_size = $HISTORY_SIZE
pad = $PAD
action_dim = $ACTION_DIM

clo_env_map = {
    "cartpole_2d": "cartpole",
    "pendulum_2d": "pendulum",
}

stress_out_dir = "$STRESS_OUT_BASE/$MODEL/seed_$SEED"
os.makedirs(stress_out_dir, exist_ok=True)

n_total = len(spec)
for i, entry in enumerate(spec):
    env_id = entry["env_id"]
    data_path = entry["path"]
    goal_offset = entry.get("goal_offset", 25)
    clo_env = entry.get("clo_env") or clo_env_map.get(env_id, env_id)
    extra = entry.get("extra_flags", [])
    # Stress envs go to a separate results tree so the ID / stress split
    # is clean downstream.
    is_stress = bool(extra)
    this_out_dir = stress_out_dir if is_stress else out_dir
    out_json = os.path.join(this_out_dir, f"eval_{env_id}.json")
    if os.path.exists(out_json):
        print(f"[skip] {env_id} (already exists: {out_json})", flush=True)
        continue
    cmd = [
        "/home/lx/miniconda3/envs/snn/bin/python", "-m", "code.eval.closed_loop",
        "--env", clo_env,
        "--ckpt", ckpt,
        "--data", data_path,
        "--out", out_json,
        "--n-episodes", str(n_episodes),
        "--n-seeds", str(n_seeds),
        "--horizon", str(horizon),
        "--eval-budget", str(eval_budget),
        "--history-size", str(history_size),
        "--goal-offset", str(goal_offset),
        "--pad-obs-eval", str(pad),
        "--action-dim-eval", str(action_dim),
    ]
    cmd.extend(extra)
    print(f"[eval {i+1}/{n_total}] {env_id} -> {out_json}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[WARN] eval for {env_id} exited rc={rc}", file=sys.stderr, flush=True)
PY
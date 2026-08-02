#!/bin/bash
# B2 eval runner (v3): same protocol as eval_one.sh but writes to seed_0/
# (matches where the training checkpoints saved final.pt).
#
# Usage:
#   bash code/scripts/generalist_v0_7_5_5m/B2_multiseed_eval.sh <model_name> <ckpt_path> <eval_spec.json> <seed_label> [out_seed_dir]
#
# seed_label is informational. out_seed_dir defaults to seed_0.

set -e
cd /home/lx/snn

MODEL=${1:?usage}
CKPT=${2:?usage}
SPEC=${3:?usage}
SEED=${4:?usage}
OUT_SEED=${5:-seed_0}

PAD=${PAD:-128}
ACTION_DIM=${ACTION_DIM:-56}
N_EPISODES=${N_EPISODES:-5}
N_SEEDS=${N_SEEDS:-1}
HORIZON=${HORIZON:-5}
EVAL_BUDGET=${EVAL_BUDGET:-50}
HISTORY_SIZE=${HISTORY_SIZE:-1}

# Read OUT_PARENT from env (caller sets it: results/5m_seedN or results/5m).
# We hard-rewrite path to use $OUT_SEED (=seed_0) regardless of SEED.

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

# Always use seed_0 to align with training ckpt dir
out_dir = os.path.join("$OUT_PARENT", split_name, "$MODEL", "$OUT_SEED")
os.makedirs(out_dir, exist_ok=True)
ckpt = "$CKPT"
n_episodes = $N_EPISODES
n_seeds = $N_SEEDS
horizon = $HORIZON
eval_budget = $EVAL_BUDGET
history_size = $HISTORY_SIZE
pad = $PAD
action_dim = $ACTION_DIM

clo_env_map = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum"}
n_total = len(spec)
for i, entry in enumerate(spec):
    env_id = entry["env_id"]
    data_path = entry["path"]
    goal_offset = entry.get("goal_offset", 25)
    clo_env = entry.get("clo_env") or clo_env_map.get(env_id, env_id)
    extra = entry.get("extra_flags", [])
    out_json = os.path.join(out_dir, f"eval_{env_id}.json")
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
    ]
    cmd.extend(extra)
    print(f"[eval {i+1}/{n_total}] {env_id} -> {out_json}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[WARN] eval for {env_id} exited rc={rc}", file=sys.stderr, flush=True)
PY

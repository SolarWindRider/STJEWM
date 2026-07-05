#!/bin/bash
# Run per-env closed-loop eval of a generalist ckpt.
# Reads the same multi-env spec used at training time so env/data/goal-offset
# all stay in sync. For stress envs (e.g. cartpole_flicker) it adds the right
# --flicker-mask-ratio / --vel-hidden-mask-obs-ratio so the env wrapper fires.
#
# Usage:
#   ./eval_generalist.sh stjewm_trace_only                              # default spec
#   ./eval_generalist.sh stjewm_trace_only configs/generalist_20env.json
#   N_EPISODES=10 ./eval_generalist.sh gru_baseline                     # quick smoke
set -e
cd /home/lx/snn

SPEC=${SPEC:-configs/generalist_16env.json}
PAD=${PAD:-128}
ACTION_DIM=${ACTION_DIM:-56}
N_EPISODES=${N_EPISODES:-50}
N_SEEDS=${N_SEEDS:-3}
HORIZON=${HORIZON:-5}
EVAL_BUDGET=${EVAL_BUDGET:-50}
HISTORY_SIZE=${HISTORY_SIZE:-1}
RESULTS_DIR=${RESULTS_DIR:-/home/lx/snn/results/generalist}
CKPT_DIR_BASE=${CKPT_DIR_BASE:-/home/lx/snn/results/generalist}

# CLI filter (model names)
FILTER="$*"

# Map env_kind -> closed_loop --env argument. For DMC envs, closed_loop.py uses
# DMCStateEnv which is keyed on the env_id string passed in --env. Same id used
# at training and eval, so we just echo it.
declare -A ENV_KIND_TO_CLO=(
    [dmc]="<env_id>"  # placeholder; replaced per-row below
    [reacher_4d]="<env_id>"
    [pusht]="<env_id>"
    [tworoom]="<env_id>"
)

# Per-env stress flags: env_id -> "extra closed_loop flags"
# (e.g. cartpole_flicker requires --flicker-mask-ratio, cheetah_velhidden requires --vel-hidden-mask-obs-ratio)
declare -A STRESS_FLAGS=(
    [cartpole_flicker]="--flicker-mask-ratio 0.5"
    [cheetah_velhidden]="--vel-hidden-mask-obs-ratio 0.0"
    [tworoom_long]=""  # handled via --goal-offset 200 in the spec
)

if [ -n "$FILTER" ]; then
    MODEL_NAMES=($FILTER)
else
    MODEL_NAMES=(stjewm_trace_only stjewm_hidden_leak lewm_baseline_v2 gru_baseline)
fi

for model_name in "${MODEL_NAMES[@]}"; do
    ckpt="$CKPT_DIR_BASE/$model_name/final.pt"
    if [ ! -f "$ckpt" ]; then
        echo "[skip] $model_name: ckpt not found at $ckpt"
        continue
    fi
    out_dir="$RESULTS_DIR/$model_name"
    mkdir -p "$out_dir"

    # Read each line of the spec and run one eval per env
    /home/lx/miniconda3/envs/snn/bin/python - <<PY
import json, subprocess, sys, os
stress_flags = {
    "cartpole_flicker": "--flicker-mask-ratio 0.5",
    "cheetah_velhidden": "--vel-hidden-mask-obs-ratio 0.0",
    "tworoom_long": "",
}
# env_id used at training -> closed_loop --env argument. DMC 2D envs (cartpole_2d,
# pendulum_2d) share the same env implementation as their non-2D names, so we map.
clo_env_map = {
    "cartpole_2d": "cartpole",
    "pendulum_2d": "pendulum",
}
action_dim = $ACTION_DIM

for entry in spec:
    env_id = entry["env_id"]
    data_path = entry["path"]
    history_size = entry.get("history_size", history_size_default)
    goal_offset = entry.get("goal_offset", 25)
    out_json = os.path.join(out_dir, f"eval_{env_id}.json")
    if os.path.exists(out_json):
        print(f"[skip] {env_id}: {out_json} already exists", flush=True)
        continue
    clo_env = clo_env_map.get(env_id, env_id)
    extra = stress_flags.get(env_id, "")
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
    ]
    if extra:
        cmd.extend(extra.split())
    short = " ".join(cmd[6:])
    print(f"[eval] {env_id}: {short}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[WARN] eval for {env_id} exited with rc={rc}", file=sys.stderr)
PY
done

echo ""
echo "============================================="
echo "GENERALIST EVAL COMPLETE"
echo "============================================="

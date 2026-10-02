#!/bin/bash
# Run per-env closed-loop eval of a generalist ckpt.
# Reads the same multi-env spec used at training time so env/data/goal-offset
# all stay in sync. For stress envs (e.g. cartpole_flicker) it adds the right
# --flicker-mask-ratio / --qpos-mask-obs-ratio so the env wrapper fires.
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

# An optional spec follows a single model; otherwise all arguments are models.
if [ "$#" -eq 2 ] && [[ "$2" == *.json ]]; then
    FILTER="$1"
    SPEC="$2"
else
    FILTER="$*"
fi


if [ -n "$FILTER" ]; then
    MODEL_NAMES=($FILTER)
else
    MODEL_NAMES=(stjewm_trace_only stjewm_hidden_leak lewm_baseline_v2 gru_baseline)
fi

for model_name in "${MODEL_NAMES[@]}"; do
    ckpt="$CKPT_DIR_BASE/$model_name/final.pt"
    if [ ! -f "$ckpt" ]; then
        echo "[error] $model_name: ckpt not found at $ckpt" >&2
        exit 1
    fi
    out_dir="$RESULTS_DIR/$model_name"
    mkdir -p "$out_dir"

    # Read each line of the spec and run one eval per env
    /home/lx/miniconda3/envs/snn/bin/python - <<PY
import json, subprocess, sys, os
spec_raw = json.load(open("$SPEC"))
spec = spec_raw.get("specs", spec_raw.get("train_specs", [])) if isinstance(spec_raw, dict) else spec_raw
ckpt = "$ckpt"
out_dir = "$out_dir"
history_size_default = $HISTORY_SIZE
n_episodes = $N_EPISODES
n_seeds = $N_SEEDS
horizon = $HORIZON
eval_budget = $EVAL_BUDGET
pad = $PAD
stress_flags = {
    "cartpole_flicker": "--flicker-mask-ratio 0.5",
    "cheetah_qpos_masked": "--qpos-mask-obs-ratio 0.0",
    "tworoom_long": "",
    "pusht_ood": "--split unseen_goal",
}
# env_id used at training -> closed_loop --env argument. DMC 2D envs (cartpole_2d,
# pendulum_2d) share the same env implementation as their non-2D names, so we map.
clo_env_map = {
    "cartpole_2d": "cartpole",
    "pendulum_2d": "pendulum",
    "humanoid_CMU": "humanoid_cmu",
    # Translate preserved training-spec provenance, not a canonical env alias.
    "cheetah_velhidden": "cheetah_qpos_masked",
}
action_dim = $ACTION_DIM

for entry in spec:
    source_env_id = entry["env_id"]
    env_id = "cheetah_qpos_masked" if source_env_id == "cheetah_velhidden" else source_env_id
    data_path = entry["path"]
    history_size = entry.get("history_size", history_size_default)
    goal_offset = entry.get("goal_offset", 25)
    out_json = os.path.join(out_dir, f"eval_{env_id}.json")
    if os.path.exists(out_json):
        print(f"[skip] {env_id}: {out_json} already exists", flush=True)
        continue
    declared_clo_env = entry.get("clo_env") or source_env_id
    clo_env = clo_env_map.get(declared_clo_env, declared_clo_env)
    extra = [
        "--qpos-mask-obs-ratio" if flag == "--vel-hidden-mask-obs-ratio" else flag
        for flag in entry.get("extra_flags", stress_flags.get(env_id, "").split())
    ]
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
    if extra:
        cmd.extend(extra)
    short = " ".join(cmd[6:])
    print(f"[eval] {env_id}: {short}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Evaluation for {env_id} exited with rc={rc}")
PY
done

echo ""
echo "============================================="
echo "GENERALIST EVAL COMPLETE"
echo "============================================="

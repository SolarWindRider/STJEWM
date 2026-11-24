#!/usr/bin/env python
"""H8-lite (no model): how much does the zero-velocity initialisation of our DMC protocol matter?

Our DMC datasets store qpos only, and _set_env_state restores qpos with qvel = 0. The goal is the
qpos 25 steps further along a recorded random trajectory that started with the TRUE velocity.
Replay the recorded 25 (and 50) actions from the zero-velocity state and measure the distance to
the recorded goal. From the true state this replay is exact by construction.
"""
import json
import sys

sys.path.insert(0, "/home/lx/snn")
import numpy as np

from code.eval.closed_loop import make_env, _set_env_state

ENVS = [("cartpole", "data/dm_control/cartpole_250k.npz"), ("pendulum", "data/dm_control/pendulum_250k.npz"),
        ("cheetah", "data/dm_control/3d_rollouts_250k/cheetah_250k.npz"),
        ("walker", "data/dm_control/3d_rollouts_250k/walker_250k.npz"),
        ("hopper", "data/dm_control/3d_rollouts_250k/hopper_250k.npz"),
        ("quadruped", "data/dm_control/3d_rollouts_250k/quadruped_250k.npz"),
        ("humanoid", "data/dm_control/3d_rollouts_250k/humanoid_250k.npz"),
        ("humanoid_cmu", "data/dm_control/3d_rollouts_250k/humanoid_CMU_250k.npz"),
        ("finger", "data/dm_control/3d_rollouts_250k/finger_250k.npz"),
        ("fish", "data/dm_control/3d_rollouts_250k/fish_250k.npz"),
        ("stacker", "data/dm_control/3d_rollouts_250k/stacker_250k.npz"),
        ("ball_in_cup", "data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz")]
N = 400
out = {}
for kind, path in ENVS:
    env = make_env(kind)
    env.reset(seed=0)
    tol = env._success_tol
    z = np.load(f"/home/lx/snn/{path}")
    obs = z["observations"].reshape(len(z["dones"].reshape(-1)), -1).astype(np.float32)
    act = z["actions"].reshape(len(obs), -1).astype(np.float32)
    done = z["dones"].reshape(-1).astype(bool)
    ok = np.flatnonzero((~done[:-50]) & (~done[25:-25]))            # window must stay inside one episode
    rng = np.random.default_rng(0)
    starts = rng.choice(ok, size=min(N, len(ok)), replace=False)
    d_start, d_stay25, d_rec25, d_stay50, d_rec50 = [], [], [], [], []
    for i in starts:
        q0, g = obs[i], obs[i + 25]
        _set_env_state(env, q0)
        d_start.append(env.check_success(obs[i], g)[1])
        _set_env_state(env, q0)
        for t in range(25):
            env.step(np.zeros(act.shape[1], np.float32))
        d_stay25.append(env.check_success(env.get_state(), g)[1])
        _set_env_state(env, q0)
        for t in range(25):
            env.step(act[i + t])
        d_rec25.append(env.check_success(env.get_state(), g)[1])
        for t in range(25, 50):
            env.step(act[i + t])
        d_rec50.append(env.check_success(env.get_state(), g)[1])
        _set_env_state(env, q0)
        for t in range(50):
            env.step(np.zeros(act.shape[1], np.float32))
        d_stay50.append(env.check_success(env.get_state(), g)[1])
    r = dict(tol=float(tol), n=len(starts),
             start_within_tol=float(np.mean(np.array(d_start) < tol)),
             stay25_within_tol=float(np.mean(np.array(d_stay25) < tol)),
             replay25_from_standstill_within_tol=float(np.mean(np.array(d_rec25) < tol)),
             replay50_from_standstill_within_tol=float(np.mean(np.array(d_rec50) < tol)),
             stay50_within_tol=float(np.mean(np.array(d_stay50) < tol)),
             median_d_start_over_tol=float(np.median(np.array(d_start) / tol)),
             median_d_replay25_over_tol=float(np.median(np.array(d_rec25) / tol)))
    out[kind] = r
    print("RESULT", json.dumps(dict(env=kind, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})), flush=True)
json.dump(out, open("/tmp/lewm_check/h8_zero_velocity.json", "w"), indent=1)

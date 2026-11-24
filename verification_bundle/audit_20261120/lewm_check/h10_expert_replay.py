#!/usr/bin/env python
"""H10: (a) replay the RECORDED expert actions from our restored init state in our env loop -> success?
            (no model; validates init restore + dynamics + success semantics for the H9 episodes)
       (b) for the H9 checkpoint: latent cost of recorded plan vs CEM plan vs zero plan, and whether
            a low latent cost corresponds to physical success.
usage: h10_expert_replay.py <h9 ckpt>"""
import sys

sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/tmp/lewm_check")
import numpy as np
import torch

from h9_faithful_state_lewm import StateLeWM, FS, DEV
from code.data.loaders import h5_episode_index, h5_load_field
from code.core.envs.swm_envs import PushTEnv
from code.eval.closed_loop import _set_env_state
import stable_worldmodel as swm

H5 = "/home/lx/LeWM/data/pusht_expert_train.h5"
state = h5_load_field(H5, "state").astype(np.float32) / 500.0
act = h5_load_field(H5, "action").astype(np.float32)
ep_idx, _, _, _, _ = h5_episode_index(H5)
n = len(state)
ok = np.ones(n - 21, bool)
for k in (5, 10, 15, 20):
    ok &= ep_idx[: n - 21] == ep_idx[k: k + n - 21]
starts_all = np.flatnonzero(ok)
ho_starts = starts_all[starts_all >= starts_all[-len(starts_all) // 20]]
rs = np.random.default_rng(0)
pick = ho_starts[rs.choice(len(ho_starts), size=50, replace=False)]
pick = pick[ep_idx[pick] == ep_idx[pick + 25]]

env = PushTEnv()
succ, d_final, d_start = [], [], []
for i, s in enumerate(pick):
    init, goal = state[s] * 500.0, state[s + 25] * 500.0
    env.reset(seed=int(i))
    _set_env_state(env, init.astype(np.float32))
    env._env.unwrapped._set_goal_state(goal.astype(np.float64))
    d_start.append(env.check_success(env.get_state(), goal)[1])
    done = False
    for t in range(25):
        env.step(act[s + t].astype(np.float32))
        if env.check_success(env.get_state(), goal)[0]:
            done = True
            break
    succ.append(done)
    d_final.append(env.check_success(env.get_state(), goal)[1])
print("RESULT", dict(what="recorded expert actions replayed from our restored init (25 steps)",
                    n=len(pick), success_anytime=float(np.mean(succ)),
                    median_start_dist=float(np.median(d_start)), median_final_dist=float(np.median(d_final))), flush=True)

if len(sys.argv) > 1:
    ck = torch.load(sys.argv[1], map_location=DEV, weights_only=False)
    model = StateLeWM().to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    mu_s, sd_s = (torch.from_numpy(ck[k]).float().to(DEV) for k in ("mu_s", "sd_s"))
    mu_a, sd_a = (torch.from_numpy(ck[k]).float().to(DEV) for k in ("mu_a", "sd_a"))
    import gymnasium as gym
    solver = swm.solver.CEMSolver(model, device=DEV, num_samples=300, topk=30, n_steps=30)
    solver.configure(action_space=gym.spaces.Box(-1.0, 1.0, (1, 2), np.float32), n_envs=1,
                     config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5))
    zeros = torch.zeros(1, 1, 1, device=DEV)
    rows = []
    for i, s in enumerate(pick):
        init, goal = state[s] * 500.0, state[s + 25] * 500.0
        env.reset(seed=int(i))
        _set_env_state(env, init.astype(np.float32))
        env._env.unwrapped._set_goal_state(goal.astype(np.float64))
        cur = (torch.from_numpy(state[s]).to(DEV) - mu_s) / sd_s
        g = (torch.from_numpy(state[s + 25]).to(DEV) - mu_s) / sd_s
        info = {"state": cur[None, None], "goal_state": g[None, None], "pixels": zeros, "goal": zeros,
                "action": torch.zeros(1, 1, 10, device=DEV)}
        exp_blocks = torch.from_numpy(act[s:s + 25].reshape(5, 10)).to(DEV)
        exp_blocks = (exp_blocks - mu_a.repeat(5)[None, :]) / sd_a.repeat(5)[None, :]
        with torch.no_grad():
            c_exp = float(model.get_cost(dict(info), exp_blocks[None, None])[0, 0])   # recorded plan (B,S,T,D)
            c_zero = float(model.get_cost(dict(info), torch.zeros(1, 1, 5, 10, device=DEV))[0, 0])
        out = solver.solve(info)
        plan = out["actions"][0].to(DEV)
        with torch.no_grad():
            c_cem = float(model.get_cost(dict(info), plan[None, None])[0, 0])
        done, taken = False, 0
        for _plan in range(2):                                   # official budget 50 = 2 plans of 25 steps
            if _plan == 1:                                        # replan from the observed state
                cur = (torch.from_numpy(np.asarray(env.get_state(), np.float32) / 500.0).to(DEV) - mu_s) / sd_s
                info = {"state": cur[None, None], "goal_state": g[None, None], "pixels": zeros, "goal": zeros,
                        "action": torch.zeros(1, 1, 10, device=DEV)}
                plan = solver.solve(info)["actions"][0].to(DEV)
            raw = (plan.reshape(25, 2).to(DEV) * sd_a + mu_a).cpu().numpy()
            for a_raw in raw:
                env.step(np.clip(a_raw, -1, 1).astype(np.float32))
                taken += 1
                if env.check_success(env.get_state(), goal)[0]:
                    done = True
                    break
            if done or taken >= 50:
                break
        rows.append(dict(i=i, c_init=0.0, c_expert=c_exp, c_zero=c_zero, c_cem=c_cem, success=bool(done),
                         plan_abs=float(np.abs(raw).mean()), exp_abs=float(np.abs(act[s:s + 25]).mean())))
    import json
    S = [r for r in rows if r["success"]]; F = [r for r in rows if not r["success"]]
    print("RESULT", json.dumps(dict(what="H9 checkpoint in our loop", step=int(ck["step"]), n=len(rows),
        success=float(np.mean([r["success"] for r in rows])),
        median_cost=dict(recorded=float(np.median([r["c_expert"] for r in rows])), zero=float(np.median([r["c_zero"] for r in rows])),
                         cem=float(np.median([r["c_cem"] for r in rows]))),
        frac_cem_below_recorded=float(np.mean([r["c_cem"] < r["c_expert"] for r in rows])),
        median_c_cem_success=(float(np.median([r["c_cem"] for r in S])) if S else None),
        median_c_cem_failure=(float(np.median([r["c_cem"] for r in F])) if F else None),
        median_c_recorded_success=(float(np.median([r["c_expert"] for r in S])) if S else None),
        median_c_recorded_failure=(float(np.median([r["c_expert"] for r in F])) if F else None),
        mean_abs_action=dict(cem=float(np.mean([r["plan_abs"] for r in rows])), recorded=float(np.mean([r["exp_abs"] for r in rows]))))), flush=True)
    json.dump(rows, open(f"/tmp/lewm_check/h10_rows_step{int(ck['step'])}.json", "w"), indent=1)

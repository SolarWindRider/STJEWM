#!/usr/bin/env python
"""Bisect why H9.env_eval (2%) and H10 (67%) disagree on the same checkpoint.
usage: h9_bisect.py <ckpt>"""
import sys

sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/tmp/lewm_check")
import numpy as np
import torch
import gymnasium as gym
import stable_worldmodel as swm

from h9_faithful_state_lewm import StateLeWM, DEV
from code.data.loaders import h5_episode_index, h5_load_field
from code.core.envs.swm_envs import PushTEnv
from code.eval.closed_loop import _set_env_state

H5 = "/home/lx/LeWM/data/pusht_expert_train.h5"
state = h5_load_field(H5, "state").astype(np.float32) / 500.0
ep_idx, _, _, _, _ = h5_episode_index(H5)
n = len(state)
ok = np.ones(n - 21, bool)
for k in (5, 10, 15, 20):
    ok &= ep_idx[: n - 21] == ep_idx[k: k + n - 21]
starts_all = np.flatnonzero(ok)
ho_starts = starts_all[starts_all >= starts_all[-len(starts_all) // 20]]
rs = np.random.default_rng(0)
pick = ho_starts[rs.choice(len(ho_starts), size=50, replace=False)]

ck = torch.load(sys.argv[1], map_location=DEV, weights_only=False)
model = StateLeWM().to(DEV)
model.load_state_dict(ck["model"])
model.eval()
mu_s, sd_s, mu_a, sd_a = (torch.from_numpy(ck[k]).float().to(DEV) for k in ("mu_s", "sd_s", "mu_a", "sd_a"))
print("ckpt step", ck["step"], "| sd_a", sd_a.tolist(), flush=True)


def make_solver():
    s = swm.solver.CEMSolver(model, device=DEV, num_samples=300, topk=30, n_steps=30)
    s.configure(action_space=gym.spaces.Box(-1.0, 1.0, (1, 2), np.float32), n_envs=1,
                config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5))
    return s


env = PushTEnv()
zeros = torch.zeros(1, 1, 1, device=DEV)
for i, s in enumerate(pick[:4]):
    init, goal = state[s] * 500.0, state[s + 25] * 500.0
    env.reset(seed=int(i))
    _set_env_state(env, init.astype(np.float32))
    env._env.unwrapped._set_goal_state(goal.astype(np.float64))
    cur_env = (torch.from_numpy(np.asarray(env.get_state(), np.float32) / 500.0).to(DEV) - mu_s) / sd_s       # H9 style
    g64 = (torch.from_numpy(goal).to(DEV) - mu_s) / sd_s                                                      # float64 (H9)
    cur_ds = (torch.from_numpy(state[s]).to(DEV) - mu_s) / sd_s                                               # H10 style
    g32 = (torch.from_numpy(state[s + 25]).to(DEV) - mu_s) / sd_s
    print(f"ep {i}: |cur_env-cur_ds| max {float((cur_env - cur_ds).abs().max()):.2e} | |g64-g32| max {float((g64.float() - g32).abs().max()):.2e}"
          f" | dtypes cur_env {cur_env.dtype} g64 {g64.dtype}", flush=True)
    variants = {
        "H9  : cur=env, goal=float64": (cur_env, g64),
        "    : cur=env, goal=float32": (cur_env, g32),
        "    : cur=data, goal=float64": (cur_ds, g64),
        "H10 : cur=data, goal=float32": (cur_ds, g32),
    }
    for name, (c, g) in variants.items():
        info = {"state": c[None, None], "goal_state": g[None, None], "pixels": zeros, "goal": zeros,
                "action": torch.zeros(1, 1, 10, device=DEV)}
        out = make_solver().solve(info)
        plan = out["actions"][0].to(DEV)
        with torch.no_grad():
            c_cem = float(model.get_cost(dict(info), plan[None, None])[0, 0])
        raw = (plan.reshape(25, 2) * sd_a + mu_a)
        print(f"   {name:32s}: plan|a| raw-norm {float(plan.abs().mean()):.3f}  denorm {float(raw.abs().mean()):.3f}  "
              f"max|raw| {float(raw.abs().max()):.2f}  cost {c_cem:.3f}", flush=True)

#!/usr/bin/env python
"""H1: official LeWM weights + official planner (CEMSolver + LeWM.get_cost + official action
StandardScaler), but driven through OUR env wrapper (code.core.envs.swm_envs.PushTEnv), OUR state
restore (_set_env_state), OUR goal setting, OUR success check and OUR episode loop.

Same 50 (episode, start) pairs as the official run (seed 42). Goal image = recorded dataset frame
(exactly as in the official pipeline). Variants:
  init=ours|official      how the initial state is restored
  frame0=render|dataset   first observation: render of restored env vs recorded dataset frame
Isolates "our env handling" from "our model / our planner".
"""
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STABLEWM_HOME"] = "/tmp/lewm_check/home"
sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/tmp/lewm_check")

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from sklearn import preprocessing
import stable_worldmodel as swm
from stable_worldmodel.wrapper.default import MegaWrapper

from code.core.envs.swm_envs import PushTEnv
from code.eval.closed_loop import _set_env_state
from official_eval import TASKS, build_model, img_transform

GOAL_OFFSET, BUDGET = 25, 50


def shrink(img):
    return np.array(Image.fromarray(img).resize((224, 224), Image.BILINEAR))


def to_hwc(img):  # (3,H,W) torch/np -> (H,W,3) uint8
    a = img.numpy() if torch.is_tensor(img) else np.asarray(img)
    return np.transpose(a, (1, 2, 0)).astype(np.uint8)


class _Shim:  # what WorldModelPolicy needs to know about the environment
    num_envs = 1
    action_space = gym.spaces.Box(-1.0, 1.0, (1, 2), np.float32)      # official EnvPool exposes (num_envs, A)
    single_action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)


def main():
    args = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
    task = args.get("task", "pusht")
    init_mode = args.get("init", "ours")          # ours | official
    frame0 = args.get("frame0", "render")         # render | dataset
    n_steps = int(args.get("iters", 30))
    limit = int(args.get("n", 50))
    tag = args.get("tag", f"{init_mode}_{frame0}")
    T = TASKS[task]

    idx = json.loads(Path(f"/tmp/lewm_check/idx_{task}_n50_off{GOAL_OFFSET}_s42.json").read_text())
    eps, starts = np.array(idx["episodes"]), np.array(idx["start"])
    n = min(limit, len(eps))
    eps, starts = eps[:n], starts[:n]

    dataset = swm.data.HDF5Dataset(T["dataset"], keys_to_cache=T["keys"], cache_dir=Path(os.environ["STABLEWM_HOME"]))
    chunks = dataset.load_chunk(eps, starts, starts + GOAL_OFFSET + 1)

    # official eval.py: the planner works in StandardScaler-normalised action space; denormalise before env.step
    _act = dataset.get_col_data("action")
    _act = _act[~np.isnan(_act).any(axis=1)]
    process = {"action": preprocessing.StandardScaler().fit(_act)}
    print(f"action scaler: mean={process['action'].mean_}, std={process['action'].scale_}", flush=True)

    model = build_model(T["hf"]).to("cuda").eval()
    model.requires_grad_(False)
    tf = img_transform()

    env = PushTEnv()
    env._env = MegaWrapper(gym.make("swm/PushT-v1", max_episode_steps=2 * BUDGET, render_mode="rgb_array"))

    def make_policy():
        solver = swm.solver.CEMSolver(model, device="cuda", num_samples=300, topk=30, n_steps=n_steps)
        cfg = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
        return swm.policy.WorldModelPolicy(solver=solver, config=cfg, process=process,
                                           transform={"pixels": tf, "goal": tf})

    rows = []
    t_all = time.time()
    for i in range(n):
        ch = chunks[i]
        init_native = ch["state"][0].numpy().astype(np.float64)
        goal_native = ch["state"][-1].numpy().astype(np.float64)
        goal_img = to_hwc(ch["pixels"][-1])
        rec_init_img = to_hwc(ch["pixels"][0])

        env.reset(seed=int(i) + 1000)
        if init_mode == "ours":
            _set_env_state(env, init_native.astype(np.float32))
        else:
            env._env.unwrapped._set_state(init_native)
            env._current_obs = {"state": env._env.unwrapped._get_obs()}
        env._env.unwrapped._set_goal_state(goal_native)
        s0 = env.get_state()
        render0 = shrink(env._env.unwrapped.render())
        rec0_small = shrink(rec_init_img)
        frame_mse0 = float(np.mean((render0.astype(np.float32) - rec0_small.astype(np.float32)) ** 2))
        restored_err = float(np.linalg.norm(s0[:5] - init_native[:5]))

        pol = make_policy()
        pol.set_env(_Shim())
        cur = render0 if frame0 == "render" else rec_init_img
        goal_small = shrink(goal_img)
        done = False
        t_ep = time.time()
        t = -1
        for t in range(BUDGET):
            info = {"pixels": cur[None, None], "goal": goal_small[None, None]}
            # 2-dim placeholder: get_cost() pops it and rollout() overwrites it with candidate block 0
            info["action"] = np.zeros((1, 1, 2), np.float32)
            a = pol.get_action(info)
            out, r, term, trunc, info = env._env.step(np.asarray(a).reshape(-1)[:2])
            env._current_obs = out                      # identical to PushTEnv.step(), but keeps info['pixels']
            cur = np.asarray(info["pixels"])
            done = bool(term or trunc)
            if done:
                break
        final = env.get_state()
        succ_final, dist = env.check_success(final, goal_native)
        rows.append(dict(i=i, success_any=bool(done), success_final=bool(succ_final), steps=t + 1,
                         final_dist=dist, frame0_mse=frame_mse0, restore_err=restored_err,
                         init_block=init_native[2:4].tolist(), goal_block=goal_native[2:4].tolist(),
                         final_block=final[2:4].tolist()))
        print(f"[{i:2d}] any={done} final={succ_final} steps={t+1} dist={dist:.1f} frame0_mse={frame_mse0:.1f} "
              f"restore_err={restored_err:.2e} ({time.time()-t_ep:.1f}s)", flush=True)

    out = dict(task=task, init=init_mode, frame0=frame0, n=n, cem_iters=n_steps,
               success_any=float(np.mean([r["success_any"] for r in rows])),
               success_final=float(np.mean([r["success_final"] for r in rows])),
               mean_frame0_mse=float(np.mean([r["frame0_mse"] for r in rows])),
               max_restore_err=float(np.max([r["restore_err"] for r in rows])), rows=rows)
    print("RESULT", json.dumps({k: v for k, v in out.items() if k != "rows"}))
    Path(f"/tmp/lewm_check/h1_{task}_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

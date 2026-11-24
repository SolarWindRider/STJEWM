#!/usr/bin/env python
"""Official LeWM evaluation pipeline (stable-worldmodel), hydra-free.

Mirrors refs/le-wm/eval.py + config/eval/<task>.yaml line by line:
  World + HDF5Dataset + StandardScaler + ImageNet transform + WorldModelPolicy(CEMSolver) + LeWM.get_cost.
Official HF weights (quentinll/lewm-*) are loaded strict=True into the official LeWM class.
"""
import json
import os
import re
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STABLEWM_HOME"] = "/tmp/lewm_check/home"

import numpy as np
import torch
import stable_worldmodel as swm
from pathlib import Path
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from transformers import ViTConfig, ViTModel
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm import module as LMOD

TASKS = {
    "pusht": dict(
        env_name="swm/PushT-v1", dataset="pusht_expert_train", hf="lewm-pusht",
        keys=["action", "proprio", "state"], n_steps=30, goal_offset=25, budget=50, world_kwargs={},
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    ),
    "tworoom": dict(
        env_name="swm/TwoRoom-v1", dataset="tworoom", hf="lewm-tworooms",
        keys=["action", "proprio"], n_steps=10, goal_offset=25, budget=50, world_kwargs={},
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
        ],
    ),
    "reacher": dict(
        env_name="swm/ReacherDMControl-v0", dataset="dmc/reacher_random", hf="lewm-reacher",
        keys=["action"], n_steps=10, goal_offset=25, budget=50, world_kwargs={"task": "qpos_match"},
        callables=[
            {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_qpos", "args": {"target_qpos": {"value": "goal_qpos"}}},
        ],
    ),
}


def _remap(k):
    if not k.startswith("encoder."):
        return k
    k = k.replace("encoder.encoder.layer.", "encoder.layers.")
    k = re.sub(r"\.attention\.attention\.query\.", ".attention.q_proj.", k)
    k = re.sub(r"\.attention\.attention\.key\.", ".attention.k_proj.", k)
    k = re.sub(r"\.attention\.attention\.value\.", ".attention.v_proj.", k)
    k = re.sub(r"\.attention\.output\.dense\.", ".attention.o_proj.", k)
    k = re.sub(r"\.intermediate\.dense\.", ".mlp.fc1.", k)
    k = re.sub(r"\.output\.dense\.", ".mlp.fc2.", k)
    return k


def build_model(hf_name):
    src = Path(f"/home/lx/LeWM/data/_hf_models/{hf_name}")
    c = json.loads((src / "config.json").read_text())
    enc = ViTModel(ViTConfig(
        image_size=c["encoder"]["image_size"], patch_size=c["encoder"]["patch_size"],
        hidden_size=192, num_hidden_layers=12, num_attention_heads=3, intermediate_size=768,
        hidden_act="gelu", layer_norm_eps=1e-12, add_pooling_layer=False))
    enc.pooler = None
    mlp = lambda k: LMOD.MLP(input_dim=c[k]["input_dim"], output_dim=c[k]["output_dim"],
                             hidden_dim=c[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
    m = LeWM(encoder=enc, predictor=LMOD.Predictor(**{k: v for k, v in c["predictor"].items() if k != "_target_"}),
             action_encoder=LMOD.Embedder(**{k: v for k, v in c["action_encoder"].items() if k != "_target_"}),
             projector=mlp("projector"), pred_proj=mlp("pred_proj"))
    sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
    print("load_state_dict:", m.load_state_dict({_remap(k): v for k, v in sd.items()}, strict=True))
    n = sum(p.numel() for p in m.parameters())
    print(f"official LeWM params: {n/1e6:.2f}M (enc {sum(p.numel() for p in enc.parameters())/1e6:.2f}M, "
          f"pred {sum(p.numel() for p in m.predictor.parameters())/1e6:.2f}M)")
    return m


def img_transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=224),
    ])


def get_episodes_length(dataset, episodes):
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    epi = dataset.get_col_data(col)
    step = dataset.get_col_data("step_idx")
    return np.array([np.max(step[epi == e]) + 1 for e in episodes])


def main():
    task = sys.argv[1]
    policy_name = sys.argv[2] if len(sys.argv) > 2 else "lewm"
    num_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    T = TASKS[task]
    n_steps = T["n_steps"] if len(sys.argv) <= 5 else int(sys.argv[5])
    goal_offset = T["goal_offset"] if len(sys.argv) <= 6 else int(sys.argv[6])
    budget = T["budget"] if len(sys.argv) <= 7 else int(sys.argv[7])

    world = swm.World(env_name=T["env_name"], num_envs=num_eval, max_episode_steps=2 * budget,
                      image_shape=(224, 224), **T["world_kwargs"])
    transform = {"pixels": img_transform(), "goal": img_transform()}
    dataset = swm.data.HDF5Dataset(T["dataset"], keys_to_cache=T["keys"], cache_dir=Path(os.environ["STABLEWM_HOME"]))

    process = {}
    for col in T["keys"]:
        p = preprocessing.StandardScaler()
        d = dataset.get_col_data(col)
        d = d[~np.isnan(d).any(axis=1)]
        p.fit(d)
        process[col] = p
        if col != "action":
            process[f"goal_{col}"] = p

    if policy_name == "random":
        policy = swm.policy.RandomPolicy()
    else:
        model = build_model(T["hf"]).to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
        solver = swm.solver.CEMSolver(model, device="cuda", num_samples=300, topk=30, n_steps=n_steps)
        policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    ep_len = get_episodes_length(dataset, ep_indices)
    max_start = ep_len - goal_offset - 1
    max_start_dict = {e: max_start[i] for i, e in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_dict[e] for e in dataset.get_col_data(col)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]
    g = np.random.default_rng(seed)
    pick = np.sort(valid[g.choice(len(valid) - 1, size=num_eval, replace=False)])
    rows = dataset.get_row_data(pick)
    eval_episodes = rows[col]
    eval_start = rows["step_idx"]
    Path(f"/tmp/lewm_check/idx_{task}_n{num_eval}_off{goal_offset}_s{seed}.json").write_text(
        json.dumps({"pick": pick.tolist(), "episodes": np.asarray(eval_episodes).tolist(),
                    "start": np.asarray(eval_start).tolist()}))

    world.set_policy(policy)
    t0 = time.time()
    res = world.evaluate(dataset=dataset, start_steps=eval_start.tolist(), goal_offset=goal_offset,
                         eval_budget=budget, episodes_idx=eval_episodes.tolist(),
                         callables=T["callables"], video=None)
    out = dict(task=task, policy=policy_name, num_eval=num_eval, seed=seed, cem_steps=n_steps,
               goal_offset=goal_offset, budget=budget, success_rate=res["success_rate"],
               n_success=int(np.sum(res["episode_successes"])), wall_sec=round(time.time() - t0, 1))
    print("RESULT", json.dumps(out))
    Path(f"/tmp/lewm_check/official_{task}_{policy_name}_n{num_eval}_it{n_steps}_off{goal_offset}_s{seed}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

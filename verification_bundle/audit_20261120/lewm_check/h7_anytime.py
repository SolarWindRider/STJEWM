#!/usr/bin/env python
"""H7: DMC protocol check. Our DMCStateEnv never terminates on success, so eval_closed_loop scores
success at the FINAL step only (step 50). The official semantics is "reached at any time within the
budget". Re-run the AGR evaluation with a non-invasive wrapper that records every visited state and
reports both final-state success and any-time success.  Everything else (ckpt, data, CEM, episodes)
is identical to the AGR grid.
usage: h7_anytime.py <shard> <n_shards> [n_episodes]
"""
import json
import os
import sys

sys.path.insert(0, "/home/lx/snn")
import numpy as np
import torch

from code.eval.closed_loop import eval_closed_loop, make_env, _set_env_state
from code.scripts.event_align import build_model

R = "/data/lx/tmp/results"
CACHE = "/data/lx/tmp/results/active_goal_20261123/_cache"
DEV = "cuda"
SPLIT = "oodc_F1"
MODELS = [("stjewm_trace_only", "5m_5mpar"), ("stjewm_spike_only", "5m_5mpar"), ("stjewm_rate_only", "5m_5mpar"),
          ("stjewm_membrane_readout", "5m_5mpar"), ("stjewm_no_trace", "5m_5mpar"), ("stjewm_hidden_leak", "5m_5mpar"),
          ("alif_timecell_baseline", "5m"), ("lif_transformer_baseline", "5m"), ("stacked_lif_trace", "5m"),
          ("stacked_lif_free", "5m"), ("gru_baseline", "5m"), ("mlp_baseline", "5m"), ("lewm_baseline_v2", "5m")]
ENVS = [("cartpole", "data/dm_control/cartpole_250k.npz"), ("finger", "data/dm_control/3d_rollouts_250k/finger_250k.npz"),
        ("walker", "data/dm_control/3d_rollouts_250k/walker_250k.npz"), ("fish", "data/dm_control/3d_rollouts_250k/fish_250k.npz"),
        ("humanoid", "data/dm_control/3d_rollouts_250k/humanoid_250k.npz"),
        ("stacker", "data/dm_control/3d_rollouts_250k/stacker_250k.npz")]


def load(name, root):
    ck = torch.load(f"{R}/{root}/{SPLIT}/{name}/seed_0/final.pt", map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    state_dim = next((int(sd[k].shape[1]) for k in ("state_encoder.proj.0.weight", "state_projector.proj.0.weight",
                                                    "state_projector.0.weight", "state_proj.0.weight") if k in sd),
                     ck_args.get("pad_obs_to"))
    action_dim = next((int(sd[k].shape[1]) for k in ("action_encoder.proj.0.weight", "action_encoder.0.weight",
                                                     "action_proj.0.weight") if k in sd), ck_args.get("action_dim"))
    if ck_args["model"] == "mlp_baseline":
        action_dim = int(sd["net.0.weight"].shape[1] - sd["state_proj.2.weight"].shape[0])
    model = build_model(ck_args["model"], state_dim, action_dim, ck_args, state_dict=sd)
    model.load_state_dict(sd, strict=True)
    return model.to(DEV).eval()


def make_anytime(inner):
    """Same class as `inner` (so every isinstance / attribute path is unchanged), logs visited states."""
    class _AnyTime(type(inner)):
        def __init__(self, i):
            self.__dict__["_inner"] = i
            self.__dict__["_traj"] = []
            self.__dict__["log"] = []

        def __getattr__(self, name):
            return getattr(self.__dict__["_inner"], name)

        def reset(self, seed=None, **kw):
            self.__dict__["_traj"] = []
            return self.__dict__["_inner"].reset(seed=seed, **kw)

        def step(self, action):
            out = self.__dict__["_inner"].step(action)
            self.__dict__["_traj"].append(np.asarray(self.__dict__["_inner"].get_state(), np.float32))
            return out

        def get_state(self):
            return self.__dict__["_inner"].get_state()

        def check_success(self, state, goal_state):
            inner = self.__dict__["_inner"]
            final_ok, final_d = inner.check_success(state, goal_state)
            ds = [inner.check_success(s, goal_state)[1] for s in self.__dict__["_traj"]] + [float(final_d)]
            self.__dict__["log"].append(dict(final=bool(final_ok), any=bool(min(ds) < inner._success_tol),
                                             min_dist=float(min(ds)), final_dist=float(final_d),
                                             first_hit=int(next((i for i, d in enumerate(ds) if d < inner._success_tol), -1))))
            return bool(min(ds) < inner._success_tol), float(final_d)
    return _AnyTime(inner)


def main():
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    os.chdir("/home/lx/snn")
    out_path = f"/tmp/lewm_check/h7_shard{shard}.json"
    out = {}
    for mi, (name, root) in enumerate(MODELS):
        if mi % n_shards != shard:
            continue
        model = load(name, root)
        for kind, data in ENVS:
            env = make_anytime(make_env(kind))
            try:
                res = eval_closed_loop(model, env, data, n_episodes=n_ep, n_seeds=1, cem_samples=300, cem_elites=30,
                                       cem_iters=10, horizon=25, eval_budget=50, goal_offset=25, history_size=1,
                                       device=DEV, split="in_dist", pad_obs_to=128, model_action_dim=56,
                                       active_goal_disp=2.0, active_goal_cache=CACHE, seed_start=0)
            except Exception as ex:
                print(f"SKIP {name}/{kind}: {type(ex).__name__}: {str(ex)[:100]}", flush=True)
                continue
            log = env.__dict__["log"]
            r = dict(n=len(log), final=float(np.mean([e["final"] for e in log])), any=float(np.mean([e["any"] for e in log])),
                     median_min_dist=round(float(np.median([e["min_dist"] for e in log])), 4),
                     tol=float(getattr(env, "_success_tol")))
            out[f"{name}/{kind}"] = r
            print("RESULT", json.dumps(dict(model=name, env=kind, **r)), flush=True)
            json.dump(out, open(out_path, "w"), indent=1)
    print("DONE H7 shard", shard, flush=True)


if __name__ == "__main__":
    main()

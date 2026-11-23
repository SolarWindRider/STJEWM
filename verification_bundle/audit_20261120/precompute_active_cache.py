"""Precompute model-independent active-goal qualifying sets (one per env/data)."""
import json, sys, glob, os
sys.path.insert(0, "/home/lx/snn")
import torch
from code.eval import closed_loop as CL
from code.scripts.event_align import build_model
CLO = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum", "humanoid_CMU": "humanoid_cmu", "humanoid_cmu": "humanoid_cmu"}
seen = {}
for p in sorted(glob.glob("/home/lx/snn/configs/oodc_5m/*.json")):
    d = json.load(open(p)); ents = d.get("specs", d) if isinstance(d, dict) else d
    for e in ents:
        if e.get("extra_flags"): continue
        clo = CLO.get(e.get("clo_env") or e["env_id"], e.get("clo_env") or e["env_id"])
        seen[(e["env_id"], clo)] = e["path"] if os.path.isabs(e["path"]) else "/home/lx/snn/" + e["path"]
seen[("cheetah_qpos_masked", "cheetah_qpos_masked")] = "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz"
ck = "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt"
ckd = torch.load(ck, map_location="cpu", weights_only=False); sd = ckd["model"]
model = build_model(ckd["args"]["model"], 128, 56, ckd["args"], state_dict=sd); model.load_state_dict(sd, strict=True); model.to("cuda:0").eval()
summary = []
for (env_id, clo), path in sorted(seen.items()):
    try:
        env = CL.make_env(clo, path)
        res = CL.eval_closed_loop(model, env, path, n_episodes=1, n_seeds=1, cem_samples=8, cem_elites=4, cem_iters=1,
                                  goal_offset=25, history_size=1, horizon=25, eval_budget=50, device="cuda:0",
                                  split="in_dist", pad_obs_to=128, model_action_dim=56, active_goal_disp=2.0,
                                  active_goal_cache="/data/lx/tmp/results/active_goal_20261123/_cache")
        f = res.protocol["active_goal_filter"]; status = "ok"
    except ValueError as ex:
        f = None; status = str(ex)[:110]
    summary.append({"env_id": env_id, "status": status, "filter": f})
    print(env_id, status if f is None else f"qualifying={f['n_qualifying']} of d0-passing={f.get('n_displacement_passing_scanned')} (zero-action already success: {f.get('n_zero_action_already_success')})", flush=True)
json.dump(summary, open("/data/lx/tmp/results/active_goal_20261123/_cache/precompute_summary.json", "w"), indent=1)

"""Distribution of init->goal displacement (in units of each env's success tolerance)."""
import json, glob, os, sys
import numpy as np
sys.path.insert(0, "/home/lx/snn")
from code.eval import closed_loop as CL
CLO = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum", "humanoid_CMU": "humanoid_cmu", "humanoid_cmu": "humanoid_cmu",
       "cheetah_velhidden": "cheetah_qpos_masked"}
seen = {}
for p in sorted(glob.glob("configs/oodc_5m/*.json")):
    d = json.load(open(p)); ents = d.get("specs", d) if isinstance(d, dict) else d
    for e in ents:
        if e.get("extra_flags"): continue
        clo = CLO.get(e.get("clo_env") or e["env_id"], e.get("clo_env") or e["env_id"])
        seen[(e["env_id"], clo)] = e["path"]
seen[("cheetah_qpos_masked", "cheetah_qpos_masked")] = "data/dm_control/3d_rollouts_250k/cheetah_250k.npz"
out = []
for (env_id, clo), path in sorted(seen.items()):
    try:
        env = CL.make_env(clo, path)
        pe = env
        while hasattr(pe, "_base"): pe = pe._base
        tol = getattr(pe, "_success_tol", None)
        if tol is None:
            out.append((env_id, clo, "no_scalar_tol", None)); print(f"{env_id:20s} no scalar tolerance ({type(pe).__name__})"); continue
        ds, _ = CL._load_eval_dataset(pe, path, 1, 25, split="in_dist", pad_obs_to=None)
        n = len(ds); nd = pe.spec.obs_dim
        r = np.zeros(n)
        for i in range(n):
            it = ds[i]
            s0 = np.asarray(it["init_state"], dtype=np.float32).reshape(-1)[:nd]; g0 = np.asarray(it["goal_state"], dtype=np.float32).reshape(-1)[:nd]
            r[i] = float(pe.check_success(s0, g0)[1]) / tol
        qs = {q: float(np.quantile(r, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
        passk = {k: int((r >= k).sum()) for k in (1, 2, 3, 5, 10)}
        out.append((env_id, clo, tol, n, qs, passk))
        print(f"{env_id:20s} tol={tol:<6g} n={n:6d} d0/tol q5/25/50/75/95 = " + " ".join(f"{qs[q]:.2f}" for q in qs) + " | >=k: " + " ".join(f"{k}:{c}" for k, c in passk.items()), flush=True)
    except Exception as ex:
        print(f"{env_id:20s} ERROR {type(ex).__name__}: {str(ex)[:80]}")
json.dump([[str(x) for x in o] for o in out], open("/tmp/goal_disp.json", "w"))

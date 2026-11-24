import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["STABLEWM_HOME"] = "/tmp/lewm_check/home"
sys.path.insert(0, "/tmp/lewm_check")
import numpy as np, torch
import official_eval as OE
from stable_worldmodel.wm.lewm import LeWM
orig = LeWM.get_cost
calls = {"n": 0}
def spy(self, info_dict, action_candidates):
    if calls["n"] < 3 or calls["n"] % 100 == 0:
        print("INFO", calls["n"], {k: v for k, v in []}, {k: (type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else (len(v), type(v[0]).__name__, tuple(getattr(v[0], "shape", ())))) for k, v in info_dict.items() if "act" in k or "term" in k or "trunc" in k or k in ("goal", "pixels")}, "cand", tuple(action_candidates.shape), flush=True)
    calls["n"] += 1
    return orig(self, info_dict, action_candidates)
LeWM.get_cost = spy
sys.argv = ["x", "pusht", "lewm", "3"]
OE.main()
print("get_cost calls:", calls["n"])

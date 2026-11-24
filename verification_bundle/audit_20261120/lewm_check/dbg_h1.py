import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["STABLEWM_HOME"] = "/tmp/lewm_check/home"
sys.path.insert(0, "/tmp/lewm_check")
import numpy as np, torch
import official_eval as OE
from stable_worldmodel.wm.lewm import LeWM
orig_roll = LeWM.rollout
def spy(self, info, seq, history_size=None):
    print("ROLLOUT in:", {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in info.items() if k in ("pixels","goal","action","state","goal_state")}, "seq", tuple(seq.shape), "H(pix.size2)=", info['pixels'].size(2), flush=True)
    return orig_roll(self, info, seq, history_size)
LeWM.rollout = spy
orig_prep = OE.swm.policy.WorldModelPolicy._prepare_info
def spy_prep(self, d):
    out = orig_prep(self, d)
    print("PREP:", {k: (tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else (type(v).__name__, len(v)) for k, v in out.items() if k in ("pixels","goal","action","state","goal_state")}, flush=True)
    return out
OE.swm.policy.WorldModelPolicy._prepare_info = spy_prep
sys.argv = ["x", "pusht", "lewm", "1"]
OE.main()

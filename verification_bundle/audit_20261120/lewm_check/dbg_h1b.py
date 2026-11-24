import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["STABLEWM_HOME"] = "/tmp/lewm_check/home"
sys.path.insert(0, "/tmp/lewm_check"); sys.path.insert(0, "/home/lx/snn")
import torch
import stable_worldmodel as swm
from stable_worldmodel.wm.lewm import LeWM
def sh(v): return (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
orig_roll = LeWM.rollout
def spy(self, info, seq, history_size=None):
    print("ROLLOUT in:", {k: sh(v) for k, v in info.items() if k in ("pixels","goal","action")}, "seq", tuple(seq.shape), flush=True)
    return orig_roll(self, info, seq, history_size)
LeWM.rollout = spy
orig_prep = swm.policy.WorldModelPolicy._prepare_info
def spy_prep(self, d):
    out = orig_prep(self, d)
    print("PREP:", {k: sh(v) for k, v in out.items()}, flush=True)
    return out
swm.policy.WorldModelPolicy._prepare_info = spy_prep
sys.argv = ["x", "task=pusht", "init=ours", "frame0=render", "n=1"]
exec(open("/tmp/lewm_check/h1_ours_env_official_planner.py").read())

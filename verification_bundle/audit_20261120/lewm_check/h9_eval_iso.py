#!/usr/bin/env python
"""Resolve the 6% (H9 in-process env_eval) vs 67% (H10) discrepancy: run the EXACT env_eval source of
h9_faithful_state_lewm.py in a fresh process on the saved checkpoint, then toggle the candidate causes.
usage: h9_eval_iso.py <ckpt>"""
import sys

sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/tmp/lewm_check")
import numpy as np
import torch
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

ck = torch.load(sys.argv[1], map_location=DEV, weights_only=False)
model = StateLeWM().to(DEV)
model.load_state_dict(ck["model"])
model.eval()
mu_s_t, sd_s_t, mu_a_t, sd_a_t = (torch.from_numpy(ck[k]).float().to(DEV) for k in ("mu_s", "sd_s", "mu_a", "sd_a"))

src = open("/tmp/lewm_check/h9_faithful_state_lewm.py").read()
a = src.index("    def env_eval(")
b = src.index("        return float(np.mean(succ)), float(np.mean(amag)), steps_taken") + len("        return float(np.mean(succ)), float(np.mean(amag)), steps_taken")
ns = dict(globals())
exec("\n".join(line[4:] if line.startswith("    ") else line for line in src[a:b].splitlines()), ns)   # dedent the nested def
print("checkpoint step", ck["step"], flush=True)
sr, am, stp = ns["env_eval"]()
print("RESULT", dict(what="exact H9 env_eval source, fresh process", success=sr, mean_abs_action=am), flush=True)

# H9's env_eval is called with the model just after opt.step(); check train()/eval() leakage paths
bn = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
print("BatchNorm modules in train mode after model.eval():", [m.training for m in bn], flush=True)

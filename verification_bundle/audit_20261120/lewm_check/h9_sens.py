#!/usr/bin/env python
"""H9-sens: why does the official-architecture state model break at the 2nd open-loop transition?
For the saved H9 checkpoint, compare the position-1 prediction given the 2nd context token
  z1 (true) | z_hat1 (model output) | z1 + random noise of the same norm | z_hat1's mean offset only
and the 1-token path predict([.], a1) at position 0."""
import sys

sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/tmp/lewm_check")
import numpy as np
import torch

from h9_faithful_state_lewm import StateLeWM, FS, HIST, DEV
from code.data.loaders import h5_episode_index, h5_load_field

H5 = "/home/lx/LeWM/data/pusht_expert_train.h5"
ck = torch.load(sys.argv[1], map_location=DEV, weights_only=False)
print("checkpoint step", ck["step"])
model = StateLeWM().to(DEV)
model.load_state_dict(ck["model"])
model.eval()

state = h5_load_field(H5, "state").astype(np.float32) / 500.0
act = h5_load_field(H5, "action").astype(np.float32)
ep_idx, _, _, _, _ = h5_episode_index(H5)
n = len(state)
ok = np.ones(n - 21, bool)
for k in (5, 10, 15, 20):
    ok &= ep_idx[: n - 21] == ep_idx[k: k + n - 21]
starts_all = np.flatnonzero(ok)
cut = starts_all[-len(starts_all) // 20]
ho = starts_all[starts_all >= cut]
rs = np.random.default_rng(5)
S = ho[rs.choice(len(ho), size=1024, replace=False)]
S = S[ep_idx[S] == ep_idx[S + 25]]                                # 6 frames = rows s..s+25 in one episode

mu_s, sd_s = (torch.from_numpy(ck[k]).float().to(DEV) for k in ("mu_s", "sd_s"))
mu_a, sd_a = (torch.from_numpy(ck[k]).float().to(DEV) for k in ("mu_a", "sd_a"))
S_t = torch.from_numpy(state).to(DEV)
A_t = torch.from_numpy(act).to(DEV)
idx = torch.from_numpy(S).to(DEV)[:, None] + FS * torch.arange(6, device=DEV)[None, :]
fr = (S_t[idx] - mu_s) / sd_s
ai = idx[..., None] + torch.arange(FS, device=DEV)
a = ((A_t[ai] - mu_a) / sd_a).reshape(*idx.shape, -1)

with torch.no_grad():
    z = model.encode({"state": fr})["emb"]                                   # (N,4,D)
    ae = model.action_encoder
    z1_hat = model.predict(z[:, :1], ae(a[:, :1]))[:, -1]                   # 1-token path -> position 0
    z1_hat2 = model.predict(z[:, :3], ae(a[:, :3]))[:, 0]                   # same quantity from the 3-token teacher-forced call
    d = z1_hat - z[:, 1]                                                     # prediction error vector
    err = lambda p, t: float(((p - t) ** 2).sum(-1).mean())
    print(f"latent norm {float(z.norm(dim=-1).mean()):.2f} | 1-step err {err(z1_hat, z[:, 1]):.3f} "
          f"| persistence err {err(z[:, 0], z[:, 1]):.3f} | |err| norm {float(d.norm(dim=-1).mean()):.3f}")
    print(f"mean offset norm (systematic bias) {float(d.mean(0).norm()):.3f} | per-sample deviation norm "
          f"{float((d - d.mean(0)).norm(dim=-1).mean()):.3f} | 1-token vs 3-token path max diff "
          f"{float((z1_hat - z1_hat2).abs().max()):.2e}")
    g = torch.Generator(device=DEV).manual_seed(0)

    def pos1(token):
        return model.predict(torch.stack([z[:, 0], token], 1), ae(a[:, :2]))[:, -1]

    rows = [("true z1", pos1(z[:, 1]))]
    for scale in (0.25, 0.5, 1.0, 2.0):
        noise = torch.randn(d.shape, generator=g, device=DEV)
        noise = noise / noise.norm(dim=-1, keepdim=True) * d.norm(dim=-1, keepdim=True) * scale
        rows.append((f"z1 + random noise x{scale}", pos1(z[:, 1] + noise)))
    rows.append(("z_hat1 (model output)", pos1(z1_hat)))
    rows.append(("z1 + mean offset only", pos1(z[:, 1] + d.mean(0))))
    rows.append(("z1 + per-sample error", pos1(z[:, 1] + (d - d.mean(0)))))
    for name, p in rows:
        print(f"position-1 prediction error with 2nd token = {name:32s}: {err(p, z[:, 2]):8.3f}")
    # 1-token path at step 2
    print(f"1-token path at step 2 with true z1 : {err(model.predict(z[:, 1:2], ae(a[:, 1:2]))[:, -1], z[:, 2]):.3f}")
    print(f"1-token path at step 2 with z_hat1  : {err(model.predict(z1_hat[:, None], ae(a[:, 1:2]))[:, -1], z[:, 2]):.3f}")
    noise = torch.randn(d.shape, generator=g, device=DEV)
    noise = noise / noise.norm(dim=-1, keepdim=True) * d.norm(dim=-1, keepdim=True)
    print(f"1-token path at step 2 with z1+noise: {err(model.predict((z[:, 1] + noise)[:, None], ae(a[:, 1:2]))[:, -1], z[:, 2]):.3f}")
    # projector vs pred_proj output statistics
    print(f"projector output  mean-norm {float(z.mean((0, 1)).norm()):.3f}  std/dim {float(z.std((0, 1)).mean()):.3f}")
    print(f"pred_proj output  mean-norm {float(z1_hat.mean(0).norm()):.3f}  std/dim {float(z1_hat.std(0).mean()):.3f}")

    seq = [z[:, 0]]
    per = []
    for t in range(5):
        lo = max(0, t + 1 - HIST)
        nxt = model.predict(torch.stack(seq[lo:], 1), ae(a[:, lo:t + 1]))[:, -1]   # token k paired with block k
        seq.append(nxt)
        per.append(round(float(((nxt - z[:, t + 1]) ** 2).sum(-1).mean()), 3))
    fin = ((seq[5] - z[:, 5]) ** 2).sum(-1).mean()
    init = ((z[:, 0] - z[:, 5]) ** 2).sum(-1).mean()
    print(f"open-loop 5 transitions (official rollout alignment): per-transition error {per} | final/init = {float(fin / init):.4f}")

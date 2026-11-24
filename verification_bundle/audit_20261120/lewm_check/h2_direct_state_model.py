#!/usr/bin/env python
"""H2: harness control. A direct state-space MLP dynamics model (no latent learning) plugged into
OUR unmodified eval_closed_loop (our CEM, our PushTEnv, our _set_env_state, our success check).

If an accurate dynamics model cannot reach goals in our harness, the harness/CEM is broken.
If it can, the harness is sound and the failures of the JEPA baselines lie in model/planning.

cost: tolerance-normalised L2 (agent+block pos within 20 px, block angle within pi/9).
usage: h2_direct_state_model.py <epochs> <hidden> <depth> <lr> [angle|posonly] [cem_sigma_init]
"""
import sys

sys.path.insert(0, "/home/lx/snn")   # must precede `import torch` (stdlib `code` module shadowing)

import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

from code.data.loaders import h5_episode_index, h5_load_field
from code.eval.closed_loop import eval_closed_loop
from code.core.cem import CEM, make_native_action_predict_hook
from code.core.encode import encode_history, encode_obs
from code.core.envs.swm_envs import PushTEnv

H5 = "/home/lx/LeWM/data/pusht_expert_train.h5"
SCALE = 500.0
TOL_POS = 20.0 / SCALE                       # 0.04 (norm over agent xy + block xy)
R_ANG = TOL_POS / (2 * np.sin(np.pi / 18))   # chord of an angle error of pi/9 equals TOL_POS
DEV = "cuda"


class DynMLP(nn.Module):
    def __init__(self, s_dim=7, a_dim=2, hidden=1024, depth=4):
        super().__init__()
        layers = [nn.Linear(s_dim + a_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, s_dim)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(s_dim))
        self.register_buffer("sd", torch.ones(s_dim))

    def delta(self, s, a):  # scaled native units
        return self.net(torch.cat([(s - self.mu) / self.sd, a], -1)) * self.sd


class DirectStateModel(nn.Module):
    """Satisfies the harness contract: forward(obs, action) -> {'emb'}; predict(ctx_emb, ctx_act).

    'latent' = cost space: [agent xy, block xy, R*cos(angle), R*sin(angle), 0, 0] (state / 500).
    """
    action_dim = 2

    def __init__(self, dyn: DynMLP, with_angle=True):
        super().__init__()
        self.dyn, self.with_angle = dyn, with_angle
        self.r = float(R_ANG)

    def _to8(self, s7):
        xy, th = s7[..., :4], s7[..., 4]
        if self.with_angle:
            c, sn = torch.cos(th) * self.r, torch.sin(th) * self.r
        else:
            c, sn = torch.zeros_like(th), torch.zeros_like(th)
        return torch.cat([xy, c[..., None], sn[..., None], torch.zeros_like(xy[..., :2])], -1)

    def _to7(self, z8):
        th = torch.atan2(z8[..., 5], z8[..., 4]) if self.with_angle else torch.zeros_like(z8[..., 4])
        return torch.cat([z8[..., :4], th[..., None], z8[..., 6:8]], -1)

    def forward(self, obs, action):
        return {"emb": self._to8(obs)}

    def predict(self, ctx_emb, ctx_act):
        s7 = self._to7(ctx_emb[:, -1])
        nxt7 = s7 + self.dyn.delta(s7, ctx_act[:, -1])
        out = ctx_emb.clone()
        out[:, -1] = self._to8(nxt7)
        return out


def main():
    epochs, hidden, depth, lr = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    with_angle = (sys.argv[5] if len(sys.argv) > 5 else "angle") == "angle"
    sig = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    tag = f"e{epochs}_h{hidden}_d{depth}_{'angle' if with_angle else 'posonly'}"

    t0 = time.time()
    state = h5_load_field(H5, "state").astype(np.float32) / SCALE
    act = h5_load_field(H5, "action").astype(np.float32)
    ep_idx, _, _, _, _ = h5_episode_index(H5)
    ok = ((ep_idx[1:] == ep_idx[:-1]) & np.isfinite(state[:-1]).all(1) & np.isfinite(state[1:]).all(1)
          & np.isfinite(act[:-1]).all(1))
    idx = np.flatnonzero(ok)
    n_hold = 60_000                                  # held-out tail for open-loop validation
    tr, ho = idx[idx < len(state) - n_hold - 30], idx[idx >= len(state) - n_hold - 30]
    s, a, ns = state[tr], act[tr], state[tr + 1]
    print(f"train transitions: {len(tr)} | held-out: {len(ho)} | action std {act[tr].std(0)} ({time.time()-t0:.0f}s)", flush=True)

    mu, sd = s.mean(0), s.std(0) + 1e-6
    dyn = DynMLP(hidden=hidden, depth=depth).to(DEV)
    dyn.mu.copy_(torch.from_numpy(mu)); dyn.sd.copy_(torch.from_numpy(sd))
    S, A, NS = (torch.from_numpy(x).to(DEV) for x in (s, a, ns))
    opt = torch.optim.AdamW(dyn.parameters(), lr=lr, weight_decay=1e-5)
    bs = 8192
    steps_per_epoch = (len(S) + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps_per_epoch * epochs, pct_start=0.05)

    def openloop_err():
        dyn.eval()
        with torch.no_grad():  # 25-step open-loop rollouts (expert actions) on held-out rows
            k = 4000
            st = torch.from_numpy(state[ho[:k]]).to(DEV)
            aa = torch.from_numpy(np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(act, 25, axis=0)[ho[:k]].transpose(0, 2, 1))).to(DEV)
            gt = torch.from_numpy(state[ho[:k] + 25]).to(DEV)
            cur = st
            for t in range(25):
                cur = cur + dyn.delta(cur, aa[:, t])
            e = torch.sqrt(((cur[:, :4] - gt[:, :4]) ** 2).sum(-1)) * SCALE
            nxt1 = st + dyn.delta(st, aa[:, 0])
            g1 = torch.from_numpy(state[ho[:k] + 1]).to(DEV)
            e1 = torch.sqrt(((nxt1[:, :4] - g1[:, :4]) ** 2).sum(-1)) * SCALE
            return e.cpu().numpy(), e1.cpu().numpy()

    for ep in range(epochs):
        dyn.train()
        perm = torch.randperm(len(S), device=DEV)
        tot = 0.0
        for i in range(0, len(S), bs):
            j = perm[i:i + bs]
            loss = ((dyn.delta(S[j], A[j]) - (NS[j] - S[j])) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(j)
        if ep % 5 == 4 or ep == epochs - 1:
            errs, e1 = openloop_err()
            print(f"epoch {ep+1}: train mse {tot/len(S):.6f} | 1-step pos err px median {np.median(e1):.2f} | "
                  f"25-step open-loop pos err px: median {np.median(errs):.1f}, p90 {np.percentile(errs,90):.1f}, "
                  f"frac<20px {np.mean(errs<20):.2f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"dyn": dyn.state_dict(), "with_angle": with_angle, "hidden": hidden, "depth": depth}, f"/tmp/lewm_check/h2_dyn_{tag}.pt")
    model = DirectStateModel(dyn, with_angle).to(DEV).eval()

    # --- our harness, twice: (i) eval_closed_loop as is (sigma_init=1.0); (ii) same loop body with
    #     CEM sigma_init matched to the dataset action std (0.21). eval_closed_loop hard-codes
    #     sigma_init=1.0, so (ii) re-runs its loop with the only change being the CEM object.
    env = PushTEnv()
    pe = env
    action_dim, low, high = pe.spec.action_dim, pe.spec.action_low, pe.spec.action_high
    for iters, sigma in ((30, 1.0), (30, sig), (10, 1.0)):
        torch.manual_seed(0); np.random.seed(0)
        cem = CEM(model, action_dim=action_dim, horizon=25, n_samples=300, n_elites=30, n_iters=iters,
                  history_size=1, sigma_init=sigma, device=DEV,
                  predict_hook=make_native_action_predict_hook(2, low, high, DEV, None))
        from code.eval.closed_loop import _load_eval_dataset, _set_env_state
        ds, _ = _load_eval_dataset(env, H5, 1, 25, split="in_dist", pad_obs_to=None)
        rng = np.random.default_rng(0 * 7919 + 42)
        episode_indices = rng.choice(len(ds), size=50, replace=False)
        succ, steps_taken, acts = [], [], []
        for ep_i in episode_indices:
            item = ds[int(ep_i)]
            init_state = np.asarray(item["init_state"], dtype=np.float32)[:7] * SCALE
            goal_state = np.asarray(item["goal_state"], dtype=np.float32)[:7] * SCALE
            env.reset(seed=int(ep_i))
            _set_env_state(env, init_state)
            env._env.unwrapped._set_goal_state(goal_state)
            hist = [torch.from_numpy(np.asarray(env.get_state(), np.float32) / SCALE)]
            z_goal = encode_obs(model, torch.from_numpy(np.asarray(goal_state, np.float32) / SCALE), 2, DEV)
            taken, done, t = 0, False, -1
            while taken < 50:
                seq = cem.plan(encode_history(model, hist, 2, DEV), z_goal)
                for cand in seq[:min(25, 50 - taken)].cpu().numpy():
                    actn = np.clip(cand, low, high).astype(np.float32)
                    obs, r, done, info = env.step(actn)
                    taken += 1
                    hist = hist[1:] + [torch.from_numpy(np.asarray(env.get_state(), np.float32) / SCALE)]
                    acts.append(float(np.abs(actn).mean()))
                    t += 1
                    if done:
                        break
                if done:
                    break
            final = env.get_state()
            okk, dist = env.check_success(final, goal_state)
            succ.append(bool(okk)); steps_taken.append(taken + 1)
        print("RESULT", json.dumps(dict(tag=tag, cem_iters=iters, sigma_init=sigma, n=50,
                                        success_rate_env=float(np.mean(succ)),
                                        mean_abs_action=float(np.mean(acts)))), flush=True)
        open(f"/tmp/lewm_check/h2_{tag}_it{iters}_s{sigma}.json", "w").write(json.dumps(
            dict(tag=tag, cem_iters=iters, sigma_init=sigma, success=succ, steps=steps_taken)))


if __name__ == "__main__":
    main()

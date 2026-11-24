#!/usr/bin/env python
"""H9: faithful state-input LeWM control.

Official LeWM pieces used verbatim: predictor (Predictor / ConditionalBlock / Embedder / BatchNorm MLP
from stable_worldmodel.wm.lelm.module), LeWM.rollout/get_cost (subclass only overrides encode()),
official loss (pred MSE + lambda*SIGReg on the SAME embedding), official frame-skip-5 action blocks,
official CEMSolver (N(0,1) in standardised action space, 300/30/30). Only the ViT is replaced by an
MLP state encoder. Evaluated in OUR env loop (PushTEnv + _set_env_state + check_success).
usage: h9_faithful_state_lewm.py <steps> [eval_every]
"""
import json
import sys
import time

sys.path.insert(0, "/home/lx/snn")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm import module as LMOD
import stable_worldmodel as swm

from code.data.loaders import h5_episode_index, h5_load_field
from code.sigreg import SIGReg
from code.core.envs.swm_envs import PushTEnv
from code.eval.closed_loop import _set_env_state

H5 = "/home/lx/LeWM/data/pusht_expert_train.h5"
FS, D, HIST = 5, 192, 3          # frame skip, latent dim, predictor history (official config.json)
DEV = "cuda"
OUT = "/tmp/lewm_check/h9_state_lewm.pt"


class StateLeWM(LeWM):
    """Official LeWM with the ViT swapped for an MLP state encoder. encode() is the only override."""

    def __init__(self, s_dim=7, a_block_dim=10, depth=6, heads=16, mlp_dim=2048, dropout=0.1):
        enc = nn.Sequential(nn.Linear(s_dim, 4 * D), nn.LayerNorm(4 * D), nn.GELU(),
                            nn.Linear(4 * D, D), nn.LayerNorm(D))
        super().__init__(
            encoder=enc,
            predictor=LMOD.Predictor(num_frames=HIST, input_dim=D, hidden_dim=D, output_dim=D, depth=depth,
                                     heads=heads, mlp_dim=mlp_dim, dim_head=64, dropout=dropout, emb_dropout=0.0),
            action_encoder=LMOD.Embedder(input_dim=a_block_dim, smoothed_dim=a_block_dim, emb_dim=D, mlp_scale=4),
            projector=LMOD.MLP(input_dim=D, hidden_dim=2048, output_dim=D, norm_fn=nn.BatchNorm1d),
            pred_proj=LMOD.MLP(input_dim=D, hidden_dim=2048, output_dim=D, norm_fn=nn.BatchNorm1d),
        )

    def encode(self, info):
        s = info["state"]
        shp = s.shape                                                   # (..., 7): works for 2-D, 3-D and 4-D inputs
        emb = self.projector(self.encoder(s.reshape(-1, shp[-1])))
        info["emb"] = emb.reshape(*shp[:-1], -1)
        return info


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    eval_every = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-5
    tag = sys.argv[4] if len(sys.argv) > 4 else "lr5e-5"
    t0 = time.time()
    state = h5_load_field(H5, "state").astype(np.float32) / 500.0
    act = h5_load_field(H5, "action").astype(np.float32)
    ep_idx, _, _, _, _ = h5_episode_index(H5)
    n = len(state)
    # frame-skip-5 windows: frames at s, +5, +10, +15 (+ target +20); blocks = 5 raw actions each
    ok = np.ones(n - 21, bool)
    for k in (5, 10, 15, 20):
        ok &= ep_idx[: n - 21] == ep_idx[k: k + n - 21]
    ok &= np.isfinite(state[: n - 21]).all(1) & np.isfinite(act[: n - 21]).all(1)
    for k in (5, 10, 15, 20):
        ok &= np.isfinite(state[k: k + n - 21]).all(1)
    starts_all = np.flatnonzero(ok)
    cut = starts_all[-len(starts_all) // 20]                  # last 5% of valid starts = held-out
    tr_starts, ho_starts = starts_all[starts_all < cut], starts_all[starts_all >= cut]
    print(f"valid starts {len(starts_all)} | train {len(tr_starts)} | held-out {len(ho_starts)} ({time.time()-t0:.0f}s)", flush=True)

    def window(S, K):
        """S (B,) starts -> frames (B,K,7), blocks (B,K,10)."""
        fr = np.stack([state[S[:, None] + FS * np.arange(K)[None, :]] for _ in [0]], 0)[0]        # (B,K,7)
        blk = np.stack([act[S[:, None] + FS * np.arange(K)[None, :] + j] for j in range(FS)], -1)  # (B,K,FS,2)
        return fr, blk.reshape(*blk.shape[:2], -1)

    f_tr, _ = window(tr_starts[:200000], 1)
    mu_s, sd_s = f_tr[:, 0].mean(0), f_tr[:, 0].std(0) + 1e-6
    a0 = act[tr_starts[:200000]]
    mu_a, sd_a = a0.mean(0), a0.std(0) + 1e-6                    # official: StandardScaler per raw action dim
    S_tr = torch.from_numpy(state).to(DEV)
    A_tr = torch.from_numpy(act).to(DEV)
    ST_tr = torch.from_numpy(tr_starts).to(DEV)
    mu_s_t, sd_s_t = (torch.from_numpy(x).float().to(DEV) for x in (mu_s, sd_s))
    mu_a_t, sd_a_t = (torch.from_numpy(x).float().to(DEV) for x in (mu_a, sd_a))

    def batch(starts_t, K=4):
        idx = starts_t[:, None] + FS * torch.arange(K, device=DEV)[None, :]                # (B,K)
        fr = (S_tr[idx] - mu_s_t) / sd_s_t                                                 # (B,K,7)
        ai = idx[..., None] + torch.arange(FS, device=DEV)                                  # (B,K,FS)
        a = (A_tr[ai] - mu_a_t) / sd_a_t                                                    # (B,K,FS,2)
        return fr, a.reshape(*a.shape[:2], -1)

    model = StateLeWM().to(DEV)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"(pred {sum(p.numel() for p in model.predictor.parameters())/1e6:.2f}M)", flush=True)
    sigreg = SIGReg(knots=17, num_proj=1024).to(DEV) if hasattr(SIGReg(knots=17, num_proj=1024), "to") else SIGReg(knots=17, num_proj=1024)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    rng = np.random.default_rng(0)
    ho = torch.from_numpy(ho_starts[rng.choice(len(ho_starts), size=1024, replace=False)]).to(DEV)
    B = 128

    @torch.no_grad()
    def diagnose():
        model.eval()
        fr, a = batch(ho, K=9)                                   # frames 0..8 (rows s..s+40), blocks 0..8
        z = model.encode({"state": fr})["emb"]                   # (N,9,D)
        p1 = model.predict(z[:, :3], model.action_encoder(a[:, :3]))[:, -1]
        err1 = ((p1 - z[:, 3]) ** 2).sum(-1)
        pers1 = ((z[:, 2] - z[:, 3]) ** 2).sum(-1)
        # planning-style rollout: start from ONE frame (official eval feeds T=1), 5 transitions
        h = z[:, :1]
        errs, track = [], []
        for t in range(5):
            lo = max(0, h.shape[1] + 1 - HIST)
            zi = z[:, :1] if t == 0 else None
            ctx = torch.cat([z[:, :1], torch.stack(errs, 1)], 1) if t > 0 else z[:, :1]
            ctx = ctx[:, -HIST:]
            nxt = model.predict(ctx, model.action_encoder(a[:, t:t + ctx.shape[1]]))[:, -1]
            errs.append(nxt)
        pred = torch.stack(errs, 1)                              # (N,5,D): 1..5 steps (5..25 env steps)
        fin = ((pred[:, -1] - z[:, 5]) ** 2).sum(-1)
        init = ((z[:, 0] - z[:, 5]) ** 2).sum(-1)
        per = ((pred - z[:, 1:6]) ** 2).sum(-1).mean(0)
        return dict(r2_1step=round(float(1 - err1.sum() / pers1.sum()), 4),
                    rollout5_rel_err=round(float(fin.mean() / init.mean()), 4),
                    per_transition_err=[round(float(x), 3) for x in per],
                    latent_norm=round(float(z.norm(dim=-1).mean()), 3))

    @torch.no_grad()
    def env_eval(n_ep=50, n_steps=30):
        """Official CEMSolver on the state model, executed in OUR env loop (2 plans of 25 steps)."""
        model.eval()
        import gymnasium as gym
        solver = swm.solver.CEMSolver(model, device=DEV, num_samples=300, topk=30, n_steps=n_steps)
        solver.configure(action_space=gym.spaces.Box(-1.0, 1.0, (1, 2), np.float32), n_envs=1,
                         config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5))
        env = PushTEnv()
        rs = np.random.default_rng(0)
        pick = ho_starts[rs.choice(len(ho_starts), size=n_ep, replace=False)]
        succ, steps_taken, amag = [], [], []
        for i, s in enumerate(pick):
            init = state[s] * 500.0
            goal = state[s + 25] * 500.0
            env.reset(seed=int(i))
            _set_env_state(env, init.astype(np.float32))
            env._env.unwrapped._set_goal_state(goal.astype(np.float64))
            done, taken = False, 0
            for _plan in range(2):
                cur = (torch.from_numpy(np.asarray(env.get_state(), np.float32) / 500.0).to(DEV) - mu_s_t) / sd_s_t
                g = (torch.from_numpy(goal / 500.0).to(DEV) - mu_s_t) / sd_s_t   # [FIX] goal is in px; model input is px/500
                zeros = torch.zeros(1, 1, 1, device=DEV)                       # placeholders the official get_cost() expects
                info = {"state": cur[None, None], "goal_state": g[None, None], "pixels": zeros, "goal": zeros,
                        "action": torch.zeros(1, 1, 10, device=DEV)}
                plan = solver.solve(info)["actions"][0]                          # (5,10) normalised
                raw = (plan.reshape(25, 2).to(DEV) * sd_a_t + mu_a_t).cpu().numpy()   # solver returns CPU tensors
                amag.append(float(np.abs(raw).mean()))
                for a_raw in raw:
                    env.step(np.clip(a_raw, -1, 1).astype(np.float32))
                    taken += 1
                    if env.check_success(env.get_state(), goal)[0]:
                        done = True
                        break
                if done or taken >= 50:
                    break
            succ.append(bool(done)); steps_taken.append(taken)
        return float(np.mean(succ)), float(np.mean(amag)), steps_taken

    log = []
    for step in range(1, steps + 1):
        model.train()
        st = ST_tr[torch.randint(0, len(ST_tr), (B,), device=DEV)]
        fr, a = batch(st, K=4)
        info = model.encode({"state": fr})
        emb = info["emb"]                                          # (B,4,D)
        pred = model.predict(emb[:, :HIST], model.action_encoder(a[:, :HIST]))   # (B,3,D) positions 0..2 -> frames 1..3
        pred_loss = (pred - emb[:, 1:]).pow(2).mean()
        sig = sigreg(emb.transpose(0, 1))
        loss = pred_loss + 0.09 * sig
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            log.append(dict(step=step, pred=round(float(pred_loss), 5), sigreg=round(float(sig), 4)))
            print(f"step {step} pred={float(pred_loss):.5f} sigreg={float(sig):.4f} ({time.time()-t0:.0f}s)", flush=True)
        if step % eval_every == 0 or step == steps:
            d = diagnose()
            sr, am, stp = env_eval()
            print("RESULT", json.dumps(dict(step=step, **d, env_success_anytime=sr, mean_abs_action=round(am, 3))), flush=True)
            torch.save({"model": model.state_dict(), "mu_s": mu_s, "sd_s": sd_s, "mu_a": mu_a, "sd_a": sd_a, "step": step},
                       f"/tmp/lewm_check/h9_state_lewm_{tag}.pt")
    print("DONE H9", flush=True)


if __name__ == "__main__":
    main()

"""Per-window dynamics export: Trace/GRU/LIF-Tx x train seeds 0/1/2 x cartpole+cheetah.

Diagnostics trained on cross_benchmark_F1 checkpoints (all three seeds exist);
S1 protocol (stab0917_rollout: eps=0.05*mean||z*||, paired twin rollouts,
128 windows/env, recorded future actions) and FD protocol (cls0917_scaling:
paired +/-0.1*per-dim-std on last context frame, obs re-encoded, 64 windows/env,
H=5) captured PER WINDOW. Seed-0 S1/FD values reproduce the published panels'
protocol exactly; seeds 1/2 use identical windows and streams.

Run: cd /home/lx/snn && CUDA_VISIBLE_DEVICES=0 python /home/lx/audit_20261120/export_dynamics_perwindow.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/home/lx/snn/code/scripts")

import mech0917_multistep as ms
import stab0917_rollout as S1
import cls0917_scaling as FD

OUT = Path("/home/lx/audit_20261120")
DEVICE = "cuda:0"
MODELS = [("trace", "stjewm_trace_only", "stjewm"),
          ("gru", "gru_baseline", "baseline"),
          ("lif_tx", "lif_transformer_baseline", "baseline")]
BASE0 = {"stjewm": "/data/lx/tmp/results/5m_5mpar", "baseline": "/data/lx/tmp/results/5m"}
SEED_ROOT = {0: "", 1: "/data/lx/tmp/results/5m_seed1", 2: "/home/lx/snn/../lx_seeds/"}
SEED_ROOT = {0: "", 1: "/data/lx/tmp/results/5m_seed1", 2: "/data/lx/tmp/results/5m_seed2"}
SPLIT = "cross_benchmark_F1"
OUT_PREFIX = "cross"
if len(sys.argv) > 1 and sys.argv[1] == "oodc_F1":
    SPLIT = "oodc_F1"
    OUT_PREFIX = "oodc_F1"
N_WIN_S1 = 128
N_WIN_FD = 64
SEED = 0


def ckpt_path(hint: str, family: str, train_seed: int) -> Path:
    if train_seed == 0:
        base = Path(BASE0[family])
    elif SPLIT == "oodc_F1":
        base = Path("/home/lx/snn/results") / f"5m_seed{train_seed}"  # P0-1 new ckpts (repo-local)
    else:
        base = Path(f"/data/lx/tmp/results/5m_seed{train_seed}")
    return base / SPLIT / hint / "seed_0" / "final.pt"


def main() -> int:
    S1_ARGS = type("A", (), {"device": DEVICE, "batch": 64, "ep_batch": 32})()
    env_data = {}
    for env_name, cfg in S1.ENVS.items():
        obs, act, episodes = ms.load_env_data(cfg["npz"], S1.CTX + S1.HMAX)
        starts, ends, ts = ms.sample_windows(episodes, N_WIN_S1,
                                             np.random.default_rng(SEED))
        env_data[env_name] = (obs, act, episodes, starts, ends, ts)
    fd_data = {}
    for env_name, cfg in FD.ENVS.items():
        obs, act, episodes = ms.load_env_data(cfg["npz"], FD.CTX + FD.H_E4)
        fd_data[env_name] = (obs, act, episodes)

    win_rows = []
    sum_rows = []
    for disp, hint, family in MODELS:
        for train_seed in (0, 1, 2):
            ck = ckpt_path(hint, family, train_seed)
            if not ck.exists():
                print(f"[skip] missing {ck}")
                continue
            model = S1.load_model(hint, str(ck), DEVICE)
            for env_name in S1.ENVS:
                obs, act, episodes, starts, ends, ts = env_data[env_name]
                # ---------------- S1 (single deterministic pass; identical
                # protocol to stab0917_rollout: same windows, eps stream, twin)
                rng = np.random.default_rng(SEED)
                n = len(ts)
                ep_map = dict(episodes)
                spans = [(s, ep_map[s]) for s in sorted(set(starts.tolist()))]
                ep_emb = ms.forward_episodes(model, obs, act, spans, DEVICE, batch=32)
                for i in range(0, n, 64):
                    sl = slice(i, min(i + 64, n))
                    ts_b = ts[sl]
                    starts_b = starts[sl]
                    o_seqs = [obs[t - S1.CTX + 1: t + 1] for t in ts_b]
                    a_seqs = [act[t - S1.CTX + 1: t + 1] for t in ts_b]
                    with torch.inference_mode():
                        ctx = S1.forward_windows(model, o_seqs, a_seqs, DEVICE)
                        fut = np.stack([act[t + 1: t + S1.HMAX + 1] for t in ts_b])
                        fut = np.pad(fut, ((0, 0), (0, 0), (0, S1.PAD_ACT - fut.shape[-1])))
                        fut_t = torch.from_numpy(fut).to(DEVICE)
                        a_win = S1.pad_batch(a_seqs, S1.PAD_ACT).to(DEVICE)
                        d_emb = ctx.shape[-1]
                        b = ctx.shape[0]
                        eps = np.empty((b,), dtype=np.float32)
                        for j, s_ep in enumerate(starts_b):
                            embs = ep_emb[int(s_ep)]
                            eps[j] = S1.EPS_FRAC * float(embs.norm(dim=-1).mean())
                        u = rng.normal(size=(b, d_emb)).astype(np.float32)
                        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
                        u_t = torch.from_numpy(u).to(DEVICE)
                        ctx_pert = ctx.clone()
                        ctx_pert[:, -1, :] += torch.from_numpy(eps).to(DEVICE).unsqueeze(1) * u_t
                        both = torch.cat([ctx, ctx_pert], dim=0)
                        bp = S1.rollout(model, both, torch.cat([a_win, a_win], 0),
                                        torch.cat([fut_t, fut_t], 0))
                        bb = bp.shape[0] // 2
                        zhat = bp[:bb].cpu().numpy()
                        zpert = bp[bb:].cpu().numpy()
                    dz = np.linalg.norm(zpert - zhat, axis=-1)  # (B,HMAX)
                    bnorm = np.linalg.norm(zhat, axis=-1)
                    if "S1_acc" not in dir():
                        pass
                    S1_acc = globals().setdefault("_s1_acc", {})
                    key = (disp, train_seed, env_name)
                    acc_dz = S1_acc.setdefault(key, np.zeros((n, S1.HMAX)))
                    acc_eps = S1_acc.setdefault(key + ("eps",), np.zeros(n))
                    acc_zhat = S1_acc.setdefault(key + ("zh",), np.zeros((n, S1.HMAX, zhat.shape[-1])))
                    acc_zpert = S1_acc.setdefault(key + ("zp",), np.zeros((n, S1.HMAX, zhat.shape[-1])))
                    acc_dz[sl] = dz
                    acc_eps[sl] = eps
                    acc_zhat[sl] = zhat
                    acc_zpert[sl] = zpert
                    if "s_emb" not in dir():
                        pass
                    for wi in range(sl.start, sl.stop):
                        k = wi - sl.start
                        for h in range(1, S1.HMAX + 1):
                            win_rows.append({
                                "kind": "S1", "model": disp, "train_seed": train_seed,
                                "env": env_name, "window": wi, "branch": "paired_twin",
                                "h": h, "dz_norm": float(dz[k, h - 1]),
                                "eps": float(eps[k]), "base_norm": float(bnorm[k, h - 1]),
                                "gain": float(dz[k, h - 1] / eps[k])})
                key = (disp, train_seed, env_name)
                acc_dz = S1_acc[key]; acc_eps = S1_acc[key + ("eps",)]
                c_h = (acc_dz / acc_eps[:, None]).mean(axis=0)  # c(h), h=1..20
                # E_lat from teacher-forced true latents
                ep_map2 = dict(episodes)
                elat = {1: [], 20: []}
                for i2, (s_ep, t) in enumerate(zip(starts, ts)):
                    embs = ep_emb[int(s_ep)]
                    for h in elat:
                        zt = embs[t + h - int(s_ep)].numpy()
                        zh = S1_acc[key + ("zh",)][i2, h - 1]
                        elat[h].append(np.linalg.norm(zh - zt) / (np.linalg.norm(zt) + 1e-8))
                sum_rows.append({"kind": "S1", "model": disp, "train_seed": train_seed,
                                 "env": env_name, "n_windows": n,
                                 "E_lat_h1": float(np.mean(elat[1])),
                                 "E_lat_h20": float(np.mean(elat[20])),
                                 "c20": float(c_h[19]),
                                 "eps_mean": float(acc_eps.mean()),
                                 "near_zero_denominator": 0})
                # ---------------- FD (per window, h=1..5, both channels)
                fobs, fact, fepisodes = fd_data[env_name]
                # deterministic per-window capture (mirrors cls0917_scaling.fd_env
                # exactly: SEED_WIN windows, batch 16, da/do on last frame native
                # dims, obs re-encoded, roll_n_steps with recorded future acts)
                f_rng = np.random.default_rng(FD.SEED_WIN)
                fstarts, _fe, fts = ms.sample_windows(fepisodes, N_WIN_FD, f_rng)
                fo_seqs = [fobs[t - FD.CTX + 1: t + 1] for t in fts]
                fa_seqs = [fact[t - FD.CTX + 1: t + 1] for t in fts]
                ffut = np.stack([fact[t + 1: t + FD.H_E4 + 1] for t in fts])
                ffut = np.pad(ffut, ((0, 0), (0, 0), (0, ms.PAD_ACT - fact.shape[1])))
                ffut = torch.from_numpy(ffut).to(DEVICE)
                obs_std = fobs.std(axis=0)
                act_std = fact.std(axis=0)
                d_obs, d_act = fobs.shape[1], fact.shape[1]
                acc = {ch: np.zeros((N_WIN_FD, FD.H_E4)) for ch in ("dza", "dzo")}
                acc["zb"] = np.zeros((N_WIN_FD, FD.H_E4))
                for i in range(0, N_WIN_FD, 16):
                    sl = slice(i, min(i + 16, N_WIN_FD))
                    with torch.inference_mode():
                        a_win = ms.pad_batch(fa_seqs[sl], ms.PAD_ACT).to(DEVICE)
                        ctx_b = ms.forward_windows(model, fo_seqs[sl], fa_seqs[sl], DEVICE)
                        da = torch.zeros_like(a_win)
                        da[:, -1, :d_act] = FD.REL * torch.from_numpy(act_std).float().to(DEVICE)
                        zb = FD.roll_n_steps(model, ctx_b, a_win, ffut[sl], FD.H_E4)
                        za_p = FD.roll_n_steps(model, ctx_b, a_win + da, ffut[sl], FD.H_E4)
                        za_m = FD.roll_n_steps(model, ctx_b, a_win - da, ffut[sl], FD.H_E4)
                        do = FD.REL * torch.from_numpy(obs_std).float().to(DEVICE)
                        ctx_op, ctx_om = [], []
                        for o_seq in fo_seqs[sl]:
                            o_p = o_seq.copy(); o_p[-1, :d_obs] += do.cpu().numpy()
                            o_m = o_seq.copy(); o_m[-1, :d_obs] -= do.cpu().numpy()
                            ctx_op.append(o_p); ctx_om.append(o_m)
                        zo_p = FD.roll_n_steps(model, ms.forward_windows(model, ctx_op, fa_seqs[sl], DEVICE),
                                               a_win, ffut[sl], FD.H_E4)
                        zo_m = FD.roll_n_steps(model, ms.forward_windows(model, ctx_om, fa_seqs[sl], DEVICE),
                                               a_win, ffut[sl], FD.H_E4)
                    acc["dza"][sl] = (za_p - za_m).norm(dim=-1).cpu().numpy()
                    acc["dzo"][sl] = (zo_p - zo_m).norm(dim=-1).cpu().numpy()
                    acc["zb"][sl] = zb.norm(dim=-1).cpu().numpy()
                for wi in range(N_WIN_FD):
                    for h in range(1, FD.H_E4 + 1):
                        for ch, arr in (("action", acc["dza"]), ("obs", acc["dzo"])):
                            win_rows.append({
                                "kind": f"FD_{ch}", "model": disp, "train_seed": train_seed,
                                "env": env_name, "window": wi, "branch": "paired_pm",
                                "h": h, "dz_norm": float(arr[wi, h - 1]),
                                "eps": float(FD.REL), "base_norm": float(acc["zb"][wi, h - 1]),
                                "gain": float(arr[wi, h - 1] / (acc["zb"][wi, h - 1] + FD.EPS))})
                gain_a = (acc["dza"] / (acc["zb"] + FD.EPS)).mean(axis=0)
                gain_o = (acc["dzo"] / (acc["zb"] + FD.EPS)).mean(axis=0)
                sum_rows.append({"kind": "FD", "model": disp, "train_seed": train_seed,
                                 "env": env_name, "n_windows": N_WIN_FD,
                                 "E_lat_h1": "", "E_lat_h20": "",
                                 "c20": "", "eps_mean": float(FD.REL),
                                 "near_zero_denominator": int(gain_o[0] < 1e-5),
                                 "gain_a_h1": float(gain_a[0]), "gain_a_h5": float(gain_a[FD.H_E4 - 1]),
                                 "gain_o_h1": float(gain_o[0]), "gain_o_h5": float(gain_o[FD.H_E4 - 1]),
                                 "dza_decay5_over_1": float(acc["dza"][:, FD.H_E4 - 1].mean() / (acc["dza"][:, 0].mean() + FD.EPS)),
                                 "dzo_decay5_over_1": float(acc["dzo"][:, FD.H_E4 - 1].mean() / (acc["dzo"][:, 0].mean() + FD.EPS))})
                print(f"[done] {disp} seed{train_seed} {env_name}", flush=True)
    with open(OUT / f"{OUT_PREFIX}_dynamics_per_window.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
        w.writeheader(); w.writerows(win_rows)
    SUM_FIELDS = ["kind", "model", "train_seed", "env", "n_windows",
                  "E_lat_h1", "E_lat_h20", "c20", "eps_mean",
                  "near_zero_denominator", "gain_a_h1", "gain_a_h5",
                  "gain_o_h1", "gain_o_h5", "dza_decay5_over_1",
                  "dzo_decay5_over_1"]
    with open(OUT / "dynamics_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUM_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(sum_rows)
    print(f"win rows: {len(win_rows)}; summary rows: {len(sum_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

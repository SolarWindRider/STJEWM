#!/usr/bin/env python
"""R1-retest — S1 protocol on the trace_proj-repaired checkpoint vs original.

Paired evaluation on identical windows/episodes (oodc_F1, cartpole_2d + cheetah):
  1. Headroom: ridge probes on TRUE latents z* -> s_{t+Delta} (original vs repaired).
  2. S1 stability: closed-loop rollout error growth (relative-L2 + state domain),
     lambda fits, epsilon-contraction c(20).
  3. env-SR: standard closed_loop CEM on both envs, 5 episodes, original vs repaired
     (paired episodes), LeWM reference from E1 cells.

Run: python -m code.scripts.stab0917_rollout_repair --device cuda:0
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")

from code.scripts.mech0917_multistep import (  # noqa: E402
    CTX, HMAX, forward_episodes, load_env_data, load_model, probe_eval,
    rollout, sample_windows,
)
from code.scripts.stab0917_rollout import log_slope  # noqa: E402

ORIG = "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt"
REPAIRED = "/data/lx/tmp/results/traceproj_repair_20260917/oodc_F1/stjewm_trace_only/seed_0/final.pt"
RECEIPT = Path("/data/lx/tmp/results/traceproj_repair_20260917/oodc_F1/stjewm_trace_only/seed_0/repair_receipt.json")
OUT_DIR = Path("/home/lx/snn/results/mech_0917_b/traceproj_repair")
ENVS = {
    "cartpole_2d": "/home/lx/snn/data/dm_control/cartpole_250k.npz",
    "cheetah": "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz",
}
DELTAS = [1, 5, 10, 20]


def eval_model(model, env_name: str, obs, act, episodes, win_starts, win_ends, win_ts, device):
    n = len(win_ts)
    spans = sorted(set(win_starts.tolist()))
    ep_map = dict(episodes)
    spans = [(s, ep_map[s]) for s in spans]
    ep_emb = forward_episodes(model, obs, act, spans, device, batch=32)
    zhat = np.zeros((n, HMAX, 192), dtype=np.float32)
    zstar = np.zeros((n, HMAX, 192), dtype=np.float32)
    sfut = np.zeros((n, HMAX, obs.shape[1]), dtype=np.float32)
    ep_starts = win_starts.copy()
    for i in range(0, n, 32):
        sl = slice(i, min(i + 32, n))
        o_seqs = [obs[t - CTX + 1: t + 1] for t in win_ts[sl]]
        a_seqs = [act[t - CTX + 1: t + 1] for t in win_ts[sl]]
        fut_seqs = [act[t + 1: t + 1 + HMAX] for t in win_ts[sl]]
        zstar_seqs = []
        for j, t in enumerate(win_ts[sl]):
            s0 = int(win_starts[sl][j])
            emb = ep_emb[s0]
            base = t - s0
            zstar_seqs.append(emb[base + 1: base + 1 + HMAX].numpy())
            for k, h in enumerate(range(1, HMAX + 1)):
                sfut[sl.start + j, k] = obs[t + h]
        with torch.inference_mode():
            o = _pad(o_seqs, 128).to(device)
            a = _pad(a_seqs, 56).to(device)
            fut = _pad(fut_seqs, 56).to(device)[:, :HMAX]
            ctx = model(o, a)["emb"][:, -CTX:]
            pred = rollout(model, ctx, a[:, -CTX:], fut).cpu().numpy()
        for j in range(len(o_seqs)):
            zhat[sl.start + j] = pred[j]
            zstar[sl.start + j] = zstar_seqs[j]
    return zhat, zstar, sfut, ep_starts


def _pad(seqs, pad):
    b = len(seqs)
    t_max = max(s.shape[0] for s in seqs)
    out = np.zeros((b, t_max, pad), dtype=np.float32)
    for j, s in enumerate(seqs):
        out[j, : s.shape[0], : s.shape[1]] = s
    return torch.from_numpy(out)


def main() -> int:
    device = "cuda:0"
    rng = np.random.default_rng(0)
    results = {"experiment": "R1_traceproj_repair_retest", "receipt": json.loads(RECEIPT.read_text()),
               "envs": {}}
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_orig = load_model("stjewm_trace_only", ORIG, device)
    model_rep = load_model("stjewm_trace_only", REPAIRED, device)

    for env_name, npz in ENVS.items():
        obs, act, episodes = load_env_data(npz, min_len=CTX + HMAX)
        ws, we, wt = sample_windows(episodes, 128, rng)
        row = {}
        for tag, model in (("orig", model_orig), ("repaired", model_rep)):
            zhat, zstar, sfut, ep_starts = eval_model(model, env_name, obs, act, episodes, ws, we, wt, device)
            train_mask = np.ones(len(zhat), dtype=bool)
            uniq = np.unique(ep_starts)
            perm = rng.permutation(len(uniq))
            tr = set(uniq[perm[: int(0.75 * len(uniq))]].tolist())
            train_mask = np.array([s in tr for s in ep_starts], dtype=bool)
            head, e_lat, e_state = {}, {}, {}
            for delta in DELTAS:
                r2_true, _ = probe_eval(zstar[:, delta - 1], zhat[:, 0] * 0.0, sfut[:, delta - 1],
                                        train_mask, delta)
                head[delta] = r2_true
            for h in range(1, HMAX + 1):
                num = np.linalg.norm(zhat[:, h - 1] - zstar[:, h - 1], axis=1)
                den = np.linalg.norm(zstar[:, h - 1], axis=1) + 1e-8
                e_lat[h] = float((num / den).mean())
            for delta in DELTAS:
                _, r2_pred = probe_eval(zstar[:, delta - 1], zhat[:, delta - 1], sfut[:, delta - 1],
                                        train_mask, delta)
                e_state[delta] = r2_pred
            # contraction
            with torch.inference_mode():
                o_seqs = [obs[t - CTX + 1: t + 1] for t in wt[:32]]
                a_seqs = [act[t - CTX + 1: t + 1] for t in wt[:32]]
                o = _pad(o_seqs, 128).to(device)
                a = _pad(a_seqs, 56).to(device)
                ctx = model(o, a)["emb"][:, -CTX:]
                fut_seqs_c = [act[t + 1: t + 1 + HMAX] for t in wt[:32]]
                fut = _pad(fut_seqs_c, 56).to(device)[:, :HMAX]
                spread = float(zstar.std())
                u = torch.randn_like(ctx[:, -1])
                u = u / u.norm(dim=-1, keepdim=True)
                eps = 0.05 * spread
                base = rollout(model, ctx, a[:, -CTX:], fut)
                ctx_p = ctx.clone()
                ctx_p[:, -1] = ctx_p[:, -1] + eps * u.to(ctx.dtype)
                pert = rollout(model, ctx_p, a[:, -CTX:], fut)
                disp = (pert - base).norm(dim=-1).mean(dim=0)  # (HMAX,)
                c20 = float(disp[-1] / eps)
            row[tag] = {
                "headroom_r2": head,
                "E_lat": e_lat,
                "lambda_lat": log_slope(list(range(2, HMAX + 1)), [e_lat[h] for h in range(2, HMAX + 1)]),
                "state_r2_pred": e_state,
                "lambda_state_r2": log_slope(DELTAS, [e_state[d] for d in DELTAS]),
                "c20_contraction": c20,
                "lambda_emp": float(np.log(max(c20, 1e-12)) / 20),
            }
        results["envs"][env_name] = row
        print(f"[retest] {env_name}: headroom orig {row['orig']['headroom_r2']} vs repaired {row['repaired']['headroom_r2']}", flush=True)
        print(f"[retest] {env_name}: c20 orig {row['orig']['c20_contraction']:.4g} vs repaired {row['repaired']['c20_contraction']:.4g}", flush=True)

    # ---- paired env-SR (standard protocol)
    sr = {}
    for env_name, clo in (("cartpole_2d", "cartpole"), ("cheetah", "cheetah")):
        for tag, ckpt in (("orig", ORIG), ("repaired", REPAIRED)):
            out = out_dir / f"eval_{env_name}_{tag}.json"
            if not out.exists():
                cmd = [sys.executable, "-u", "-m", "code.eval.closed_loop", "--env", clo,
                       "--ckpt", ckpt, "--data", ENVS[env_name], "--out", str(out),
                       "--device", "cuda:0", "--split", "in_dist",
                       "--n-episodes", "5", "--n-seeds", "1", "--horizon", "25",
                       "--eval-budget", "50", "--history-size", "1",
                       "--cem-samples", "300", "--cem-elites", "30", "--cem-iters", "10",
                       "--goal-offset", "25", "--pad-obs-eval", "128", "--action-dim-eval", "56"]
                subprocess.run(cmd, cwd="/home/lx/snn", check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            payload = json.loads(out.read_text())
            sr.setdefault(env_name, {})[tag] = {"env_sr": payload["success_rate_env"],
                                                "cos": payload["mean_cos_dist"]}
    results["env_sr"] = sr
    (out_dir / "repair_retest_summary.json").write_text(json.dumps(results, indent=2))
    print(f"[retest] wrote {out_dir}/repair_retest_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""R2-eval — dose-response evaluation for accessibility-trained checkpoints.

For lambda in {0 (existing final ckpt), 0.1, 1.0, 10.0}:
  1. Headroom: held-out ridge probes on TRUE latents z* -> s_{t+Delta}, Delta in {0,1,5,10,20}.
  2. Stability: closed-loop rollout relative-L2 error growth lambda, state-domain R2 slope,
     epsilon-contraction c(20).
  3. Action sensitivity (S3 protocol): relative 1-step displacement under paired action deltas.
  4. env-SR: standard closed_loop CEM (cartpole_2d + cheetah, 5 episodes).

Run: python -m code.scripts.access_eval --device cuda:0
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
    CTX, HMAX, forward_episodes, load_env_data, load_model, rollout, sample_windows,
)
from code.scripts.stab0917_rollout import log_slope  # noqa: E402

OUT_ROOT = Path("/data/lx/tmp/results/accessibility_20260917/oodc_F1")
OUT_DIR = Path("/home/lx/snn/results/mech_0917_b/accessibility")
ARMS = {
    "lam0_ref": "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt",
    "lam0.1": OUT_ROOT / "stjewm_trace_only_lam01" / "seed_0" / "final.pt",
    "lam1.0": OUT_ROOT / "stjewm_trace_only_lam10" / "seed_0" / "final.pt",
    "lam10.0": OUT_ROOT / "stjewm_trace_only_lam100" / "seed_0" / "final.pt",
}
ENVS = {
    "cartpole_2d": "/home/lx/snn/data/dm_control/cartpole_250k.npz",
    "cheetah": "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz",
}
DELTAS = [0, 1, 5, 10, 20]


def _pad(seqs, pad):
    b = len(seqs)
    t_max = max(s.shape[0] for s in seqs)
    out = np.zeros((b, t_max, pad), dtype=np.float32)
    for j, s in enumerate(seqs):
        out[j, : s.shape[0], : s.shape[1]] = s
    return torch.from_numpy(out)


def probe_r2(z: np.ndarray, s: np.ndarray, ep_starts: np.ndarray, rng, delta: int, fut_offset: int):
    """Ridge z -> s on row-aligned arrays; episode-level 75/25; held-out R2."""
    uniq = np.unique(ep_starts)
    perm = rng.permutation(len(uniq))
    tr = set(uniq[perm[: int(0.75 * len(uniq))]].tolist())
    m = np.array([e in tr for e in ep_starts], dtype=bool)
    xtr, ytr = z[m], s[m]
    xte, yte = z[~m], s[~m]
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    sc = StandardScaler().fit(xtr)
    rg = Ridge(alpha=1.0).fit(sc.transform(xtr), ytr)
    return float(r2_score(yte, rg.predict(sc.transform(xte)), multioutput="uniform_average"))


def main() -> int:
    device = "cuda:0"
    rng = np.random.default_rng(0)
    results = {"arms": {}}
    env_data = {e: load_env_data(p, min_len=CTX + HMAX) for e, p in ENVS.items()}
    windows = {e: sample_windows(ed[2], 128, np.random.default_rng(0)) for e, ed in env_data.items()}

    for tag, ckpt in ARMS.items():
        model = load_model("stjewm_trace_only", str(ckpt), device)
        arm = {}
        for env_name, (obs, act, episodes) in env_data.items():
            ws, we, wt = windows[env_name]
            spans = sorted(set(ws.tolist()))
            ep_map = dict(episodes)
            spans = [(s, ep_map[s]) for s in spans]
            ep_emb = forward_episodes(model, obs, act, spans, device, batch=32)
            n = len(wt)
            zhat = np.zeros((n, HMAX, 192), dtype=np.float32)
            zstar = np.zeros((n, HMAX, 192), dtype=np.float32)
            sfut = np.zeros((n, HMAX, obs.shape[1]), dtype=np.float32)
            ep_starts = ws.copy()
            for i in range(0, n, 32):
                sl = slice(i, min(i + 32, n))
                o_seqs = [obs[t - CTX + 1: t + 1] for t in wt[sl]]
                a_seqs = [act[t - CTX + 1: t + 1] for t in wt[sl]]
                fut_seqs = [act[t + 1: t + 1 + HMAX] for t in wt[sl]]
                for j, t in enumerate(wt[sl]):
                    s0 = int(ws[sl][j])
                    emb = ep_emb[s0]
                    base = t - s0
                    zstar[sl.start + j] = emb[base + 1: base + 1 + HMAX].numpy()
                    for k, h in enumerate(range(1, HMAX + 1)):
                        sfut[sl.start + j, k] = obs[t + h]
                with torch.inference_mode():
                    o = _pad(o_seqs, 128).to(device)
                    a = _pad(a_seqs, 56).to(device)
                    fut = _pad(fut_seqs, 56).to(device)[:, :HMAX]
                    ctx = model(o, a)["emb"][:, -CTX:]
                    pred = rollout(model, ctx, a[:, -CTX:], fut).cpu().numpy()
                zhat[sl.start: sl.start + len(o_seqs)] = pred
            # per-window position 0: z at t, z* at t (from episode emb) for Delta=0
            z_t = np.zeros((n, 192), dtype=np.float32)
            s_t = np.zeros((n, obs.shape[1]), dtype=np.float32)
            for j, t in enumerate(wt):
                s0 = int(ws[j])
                z_t[j] = ep_emb[s0][t - s0].numpy()
                s_t[j] = obs[t]
            head = {0: probe_r2(z_t, s_t, ep_starts, rng, 0, 0)}
            for delta in DELTAS[1:]:
                head[delta] = probe_r2(zstar[:, delta - 1], sfut[:, delta - 1],
                                       ep_starts, rng, delta, 0)
            e_lat = {}
            for h in range(1, HMAX + 1):
                num = np.linalg.norm(zhat[:, h - 1] - zstar[:, h - 1], axis=1)
                den = np.linalg.norm(zstar[:, h - 1], axis=1) + 1e-8
                e_lat[h] = float((num / den).mean())
            # contraction + action sensitivity
            idx32 = np.arange(0, 32)
            o_seqs = [obs[t - CTX + 1: t + 1] for t in wt[idx32]]
            a_seqs = [act[t - CTX + 1: t + 1] for t in wt[idx32]]
            fut_seqs = [act[t + 1: t + 1 + HMAX] for t in wt[idx32]]
            with torch.inference_mode():
                o = _pad(o_seqs, 128).to(device)
                a = _pad(a_seqs, 56).to(device)
                ctx = model(o, a)["emb"][:, -CTX:]
                fut = _pad(fut_seqs, 56).to(device)[:, :HMAX]
                spread = float(zstar.std())
                u = torch.randn_like(ctx[:, -1])
                u = u / u.norm(dim=-1, keepdim=True)
                eps = 0.05 * spread
                base = rollout(model, ctx, a[:, -CTX:], fut)
                ctx_p = ctx.clone()
                ctx_p[:, -1] = ctx_p[:, -1] + eps * u.to(ctx.dtype)
                pert = rollout(model, ctx_p, a[:, -CTX:], fut)
                disp = (pert - base).norm(dim=-1).mean(dim=0)
                c20 = float(disp[-1] / eps)
                # action sensitivity (paired deltas on final context action)
                a_std = float(act.std()) + 1e-8
                delta_a = 0.1 * (float(act.max()) - float(act.min()) + 2 * a_std)
                z1 = model.predict(ctx, a[:, -CTX:])[:, -1]
                a_pos = a.clone()
                a_neg = a.clone()
                a_pos[:, -1] = a_pos[:, -1] + delta_a
                a_neg[:, -1] = a_neg[:, -1] - delta_a
                z1p = model.predict(ctx, a_pos[:, -CTX:])[:, -1]
                z1n = model.predict(ctx, a_neg[:, -CTX:])[:, -1]
                dz = (z1p - z1n).norm(dim=-1)
                act_sens = float((dz / (z1.norm(dim=-1) + 1e-8)).mean())
            arm[env_name] = {
                "headroom_r2": head,
                "lambda_lat": log_slope(list(range(2, HMAX + 1)),
                                        [e_lat[h] for h in range(2, HMAX + 1)]),
                "E_lat20": e_lat[HMAX],
                "c20_contraction": c20,
                "action_sens_rel": act_sens,
            }
            print(f"[access-eval] {tag}/{env_name}: head0={head[0]:.3f} head20={head[20]:.3f} "
                  f"c20={c20:.3g} act={act_sens:.3g}", flush=True)
        results["arms"][tag] = arm

    # env-SR
    sr = {}
    for env_name, clo in (("cartpole_2d", "cartpole"), ("cheetah", "cheetah")):
        for tag, ckpt in ARMS.items():
            out = OUT_DIR / f"eval_{env_name}_{tag.replace('.', '')}.json"
            if not out.exists():
                cmd = [sys.executable, "-u", "-m", "code.eval.closed_loop", "--env", clo,
                       "--ckpt", str(ckpt), "--data", ENVS[env_name], "--out", str(out),
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "access_summary.json").write_text(json.dumps(results, indent=2))
    print(f"[access-eval] wrote {OUT_DIR}/access_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""S1 (stab0917_b): closed-loop rollout stability on frozen world models.

Hypothesis under test: STJEWM's advantage is more stable / sustainable latent
dynamics (slower error growth under self-feeding closed-loop rollout), not more
accurate one-step prediction.

Protocol (reuses the mech0917_multistep.py rollout/probe skeleton):
  1. 128 windows/env; context = 27 true frames [t-26..t]; closed-loop rollout
     h=1..20 via model.predict() with the prediction and the teacher-forced
     dataset action appended to a sliding 27-position buffer.
  2. Ground truth z*_{t+h} from one warm full-episode forward at absolute
     positions (prefix-causal); s_{t+h} from the dataset state field.
  3. Metrics per (model, env, h):
       - E_lat(h) = mean_i ||z_hat_ih - z*_ih|| / (||z*_ih|| + 1e-8)  [primary]
       - state-domain: per-Delta ridge probes (z*_{t+Delta} -> s_{t+Delta},
         features AND targets standardized, alpha=1.0, episode-level 75/25
         split, held-out) applied to z_hat at h == Delta: RMSE (standardized
         state space) and R2  [primary]
       - mean cosine distance  [auxiliary only; saturates for trace models]
  4. lambda = least-squares slope of log E(h) vs h over h in [2,20]
       (latent: every integer h; state: Delta grid {2,5,10,20}).
  5. Contraction test: two rollouts from the same start; initial latent
     (last context position) perturbed by eps*u, u a random unit vector,
     eps = 0.05 * mean_j ||z*_j|| over the warm episode. c(h) =
     mean_i ||dz_hat_ih|| / eps_i; lambda_emp = log c(20)/20; c20 < 1 =>
     contractive.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, "/home/lx/snn")
# torch/numpy may pull a module named `code` into sys.modules before the
# project package is importable; evict any non-package binding so `code.*`
# below resolves to /home/lx/snn/code (same guard as mech0917_multistep).
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from code.scripts.mech0917_multistep import (
    CTX,
    ENVS,
    HMAX,
    PAD_ACT,
    PAD_OBS,
    episode_split,
    forward_episodes,
    forward_windows,
    load_env_data,
    load_model,
    pad_batch,
    sample_windows,
)

# ---------------------------------------------------------------- constants
PROBE_DELTAS = [1, 2, 5, 10, 20]     # state-probe horizons (h grid)
LAMBDA_DELTAS = [2, 5, 10, 20]       # state-domain h grid for the lambda fit (h in [2,20])
LAMBDA_HS = list(range(2, HMAX + 1))  # latent-domain h grid for the lambda fit
EPS_FRAC = 0.05                       # perturbation = eps_frac * mean ||z*|| over episode

MODELS = [
    # (display name, build_model hint, checkpoint)
    ("trace", "stjewm_trace_only",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt"),
    ("spike", "stjewm_spike_only",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_spike_only/seed_0/final.pt"),
    ("membrane", "stjewm_membrane_readout",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_membrane_readout/seed_0/final.pt"),
    ("hidden_leak", "stjewm_hidden_leak",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_hidden_leak/seed_0/final.pt"),
    ("lewm", "lewm_baseline_v2",
     "/data/lx/tmp/results/5m/oodc_F1/lewm_baseline_v2/seed_0/final.pt"),
    ("gru", "gru_baseline",
     "/data/lx/tmp/results/5m/oodc_F1/gru_baseline/seed_0/final.pt"),
    ("mlp", "mlp_baseline",
     "/data/lx/tmp/results/5m/oodc_F1/mlp_baseline/seed_0/final.pt"),
    ("slif_trace", "stacked_lif_trace",
     "/data/lx/tmp/results/5m/oodc_F1/stacked_lif_trace/seed_0/final.pt"),
]


# ---------------------------------------------------------------- rollout (skeleton reuse)
def rollout(model, ctx: torch.Tensor, act_win: torch.Tensor,
            future_acts: torch.Tensor) -> torch.Tensor:
    """Closed-loop rollout with teacher-forced actions, sliding CTX-position context.

    ctx (B,CTX,D), act_win (B,CTX,A) on device; future_acts (B,HMAX,A) with
    future_acts[:, i] = a_{t+1+i}. Returns (B,HMAX,D) predicted latents.
    """
    buf_e, buf_a = ctx, act_win
    preds = []
    for h in range(1, HMAX + 1):
        p = model.predict(buf_e[:, -CTX:], buf_a[:, -CTX:])[:, -1]
        preds.append(p)
        if h < HMAX:
            buf_e = torch.cat([buf_e, p.unsqueeze(1)], dim=1)
            buf_a = torch.cat([buf_a, future_acts[:, h - 1].unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


# ---------------------------------------------------------------- state probe
def probe_state(z_true_d: np.ndarray, z_hat_d: np.ndarray, s_d: np.ndarray,
                train_mask: np.ndarray) -> dict:
    """Ridge probe z*_{t+Delta} -> s_{t+Delta}; features and targets standardized.

    Returns held-out RMSE (standardized state space) and R2, both for the probe
    applied to true held-out latents (headroom) and to z_hat at h == Delta.
    """
    xtr, ytr = z_true_d[train_mask], s_d[train_mask]
    xte, yte = z_true_d[~train_mask], s_d[~train_mask]
    sx = StandardScaler().fit(xtr)
    sy = StandardScaler().fit(ytr)
    rg = Ridge(alpha=1.0).fit(sx.transform(xtr), sy.transform(ytr))
    yte_s = sy.transform(yte)
    p_true = rg.predict(sx.transform(xte))
    p_hat = rg.predict(sx.transform(z_hat_d[~train_mask]))
    return {
        "rmse_true": float(np.sqrt(np.mean((yte_s - p_true) ** 2))),
        "rmse_pred": float(np.sqrt(np.mean((yte_s - p_hat) ** 2))),
        "r2_true": float(r2_score(yte_s, p_true, multioutput="uniform_average")),
        "r2_pred": float(r2_score(yte_s, p_hat, multioutput="uniform_average")),
    }


def log_slope(hs: list[int], errs: list[float]) -> float | None:
    """Least-squares slope of log E(h) vs h; None if any value is non-finite."""
    lh = np.asarray(hs, dtype=np.float64)
    le = np.asarray([np.log(e) for e in errs], dtype=np.float64)
    if not np.all(np.isfinite(le)) or len(lh) < 2:
        return None
    return float(np.polyfit(lh, le, 1)[0])


# ---------------------------------------------------------------- per (model, env)
def run_env_model(model, env_name: str, pos: tuple[int, int], obs, act, episodes,
                  win_starts, win_ends, win_ts, args, rng: np.random.Generator) -> dict:
    device = args.device
    n = len(win_ts)
    ep_map = dict(episodes)
    spans = [(s, ep_map[s]) for s in sorted(set(win_starts.tolist()))]
    ep_emb = forward_episodes(model, obs, act, spans, device, batch=args.ep_batch)

    # ---- context forwards + closed-loop rollout (+ perturbed twin), batched
    ctx_chunks, pred_chunks, pred_pert_chunks, eps_parts = [], [], [], []
    for i in range(0, n, args.batch):
        sl = slice(i, min(i + args.batch, n))
        ts_b = win_ts[sl]
        starts_b = win_starts[sl]
        o_seqs = [obs[t - CTX + 1 : t + 1] for t in ts_b]
        a_seqs = [act[t - CTX + 1 : t + 1] for t in ts_b]
        with torch.inference_mode():
            ctx = forward_windows(model, o_seqs, a_seqs, device)      # (B,CTX,D)
            fut = np.stack([act[t + 1 : t + HMAX + 1] for t in ts_b])  # (B,HMAX,A)
            fut = np.pad(fut, ((0, 0), (0, 0), (0, PAD_ACT - fut.shape[-1])))
            fut = torch.from_numpy(fut).to(device)
            a_win = pad_batch(a_seqs, PAD_ACT).to(device)

            # contraction perturbation: last context position z_t -> z_t + eps*u
            d_emb = ctx.shape[-1]
            b = ctx.shape[0]
            eps = np.empty((b, 1, 1), dtype=np.float32)
            for j, s_ep in enumerate(starts_b):
                embs = ep_emb[int(s_ep)]                               # (T_ep, D)
                eps[j, 0, 0] = EPS_FRAC * float(embs.norm(dim=-1).mean())
            u = rng.normal(size=(b, d_emb)).astype(np.float32)
            u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
            u_t = torch.from_numpy(u).to(device)
            ctx_pert = ctx.clone()
            ctx_pert[:, -1, :] += torch.from_numpy(eps[:, 0, 0]).to(device).unsqueeze(1) * u_t

            both = torch.cat([ctx, ctx_pert], dim=0)
            both_pred = rollout(model, both, torch.cat([a_win, a_win], 0),
                                torch.cat([fut, fut], 0))              # (2B,HMAX,D)
            bb = both_pred.shape[0] // 2
            pred_chunks.append((sl, both_pred[:bb].cpu()))
            pred_pert_chunks.append((sl, both_pred[bb:].cpu()))
            eps_parts.append(eps[:, 0, 0])

    d_emb = pred_chunks[0][1].shape[-1]
    zhat = np.zeros((n, HMAX, d_emb), dtype=np.float32)
    zhat_pert = np.zeros((n, HMAX, d_emb), dtype=np.float32)
    eps_all = np.empty(n, dtype=np.float64)
    for sl, pr in pred_chunks:
        zhat[sl] = pr.numpy()
    for sl, pr in pred_pert_chunks:
        zhat_pert[sl] = pr.numpy()
    eps_all = np.concatenate(eps_parts)

    # ---- warm ground-truth latents z*_{t+h} and future states s_{t+h}
    z_true = np.zeros((n, HMAX, d_emb), dtype=np.float32)
    p0, p1 = pos
    s_future = np.zeros((n, HMAX, p1 - p0), dtype=np.float32)
    for i, (s_ep, t) in enumerate(zip(win_starts, win_ts)):
        embs = ep_emb[int(s_ep)]
        for h in range(1, HMAX + 1):
            z_true[i, h - 1] = embs[t + h - int(s_ep)].numpy()
            s_future[i, h - 1] = obs[t + h, p0:p1]

    train_mask = episode_split(win_starts, np.random.default_rng(123))

    zh = torch.from_numpy(zhat)
    zt = torch.from_numpy(z_true)

    res: dict = {
        "emb_dim": int(d_emb), "n_windows": int(n),
        "n_windows_test": int((~train_mask).sum()),
        "E_lat": {}, "cos_dist": {}, "c": {},
    }
    # ---- primary: relative L2 in latent space, all h in 1..20
    for h in range(1, HMAX + 1):
        dz = (zh[:, h - 1] - zt[:, h - 1]).norm(dim=-1)
        denom = zt[:, h - 1].norm(dim=-1) + 1e-8
        res["E_lat"][str(h)] = float((dz / denom).mean().item())
        res["cos_dist"][str(h)] = float(
            (1.0 - F.cosine_similarity(zh[:, h - 1], zt[:, h - 1], dim=-1)).mean().item())
    # ---- primary: state-domain probe RMSE / R2 on the Delta grid
    res["state"] = {}
    for delta in PROBE_DELTAS:
        res["state"][str(delta)] = probe_state(
            z_true[:, delta - 1], zhat[:, delta - 1], s_future[:, delta - 1], train_mask)
    # ---- lambda fits
    res["lambda_lat"] = log_slope(
        LAMBDA_HS, [res["E_lat"][str(h)] for h in LAMBDA_HS])
    res["lambda_state"] = log_slope(
        LAMBDA_DELTAS, [res["state"][str(d)]["rmse_pred"] for d in LAMBDA_DELTAS])
    # ---- contraction test
    zhp = torch.from_numpy(zhat_pert)
    for h in range(1, HMAX + 1):
        dz = (zhp[:, h - 1] - zh[:, h - 1]).norm(dim=-1).double() / torch.from_numpy(eps_all)
        res["c"][str(h)] = float(dz.mean().item())
    res["lambda_emp"] = float(np.log(res["c"]["20"]) / 20.0)
    res["contractive"] = bool(res["c"]["20"] < 1.0)
    return res


# ---------------------------------------------------------------- summary
def pooled(res_by_env: dict[str, dict]) -> dict:
    keys = ["lambda_lat", "lambda_state", "lambda_emp"]
    out = {k: float(np.mean([r[k] for r in res_by_env.values()]))
           for k in keys if all(r.get(k) is not None for r in res_by_env.values())}
    out["E_state20"] = float(np.mean(
        [r["state"]["20"]["rmse_pred"] for r in res_by_env.values()]))
    out["E_lat20"] = float(np.mean([r["E_lat"]["20"] for r in res_by_env.values()]))
    out["c20"] = float(np.mean([r["c"]["20"] for r in res_by_env.values()]))
    out["contractive"] = bool(out["c20"] < 1.0)
    return out


def verdict(results: dict) -> dict:
    v: dict = {}
    tr = {e: results["trace"][e] for e in results["trace"]}
    lw = {e: results["lewm"][e] for e in results["lewm"]}
    for scope, agg in (("per_env", None), ("pooled", None)):
        if scope == "pooled":
            tr_s, lw_s = {"pooled": pooled(results["trace"])}, {"pooled": pooled(results["lewm"])}
            items = [("pooled", tr_s["pooled"], lw_s["pooled"])]
        else:
            items = [(e, tr[e], lw[e]) for e in tr]
        for name, t, l in items:
            c20t = t["c20"] if "c20" in t else t["c"]["20"]
            c20l = l["c20"] if "c20" in l else l["c"]["20"]
            st_t = t["state"]["20"]["rmse_pred"] if "state" in t else t["E_state20"]
            st_l = l["state"]["20"]["rmse_pred"] if "state" in l else l["E_state20"]
            v[f"{scope}:{name}"] = {
                "lambda_lat_trace": t["lambda_lat"], "lambda_lat_lewm": l["lambda_lat"],
                "lambda_state_trace": t["lambda_state"], "lambda_state_lewm": l["lambda_state"],
                "trace_slower_latent": t["lambda_lat"] < l["lambda_lat"],
                "trace_slower_state": t["lambda_state"] < l["lambda_state"],
                "ordering_consistent": (t["lambda_lat"] < l["lambda_lat"])
                == (t["lambda_state"] < l["lambda_state"]),
                "E_state20_trace": st_t, "E_state20_lewm": st_l,
                "state_level_favors": "lewm" if st_l < st_t else "trace",
                "trace_contractive": t["contractive"],
                "trace_c20": c20t, "lewm_c20": c20l,
            }
    p = pooled(results["trace"])
    pl = pooled(results["lewm"])
    # probe headroom on TRUE latents: can any ridge decode state from z* at all?
    head_tr = float(np.mean([results["trace"][e]["state"]["1"]["rmse_true"]
                             for e in results["trace"]]))
    head_lw = float(np.mean([results["lewm"][e]["state"]["1"]["rmse_true"]
                             for e in results["lewm"]]))
    v["probe_headroom_rmse_true_D1"] = {"trace": head_tr, "lewm": head_lw}
    v["headline"] = (
        f"trace lambda_lat={p['lambda_lat']:+.4f} vs lewm {pl['lambda_lat']:+.4f}; "
        f"lambda_state={p['lambda_state']:+.4f} vs {pl['lambda_state']:+.4f}; "
        f"trace c20={p['c20']:.3f} ({'contractive' if p['contractive'] else 'expanding'}); "
        f"E_state(20) trace={p['E_state20']:.3f} lewm={pl['E_state20']:.3f} "
        f"(probe headroom on true latents, D1 RMSE: trace={head_tr:.3f}, "
        f"lewm={head_lw:.3f})"
    )
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="S1: closed-loop rollout stability")
    ap.add_argument("--n-windows", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ep-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="all", help="comma list of display names")
    ap.add_argument("--out-dir", default="/home/lx/snn/results/mech_0917_b/rollout")
    args = ap.parse_args()

    models = MODELS if args.models == "all" else \
        [m for m in MODELS if m[0] in set(args.models.split(","))]
    unknown = set(args.models.split(",")) - {m[0] for m in MODELS} if args.models != "all" else set()
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env_data = {}
    for env_name, cfg in ENVS.items():
        min_len = CTX + HMAX
        obs, act, episodes = load_env_data(cfg["npz"], min_len)
        starts, ends, ts = sample_windows(episodes, args.n_windows,
                                          np.random.default_rng(args.seed))
        env_data[env_name] = (obs, act, episodes, starts, ends, ts)
        print(f"[s1] {env_name}: {len(episodes)} usable episodes, {len(ts)} windows",
              flush=True)

    results: dict = {}
    for disp, hint, ckpt in models:
        model = load_model(hint, ckpt, args.device)
        results[disp] = {}
        for env_name, cfg in ENVS.items():
            obs, act, episodes, starts, ends, ts = env_data[env_name]
            rng = np.random.default_rng(args.seed)  # eps/u stream, per (model, env)
            res = run_env_model(model, env_name, cfg["pos"], obs, act, episodes,
                                starts, ends, ts, args, rng)
            results[disp][env_name] = res
            print(f"[s1] {disp} x {env_name}: Elat1={res['E_lat']['1']:.3f} "
                  f"Elat20={res['E_lat']['20']:.3f} lam_lat={res['lambda_lat']:+.4f} "
                  f"lam_state={res['lambda_state']:+.4f} c20={res['c']['20']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    pooled_table = {disp: pooled(results[disp]) for disp in results}
    payload = {
        "experiment": "S1 closed-loop rollout stability (stab0917_b)",
        "protocol": {
            "context_frames": CTX, "horizons": list(range(1, HMAX + 1)),
            "probe_deltas": PROBE_DELTAS, "lambda_hs_latent": LAMBDA_HS,
            "lambda_hs_state": LAMBDA_DELTAS,
            "pad_obs": PAD_OBS, "pad_act": PAD_ACT,
            "rollout": "closed-loop predict() rollout, sliding 27-position context, "
                       "teacher-forced dataset actions; warm full-episode z*",
            "E_lat": "mean_i ||z_hat - z*|| / (||z*|| + 1e-8)",
            "state_probe": "ridge alpha=1.0, standardized features AND targets, "
                           "episode-level 75/25 split, held-out RMSE in standardized "
                           "state space",
            "contraction": f"z_t -> z_t + eps*u, u unit gaussian, eps={EPS_FRAC}*"
                           "mean episode ||z*||; c(h)=mean ||dz_hat_h||/eps; "
                           "lambda_emp=log c(20)/20",
            "seed": args.seed, "n_windows": args.n_windows,
            "checkpoints": {d: c for d, _, c in models},
            "data": {k: v["npz"] for k, v in ENVS.items()},
        },
        "results": results,
        "pooled": pooled_table,
        "verdict": verdict(results),
    }
    json_path = out_dir / "rollout_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False))

    # ---------------- markdown
    lines = [
        "# S1: Closed-loop rollout stability",
        "",
        f"{args.n_windows} windows/env, 27-frame true context, closed-loop rollout "
        f"h=1..20 (teacher-forced actions). lambda = slope of log E(h) vs h over "
        f"h in [2,20] (state domain: Delta grid {LAMBDA_DELTAS}).",
        "",
    ]
    for env_name in ENVS:
        lines += [f"## {env_name}", "",
                  "| model | lambda_lat | lambda_state | E_lat(1) | E_lat(20) | "
                  "state RMSE(20) | state RMSE headroom(20) | state R2(20) | "
                  "c(20) | lambda_emp | contractive |",
                  "|---|" + "---|" * 10]
        for disp, _, _ in models:
            r = results[disp][env_name]
            st = r["state"]["20"]
            lines.append(
                f"| {disp} | {r['lambda_lat']:+.4f} | {r['lambda_state']:+.4f} | "
                f"{r['E_lat']['1']:.3f} | {r['E_lat']['20']:.3f} | "
                f"{st['rmse_pred']:.3f} | {st['rmse_true']:.3f} | {st['r2_pred']:.3f} | "
                f"{r['c']['20']:.3f} | {r['lambda_emp']:+.4f} | "
                f"{'yes' if r['contractive'] else 'no'} |")
        lines += ["", "### State-domain RMSE (standardized) on the Delta grid", "",
                  "| model | " + " | ".join(f"D={d}" for d in PROBE_DELTAS) + " |",
                  "|---|" + "---|" * len(PROBE_DELTAS)]
        for disp, _, _ in models:
            r = results[disp][env_name]
            lines.append(f"| {disp} | " + " | ".join(
                f"{r['state'][str(d)]['rmse_pred']:.3f}" for d in PROBE_DELTAS) + " |")
        lines += ["", "### E_lat on the h grid (relative L2)", "",
                  "| model | " + " | ".join(f"h={h}" for h in [1, 2, 5, 10, 20]) + " |",
                  "|---|" + "---|" * 5]
        for disp, _, _ in models:
            r = results[disp][env_name]
            lines.append(f"| {disp} | " + " | ".join(
                f"{r['E_lat'][str(h)]:.3f}" for h in [1, 2, 5, 10, 20]) + " |")
        lines += ["", "### Contraction c(h)", "",
                  "| model | " + " | ".join(f"h={h}" for h in [1, 5, 10, 20]) + " |",
                  "|---|" + "---|" * 4]
        for disp, _, _ in models:
            r = results[disp][env_name]
            lines.append(f"| {disp} | " + " | ".join(
                f"{r['c'][str(h)]:.3f}" for h in [1, 5, 10, 20]) + " |")
        lines += ["", "### Cosine distance to warm z* (auxiliary; saturates for "
                  "small-norm readouts)", "",
                  "| model | " + " | ".join(f"h={h}" for h in [1, 5, 10, 20]) + " |",
                  "|---|" + "---|" * 4]
        for disp, _, _ in models:
            r = results[disp][env_name]
            lines.append(f"| {disp} | " + " | ".join(
                f"{r['cos_dist'][str(h)]:.4f}" for h in [1, 5, 10, 20]) + " |")
        lines.append("")

    lines += ["## Pooled over envs", "",
              "| model | lambda_lat | lambda_state | E_lat(20) | state RMSE(20) | "
              "c(20) | lambda_emp | contractive |",
              "|---|" + "---|" * 7]
    for disp, _, _ in models:
        p = pooled_table[disp]
        lines.append(
            f"| {disp} | {p['lambda_lat']:+.4f} | {p['lambda_state']:+.4f} | "
            f"{p['E_lat20']:.3f} | {p['E_state20']:.3f} | {p['c20']:.3f} | "
            f"{p['lambda_emp']:+.4f} | {'yes' if p['contractive'] else 'no'} |")
    lines += ["", f"Verdict: {payload['verdict']['headline']}", "",
              f"({time.time() - t0:.0f}s total)", ""]
    md_path = out_dir / "rollout_summary.md"
    md_path.write_text("\n".join(lines))
    print(f"[s1] wrote {json_path} and {md_path} ({time.time() - t0:.0f}s total)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

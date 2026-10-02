"""M2 (mech0917): Open-loop multi-step latent prediction on frozen world models.

Protocol (per model x env):
  1. Sample N windows per env: context = 27 frames [t-26..t] (obs+actions, zero-padded
     to 128 obs dims / 56 action dims, probe.py padding convention), sampled only where
     [t-26..t+20] stays inside one episode.
  2. Context latents: one 27-frame forward -> ctx = out["emb"] (27 positions).
  3. Open-loop rollout h=1..20 with teacher-forced dataset actions: at step h call
     model.predict(ctx[:, -27:], act[:, -27:])[:, -1] (per-position next-latent head,
     the trained CEM/rollout convention), append the prediction and the true action
     a_{t+h} to the sliding buffers. Every model exposes the uniform
     predict(ctx_emb, ctx_act) -> (B, H, D) contract (verified in model code).
  4. Ground-truth latents z*_{t+h}: one warm full-episode forward per episode (all four
     models are prefix-causal), emb at absolute position t+h.
  5. Metrics per (model, env, h):
       (a) mean cosine distance(ẑ_{t+h}, z*_{t+h})  [lower = better]
       (b) state decoding: per Delta in {1,5,10,20}, sklearn Ridge (standardized
           features, alpha=1.0) fit on (z*_{t'+Delta} -> s_{t'+Delta}) with s from the
           dataset state field, probe.py pos slice; episode-level 75/25 split.
           Reported: probe R2 on held-out TRUE latents (headroom) and R2 of the probe
           applied to ẑ at the matching horizon h == Delta (held-out windows).
       (c) normalized degradation E(h) = metric(h) / metric(1).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/lx/snn")
# torch/numpy may pull a module named `code` into sys.modules (stdlib or cwd
# artifact) before the project package is importable; evict any non-package
# binding so `code.*` below resolves to /home/lx/snn/code.
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from code.scripts.event_align import build_model

# ---------------------------------------------------------------- constants
CTX = 27                     # context window length (training window shape)
HORIZONS = [1, 2, 5, 10, 20]  # rollout horizons for latent metrics
DELTAS = [1, 5, 10, 20]       # probe horizons (state decoding)
HMAX = max(HORIZONS)
PAD_OBS, PAD_ACT = 128, 56
STATE_DIM, ACTION_DIM = PAD_OBS, PAD_ACT

MODELS = [
    # (display name, build_model hint, checkpoint)
    ("stjewm_trace_only", "stjewm_trace_only",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt"),
    ("lewm_baseline_v2", "lewm_baseline_v2",
     "/data/lx/tmp/results/5m/oodc_F1/lewm_baseline_v2/seed_0/final.pt"),
    ("gru_baseline", "gru_baseline",
     "/data/lx/tmp/results/5m/oodc_F1/gru_baseline/seed_0/final.pt"),
    ("mlp_baseline", "mlp_baseline",
     "/data/lx/tmp/results/5m/oodc_F1/mlp_baseline/seed_0/final.pt"),
]

ENVS = {
    "cartpole_2d": {
        "npz": "/home/lx/snn/data/dm_control/cartpole_250k.npz",
        "pos": (0, 2),   # ENV_PROBE["cartpole_2d"]["pos"]
    },
    "cheetah": {
        "npz": "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz",
        "pos": (0, 9),   # ENV_PROBE["cheetah"]["pos"]
    },
}


# ---------------------------------------------------------------- data
def load_env_data(npz_path: str, min_len: int):
    """Return obs (N,D) f32, act (N,A) f32, episodes [(start, end_incl)] with len >= min_len."""
    d = np.load(npz_path)
    obs = d["observations"][:, 0, :].astype(np.float32)
    act = d["actions"][:, 0, :].astype(np.float32)
    done = np.asarray(d["dones"]).reshape(-1).astype(bool)
    episodes = []
    pos = 0
    for e in np.flatnonzero(done):
        e = int(e)
        if e >= pos:
            episodes.append((pos, e))
            pos = e + 1
    if pos < len(done):
        episodes.append((pos, len(done) - 1))
    episodes = [(s, e) for (s, e) in episodes if (e - s + 1) >= min_len]
    if not episodes:
        raise ValueError(f"{npz_path}: no episode of length >= {min_len}")
    return obs, act, episodes


def sample_windows(episodes, n: int, rng: np.random.Generator):
    """Uniformly sample n within-episode windows (start, t) needing [t-26..t+20] in-episode."""
    starts, ends, ts = [], [], []
    for (s, e) in episodes:
        lo, hi = s + CTX - 1, e - HMAX
        if hi < lo:
            continue
        starts.append(np.full(hi - lo + 1, s, dtype=np.int64))
        ends.append(np.full(hi - lo + 1, e, dtype=np.int64))
        ts.append(np.arange(lo, hi + 1, dtype=np.int64))
    starts = np.concatenate(starts)
    ends = np.concatenate(ends)
    ts = np.concatenate(ts)
    n = min(n, len(ts))
    idx = rng.choice(len(ts), size=n, replace=False)
    return starts[idx], ends[idx], ts[idx]


# ---------------------------------------------------------------- padding / forward
def pad_batch(seqs: list[np.ndarray], pad: int) -> torch.Tensor:
    """Zero-pad a list of (T,D) arrays to (B,T_max,pad). probe.py padding convention."""
    b = len(seqs)
    t_max = max(s.shape[0] for s in seqs)
    out = np.zeros((b, t_max, pad), dtype=np.float32)
    for j, s in enumerate(seqs):
        out[j, : s.shape[0], : s.shape[1]] = s
    return torch.from_numpy(out)


def forward_windows(model, obs_seqs, act_seqs, device):
    with torch.inference_mode():
        o = pad_batch(obs_seqs, PAD_OBS).to(device)
        a = pad_batch(act_seqs, PAD_ACT).to(device)
        return model(o, a)["emb"]


def forward_episodes(model, obs, act, spans, device, batch: int):
    """Warm full-episode forwards. Returns {start: emb (T,D) cpu float32}."""
    out = {}
    for i in range(0, len(spans), batch):
        chunk = spans[i : i + batch]
        o_seqs = [obs[s : e + 1] for (s, e) in chunk]
        a_seqs = [act[s : e + 1] for (s, e) in chunk]
        with torch.inference_mode():
            o = pad_batch(o_seqs, PAD_OBS).to(device)
            a = pad_batch(a_seqs, PAD_ACT).to(device)
            emb = model(o, a)["emb"]
        for j, (s, e) in enumerate(chunk):
            out[s] = emb[j, : e - s + 1].float().cpu()
    return out


def rollout(model, ctx: torch.Tensor, act_win: torch.Tensor,
            future_acts: torch.Tensor) -> torch.Tensor:
    """Open-loop rollout with teacher-forced actions, sliding CTX-position context.

    ctx (B,CTX,D), act_win (B,CTX,A) on device; future_acts (B,HMAX,A) with
    future_acts[:, i] = a_{t+1+i}. Returns (B,HMAX,D) predicted latents ẑ_{t+1..t+HMAX}.
    """
    buf_e, buf_a = ctx, act_win
    preds = []
    for h in range(1, HMAX + 1):
        # per-position next-latent head: last position predicts z_{t+h} given prefix
        p = model.predict(buf_e[:, -CTX:], buf_a[:, -CTX:])[:, -1]
        preds.append(p)
        if h < HMAX:
            buf_e = torch.cat([buf_e, p.unsqueeze(1)], dim=1)
            buf_a = torch.cat([buf_a, future_acts[:, h - 1].unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


def mean_cos_dist(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((1.0 - F.cosine_similarity(pred, target, dim=-1)).mean().item())


# ---------------------------------------------------------------- probe
def episode_split(ep_starts: np.ndarray, rng: np.random.Generator, frac: float = 0.75):
    """Episode-level 75/25 split. Returns boolean train mask over windows."""
    uniq = np.unique(ep_starts)
    perm = rng.permutation(len(uniq))
    n_train = max(1, min(len(uniq) - 1, int(round(frac * len(uniq)))))
    train_eps = set(uniq[perm[:n_train]].tolist())
    return np.array([s in train_eps for s in ep_starts], dtype=bool)


def probe_eval(z_true_future: np.ndarray, z_pred_h: np.ndarray, s_future: np.ndarray,
               train_mask: np.ndarray, delta: int):
    """Ridge probe z*_{t+Delta} -> s_{t+Delta}; returns (probe R2 on true latents,
    R2 of probe applied to ẑ at horizon h == delta), both held-out."""
    xtr, ytr = z_true_future[train_mask], s_future[train_mask]
    xte, yte = z_true_future[~train_mask], s_future[~train_mask]
    sc = StandardScaler().fit(xtr)
    rg = Ridge(alpha=1.0).fit(sc.transform(xtr), ytr)
    r2_true = float(rg.score(sc.transform(xte), yte))
    r2_pred = float(r2_score(yte, rg.predict(sc.transform(z_pred_h[~train_mask])),
                             multioutput="uniform_average"))
    return r2_true, r2_pred


# ---------------------------------------------------------------- model loading
def load_model(hint: str, ckpt_path: str, device: str):
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(hint, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                        ck_args=ck["args"], state_dict=ck["model"])
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------- main
def run_env_model(model, env_name: str, pos: tuple[int, int], obs, act, episodes,
                  win_starts, win_ends, win_ts, args) -> dict:
    device = args.device
    n = len(win_ts)
    spans = sorted(set(win_starts.tolist()))
    ep_map = dict(episodes)
    spans = [(s, ep_map[s]) for s in spans]
    ep_emb = forward_episodes(model, obs, act, spans, device, batch=args.ep_batch)

    # ---- context forwards + open-loop rollout, batched over windows
    zhat = np.zeros((n, HMAX, 1), dtype=np.float32)  # D filled after first batch
    ctx_reps = []
    bs = args.batch
    pred_batches = []
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        o_seqs = [obs[t - CTX + 1 : t + 1] for t in win_ts[sl]]
        a_seqs = [act[t - CTX + 1 : t + 1] for t in win_ts[sl]]
        with torch.inference_mode():
            ctx = forward_windows(model, o_seqs, a_seqs, device)          # (B,CTX,D)
            # teacher-forced future actions a_{t+1..t+HMAX}
            fut = np.stack([act[t + 1 : t + HMAX + 1] for t in win_ts[sl]])  # (B,HMAX,A)
            fut = np.pad(fut, ((0, 0), (0, 0), (0, PAD_ACT - fut.shape[-1])))
            fut = torch.from_numpy(fut).to(device)
            a_win = pad_batch(a_seqs, PAD_ACT).to(device)
            pred = rollout(model, ctx, a_win, fut).cpu()
        pred_batches.append((sl, pred))

    d_emb = pred_batches[0][1].shape[-1]
    zhat = np.zeros((n, HMAX, d_emb), dtype=np.float32)
    for sl, pr in pred_batches:
        zhat[sl] = pr.numpy()

    # ---- warm ground-truth latents z*_{t+h} and future states s_{t+h}
    z_true = np.zeros((n, HMAX, d_emb), dtype=np.float32)
    p0, p1 = pos
    s_future = np.zeros((n, HMAX, p1 - p0), dtype=np.float32)
    for i, (s_ep, t) in enumerate(zip(win_starts, win_ts)):
        embs = ep_emb[int(s_ep)]
        for h in HORIZONS:
            z_true[i, h - 1] = embs[t + h - int(s_ep)].numpy()
            s_future[i, h - 1] = obs[t + h, p0:p1]

    train_mask = episode_split(win_starts, np.random.default_rng(123))

    zh = torch.from_numpy(zhat).to(device)
    zt = torch.from_numpy(z_true).to(device)
    res = {
        "emb_dim": int(d_emb),
        "n_windows": int(n),
        "n_windows_test": int((~train_mask).sum()),
        "cos_dist": {}, "cos_dist_test": {}, "E_cos": {},
        "probe_true_r2": {}, "state_r2_pred": {}, "E_r2": {},
    }
    cd_all = {h: mean_cos_dist(zh[:, h - 1], zt[:, h - 1]) for h in HORIZONS}
    tm = torch.from_numpy(train_mask).to(device)
    cd_test = {h: mean_cos_dist(zh[:, h - 1][tm], zt[:, h - 1][tm]) for h in HORIZONS}
    res["cos_dist"] = {str(h): cd_all[h] for h in HORIZONS}
    res["cos_dist_test"] = {str(h): cd_test[h] for h in HORIZONS}
    res["E_cos"] = {str(h): cd_all[h] / cd_all[1] for h in HORIZONS}

    for delta in DELTAS:
        r2_true, r2_pred = probe_eval(
            z_true[:, delta - 1], zhat[:, delta - 1], s_future[:, delta - 1],
            train_mask, delta)
        res["probe_true_r2"][str(delta)] = r2_true
        res["state_r2_pred"][str(delta)] = r2_pred
    r1 = res["state_r2_pred"]["1"]
    res["E_r2"] = {str(h): (res["state_r2_pred"][str(h)] / r1 if abs(r1) > 1e-12 else 0.0)
                   for h in DELTAS}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="M2: open-loop multi-step latent prediction")
    ap.add_argument("--n-windows", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64, help="windows per forward batch")
    ap.add_argument("--ep-batch", type=int, default=16, help="episodes per forward batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/home/lx/snn/results/mech_0917/multistep")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    # per-env data + window sampling (shared across models, rng seeded per env)
    env_data = {}
    for env_name, cfg in ENVS.items():
        min_len = CTX + HMAX  # 27 context + 20 future
        obs, act, episodes = load_env_data(cfg["npz"], min_len)
        starts, ends, ts = sample_windows(episodes, args.n_windows,
                                          np.random.default_rng(args.seed))
        env_data[env_name] = (obs, act, episodes, starts, ends, ts)
        print(f"[m2] {env_name}: {len(episodes)} usable episodes, "
              f"{len(ts)} windows sampled", flush=True)

    results: dict = {}
    for disp, hint, ckpt in MODELS:
        model = load_model(hint, ckpt, args.device)
        results[disp] = {}
        for env_name, cfg in ENVS.items():
            obs, act, episodes, starts, ends, ts = env_data[env_name]
            res = run_env_model(model, env_name, cfg["pos"], obs, act, episodes,
                                starts, ends, ts, args)
            results[disp][env_name] = res
            print(f"[m2] {disp} x {env_name}: cos1={res['cos_dist']['1']:.4f} "
                  f"cos20={res['cos_dist']['20']:.4f} E20={res['E_cos']['20']:.2f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "experiment": "M2 open-loop multi-step latent prediction",
        "protocol": {
            "context_frames": CTX, "horizons": HORIZONS, "probe_deltas": DELTAS,
            "pad_obs": PAD_OBS, "pad_act": PAD_ACT,
            "rollout": "open-loop latent rollout via predict(), sliding 27-position "
                       "context, teacher-forced dataset actions a_{t+h-1}",
            "zstar": "warm full-episode forward (prefix-causal), emb at absolute t+h",
            "probe": "sklearn Ridge alpha=1.0 on standardized features, "
                     "episode-level 75/25 split, held-out multioutput R2 (uniform avg)",
            "cosine": "mean cosine distance over all windows (lower=better)",
            "seed": args.seed, "n_windows_requested": args.n_windows,
            "checkpoints": {d: c for d, _, c in MODELS},
            "data": {k: v["npz"] for k, v in ENVS.items()},
        },
        "results": results,
    }
    json_path = out_dir / "multistep_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False))

    # ---------------- markdown
    lines = [
        "# M2: Open-loop multi-step latent prediction",
        "",
        f"Windows/env: {args.n_windows} (contexts of {CTX} frames; warm full-episode "
        f"z*; open-loop predict() rollout to h=20 with teacher-forced actions).",
        "",
    ]
    for env_name in ENVS:
        lines += [f"## {env_name}", "",
                  "### Mean cosine distance to warm z* (lower=better)",
                  "",
                  "| model | " + " | ".join(f"h={h}" for h in HORIZONS) + " |",
                  "|---|" + "---|" * len(HORIZONS)]
        for disp, _, _ in MODELS:
            r = results[disp][env_name]
            lines.append("| " + disp + " | " +
                         " | ".join(f"{r['cos_dist'][str(h)]:.4f}" for h in HORIZONS) + " |")
        lines += ["", "### Normalized degradation E(h) = cos(h)/cos(1)",
                  "", "| model | " + " | ".join(f"h={h}" for h in HORIZONS) + " |",
                  "|---|" + "---|" * len(HORIZONS)]
        for disp, _, _ in MODELS:
            r = results[disp][env_name]
            lines.append("| " + disp + " | " +
                         " | ".join(f"{r['E_cos'][str(h)]:.2f}" for h in HORIZONS) + " |")
        lines += ["", "### State decoding R2: ridge(z* -> s) applied to ẑ at h (held-out)",
                  "", "| model | " + " | ".join(f"h={d}" for d in DELTAS) + " |",
                  "|---|" + "---|" * len(DELTAS)]
        for disp, _, _ in MODELS:
            r = results[disp][env_name]
            lines.append("| " + disp + " (ẑ) | " +
                         " | ".join(f"{r['state_r2_pred'][str(d)]:.3f}" for d in DELTAS) + " |")
            lines.append("| " + disp + " (z*, probe headroom) | " +
                         " | ".join(f"{r['probe_true_r2'][str(d)]:.3f}" for d in DELTAS) + " |")
        lines += ["", "### Normalized degradation E(h) = R2(h)/R2(1) on ẑ decode",
                  "", "| model | " + " | ".join(f"h={d}" for d in DELTAS) + " |",
                  "|---|" + "---|" * len(DELTAS)]
        for disp, _, _ in MODELS:
            r = results[disp][env_name]
            lines.append("| " + disp + " | " +
                         " | ".join(f"{r['E_r2'][str(d)]:.2f}" for d in DELTAS) + " |")
        n_win = results[MODELS[0][0]][env_name]["n_windows"]
        n_te = results[MODELS[0][0]][env_name]["n_windows_test"]
        lines += ["", f"n windows = {n_win} ({n_te} held-out for probes).", ""]

    md_path = out_dir / "multistep_summary.md"
    md_path.write_text("\n".join(lines))
    print(f"[m2] wrote {json_path} and {md_path} ({time.time() - t_start:.0f}s total)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""S2 (stab0917_gain): Latent gain / Jacobian stability on frozen world models.

Question: is STJEWM's advantage a more stable (contractive) effective Jacobian
rather than more accurate one-step prediction?

Protocol (per model x env):
  1. Sample 32 contexts (27-frame windows [t-26..t], [t-26..t+20] inside one episode).
     eps_i = 0.01 * std of the TRUE latents (warm full-episode emb) of context i's
     episode — keeps relative gain comparable across near-zero-norm trace readouts.
  2. Latent FD gain: perturb z_buf[-1] by eps*u (u = random unit vector, 32 draws),
     closed-loop rollout with teacher-forced dataset actions (sliding 27-buffer,
     mech0917_multistep convention). gain(h) = ||dz_hat_h|| / eps for h in {1,5,10,20}.
     Mean/max over (context, direction) pairs. This is the empirical product of
     Jacobians along the closed-loop trajectory.
  3. Obs-side gain: perturb the LAST observation frame by gaussian noise with per-dim
     std sigma = 0.1 * dataset obs std, re-encode the 27-frame window, one predict
     step: gain_obs = ||dz_1|| / ||dobs|| (32 noise draws).
  4. Optional autograd spectral estimate: 8 contexts, 6-iteration power iteration on
     J = d predict-output / d ctx[-1] via torch.func jvp/vjp; top singular value.
     Skipped per model if functorch cannot trace predict (FD above is primary).
  5. Verdict: divergent if mean gain(20) > 1, else contractive.
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

from code.scripts.event_align import build_model

# ---------------------------------------------------------------- constants
CTX = 27
HS = [1, 5, 10, 20]          # gain horizons
HMAX = max(HS)               # 20
PAD_OBS, PAD_ACT = 128, 56
STATE_DIM, ACTION_DIM = PAD_OBS, PAD_ACT
EPS_FRAC = 0.01              # eps = EPS_FRAC * episode latent std
OBS_SIGMA_FRAC = 0.1         # obs perturbation = OBS_SIGMA_FRAC * per-dim obs std
SPEC_CTX, SPEC_ITERS = 8, 6  # power iteration budget

MODELS = [
    # (display name, build_model hint, checkpoint)
    ("stjewm_trace_only", "stjewm_trace_only",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt"),
    ("stjewm_spike_only", "stjewm_spike_only",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_spike_only/seed_0/final.pt"),
    ("stjewm_membrane_readout", "stjewm_membrane_readout",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_membrane_readout/seed_0/final.pt"),
    ("stjewm_hidden_leak", "stjewm_hidden_leak",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_hidden_leak/seed_0/final.pt"),
    ("stacked_lif_trace", "stacked_lif_trace",
     "/data/lx/tmp/results/5m/oodc_F1/stacked_lif_trace/seed_0/final.pt"),
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
        "pos": (0, 2),
    },
    "cheetah": {
        "npz": "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz",
        "pos": (0, 9),
    },
}


# ---------------------------------------------------------------- data (mech0917_multistep skeleton)
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


def load_model(hint: str, ckpt_path: str, device: str):
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(hint, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                        ck_args=ck["args"], state_dict=ck["model"])
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------- S2 gains
def latent_gain(model, ctx: torch.Tensor, act_win: torch.Tensor, fut: torch.Tensor,
                eps: torch.Tensor, dirs: torch.Tensor,
                chunk: int) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """FD gain of the closed-loop rollout wrt a perturbation of ctx[:, -1].

    ctx (B,CTX,D), act_win (B,CTX,A), fut (B,HMAX,A), eps (B,), dirs (K,D) on device.
    Returns {h: gains (B,K)} with gain_{i,j}(h) = ||dz_hat_h|| / eps_i.
    """
    B, C, D = ctx.shape
    K = dirs.shape[0]
    BK = B * K
    ii = torch.arange(BK, device=ctx.device) // K
    jj = torch.arange(BK, device=ctx.device) % K
    with torch.inference_mode():
        base = rollout(model, ctx, act_win, fut)                       # (B,HMAX,D)
        gains = {h: torch.zeros(BK, device=ctx.device) for h in HS}
        for s in range(0, BK, chunk):
            sl = slice(s, min(s + chunk, BK))
            i, j = ii[sl], jj[sl]
            c = ctx[i].clone()
            c[:, -1] += eps[i].unsqueeze(1) * dirs[j]                  # eps * u
            p = rollout(model, c, act_win[i], fut[i])                  # (b,HMAX,D)
            for h in HS:
                dz = (p[:, h - 1] - base[i][:, h - 1]).norm(dim=-1)
                gains[h][sl] = dz / eps[i]
    return {h: g.reshape(B, K).float().cpu() for h, g in gains.items()}, \
        {h: base[:, h - 1].norm(dim=-1).mean().item() for h in HS}


def obs_gain(model, o_win: torch.Tensor, act_pad: torch.Tensor, sigma_vec: torch.Tensor,
             draws: int, chunk: int, gen: torch.Generator) -> torch.Tensor:
    """Obs-side gain: perturb last obs frame by sigma_vec-scaled gaussian, re-encode,
    one predict step. Returns gains (B,draws) = ||dz_1|| / ||dobs||.

    o_win (B,CTX,D_true) on device (true dims only), act_pad (B,CTX,PAD_ACT).
    """
    B, C, Dt = o_win.shape
    BK = B * draws
    ii = torch.arange(BK, device=o_win.device) // draws
    jj = torch.arange(BK, device=o_win.device) % draws
    with torch.inference_mode():
        ctx0 = model(F.pad(o_win, (0, PAD_OBS - Dt)), act_pad)["emb"]
        p0 = model.predict(ctx0[:, -CTX:], act_pad[:, -CTX:])[:, -1]   # (B,D)
        gains = torch.zeros(BK, device=o_win.device)
        for s in range(0, BK, chunk):
            sl = slice(s, min(s + chunk, BK))
            i, _ = ii[sl], jj[sl]
            b = sl.stop - sl.start
            noise = torch.randn((b, Dt), generator=gen).to(o_win.device)
            dobs = noise * sigma_vec.unsqueeze(0)
            o = o_win[i].clone()
            o[:, -1] += dobs
            emb = model(F.pad(o, (0, PAD_OBS - Dt)), act_pad[i])["emb"]
            p1 = model.predict(emb[:, -CTX:], act_pad[i][:, -CTX:])[:, -1]
            gains[sl] = (p1 - p0[i]).norm(dim=-1) / dobs.norm(dim=-1).clamp_min(1e-12)
    return gains.reshape(B, draws).float().cpu()


def _spectral_power(model, ctx: torch.Tensor, act: torch.Tensor, iters: int,
                    gen: torch.Generator) -> list[float]:
    """6-iteration power iteration on J = d predict-output / d ctx[-1], per context."""
    from torch.func import jvp, vjp
    try:  # flash/efficient SDPA lack forward AD; MATH backend is functorch-traceable
        from torch.nn.attention import SDPBackend, sdpa_kernel
        sdpa_ctx = lambda: sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        import contextlib
        sdpa_ctx = contextlib.nullcontext
    per = []
    for i in range(ctx.shape[0]):
        c, a = ctx[i : i + 1], act[i : i + 1]
        prefix = c[:, :-1]
        last = c[:, -1]

        def f(v, prefix=prefix, last=last, a=a):
            full = torch.cat([prefix, (last + v).unsqueeze(1)], dim=1)
            return model.predict(full, a)[0, -1]

        D = last.shape[-1]
        v = torch.randn(D, generator=gen)
        v = (v / v.norm()).to(c.device)
        for _ in range(iters):
            with sdpa_ctx():
                _, Jv = jvp(f, (v,), (v,))
                _, pull = vjp(f, v)
                JtJv = pull(Jv)[0]
            v = (JtJv / JtJv.norm()).detach()
        with sdpa_ctx():
            _, Jv = jvp(f, (v,), (v,))
        per.append(float(Jv.norm()))
    return per


def spectral_gain(model, ctx: torch.Tensor, act_win: torch.Tensor, gen: torch.Generator,
                  n_ctx: int = SPEC_CTX, iters: int = SPEC_ITERS) -> dict:
    """Top singular value of the one-step Jacobian via autograd power iteration.
    Falls back to CPU if functorch cannot trace the model on GPU."""
    n = min(n_ctx, ctx.shape[0])
    try:
        per = _spectral_power(model, ctx[:n], act_win[:n], iters, gen)
        return {"status": "ok", "device": str(ctx.device), "per_ctx": per,
                "top_sv_mean": float(np.mean(per))}
    except Exception as e_gpu:  # functorch tracing failure: retry on CPU
        dev = next(model.parameters()).device
        try:
            model.to("cpu")
            per = _spectral_power(model, ctx[:n].cpu(), act_win[:n].cpu(), iters, gen)
            return {"status": "ok", "device": "cpu", "per_ctx": per,
                    "top_sv_mean": float(np.mean(per))}
        except Exception as e_cpu:
            return {"status": "skipped", "reason": f"cuda: {type(e_gpu).__name__}: "
                    f"{str(e_gpu)[:200]} | cpu: {type(e_cpu).__name__}: {str(e_cpu)[:200]}"}
        finally:
            model.to(dev)


def pair_stats(g: np.ndarray) -> dict:
    return {"mean": float(g.mean()), "max": float(g.max()),
            "median": float(np.median(g))}


# ---------------------------------------------------------------- per (model, env)
def run_env_model(model, env_name: str, obs, act, episodes, win_starts, win_ts, args) -> dict:
    device = args.device
    n = len(win_ts)
    ep_map = dict(episodes)
    spans = sorted(set(win_starts.tolist()))
    spans = [(s, ep_map[s]) for s in spans]

    # eps from TRUE latent scale of each context's episode (warm forward)
    ep_emb = forward_episodes(model, obs, act, spans, device, batch=args.ep_batch)
    eps_np = np.array([float(ep_emb[s].std()) for s in win_starts], dtype=np.float32)
    eps = (torch.from_numpy(eps_np) * EPS_FRAC).to(device)

    o_seqs = [obs[t - CTX + 1 : t + 1] for t in win_ts]
    a_seqs = [act[t - CTX + 1 : t + 1] for t in win_ts]
    # no_grad (NOT inference_mode): ctx feeds the autograd jvp/vjp spectral phase,
    # and inference tensors cannot participate in torch.func transforms.
    with torch.no_grad():
        o = pad_batch(o_seqs, PAD_OBS).to(device)
        a = pad_batch(a_seqs, PAD_ACT).to(device)
        ctx = model(o, a)["emb"]                                        # (B,CTX,D)
    fut_np = np.stack([act[t + 1 : t + HMAX + 1] for t in win_ts])
    fut_np = np.pad(fut_np, ((0, 0), (0, 0), (0, PAD_ACT - fut_np.shape[-1])))
    fut = torch.from_numpy(fut_np).to(device)
    act_pad = pad_batch(a_seqs, PAD_ACT).to(device)

    D = ctx.shape[-1]
    dirs = torch.randn((args.dirs, D), generator=args.gen)
    dirs = (dirs / dirs.norm(dim=-1, keepdim=True)).to(device)

    gains, base_norm = latent_gain(model, ctx, act_pad, fut, eps, dirs, args.chunk)

    # obs-side: last-frame perturbation, sigma = 0.1 * per-dim dataset obs std
    Dt = obs.shape[1]
    sigma_vec = (OBS_SIGMA_FRAC * obs.std(axis=0).astype(np.float32))
    sigma = torch.from_numpy(sigma_vec).to(device)
    o_win = torch.from_numpy(np.stack(o_seqs)).to(device)               # (B,CTX,Dt)
    g_obs = obs_gain(model, o_win, act_pad, sigma, args.obs_draws, args.chunk, args.gen)

    spec = spectral_gain(model, ctx, act_pad, args.gen)

    g20 = gains[HMAX].numpy()
    res = {
        "emb_dim": int(D),
        "n_contexts": int(n), "n_dirs": int(args.dirs), "n_obs_draws": int(args.obs_draws),
        "eps_mean": float(eps_np.mean()), "eps_median": float(np.median(eps_np)),
        "base_pred_norm": {str(h): base_norm[h] for h in HS},
        "gain_h": {str(h): pair_stats(gains[h].numpy().ravel()) for h in HS},
        "gain20_gt1_frac": float((g20 > 1.0).mean()),
        "gain_obs": pair_stats(g_obs.numpy().ravel()),
        "verdict": "divergent" if g20.mean() > 1.0 else "contractive",
        "spectral": spec,
        "raw_gain1": gains[1].numpy().ravel().tolist(),
        "raw_gain20": gains[HMAX].numpy().ravel().tolist(),
        "raw_gain_obs": g_obs.numpy().ravel().tolist(),
    }
    return res


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="S2: latent gain / Jacobian stability")
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--dirs", type=int, default=32)
    ap.add_argument("--obs-draws", type=int, default=32)
    ap.add_argument("--spec-ctx", type=int, default=SPEC_CTX)
    ap.add_argument("--spec-iters", type=int, default=SPEC_ITERS)
    ap.add_argument("--chunk", type=int, default=256, help="rollout rows per forward batch")
    ap.add_argument("--ep-batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0, help="window sampling rng")
    ap.add_argument("--gen-seed", type=int, default=123, help="torch rng: dirs/noise/power init")
    ap.add_argument("--models", default=None, help="comma-separated display-name filter (smoke)")
    ap.add_argument("--out-dir", default="/home/lx/snn/results/mech_0917_b/gain")
    args = ap.parse_args()
    args.gen = torch.Generator().manual_seed(args.gen_seed)

    keep = None
    if args.models:
        keep = {s.strip() for s in args.models.split(",") if s.strip()}
    models = [(d, h, c) for (d, h, c) in MODELS if keep is None or d in keep]
    if not models:
        print(f"[s2gain] model filter {keep} matched nothing", flush=True)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    env_data = {}
    for env_name, cfg in ENVS.items():
        min_len = CTX + HMAX
        obs, act, episodes = load_env_data(cfg["npz"], min_len)
        starts, _, ts = sample_windows(episodes, args.contexts,
                                       np.random.default_rng(args.seed))
        env_data[env_name] = (obs, act, episodes, starts, ts)
        print(f"[s2gain] {env_name}: {len(episodes)} usable episodes, "
              f"{len(ts)} contexts sampled", flush=True)

    results: dict = {}
    for disp, hint, ckpt in models:
        model = load_model(hint, ckpt, args.device)
        results[disp] = {}
        for env_name in ENVS:
            obs, act, episodes, starts, ts = env_data[env_name]
            res = run_env_model(model, env_name, obs, act, episodes, starts, ts, args)
            results[disp][env_name] = res
            gh = res["gain_h"]
            print(f"[s2gain] {disp} x {env_name}: gain1={gh['1']['mean']:.3f} "
                  f"gain20={gh['20']['mean']:.3f} obs={res['gain_obs']['mean']:.3f} "
                  f"spec={res['spectral'].get('top_sv_mean', 'skip')} -> "
                  f"{res['verdict']} ({time.time() - t_start:.0f}s)", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # pooled across envs (equal pair counts: simple average of env stats)
    pooled = {}
    for disp in results:
        r = results[disp]
        g1 = np.concatenate([r[e]["raw_gain1"] for e in ENVS])
        g20 = np.concatenate([r[e]["raw_gain20"] for e in ENVS])
        go = np.concatenate([r[e]["raw_gain_obs"] for e in ENVS])
        pooled[disp] = {
            "gain1": pair_stats(g1), "gain20": pair_stats(g20), "gain_obs": pair_stats(go),
            "verdict": "divergent" if g20.mean() > 1.0 else "contractive",
            "spectral": {e: {"status": r[e]["spectral"]["status"],
                             "top_sv_mean": r[e]["spectral"].get("top_sv_mean")}
                         for e in ENVS},
        }

    payload = {
        "experiment": "S2 latent gain / Jacobian stability (stab0917_gain)",
        "protocol": {
            "context_frames": CTX, "gain_horizons": HS,
            "n_contexts": args.contexts, "n_dirs": args.dirs, "n_obs_draws": args.obs_draws,
            "eps": f"{EPS_FRAC} * std of warm true-latent episode emb (per context)",
            "latent_gain": "FD: perturb z_buf[-1] += eps*u (random unit), closed-loop "
                           "predict() rollout to h=20, teacher-forced dataset actions; "
                           "gain(h)=||dz_hat_h||/eps",
            "obs_gain": f"gaussian noise, per-dim std = {OBS_SIGMA_FRAC} * dataset obs std, "
                        "on last obs frame, re-encode 27-frame window, 1 predict step; "
                        "gain_obs=||dz_1||/||dobs||",
            "spectral": f"torch.func jvp/vjp power iteration ({SPEC_ITERS} iters) on "
                        f"J=d predict/d ctx[-1], {SPEC_CTX} contexts; skipped if untraceable",
            "verdict_rule": "divergent iff mean gain(20) > 1 over (context, direction) pairs",
            "seed_window": args.seed, "seed_gen": args.gen_seed,
            "checkpoints": {d: c for d, _, c in models},
            "data": {k: v["npz"] for k, v in ENVS.items()},
        },
        "results": results,
        "pooled": pooled,
    }
    json_path = out_dir / "gain_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False))

    # ---------------- markdown
    def gcell(s: dict) -> str:
        return f"{s['mean']:.3f} / {s['max']:.3g}"

    lines = [
        "# S2: Latent gain / Jacobian stability",
        "",
        f"{args.contexts} contexts x {args.dirs} directions per (model, env); "
        f"eps = {EPS_FRAC}x episode latent std. Cells: mean / max over pairs. "
        "Verdict: divergent iff mean gain(20) > 1.",
        "",
    ]
    div = [d for d in results if pooled[d]["verdict"] == "divergent"]
    lines += ["**Divergent (pooled gain(20) > 1):** " +
              (", ".join(div) if div else "none") , ""]

    for env_name in ENVS:
        lines += [f"## {env_name}", "",
                  "### Latent FD gain gain(h) = ||dz_hat_h||/eps (mean / max)",
                  "",
                  "| model | " + " | ".join(f"h={h}" for h in HS) +
                  " | verdict | frac gain20>1 |",
                  "|---|" + "---|" * (len(HS) + 2)]
        for disp, _, _ in models:
            r = results[disp][env_name]
            cells = [gcell(r["gain_h"][str(h)]) for h in HS]
            lines.append("| " + disp + " | " + " | ".join(cells) + " | " +
                         r["verdict"] + f" | {r['gain20_gt1_frac']:.2f} |")

        lines += ["", "### Obs-side gain ||dz_1||/||dobs|| (mean / max)", "",
                  "| model | gain_obs |", "|---|---|"]
        for disp, _, _ in models:
            lines.append("| " + disp + f" | {gcell(results[disp][env_name]['gain_obs'])} |")

        lines += ["", "### One-step Jacobian top singular value (autograd power iteration)",
                  "", "| model | top sv | device |", "|---|---|---|"]
        for disp, _, _ in models:
            sp = results[disp][env_name]["spectral"]
            if sp["status"] == "ok":
                lines.append(f"| {disp} | {sp['top_sv_mean']:.3f} | {sp['device']} |")
            else:
                lines.append(f"| {disp} | skipped | {sp.get('reason', '')[:80]} |")
        lines.append("")

    lines += ["## Pooled (both envs)", "",
              "| model | gain1 | gain5 | gain10 | gain20 | gain_obs | verdict |",
              "|---|" + "---|" * 6]
    for disp, _, _ in models:
        p = pooled[disp]
        g_mid = [np.mean([results[disp][e]["gain_h"][str(h)]["mean"]
                          for e in ENVS]) for h in HS[1:-1]]
        lines.append("| " + disp + " | " + gcell(p["gain1"]) + " | " +
                     " | ".join(f"{v:.3f}" for v in g_mid) +
                     " | " + gcell(p["gain20"]) + " | " + gcell(p["gain_obs"]) + " | " +
                     p["verdict"] + " |")
    lines.append("")

    md_path = out_dir / "gain_summary.md"
    md_path.write_text("\n".join(lines))
    print(f"[s2gain] wrote {json_path} and {md_path} ({time.time() - t_start:.0f}s total)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

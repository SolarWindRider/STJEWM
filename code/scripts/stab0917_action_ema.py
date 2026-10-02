"""S3/S6 (stab0917_b): reviewer-attack controls on frozen world models.

PART A (S3/Exp6) — action sensitivity, 8 models x 2 envs x 32 contexts x 16 pairs:
    For each 27-frame context, sample 16 action pairs a1 = a + d, a2 = a - d
  (a ~ U(act_min + d, act_max - d) per dim, d = 0.1 * per-dim action range):
    - S1step = ||F(z,a1) - F(z,a2)|| / ||(a1-a2)/act_std||  (F = one-step predict
      head; the differing action sits at the context's last position, i.e. the
      z_{t+1}-driving action under the per-position next-latent convention).
      Also reported relative to the prediction norm (cross-model comparable).
    - Impulse persistence: a1 vs a2 ONLY at that first (z_{t+1}-driving) action;
      steps 2..10 teacher-forced shared dataset actions; ||dz_h|| for h=1..10
      (dz_1 equals the S1step displacement by construction).
    - Sustained: center +- d at EVERY step (first action and all rollout steps),
      ||dz_h|| for h=1..10.
  Persistence is reported raw, relative to h=1 (wash-out shape), and relative to
  the base (dataset-action) rollout latent norm (cross-model comparable).

PART B (S6/Exp5) — EMA control, LeWM vs stjewm_trace_only:
  (a) Response matching: last-frame obs + 0.1*per-dim obs-std noise (gain_obs
      protocol); response = displacement of the one-step predicted latent when the
      context is perturbed, under EMA z' = alpha*p + (1-alpha)*z_ctx, for alpha in
      {0.1,0.3,0.5,0.7,0.9,1.0}. Primary variant anchors the EMA at the SHARED
      pre-perturbation context latent (both counterfactual worlds share history),
      so resp(alpha) = alpha*resp(1); a perturbed-anchor variant is reported as a
      diagnostic (encoder-pathway contribution, EMA-immune). alpha* = grid
      argmin |resp_LeWM(alpha) - resp_trace(1)|; alpha_required = exact ratio.
  (b) Closed-loop env-SR with EMA active in the CEM planner: cem_predict_hook
      EMA-wraps model.predict so every latent appended to the planner buffer is
      z'_t = alpha*z_t + (1-alpha)*z'_{t-1} (anchor at rollout start = observed
      history latent). Standard budget: CEM 300/30/10, horizon 25, budget 50,
      goal_offset 25, history 3; 5 episodes, 1 seed. alpha in {alpha*, 0.5, 1.0},
      plus a stjewm_trace_only anchor at the same episodes.

Anti-saturation rule: primary cross-model quantities are raw displacement AND
latent-relative displacement (dz/||z||); cosine is not used.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, "/home/lx/snn")
# torch/numpy may pull a module named `code` into sys.modules (stdlib or cwd
# artifact) before the project package is importable; evict any non-package
# binding so `code.*` below resolves to /home/lx/snn/code.
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

import numpy as np
import torch

from code.scripts.mech0917_multistep import (
    CTX, PAD_OBS, PAD_ACT, ENVS, load_env_data, sample_windows, pad_batch,
    forward_windows, load_model,
)
from code.eval.closed_loop import eval_closed_loop, make_env

# ---------------------------------------------------------------- constants
HPH = 10                      # persistence rollout horizon (Part A)
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
CL_CEM = dict(cem_samples=300, cem_elites=30, cem_iters=10,
              horizon=25, eval_budget=50, goal_offset=25, history_size=3)
CL_ENV_KIND = {"cartpole_2d": "cartpole", "cheetah": "cheetah"}

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
    ("lewm_baseline_v2", "lewm_baseline_v2",
     "/data/lx/tmp/results/5m/oodc_F1/lewm_baseline_v2/seed_0/final.pt"),
    ("gru_baseline", "gru_baseline",
     "/data/lx/tmp/results/5m/oodc_F1/gru_baseline/seed_0/final.pt"),
    ("mlp_baseline", "mlp_baseline",
     "/data/lx/tmp/results/5m/oodc_F1/mlp_baseline/seed_0/final.pt"),
    ("stacked_lif_trace", "stacked_lif_trace",
     "/data/lx/tmp/results/5m/oodc_F1/stacked_lif_trace/seed_0/final.pt"),
]
MODEL_IDX = {d: i for i, (d, _, _) in enumerate(MODELS)}


# ---------------------------------------------------------------- shared sampling
def sample_contexts(episodes, n: int):
    """Deterministic context windows (rng 0), needing [t-26..t+HPH] in-episode."""
    starts, _, ts = sample_windows(episodes, n, np.random.default_rng(0))
    return ts


def sample_action_pairs(obs, act, ts, n_pairs: int):
    """16 (a1,a2) pairs per context: a ~ U(min+d, max-d), d = 0.1*range per dim."""
    nat = act.shape[1]
    act_min = act.min(0).astype(np.float32)
    act_max = act.max(0).astype(np.float32)
    act_rng = np.maximum(act_max - act_min, 1e-6).astype(np.float32)
    act_std = np.maximum(act.std(0), 1e-6).astype(np.float32)
    d = (0.1 * act_rng).astype(np.float32)
    n = len(ts)
    rng = np.random.default_rng(123)
    centers = rng.uniform(act_min + d, act_max - d, size=(n, n_pairs, nat))
    centers = centers.astype(np.float32)
    a1 = np.clip(centers + d, act_min, act_max).astype(np.float32)
    a2 = np.clip(centers - d, act_min, act_max).astype(np.float32)
    denom = np.linalg.norm((a1 - a2) / act_std, axis=2).reshape(-1)  # (n*pairs,)
    return centers, a1, a2, d, denom, nat


def pad_nat(x: np.ndarray, nat: int, pad: int) -> np.ndarray:
    return np.pad(x, ((0, 0), (0, pad - nat))) if x.ndim == 2 else \
        np.pad(x, ((0, 0), (0, 0), (0, pad - nat)))


# ---------------------------------------------------------------- Part A
def rollout_h(model, ctx: torch.Tensor, act_win: torch.Tensor,
              future_acts: torch.Tensor, hmax: int) -> torch.Tensor:
    """Sliding-27-buffer open-loop rollout (mech0917_multistep convention), h=1..hmax."""
    buf_e, buf_a = ctx, act_win
    preds = []
    for h in range(1, hmax + 1):
        p = model.predict(buf_e[:, -CTX:], buf_a[:, -CTX:])[:, -1]
        preds.append(p)
        if h < hmax:
            buf_e = torch.cat([buf_e, p.unsqueeze(1)], dim=1)
            buf_a = torch.cat([buf_a, future_acts[:, h - 1].unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


def part_a(model, env_name: str, obs, act, episodes, device, args) -> dict:
    ts = sample_contexts(episodes, args.contexts)
    n = len(ts)
    centers, a1, a2, d, denom, nat = sample_action_pairs(obs, act, ts, args.pairs)
    P = args.pairs
    nP = n * P

    o_seqs = [obs[t - CTX + 1 : t + 1] for t in ts]
    a_seqs = [act[t - CTX + 1 : t + 1] for t in ts]
    fut = np.stack([act[t + 1 : t + HPH + 1] for t in ts])           # (n,HPH,nat)

    with torch.inference_mode():
        ctx = forward_windows(model, o_seqs, a_seqs, device)          # (n,CTX,D)
        ctx_rep = ctx.repeat_interleave(P, dim=0)                     # (nP,CTX,D)

        a1_flat = pad_nat(a1.reshape(nP, nat), nat, PAD_ACT)
        a2_flat = pad_nat(a2.reshape(nP, nat), nat, PAD_ACT)
        a_win = pad_nat(np.stack(a_seqs), nat, PAD_ACT)               # (n,CTX,56)
        a_win_rep = torch.from_numpy(np.repeat(a_win, P, axis=0)).to(device)
        fut_rep = pad_nat(np.repeat(fut, P, axis=0), nat, PAD_ACT)    # (nP,HPH,56)
        fut_As = np.repeat(a1_flat[:, None, :], HPH, axis=1)          # (nP,HPH,56)
        fut_Bs = np.repeat(a2_flat[:, None, :], HPH, axis=1)

        def roll(act_win: torch.Tensor, fut_np) -> torch.Tensor:
            fut_t = torch.from_numpy(np.ascontiguousarray(fut_np)).to(device)
            outs = []
            for i in range(0, nP, args.batch):
                sl = slice(i, min(i + args.batch, nP))
                outs.append(rollout_h(model, ctx_rep[sl], act_win[sl], fut_t[sl], HPH))
            return torch.cat(outs, dim=0)

        # ---- S1step: one-step predict with the action driving z_{t+1} set to a1 vs a2
        aw1, aw2 = a_win_rep.clone(), a_win_rep.clone()
        aw1[:, -1, :] = torch.from_numpy(a1_flat).to(device)
        aw2[:, -1, :] = torch.from_numpy(a2_flat).to(device)
        preds = []
        for aw in (aw1, aw2):
            ps = []
            for i in range(0, nP, args.batch):
                sl = slice(i, min(i + args.batch, nP))
                ps.append(model.predict(ctx_rep[sl], aw[sl])[:, -1])
            preds.append(torch.cat(ps, dim=0))
        p1, p2 = preds
        dz1 = (p1 - p2).norm(dim=1)                                    # (nP,)
        denom_t = torch.from_numpy(denom).to(device)
        s1_raw = float((dz1 / denom_t).mean())
        lat_norm = p1.norm(dim=1)
        s1_rel = float((dz1 / (lat_norm + 1e-8) / denom_t).mean())

        # ---- persistence rollouts (impulse: a1/a2 only at the z_{t+1}-driving
        # action, then shared dataset actions; sustained: a1/a2 at every step)
        rAi = roll(aw1, fut_rep)
        rBi = roll(aw2, fut_rep)
        rAs = roll(aw1, fut_As)
        rBs = roll(aw2, fut_Bs)
        rb = roll(a_win_rep, fut_rep)                                  # dataset-action base

        def curves(ra: torch.Tensor, rbb: torch.Tensor) -> dict:
            dz = (ra - rbb).norm(dim=-1).mean(0)                       # (HPH,)
            base_norm = rb.norm(dim=-1).mean(0) + 1e-8
            return {
                "raw": [float(v) for v in dz],
                "rel_to_h1": [float(v) for v in dz / (dz[0] + 1e-12)],
                "rel_to_latent": [float(v) for v in dz / base_norm],
            }

        res = {
            "n_contexts": n, "n_pairs": P, "n_rows": nP,
            "denom_mean": float(denom.mean()),
            "latent_norm_mean": float(lat_norm.mean()),
            "s1step_raw": s1_raw,
            "s1step_rel": s1_rel,
            "s1step_raw_median": float((dz1 / denom_t).median()),
            "impulse": curves(rAi, rBi),
            "sustained": curves(rAs, rBs),
        }
    return res


def pooled(res_by_env: dict) -> dict:
    """Pool Part-A metrics across envs (plain mean of per-env numbers)."""
    out = {}
    for key in ("s1step_raw", "s1step_rel", "denom_mean", "latent_norm_mean"):
        out[key] = float(np.mean([r[key] for r in res_by_env.values()]))
    for kind in ("impulse", "sustained"):
        out[kind] = {
            m: [float(np.mean([r[kind][m][h] for r in res_by_env.values()]))
                for h in range(HPH)]
            for m in ("raw", "rel_to_h1", "rel_to_latent")
        }
    return out


# ---------------------------------------------------------------- Part B(a)
def part_ba_response(model, obs, act, episodes, device, args) -> dict:
    """Obs-perturbation response of the one-step predicted latent under EMA(alpha)."""
    ts = sample_contexts(episodes, args.contexts)
    n = len(ts)
    obs_nat = obs.shape[1]
    obs_std = np.maximum(obs.std(0), 1e-6).astype(np.float32)

    o_seqs = [obs[t - CTX + 1 : t + 1] for t in ts]
    o_seqs_p = []
    for seq in o_seqs:
        s2 = seq.copy()
        s2[-1, :obs_nat] += 0.1 * obs_std
        o_seqs_p.append(s2)
    a_seqs = [act[t - CTX + 1 : t + 1] for t in ts]
    nat = act.shape[1]

    with torch.inference_mode():
        ctx = forward_windows(model, o_seqs, a_seqs, device)
        ctx_p = forward_windows(model, o_seqs_p, a_seqs, device)
        a_win = torch.from_numpy(pad_nat(np.stack(a_seqs), nat, PAD_ACT)).to(device)

        # (i) encoder response at the last context position
        enc_dz = (ctx_p[:, -1, :] - ctx[:, -1, :]).norm(dim=1)

        # (ii) one-step predicted-latent response under EMA. Primary (shared
        # anchor): both counterfactual worlds share the pre-perturbation
        # history, so the EMA anchor is the UNPERTURBED context latent and the
        # response is alpha * (p_pert - p). Diagnostic (perturbed anchor): the
        # perturbed current-obs latent anchors the EMA, isolating the
        # encoder-pathway contribution that EMA-on-predictions cannot shrink.
        p = model.predict(ctx[:, -CTX:], a_win[:, -CTX:])[:, -1]
        p_p = model.predict(ctx_p[:, -CTX:], a_win[:, -CTX:])[:, -1]
        anchor = ctx[:, -1, :]
        anchor_p = ctx_p[:, -1, :]

        def resp(alpha: float, perturb_anchor: bool) -> tuple[float, float]:
            anc_p = anchor_p if perturb_anchor else anchor
            z = alpha * p + (1 - alpha) * anchor
            zp = alpha * p_p + (1 - alpha) * anc_p
            dz = (zp - z).norm(dim=1)
            raw = float(dz.mean())
            rel = float((dz / (z.norm(dim=1) + 1e-8)).mean())
            return raw, rel

        res = {
            "n_contexts": n,
            "enc_resp_raw": float(enc_dz.mean()),
            "enc_resp_rel": float((enc_dz / (ctx[:, -1, :].norm(dim=1) + 1e-8)).mean()),
            "pred_resp_raw": {}, "pred_resp_rel": {},
            "pred_resp_pertanchor_raw": {}, "pred_resp_pertanchor_rel": {},
        }
        for alpha in ALPHAS:
            raw, rel = resp(alpha, perturb_anchor=False)
            res["pred_resp_raw"][str(alpha)] = raw
            res["pred_resp_rel"][str(alpha)] = rel
            raw, rel = resp(alpha, perturb_anchor=True)
            res["pred_resp_pertanchor_raw"][str(alpha)] = raw
            res["pred_resp_pertanchor_rel"][str(alpha)] = rel
    return res


def match_alpha(resp_lewm_pooled: dict, resp_trace_pooled: dict) -> dict:
    """alpha* = argmin_alpha |resp_LeWM(alpha) - resp_trace(1.0)| (pooled over envs).
    With the shared-anchor EMA the response is analytic, resp(alpha) = alpha*resp(1),
    so the exact-match requirement is alpha_req = resp_trace(1)/resp_LeWM(1); the
    grid-constrained alpha* is the grid value closest to it."""
    out = {}
    for metric in ("pred_resp_raw", "pred_resp_rel"):
        trace_v = resp_trace_pooled[metric]["1.0"]
        lewm_1 = resp_lewm_pooled[metric]["1.0"]
        alpha_req = trace_v / lewm_1 if abs(lewm_1) > 1e-30 else float("nan")
        diffs = {a: abs(resp_lewm_pooled[metric][a] - trace_v) for a in resp_lewm_pooled[metric]}
        best = min(diffs, key=diffs.get)
        out[metric] = {"alpha_star": float(best), "alpha_required_unclipped": alpha_req,
                       "trace_response": trace_v,
                       "lewm_at_alpha_star": resp_lewm_pooled[metric][best],
                       "abs_diff": diffs[best], "diffs": diffs}
    return out


# ---------------------------------------------------------------- Part B(b)
def ema_predict_hook(alpha: float):
    """EMA-wrap predict so every latent appended to the CEM planner buffer is
    z'_t = alpha*z_t + (1-alpha)*z'_{t-1} (buffer last entry = z'_{t-1}; at
    rollout start it is the observed history latent)."""
    def hook(model, ctx_emb, ctx_act):
        out = model.predict(ctx_emb, ctx_act)
        out = out.clone()
        out[:, -1] = alpha * out[:, -1] + (1 - alpha) * ctx_emb[:, -1]
        return out
    return hook


def run_closed_loop(model_name: str, alpha: float, env_name: str, device: str,
                    args) -> dict:
    disp, hint, ckpt = MODELS[MODEL_IDX[model_name]]
    cfg = ENVS[env_name]
    model = load_model(hint, ckpt, device)
    env = make_env(CL_ENV_KIND[env_name])
    hook = ema_predict_hook(alpha) if alpha < 1.0 else None
    t0 = time.time()
    res = eval_closed_loop(
        model, env, cfg["npz"],
        n_episodes=args.episodes, n_seeds=1, seed_start=0,
        device=device, pad_obs_to=PAD_OBS, model_action_dim=PAD_ACT,
        cem_predict_hook=hook, **CL_CEM,
    )
    out = {
        "model": disp, "alpha": alpha, "env": env_name,
        "env_success_rate": res.success_rate_env,
        "mean_cos_dist": res.mean_cos_dist,
        "mean_phys_dist": res.mean_phys_dist,
        "success_rate_cos01": res.success_rate_lewm,
        "n_episodes": len(res.per_episode),
        "per_episode": [
            {k: e[k] for k in ("episode_idx", "env_success", "cos_dist", "phys_dist",
                               "actions_taken", "planning_calls")}
            for e in res.per_episode
        ],
        "wall_time_sec": time.time() - t0,
    }
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="S3/S6: action sensitivity + EMA control")
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256, help="rows per predict batch")
    ap.add_argument("--episodes", type=int, default=5, help="closed-loop episodes")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-closed-loop", action="store_true")
    ap.add_argument("--out-dir", default="/home/lx/snn/results/mech_0917_b/action_ema")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.contexts, args.pairs, args.episodes = 4, 2, 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = args.device

    part_a_models = MODELS if not args.smoke else [MODELS[MODEL_IDX[m]] for m in
                                                   ("stjewm_trace_only", "lewm_baseline_v2")]
    envs_run = list(ENVS) if not args.smoke else ["cartpole_2d"]

    # ---------------- data (shared across models)
    env_data = {}
    for env_name, cfg in ENVS.items():
        obs, act, episodes = load_env_data(cfg["npz"], CTX + HPH)
        env_data[env_name] = (obs, act, episodes)
        print(f"[s3s6] {env_name}: {len(episodes)} usable episodes", flush=True)

    # ---------------- PART A: action sensitivity
    pa: dict = {}
    for disp, hint, ckpt in part_a_models:
        model = load_model(hint, ckpt, device)
        pa[disp] = {}
        for env_name in envs_run:
            obs, act, episodes = env_data[env_name]
            pa[disp][env_name] = part_a(model, env_name, obs, act, episodes, device, args)
            print(f"[s3s6/A] {disp} x {env_name}: S1step_raw={pa[disp][env_name]['s1step_raw']:.4g} "
                  f"S1step_rel={pa[disp][env_name]['s1step_rel']:.4g} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---------------- PART B(a): response matching (LeWM vs trace)
    ba_models = [("stjewm_trace_only", "stjewm_trace_only",
                  MODELS[MODEL_IDX["stjewm_trace_only"]][2]),
                 ("lewm_baseline_v2", "lewm_baseline_v2",
                  MODELS[MODEL_IDX["lewm_baseline_v2"]][2])]
    ba: dict = {}
    for disp, hint, ckpt in ba_models:
        model = load_model(hint, ckpt, device)
        ba[disp] = {}
        for env_name in envs_run:
            obs, act, episodes = env_data[env_name]
            ba[disp][env_name] = part_ba_response(model, obs, act, episodes, device, args)
            print(f"[s3s6/Ba] {disp} x {env_name}: resp@1.0="
                  f"{ba[disp][env_name]['pred_resp_raw']['1.0']:.4g} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    def pooled_resp(disp: str) -> dict:
        per_env = ba[disp]
        out = {k: float(np.mean([per_env[e][k] for e in per_env]))
               for k in ("enc_resp_raw", "enc_resp_rel")}
        for metric in ("pred_resp_raw", "pred_resp_rel",
                       "pred_resp_pertanchor_raw", "pred_resp_pertanchor_rel"):
            out[metric] = {a: float(np.mean([per_env[e][metric][a] for e in per_env]))
                           for a in per_env[envs_run[0]][metric]}
        return out

    resp_trace_pool = pooled_resp("stjewm_trace_only")
    resp_lewm_pool = pooled_resp("lewm_baseline_v2")
    match = match_alpha(resp_lewm_pool, resp_trace_pool)
    alpha_star = match["pred_resp_rel"]["alpha_star"]
    print(f"[s3s6/Ba] alpha* (rel match) = {alpha_star} | "
          f"alpha* (raw match) = {match['pred_resp_raw']['alpha_star']}", flush=True)

    # ---------------- PART B(b): closed-loop env-SR with EMA in the planner
    bb: dict = {}
    if not args.skip_closed_loop:
        sr_alphas = sorted({alpha_star, 0.5, 1.0})
        jobs = [(f"lewm_baseline_v2@alpha={a}", "lewm_baseline_v2", a) for a in sr_alphas]
        jobs.append(("stjewm_trace_only@alpha=1.0 (anchor)", "stjewm_trace_only", 1.0))
        for env_name in envs_run:
            for label, mname, alpha in jobs:
                key = f"{label}|{env_name}"
                bb[key] = run_closed_loop(mname, alpha, env_name, device, args)
                print(f"[s3s6/Bb] {key}: env-SR={bb[key]['env_success_rate']:.2f} "
                      f"cos={bb[key]['mean_cos_dist']:.4f} "
                      f"({time.time() - t_start:.0f}s)", flush=True)

    # ---------------- persist
    payload = {
        "experiment": "S3 action sensitivity + S6 EMA control (stab0917_b)",
        "protocol": {
            "contexts_per_env": args.contexts, "action_pairs_per_context": args.pairs,
            "persistence_horizon": HPH, "ema_alphas": ALPHAS,
            "d_perturb": "0.1 * per-dim action range (dataset min/max)",
            "s1step": "||F(z,a1)-F(z,a2)|| / ||(a1-a2)/act_std||, one-step predict head "
                      "(differing action at the z_{t+1}-driving context position)",
            "persistence": "impulse = a1/a2 only at the z_{t+1}-driving action then "
                           "shared dataset actions; sustained = center+-d at every "
                           "step; ||dz_h|| raw, /h1, /||z_base_h||",
            "response_matching": "last-frame obs += 0.1*per-dim obs std; response of "
                                 "EMA'd one-step predicted latent, alpha grid; primary "
                                 "variant anchors EMA at the shared pre-perturbation "
                                 "context latent (resp analytic in alpha); perturbed-"
                                 "anchor variant kept as encoder-pathway diagnostic; "
                                 "alpha* matches trace@1.0, alpha_required = ratio",
            "closed_loop": {**CL_CEM, "n_episodes": args.episodes, "n_seeds": 1,
                            "seed_start": 0, "split": "in_dist",
                            "hook": "cem_predict_hook EMA on last predicted position; "
                                    "anchor = planner-buffer last entry"},
            "envs": {k: v["npz"] for k, v in ENVS.items()},
            "checkpoints": {d: c for d, _, c in MODELS},
            "smoke": bool(args.smoke),
            "closed_loop_ran": not args.skip_closed_loop,
        },
        "part_a": pa,
        "part_a_pooled": {disp: pooled(r) for disp, r in pa.items()},
        "part_ba": ba,
        "part_ba_pooled": {"stjewm_trace_only": resp_trace_pool,
                           "lewm_baseline_v2": resp_lewm_pool},
        "alpha_match": match,
        "part_bb": bb,
        "wall_time_sec": time.time() - t_start,
    }
    json_path = out_dir / "action_ema_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False))

    md = render_md(payload)
    md_path = out_dir / "action_ema_summary.md"
    md_path.write_text(md)
    print(f"[s3s6] wrote {json_path} and {md_path} "
          f"({time.time() - t_start:.0f}s total)", flush=True)
    return 0


def _fmt(v: float) -> str:
    return f"{v:.3g}"


def render_md(p: dict) -> str:
    L = ["# S3 action sensitivity + S6 EMA control (stab0917_b)", ""]
    smoke = p["protocol"]["smoke"]
    if smoke:
        L += ["**SMOKE RUN** (4 contexts x 2 pairs, 1 closed-loop episode).", ""]
    L += [f"Contexts/env: {p['protocol']['contexts_per_env']} x "
          f"{p['protocol']['action_pairs_per_context']} action pairs; "
          f"persistence horizon h=1..{p['protocol']['persistence_horizon']}.", ""]

    # ---- S1step
    L += ["## Part A: one-step action sensitivity S1step", "",
          "S1step = ||dz|| / ||(a1-a2)/act_std||; `rel` additionally normalizes dz "
          "by the prediction norm (cross-model comparable).", "",
          "| model | S1step raw (pooled) | S1step rel (pooled) | latent norm |",
          "|---|---|---|---|"]
    for disp, r in p["part_a_pooled"].items():
        L.append(f"| {disp} | {_fmt(r['s1step_raw'])} | {_fmt(r['s1step_rel'])} | "
                 f"{_fmt(r['latent_norm_mean'])} |")

    # ---- persistence
    for kind, title in (("impulse", "Impulse persistence (action differs only at step 1)"),
                        ("sustained", "Sustained (center+-d at every step)")):
        L += ["", f"## Part A: {title}", "",
              "Curves pooled over envs. rel_to_h1 = ||dz_h||/||dz_1|| (wash-out shape); "
              "rel_to_latent = ||dz_h||/||z_base_h|| (cross-model comparable).", ""]
        hs = [f"h={h+1}" for h in range(p["protocol"]["persistence_horizon"])]
        for metric in ("rel_to_h1", "rel_to_latent"):
            L += [f"### {metric}", "", "| model | " + " | ".join(hs) + " |",
                  "|---|" + "---|" * len(hs)]
            for disp, r in p["part_a_pooled"].items():
                L.append(f"| {disp} | " +
                         " | ".join(_fmt(v) for v in r[kind][metric]) + " |")
            L.append("")

    # ---- response matching
    L += ["## Part B(a): EMA response matching (LeWM vs trace)", "",
          "Response = ||dz|| of the one-step predicted latent under last-frame obs "
          "perturbation (+0.1*obs std), with EMA z' = alpha*p + (1-alpha)*z_anchor. "
          "Primary table: SHARED pre-perturbation anchor (resp = alpha * resp(1)).",
          "",
          "| alpha | LeWM resp raw | LeWM resp rel | trace resp raw | trace resp rel |",
          "|---|---|---|---|---|"]
    tr, lw = p["part_ba_pooled"]["stjewm_trace_only"], p["part_ba_pooled"]["lewm_baseline_v2"]
    for a in tr["pred_resp_raw"]:
        L.append(f"| {a} | {_fmt(lw['pred_resp_raw'][a])} | {_fmt(lw['pred_resp_rel'][a])} | "
                 f"{_fmt(tr['pred_resp_raw'][a])} | {_fmt(tr['pred_resp_rel'][a])} |")
    m = p["alpha_match"]
    L += ["", f"Perturbed-anchor diagnostic (encoder pathway, EMA-immune), LeWM: "
          f"{_fmt(lw['pred_resp_pertanchor_raw']['0.1'])}..{_fmt(lw['pred_resp_pertanchor_raw']['1.0'])} "
          f"raw across alpha; trace: "
          f"{_fmt(tr['pred_resp_pertanchor_raw']['0.1'])}..{_fmt(tr['pred_resp_pertanchor_raw']['1.0'])}. "
          f"Encoder-only response: LeWM {_fmt(lw['enc_resp_raw'])} raw / "
          f"{_fmt(lw['enc_resp_rel'])} rel; trace {_fmt(tr['enc_resp_raw'])} raw / "
          f"{_fmt(tr['enc_resp_rel'])} rel.", "",
          f"**alpha\\* raw match = {m['pred_resp_raw']['alpha_star']}** "
          f"(exact ratio required: {_fmt(m['pred_resp_raw']['alpha_required_unclipped'])}); "
          f"**alpha\\* rel match = {m['pred_resp_rel']['alpha_star']}** "
          f"(exact ratio required: {_fmt(m['pred_resp_rel']['alpha_required_unclipped'])}).", ""]

    # ---- closed loop
    if p["part_bb"]:
        L += ["## Part B(b): closed-loop env-SR with EMA in the CEM planner", "",
              "Standard budget: CEM 300/30/10, horizon 25, budget 50, goal_offset 25, "
              "history 3; 5 episodes, 1 seed.", "",
              "| condition | env | env-SR | mean cos dist | mean phys dist |",
              "|---|---|---|---|---|"]
        for key, r in p["part_bb"].items():
            label, env_name = key.rsplit("|", 1)
            L.append(f"| {label} | {env_name} | {r['env_success_rate']:.2f} | "
                     f"{r['mean_cos_dist']:.4f} | {r['mean_phys_dist']:.4f} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())

"""E3 (cls0917_transfer): Cross-environment mechanism transfer on frozen world models.

Question: do the STJEWM-trace mechanism properties measured on cartpole/cheetah
(S1/S2: low obs gain, high action-to-obs gain ratio CSR, contractive lambda/c20)
persist on NEW environments, without retraining?

Environments (no retraining, frozen oodc_F1 seed-0 checkpoints):
  walker       — OOD for every model (out of the training union), zero-shot.
  ball_in_cup  — in-union for oodc_F1, different dynamics from cartpole/cheetah.

Per (model x env), 32 true 27-frame windows (seed 0):
  1. gain_a_rel / gain_o_rel / CSR  — paired +/- perturbation of the final-context
     action / observation by 0.1 * per-dim dataset std, relative one-step
     displacement ||dz_1|| / (||z_1|| + 1e-8) of the closed-loop predict head
     (same convention as cls0917_scaling so E3/E4 tables line up);
     CSR = gain_a_rel / gain_o_rel.
  2. ratio5 — same perturbations but self-fed 5 steps through the closed-loop
     predict head (perturbation at init only, dataset actions teacher-forced):
     ||dz^a_5|| / (||dz^o_5|| + 1e-8).
  3. lambda5 / c5 / c20 — latent FD protocol (stab0917_gain convention):
     ctx[:,-1] += eps * u (random unit dir), eps = 0.01 * per-context episode
     latent std; closed-loop rollout 20 steps, c(h) = ||dz_h|| / eps;
     lambda5 = log(c5) / 5 is the one-step log-Jacobian proxy; c20 the
     20-step contraction factor.
  4. mean ||z|| — mean context-latent norm (event-closure check; normalises the
     relative metrics).
  5. env-SR — standard closed_loop CEM subprocess, EXACT access_eval flag set
     (300/30/10, horizon 25, budget 50, goal_offset 25, 5 episodes, pad 128/56).
     CONTEXT COLUMN ONLY: walker is zero-shot (floor expected); ball_in_cup is
     in-union for oodc_F1 but unseen dynamics for the 5m baselines.

Verdict: mechanism persistence = trace keeps (low obs gain, CSR >> LeWM CSR,
lambda5 < 0 and c20 < 1) on BOTH walker and ball_in_cup; per-env CSR ordering
trace vs lewm vs gru reported.

Run: CUDA_VISIBLE_DEVICES=2 python -m code.scripts.cls0917_transfer [--smoke]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/lx/snn")
# torch/numpy may pull a module named `code` into sys.modules (stdlib or cwd
# artifact) before the project package is importable; evict any non-package
# binding so `code.*` below resolves to /home/lx/snn/code.
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

import numpy as np
import torch

from code.scripts.mech0917_multistep import (  # noqa: E402
    CTX, HMAX, forward_episodes, load_env_data, load_model, pad_batch, rollout,
    sample_windows,
)

OUT_DIR = Path("/home/lx/snn/results/mech_0917_b/transfer")

PAD_OBS, PAD_ACT = 128, 56
CONTEXTS = 32
N_DIRS = 16                 # latent-FD random unit directions per context
FD_EPS_FRAC = 0.01          # eps = FD_EPS_FRAC * per-context episode latent std
PERT_STD_FRAC = 0.1         # input perturbation = PERT_STD_FRAC * per-dim dataset std
LAMBDA_H = 5                # lambda proxy horizon: lambda5 = log(c5) / 5
RATIO_H = 5                 # self-fed horizon for the action/obs displacement ratio
C_HORIZONS = [1, 5, 20]     # contraction horizons recorded from the latent FD
SEED, GEN_SEED = 0, 123

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
    ("stjewm_spike_only", "stjewm_spike_only",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_spike_only/seed_0/final.pt"),
    ("stjewm_membrane_readout", "stjewm_membrane_readout",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_membrane_readout/seed_0/final.pt"),
    ("stjewm_hidden_leak", "stjewm_hidden_leak",
     "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_hidden_leak/seed_0/final.pt"),
    ("stacked_lif_trace", "stacked_lif_trace",
     "/data/lx/tmp/results/5m/oodc_F1/stacked_lif_trace/seed_0/final.pt"),
]

ENVS = {
    "walker": {
        "npz": "/home/lx/snn/data/dm_control/3d_rollouts_250k/walker_250k.npz",
        "closed_loop_env": "walker",
        "union": "ood",            # out of EVERY model's training union
    },
    "ball_in_cup": {
        "npz": "/home/lx/snn/data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz",
        "closed_loop_env": "ball_in_cup",
        "union": "in_union",       # in the oodc_F1 training union
    },
}


# ---------------------------------------------------------------- perturbation metrics
def input_gains(model, o_win: torch.Tensor, act_pad: torch.Tensor,
                sigma_o: torch.Tensor, delta_a: torch.Tensor) -> dict:
    """Paired +/- input perturbations at the final context position (E2 convention).

    o_win (B,CTX,D_true) true-dim obs windows; act_pad (B,CTX,PAD_ACT);
    sigma_o (D_true,), delta_a (PAD_ACT,) = 0.1 * per-dim dataset std.
    All displacements relative: ||dz_1|| / (||z_1|| + 1e-8).
    """
    with torch.inference_mode():
        ctx = model(pad(o_win), act_pad)["emb"]                         # (B,CTX,D)
        act_last = act_pad[:, -CTX:]
        z1 = model.predict(ctx, act_last)[:, -1]                        # (B,D)
        z1n = z1.norm(dim=-1) + 1e-8

        a_pos, a_neg = act_last.clone(), act_last.clone()
        a_pos[:, -1] = a_pos[:, -1] + delta_a
        a_neg[:, -1] = a_neg[:, -1] - delta_a
        dz_a = (model.predict(ctx, a_pos)[:, -1]
                - model.predict(ctx, a_neg)[:, -1]).norm(dim=-1)

        o_pos, o_neg = o_win.clone(), o_win.clone()
        o_pos[:, -1] = o_pos[:, -1] + sigma_o
        o_neg[:, -1] = o_neg[:, -1] - sigma_o
        ctx_p = model(pad(o_pos), act_pad)["emb"]
        ctx_n = model(pad(o_neg), act_pad)["emb"]
        dz_o = (model.predict(ctx_p, act_last)[:, -1]
                - model.predict(ctx_n, act_last)[:, -1]).norm(dim=-1)

    return {
        "gain_a_rel": (dz_a / z1n).float().cpu().numpy(),
        "gain_o_rel": (dz_o / z1n).float().cpu().numpy(),
        "z1_norm": z1.norm(dim=-1).float().cpu().numpy(),
    }


def ratio5(model, ctx_base: torch.Tensor, ctx_obs_pos: torch.Tensor,
           ctx_obs_neg: torch.Tensor, act_pad: torch.Tensor, fut: torch.Tensor,
           delta_a: torch.Tensor) -> np.ndarray:
    """Self-fed RATIO_H-step displacement ratio ||dz^a_R|| / (||dz^o_R|| + 1e-8).

    Perturbation applied at the initial context only (action +/- delta_a on the
    final context action, obs +sigma/-sigma re-encoded); predicted latents are
    self-fed, future actions teacher-forced from the dataset.
    """
    act_win = act_pad[:, -CTX:]
    a_pos, a_neg = act_win.clone(), act_win.clone()
    a_pos[:, -1] = a_pos[:, -1] + delta_a
    a_neg[:, -1] = a_neg[:, -1] - delta_a
    with torch.inference_mode():
        base = rollout(model, ctx_base, act_win, fut)[:, RATIO_H - 1]
        pa = rollout(model, ctx_base, a_pos, fut)[:, RATIO_H - 1]
        na = rollout(model, ctx_base, a_neg, fut)[:, RATIO_H - 1]
        po = rollout(model, ctx_obs_pos, act_win, fut)[:, RATIO_H - 1]
        no = rollout(model, ctx_obs_neg, act_win, fut)[:, RATIO_H - 1]
    dza = (pa - na).norm(dim=-1)
    dzo = (po - no).norm(dim=-1)
    return (dza / (dzo + 1e-8)).float().cpu().numpy()


def pad(o_win: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(o_win, (0, PAD_OBS - o_win.shape[-1]))


# ---------------------------------------------------------------- latent FD (S2 convention)
def latent_fd(model, ctx: torch.Tensor, act_pad: torch.Tensor, fut: torch.Tensor,
              eps: torch.Tensor, dirs: torch.Tensor, chunk: int) -> dict:
    """FD contraction c(h) = ||dz_h||/eps for h in C_HORIZONS; lambda5 = log(c5)/5.

    Same protocol as stab0917_gain.latent_gain (eps = FD_EPS_FRAC * per-context
    episode latent std, random unit directions, closed-loop rollout).
    """
    B, _, D = ctx.shape
    K = dirs.shape[0]
    BK = B * K
    ii = torch.arange(BK, device=ctx.device) // K
    jj = torch.arange(BK, device=ctx.device) % K
    act_win = act_pad[:, -CTX:]
    with torch.inference_mode():
        base = rollout(model, ctx, act_win, fut)
        c = {h: torch.zeros(BK, device=ctx.device) for h in C_HORIZONS}
        for s in range(0, BK, chunk):
            sl = slice(s, min(s + chunk, BK))
            i, j = ii[sl], jj[sl]
            cp = ctx[i].clone()
            cp[:, -1] = cp[:, -1] + eps[i].unsqueeze(1) * dirs[j]
            p = rollout(model, cp, act_win[i], fut[i])
            for h in C_HORIZONS:
                c[h][sl] = (p[:, h - 1] - base[i][:, h - 1]).norm(dim=-1) / eps[i]
    out = {}
    for h in C_HORIZONS:
        v = c[h].reshape(B, K).float().cpu().numpy()
        out[f"c{h}"] = {"mean": float(v.mean()), "median": float(np.median(v))}
    c5 = max(out[f"c{LAMBDA_H}"]["mean"], 1e-12)
    out["lambda5"] = float(np.log(c5) / LAMBDA_H)
    return out


# ---------------------------------------------------------------- env-SR subprocess
def closed_loop_cmd(env_key: str, ckpt: str, out: Path,
                    n_episodes: int) -> list[str]:
    """EXACT flag set from code/scripts/access_eval.py (verified protocol)."""
    return [sys.executable, "-u", "-m", "code.eval.closed_loop",
            "--env", ENVS[env_key]["closed_loop_env"],
            "--ckpt", ckpt,
            "--data", ENVS[env_key]["npz"],
            "--out", str(out),
            "--device", "cuda:0",
            "--split", "in_dist",
            "--n-episodes", str(n_episodes), "--n-seeds", "1",
            "--horizon", "25", "--eval-budget", "50", "--history-size", "1",
            "--cem-samples", "300", "--cem-elites", "30", "--cem-iters", "10",
            "--goal-offset", "25",
            "--pad-obs-eval", str(PAD_OBS), "--action-dim-eval", str(PAD_ACT)]


def run_env_sr(env_key: str, ckpt: str, out: Path, n_episodes: int) -> dict:
    if not out.exists():
        log_path = out.with_suffix(".log")
        with open(log_path, "w") as log_f:
            subprocess.run(closed_loop_cmd(env_key, ckpt, out, n_episodes),
                           cwd="/home/lx/snn", check=True, stdout=log_f,
                           stderr=subprocess.STDOUT)
    payload = json.loads(out.read_text())
    return {"env_sr": payload["success_rate_env"],
            "env_sr_std": payload["success_rate_env_std"],
            "cos": payload["mean_cos_dist"]}


# ---------------------------------------------------------------- per (model, env)
def run_env_model(model, env_key: str, obs, act, episodes, win_starts, win_ts,
                  args) -> dict:
    device = args.device
    n = len(win_ts)
    ep_map = dict(episodes)
    spans = sorted(set(win_starts.tolist()))
    spans = [(s, ep_map[s]) for s in spans]

    # eps scale from warm true-latent episode std (per context) + mean ||z||
    ep_emb = forward_episodes(model, obs, act, spans, device, batch=args.ep_batch)
    eps_np = np.array([float(ep_emb[s].std()) for s in win_starts], dtype=np.float32)

    o_seqs = [obs[t - CTX + 1: t + 1] for t in win_ts]
    a_seqs = [act[t - CTX + 1: t + 1] for t in win_ts]
    with torch.no_grad():
        o_pad = pad_batch(o_seqs, PAD_OBS).to(device)
        act_pad = pad_batch(a_seqs, PAD_ACT).to(device)
        ctx = model(o_pad, act_pad)["emb"]                              # (B,CTX,D)
    fut_np = np.stack([act[t + 1: t + HMAX + 1] for t in win_ts])
    fut_np = np.pad(fut_np, ((0, 0), (0, 0), (0, PAD_ACT - fut_np.shape[-1])))
    fut = torch.from_numpy(fut_np).to(device)

    o_win = torch.from_numpy(np.stack(o_seqs)).to(device)               # (B,CTX,Dt)
    sigma_o = (PERT_STD_FRAC * obs.std(axis=0).astype(np.float32))
    sigma_o = torch.from_numpy(sigma_o).to(device)
    delta_a_np = PERT_STD_FRAC * act.std(axis=0).astype(np.float32)
    # pad to PAD_ACT so it broadcasts against a padded (B, PAD_ACT) action row
    delta_a = torch.from_numpy(
        np.pad(delta_a_np, (0, PAD_ACT - len(delta_a_np)))).to(device)

    gains = input_gains(model, o_win, act_pad, sigma_o, delta_a)
    # self-fed ratio: +/- perturbed-obs contexts (re-encode o +/- sigma_o)
    with torch.inference_mode():
        o_pos = o_win.clone()
        o_pos[:, -1] = o_pos[:, -1] + sigma_o
        o_neg = o_win.clone()
        o_neg[:, -1] = o_neg[:, -1] - sigma_o
        ctx_obs_pos = model(pad(o_pos), act_pad)["emb"]
        ctx_obs_neg = model(pad(o_neg), act_pad)["emb"]
    r5 = ratio5(model, ctx, ctx_obs_pos, ctx_obs_neg, act_pad, fut, delta_a)

    eps = (torch.from_numpy(eps_np) * FD_EPS_FRAC).to(device)
    D = ctx.shape[-1]
    dirs = torch.randn((args.dirs, D), generator=args.gen)
    dirs = (dirs / dirs.norm(dim=-1, keepdim=True)).to(device)
    fd = latent_fd(model, ctx, act_pad, fut, eps, dirs, args.chunk)

    gar, gor = gains["gain_a_rel"], gains["gain_o_rel"]
    res = {
        "n_contexts": int(n), "n_dirs": int(args.dirs),
        "mean_z_norm": float(ctx[:, -1].norm(dim=-1).mean().item()),
        "gain_a_rel": {"mean": float(gar.mean()), "median": float(np.median(gar))},
        "gain_o_rel": {"mean": float(gor.mean()), "median": float(np.median(gor))},
        "CSR": float(gar.mean() / max(gor.mean(), 1e-18)),
        "ratio5": {"mean": float(r5.mean()), "median": float(np.median(r5))},
        "eps_mean": float(eps_np.mean()),
        **fd,
    }
    return res


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="E3: cross-environment mechanism transfer")
    ap.add_argument("--contexts", type=int, default=CONTEXTS)
    ap.add_argument("--dirs", type=int, default=N_DIRS)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--ep-batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gen-seed", type=int, default=GEN_SEED)
    ap.add_argument("--sr-workers", type=int, default=4,
                    help="concurrent closed_loop subprocesses on the GPU")
    ap.add_argument("--models", default=None, help="comma-separated display filter (smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="full mechanism protocol, env-SR trace-only x 1 episode, "
                         "out to smoke/")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    args.gen = torch.Generator().manual_seed(args.gen_seed)

    out_dir = Path(args.out_dir) if args.out_dir else (
        OUT_DIR / "smoke" if args.smoke else OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr_dir = out_dir / "evals"
    sr_dir.mkdir(exist_ok=True)
    n_episodes = 1 if args.smoke else 5

    keep = None
    if args.models:
        keep = {s.strip() for s in args.models.split(",") if s.strip()}
    models = [(d, h, c) for (d, h, c) in MODELS if keep is None or d in keep]
    if not models:
        print(f"[e3xfer] model filter {keep} matched nothing", flush=True)
        return 1

    t0 = time.time()

    # ---- env-SR pool first: subprocesses overlap with the mechanism metrics
    # smoke: trace only, both env kinds (verifies the subprocess protocol on
    # walker AND ball_in_cup without paying for the full 16-run grid)
    sr_models = models if not args.smoke else models[:1]
    sr_futures = {}
    sr_results: dict = {}
    pool = ThreadPoolExecutor(max_workers=args.sr_workers)
    for env_key in ENVS:
        for disp, _, ckpt in sr_models:
            out = sr_dir / f"eval_{env_key}_{disp}.json"
            sr_futures[(env_key, disp)] = pool.submit(
                run_env_sr, env_key, ckpt, out, n_episodes)
    print(f"[e3xfer] env-SR pool launched: {len(sr_futures)} runs, "
          f"{args.sr_workers} workers, n_episodes={n_episodes}", flush=True)

    # ---- mechanism metrics
    env_data = {}
    for env_key, cfg in ENVS.items():
        obs, act, episodes = load_env_data(cfg["npz"], CTX + HMAX)
        starts, _, ts = sample_windows(episodes, args.contexts,
                                       np.random.default_rng(args.seed))
        env_data[env_key] = (obs, act, episodes, starts, ts)
        print(f"[e3xfer] {env_key} ({cfg['union']}): {len(episodes)} usable episodes, "
              f"{len(ts)} contexts", flush=True)

    results: dict = {}
    for disp, hint, ckpt in models:
        model = load_model(hint, ckpt, args.device)
        results[disp] = {}
        for env_key in ENVS:
            obs, act, episodes, starts, ts = env_data[env_key]
            res = run_env_model(model, env_key, obs, act, episodes, starts, ts, args)
            results[disp][env_key] = res
            print(f"[e3xfer] {disp} x {env_key}: CSR={res['CSR']:.3g} "
                  f"(a={res['gain_a_rel']['mean']:.3g} o={res['gain_o_rel']['mean']:.3g}) "
                  f"lambda5={res['lambda5']:.3f} c5={res['c5']['mean']:.3g} "
                  f"c20={res['c20']['mean']:.3g} |z|={res['mean_z_norm']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- collect env-SR
    for (env_key, disp), fut in sr_futures.items():
        try:
            sr_results.setdefault(disp, {})[env_key] = fut.result()
            r = sr_results[disp][env_key]
            print(f"[e3xfer] SR {disp} x {env_key}: env_sr={r['env_sr']:.3f} "
                  f"cos={r['cos']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # keep partial results visible in the payload
            sr_results.setdefault(disp, {})[env_key] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[e3xfer] SR FAILED {disp} x {env_key}: {e}", flush=True)
    pool.shutdown()

    for disp in results:
        for env_key in ENVS:
            results[disp][env_key]["env_sr"] = sr_results.get(disp, {}).get(env_key)

    # ---- verdict
    verdict = {}
    for env_key in ENVS:
        r = {d: results[d][env_key] for d in results}
        csr = {d: r[d]["CSR"] for d in r}
        tr_csr, lw_csr = csr.get("stjewm_trace_only"), csr.get("lewm_baseline_v2")
        ratio = (tr_csr / lw_csr) if (tr_csr is not None and lw_csr) else None
        verdict[env_key] = {
            "csr_order": sorted(csr, key=csr.get, reverse=True),
            "csr_trace_over_lewm": ratio,
            "trace_lambda5": r["stjewm_trace_only"]["lambda5"],
            "trace_c20": r["stjewm_trace_only"]["c20"]["mean"],
            "trace_contractive": bool(r["stjewm_trace_only"]["lambda5"] < 0
                                      and r["stjewm_trace_only"]["c20"]["mean"] < 1),
        }

    payload = {
        "experiment": "E3 cross-environment mechanism transfer (cls0917_transfer)",
        "protocol": {
            "contexts": args.contexts, "dirs_latent_fd": args.dirs,
            "window": CTX, "pad_obs": PAD_OBS, "pad_act": PAD_ACT,
            "gain_a_rel_o_rel": f"paired +/- {PERT_STD_FRAC}*per-dim dataset std at the "
                                "final context position (cls0917_scaling convention); "
                                "relative displacement ||dz_1||/(||z_1||+1e-8); "
                                "CSR = gain_a_rel/gain_o_rel",
            "ratio5": f"same perturbation self-fed {RATIO_H} steps (init only, dataset "
                      "actions teacher-forced): ||dz^a_5||/(||dz^o_5||+1e-8)",
            "lambda5_c": f"latent FD: ctx[:,-1] += eps*u (unit gaussian dir), "
                         f"eps = {FD_EPS_FRAC} * per-context warm episode latent std; "
                         f"closed-loop rollout {HMAX} steps; c(h)=||dz_h||/eps; "
                         f"lambda5 = log(c5)/{LAMBDA_H}",
            "env_sr": "standard closed_loop CEM subprocess, access_eval flag set "
                      "(300/30/10, horizon 25, budget 50, goal_offset 25, 5 episodes, "
                      "pad 128/56, split in_dist). CONTEXT ONLY: walker is zero-shot "
                      "for every model; ball_in_cup is in-union for oodc_F1 but unseen "
                      "by the 5m baselines.",
            "no_velocity_field": "npz has no qvel; observations are qpos-based "
                                 "(walker 9-d, ball_in_cup 4-d); no state probes in E3",
            "seed_window": args.seed, "seed_gen": args.gen_seed,
            "checkpoints": {d: c for d, _, c in models},
            "env_union": {k: v["union"] for k, v in ENVS.items()},
            "smoke": bool(args.smoke),
        },
        "results": results,
        "verdict": verdict,
    }
    json_path = out_dir / "transfer_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"[e3xfer] wrote {json_path}", flush=True)

    if not args.smoke:
        write_md(out_dir, payload)
        write_scatter(out_dir, payload)
    print(f"[e3xfer] done ({time.time() - t0:.0f}s)", flush=True)
    return 0


SHORT = {"stjewm_trace_only": "trace", "lewm_baseline_v2": "lewm", "gru_baseline": "gru"}


def _load_prior_reference() -> dict | None:
    """Pooled cartpole+cheetah reference values from the S1/S2 summaries (read-only)."""
    try:
        gain = json.loads(
            Path("/home/lx/snn/results/mech_0917_b/gain/gain_summary.json").read_text())
        roll = json.loads(
            Path("/home/lx/snn/results/mech_0917_b/rollout/rollout_summary.json").read_text())
        out = {}
        for disp in ("stjewm_trace_only", "lewm_baseline_v2", "gru_baseline"):
            key = SHORT.get(disp, disp)
            g = gain.get("pooled", {}).get(disp, {})
            r = roll.get("pooled", {}).get(key, {})
            out[disp] = {
                "gain_obs": g.get("gain_obs", {}).get("mean"),
                "c20": r.get("c20"),
                "lambda_emp": r.get("lambda_emp"),
            }
        return out
    except Exception:
        return None


def write_md(out_dir: Path, payload: dict) -> None:
    results, verdict = payload["results"], payload["verdict"]
    models = [m[0] for m in MODELS if m[0] in results]
    prior = _load_prior_reference()
    lines = [
        "# E3: Cross-environment mechanism transfer",
        "",
        "Frozen oodc_F1 seed-0 checkpoints evaluated on new environments WITHOUT "
        "retraining. Envs: **walker = OOD** (out of every model's training union, "
        "zero-shot); **ball_in_cup = in_union** for oodc_F1 (different dynamics from "
        "cartpole/cheetah; unseen by the 5m baselines).",
        "",
        f"Perturbations: paired +/- 0.1*per-dim dataset std at the final context "
        f"position; relative one-step displacement ||dz_1||/(||z_1||+1e-8); "
        f"CSR = gain_a_rel/gain_o_rel (cls0917_scaling convention). Latent FD: "
        f"eps = 0.01*per-context episode latent std, {N_DIRS} unit dirs, "
        f"lambda5 = log(c5)/5, c20 = 20-step contraction factor. 32 windows/env, "
        f"seed 0.",
        "",
        "**env-SR columns are CONTEXT, not the claim: walker env-SR is zero-shot for "
        "every model (floor expected); ball_in_cup is in-union for oodc_F1.** Note: "
        "trace's latent is near-constant (mean |z| ~ 0.01), so its terminal "
        "latent-goal cosine is degenerate (~0 distance by construction); env-native "
        "success rate is the meaningful column.",
        "",
        "No velocity (qvel) field exists in the npz (observations are qpos-based: "
        "walker 9-d, ball_in_cup 4-d); E3 uses no state probes, so no velocity "
        "decomposition applies.",
        "",
    ]
    for env_key in ENVS:
        lines += [
            f"## {env_key} ({ENVS[env_key]['union']})", "",
            "| model | gain_a_rel | gain_o_rel | CSR | ratio5 | lambda5 | c5 | c20 | "
            "mean \\|\\|z\\|\\| | env-SR | cos |",
            "|---|" + "---|" * 10,
        ]
        for disp in models:
            r = results[disp][env_key]
            sr = r.get("env_sr") or {}
            sr_cell = (f"{sr['env_sr']:.2f}" if "env_sr" in sr
                       else sr.get("error", "n/a")[:30])
            cos_cell = f"{sr['cos']:.3f}" if "cos" in sr else "-"
            lines.append(
                f"| {disp} | {r['gain_a_rel']['mean']:.3g} | {r['gain_o_rel']['mean']:.3g} "
                f"| {r['CSR']:.3g} | {r['ratio5']['mean']:.3g} | {r['lambda5']:.3f} "
                f"| {r['c5']['mean']:.3g} | {r['c20']['mean']:.3g} "
                f"| {r['mean_z_norm']:.3f} | {sr_cell} | {cos_cell} |")
        v = verdict[env_key]
        tr_csr = results["stjewm_trace_only"][env_key]["CSR"]
        lw_csr = results["lewm_baseline_v2"][env_key]["CSR"]
        gr_csr = results["gru_baseline"][env_key]["CSR"]
        ratio_cell = (f"**{v['csr_trace_over_lewm']:.3g}**"
                      if v["csr_trace_over_lewm"] is not None else "n/a")
        lines += [
            "",
            f"CSR ordering: {' > '.join(v['csr_order'])}. "
            f"trace CSR / lewm CSR = {ratio_cell}; "
            f"trace CSR / gru CSR = {tr_csr / gr_csr:.3g}.",
            f"trace contractive on {env_key} (lambda5<0 and c20<1): "
            f"**{v['trace_contractive']}** "
            f"(lambda5={v['trace_lambda5']:.3f}, c20={v['trace_c20']:.3g}).",
            "",
        ]
    if prior is not None:
        lines += [
            "## Reference: prior pooled cartpole_2d + cheetah (S1/S2, same checkpoints)",
            "",
            "| model | gain_obs (S2) | c20 (S1) | lambda_emp (S1) |",
            "|---|---|---|---|",
        ]
        for disp, v in prior.items():
            go = f"{v['gain_obs']:.3g}" if v["gain_obs"] is not None else "-"
            c20 = f"{v['c20']:.3g}" if v["c20"] is not None else "-"
            le = f"{v['lambda_emp']:.3g}" if v["lambda_emp"] is not None else "-"
            lines.append(f"| {disp} | {go} | {c20} | {le} |")
        lines += [
            "",
            "Footnotes: **stacked_lif_trace on ball_in_cup**: gain_o_rel = 0 exactly "
            "(obs perturbation of 0.1*per-dim std never moves its readout), so its "
            "CSR (~9e14) is an underflow artifact, not a meaningful selectivity "
            "estimate; **mlp_baseline**: both gains ~1e-7 with near-constant latent "
            "(|z|=0.313) — a dead/constant latent, so its CSR ratio is not "
            "interpretable as selectivity.",
            "",
        ]
    md_path = out_dir / "transfer_summary.md"
    md_path.write_text("\n".join(lines))
    print(f"[e3xfer] wrote {md_path}", flush=True)


def write_scatter(out_dir: Path, payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = payload["results"]
    CSR_CLIP = 1e7
    clipped = []
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"walker": "#d62728", "ball_in_cup": "#1f77b4"}
    marks = {"stjewm_trace_only": "*", "lewm_baseline_v2": "s", "gru_baseline": "^"}
    for disp, envs in results.items():
        for env_key, r in envs.items():
            if not np.isfinite(r["CSR"]) or r["c20"]["mean"] <= 0:
                continue
            x = r["CSR"]
            label = disp.replace("_baseline", "").replace("stjewm_", "")
            if x > CSR_CLIP:
                x = CSR_CLIP
                label = f"{label} >1e7"
                clipped.append((disp, env_key))
            ax.scatter(x, r["c20"]["mean"],
                       c=colors[env_key],
                       marker=marks.get(disp, "o"), s=140 if disp in marks else 60,
                       edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(label, (x, r["c20"]["mean"]), fontsize=6,
                        xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, CSR_CLIP * 1.5)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.axvline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("CSR = gain_a_rel / gain_o_rel  (selectivity, higher = more action-selective)")
    ax.set_ylabel("c20 contraction factor  (<1 contractive)")
    title = "E3 transfer: selectivity vs contraction on new envs\n" \
            "(red = walker OOD, blue = ball_in_cup in-union)"
    if clipped:
        title += f"\nCSR clipped at {CSR_CLIP:.0e} for: " + \
            ", ".join(f"{d}/{e}" for d, e in clipped)
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png = out_dir / "transfer_scatter.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[e3xfer] wrote {png}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

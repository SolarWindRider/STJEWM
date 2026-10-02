"""E1 (cls0917-b): Nonlinear/temporal accessibility probe on frozen world models.

Question: do STJEWM-trace latents HIDE information nonlinearly (recoverable by an
MLP but not a linear probe) rather than discarding it, and does history-dependent
(temporal) context add beyond a single-latent probe?

Protocol (per model x env, frozen models, no retraining):
  Context protocols, both evaluated on the SAME sampled t positions:
    (a) WINDOW: 27-frame contexts [t-26..t] via sample_windows (planning-realistic),
        z_t = emb[:, -1]; temporal concat = emb[:, -6:] = [z_{t-5}..z_t].
    (b) WARM: full-episode forwards (prefix-causal), z_t = emb[t - ep_start];
        temporal concat = episode emb at [t-5..t].
  Targets (all from the 128-padded state matrix; native qpos dims = first
  obs_dim_native columns): s_t, pseudo-velocity s_t - s_{t-1} (no qvel field in
  the npz -> first difference, labelled as such), s_{t+5}, s_{t+20}.
  Probe families:
    linear      ridge alpha=1.0 on standardized z (baseline)
    mlp         Linear(d,256)-GELU-Dropout(0.1)-Linear(256,out), AdamW lr 1e-3
                wd 1e-4, batch 64, <=300 epochs, early stopping on a window-level
                val slice of TRAIN episodes (patience 20, restore best val R2);
                3 seeds -> mean +- std. Train R2 reported to expose overfitting.
    temporal    ridge on concatenated [z_{t-5}..z_t] (6*d latents; both protocols)
  Scoring: held-out R2 (episode-level 75/25 split), multioutput uniform average
  over NON-CONSTANT columns of the 128-d target (constant padded-zero columns are
  excluded; sklearn force_finite would otherwise score them 1.0 for ridge -- which
  predicts exactly 0 there -- and 0.0 for the MLP, a pure artifact). The literal
  128-column uniform average is recorded as "r2_full128" in the JSON only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/lx/snn")
# torch/numpy may pull a module named `code` into sys.modules before the project
# package is importable; evict any non-package binding (same guard as siblings).
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from code.scripts.mech0917_multistep import (
    CTX,
    HMAX,
    PAD_ACT,
    PAD_OBS,
    episode_split,
    forward_episodes,
    forward_windows,
    load_env_data,
    load_model,
    sample_windows,
)

# ---------------------------------------------------------------- constants
ENVS = {
    "cartpole_2d": {
        "npz": "/home/lx/snn/data/dm_control/cartpole_250k.npz",
        "native": 2,
    },
    "cheetah": {
        "npz": "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz",
        "native": 9,
    },
    "finger": {
        "npz": "/home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz",
        "native": 3,
    },
}

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

TARGETS = ["s_t", "vel", "s_t+5", "s_t+20"]     # probed by linear + mlp
TEMPORAL_TARGETS = ["s_t", "s_t+20"]            # temporal-linear targets
SPLIT_SEED = 123                                 # matches sibling experiments
HIST = 5                                         # [z_{t-5}..z_t] -> 6 latents
VAL_FRAC = 0.20                                  # window-level val inside train eps


# ---------------------------------------------------------------- targets
def build_env_arrays(npz_path: str):
    """Padded state matrix (N,128), pseudo-velocity (N,128), episodes, windows."""
    obs, act, episodes = load_env_data(npz_path, CTX + HMAX)
    n = obs.shape[0]
    s_pad = np.zeros((n, PAD_OBS), dtype=np.float32)
    s_pad[:, : obs.shape[1]] = obs
    vel = np.zeros_like(s_pad)
    for (s, e) in episodes:
        if e > s:
            vel[s + 1 : e + 1] = s_pad[s + 1 : e + 1] - s_pad[s:e]
    return obs, act, episodes, s_pad, vel


def make_targets(s_pad, vel, ts):
    """(N,128) target matrix per target name, aligned to sampled t."""
    t = np.asarray(ts)
    return {
        "s_t": s_pad[t],
        "vel": vel[t],
        "s_t+5": s_pad[t + 5],
        "s_t+20": s_pad[t + HMAX],
    }


# ---------------------------------------------------------------- scoring
def r2_uniform(y_true: np.ndarray, y_pred: np.ndarray, const_mask: np.ndarray) -> float:
    """Multioutput uniform-average R2 over non-constant (train) columns."""
    if const_mask.all():
        return float("nan")
    return float(r2_score(y_true[:, ~const_mask], y_pred[:, ~const_mask],
                          multioutput="uniform_average"))


def fit_ridge(z: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
              const_mask: np.ndarray, native: int) -> dict:
    xtr, ytr = z[train_mask], y[train_mask]
    xte, yte = z[~train_mask], y[~train_mask]
    sc = StandardScaler().fit(xtr)
    rg = Ridge(alpha=1.0).fit(sc.transform(xtr), ytr)
    pred_te = rg.predict(sc.transform(xte))
    return {
        "r2_test": r2_uniform(yte, pred_te, const_mask),
        "r2_test_full128": float(r2_score(yte, pred_te, multioutput="uniform_average")),
        "r2_test_native": float(r2_score(yte[:, :native], pred_te[:, :native],
                                         multioutput="uniform_average")),
        "r2_train": r2_uniform(ytr, rg.predict(sc.transform(xtr)), const_mask),
    }


# ---------------------------------------------------------------- MLP probe
def train_mlp(z: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
              const_mask: np.ndarray, seed: int, epochs: int, patience: int,
              device: str, native: int) -> dict:
    rng = np.random.default_rng(1000 + seed)
    tr_idx = np.flatnonzero(train_mask)
    n_val = max(2, int(round(VAL_FRAC * len(tr_idx))))
    perm = rng.permutation(len(tr_idx))
    val_idx, fit_idx = tr_idx[perm[:n_val]], tr_idx[perm[n_val:]]
    te_idx = np.flatnonzero(~train_mask)

    sc_x = StandardScaler().fit(z[fit_idx])
    sc_y = StandardScaler().fit(y[fit_idx])
    zt = {k: torch.from_numpy(sc_x.transform(z[v])).float().to(device)
          for k, v in {"fit": fit_idx, "val": val_idx, "te": te_idx,
                       "tr": tr_idx}.items()}
    yt = {k: torch.from_numpy(sc_y.transform(y[v])).float().to(device)
          for k, v in {"fit": fit_idx, "val": val_idx, "te": te_idx,
                       "tr": tr_idx}.items()}

    torch.manual_seed(seed)
    d_in, d_out = z.shape[1], y.shape[1]
    net = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(256, d_out)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    def r2_of(idx_key: str) -> float:
        net.eval()
        with torch.inference_mode():
            pred = sc_y.inverse_transform(net(zt[idx_key]).cpu().numpy())
        y_true = y[{"fit": fit_idx, "val": val_idx, "te": te_idx,
                    "tr": tr_idx}[idx_key]]
        return r2_uniform(y_true, pred, const_mask)

    best_val, best_state, best_ep, bad = -np.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(len(fit_idx)))
        for i in range(0, len(order), 64):
            sel = order[i : i + 64]
            opt.zero_grad()
            loss = loss_fn(net(zt["fit"][sel]), yt["fit"][sel])
            loss.backward()
            opt.step()
        val_r2 = r2_of("val")
        if val_r2 > best_val + 1e-6:
            best_val, best_ep, bad = val_r2, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.inference_mode():
        pred_te = sc_y.inverse_transform(net(zt["te"]).cpu().numpy())
    y_te = y[te_idx]
    return {"r2_test": r2_uniform(y_te, pred_te, const_mask),
            "r2_test_full128": float(r2_score(y_te, pred_te,
                                              multioutput="uniform_average")),
            "r2_test_native": float(r2_score(y_te[:, :native], pred_te[:, :native],
                                             multioutput="uniform_average")),
            "r2_train": r2_of("tr"),
            "r2_val_best": best_val, "best_epoch": best_ep, "seed": seed}


# ---------------------------------------------------------------- latents
def collect_latents(model, obs, act, episodes, starts, ts, device, ep_batch: int,
                    batch: int):
    """Window-context and warm-context latents on the same t positions.

    Returns z_win (n,d), hist_win (n,6,d) = emb[:,-6:], z_warm (n,d),
    hist_warm (n,6,d) = episode emb [t-5..t].
    """
    n = len(ts)
    spans = sorted({int(s) for s in starts})
    ep_map = dict(episodes)
    ep_emb = forward_episodes(model, obs, act,
                              [(s, ep_map[s]) for s in spans], device, ep_batch)

    d = None
    z_win, hist_win = [], []
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        o_seqs = [obs[t - CTX + 1 : t + 1] for t in ts[sl]]
        a_seqs = [act[t - CTX + 1 : t + 1] for t in ts[sl]]
        emb = forward_windows(model, o_seqs, a_seqs, device)   # (B,CTX,d) on device
        d = emb.shape[-1]
        z_win.append(emb[:, -1].float().cpu().numpy())
        hist_win.append(emb[:, -HIST - 1 :].float().cpu().numpy())
    z_win = np.concatenate(z_win)
    hist_win = np.concatenate(hist_win)

    z_warm = np.zeros((n, d), dtype=np.float32)
    hist_warm = np.zeros((n, HIST + 1, d), dtype=np.float32)
    for i, (s_ep, t) in enumerate(zip(starts, ts)):
        embs = ep_emb[int(s_ep)]
        off = int(t) - int(s_ep)
        assert off >= CTX - 1, f"warm position {t} has only {off} preceding frames"
        z_warm[i] = embs[off].numpy()
        hist_warm[i] = embs[off - HIST : off + 1].numpy()
    return z_win, hist_win, z_warm, hist_warm, d


# ---------------------------------------------------------------- per cell
def probe_cell(z: np.ndarray, zh: np.ndarray | None, y: np.ndarray,
               train_mask: np.ndarray, const_mask: np.ndarray, native: int,
               seeds: int, epochs: int, patience: int, device: str) -> dict:
    """linear + mlp(+temporal if zh given) probes for one target matrix y."""
    out = {"linear": fit_ridge(z, y, train_mask, const_mask, native)}
    seed_runs = [train_mlp(z, y, train_mask, const_mask, s, epochs, patience,
                           device, native)
                 for s in range(seeds)]
    r2s = [r["r2_test"] for r in seed_runs]
    out["mlp"] = {"mean": float(np.mean(r2s)), "std": float(np.std(r2s)),
                  "seeds": seed_runs}
    if zh is not None:
        out["temporal"] = fit_ridge(zh, y, train_mask, const_mask, native)
    return out


def run_model_env(model, env_name: str, obs, act, episodes, s_pad, vel,
                  starts, ts, args) -> dict:
    device = args.device
    native = ENVS[env_name]["native"]
    targets = make_targets(s_pad, vel, ts)
    train_mask = episode_split(starts, np.random.default_rng(SPLIT_SEED))
    # const columns per target (from train rows of the 128-d matrix)
    const_masks = {k: np.std(v[train_mask], axis=0) < 1e-8 for k, v in targets.items()}

    z_win, hist_win, z_warm, hist_warm, d = collect_latents(
        model, obs, act, episodes, starts, ts, device, args.ep_batch, args.batch)
    zh_win = hist_win.reshape(len(ts), -1)
    zh_warm = hist_warm.reshape(len(ts), -1)

    res = {"emb_dim": int(d), "n_windows": int(len(ts)),
           "n_windows_test": int((~train_mask).sum()), "contexts": {}}
    for ctx_name, z, zh in (("window", z_win, zh_win), ("warm", z_warm, zh_warm)):
        cell = {}
        for tname in TARGETS:
            temporal = zh if tname in TEMPORAL_TARGETS else None
            cell[tname] = probe_cell(z, temporal, targets[tname], train_mask,
                                     const_masks[tname], native,
                                     args.seeds, args.epochs, args.patience, device)
        res["contexts"][ctx_name] = cell
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 nonlinear/temporal accessibility probe")
    ap.add_argument("--models", default="all", help="'all' or comma-separated names")
    ap.add_argument("--envs", default=",".join(ENVS))
    ap.add_argument("--n-windows", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ep-batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0, help="window-sampling seed")
    ap.add_argument("--out-dir", default="/home/lx/snn/results/mech_0917_b/nonlinear_probe")
    args = ap.parse_args()

    models = MODELS if args.models == "all" else \
        [m for m in MODELS if m[0] in set(args.models.split(","))]
    unknown = set(args.models.split(",")) - {m[0] for m in MODELS} \
        if args.models != "all" else set()
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")
    env_names = [e.strip() for e in args.envs.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env_data = {}
    for env_name in env_names:
        cfg = ENVS[env_name]
        obs, act, episodes, s_pad, vel = build_env_arrays(cfg["npz"])
        starts, ends, ts = sample_windows(episodes, args.n_windows,
                                          np.random.default_rng(args.seed))
        env_data[env_name] = (obs, act, episodes, s_pad, vel, starts, ts)
        print(f"[e1] {env_name}: {len(episodes)} usable episodes, "
              f"{len(ts)} windows", flush=True)

    results: dict = {}
    for disp, hint, ckpt in models:
        model = load_model(hint, ckpt, args.device)
        results[disp] = {}
        for env_name in env_names:
            obs, act, episodes, s_pad, vel, starts, ts = env_data[env_name]
            results[disp][env_name] = run_model_env(model, env_name, obs, act,
                                                    episodes, s_pad, vel, starts,
                                                    ts, args)
            r = results[disp][env_name]
            lin = r["contexts"]["window"]["s_t"]["linear"]["r2_test"]
            ml = r["contexts"]["window"]["s_t"]["mlp"]["mean"]
            print(f"[e1] {disp} x {env_name}: s_t win linear={lin:.3f} "
                  f"mlp={ml:.3f} ({time.time() - t0:.0f}s)", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    pooled = pool_results(results, env_names)
    payload = {
        "experiment": "E1 nonlinear/temporal accessibility probe (cls0917-b)",
        "protocol": {
            "context_frames": CTX,
            "protocols": ["window (27-frame ctx, emb[:,-1])",
                          "warm (full-episode forward, emb at t)"],
            "temporal_concat": f"[z_(t-{HIST})..z_t] -> {HIST + 1} latents "
                               f"(spec said 5*192; note mandates 6 positions "
                               f"ending at t -> {(HIST + 1)}*d)",
            "targets": TARGETS + ["native sub-metric = first obs_dim_native cols"],
            "velocity": "pseudo-velocity: first difference of state within episode "
                        "(npz has no qvel field)",
            "probes": {"linear": "ridge alpha=1.0 standardized",
                       "mlp": "Linear(d,256)-GELU-Dropout(0.1)-Linear(256,out), "
                              "AdamW lr1e-3 wd1e-4, batch64, <=300 ep, early stop "
                              "patience 20 on window-level val slice of train "
                              "episodes, restore best; 3 seeds",
                       "temporal": "ridge on concatenated [z_(t-5)..z_t]"},
            "split": f"episode-level 75/25 (rng({SPLIT_SEED})), held-out R2 "
                     "uniform average over NON-CONSTANT columns of the 128-d "
                     "target; literal 128-col average kept as r2_full128 (JSON "
                     "only) -- constant padded columns would score 1.0 for ridge "
                     "vs 0.0 for MLP (force_finite artifact)",
            "n_windows_requested": args.n_windows,
            "seeds": list(range(args.seeds)),
            "checkpoints": {d: c for d, _, c in models},
            "data": {k: ENVS[k]["npz"] for k in env_names},
        },
        "results": results,
        "pooled": pooled,
    }
    json_path = out_dir / "nonlinear_probe_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    write_markdown(out_dir, results, pooled, env_names, [m[0] for m in models], args)
    print(f"[e1] wrote {json_path} ({time.time() - t0:.0f}s total)", flush=True)
    return 0


# ---------------------------------------------------------------- pooling
def pool_results(results: dict, env_names: list[str]) -> dict:
    """Per model x context x target x family: mean over envs; MLP seed std."""
    pooled: dict = {}
    for model, envs in results.items():
        pooled[model] = {}
        for ctx in ("window", "warm"):
            pc = {}
            for tname in TARGETS:
                cell = envs[env_names[0]]["contexts"][ctx][tname]
                entry: dict = {}
                fams = [f for f in ("linear", "temporal") if f in cell]
                for fam in fams:
                    per_env = [envs[e]["contexts"][ctx][tname][fam]["r2_test"]
                               for e in env_names]
                    entry[fam] = {"mean": float(np.mean(per_env)),
                                  "per_env": dict(zip(env_names, per_env))}
                seed_mat = np.array([
                    [s["r2_test"] for s in envs[e]["contexts"][ctx][tname]["mlp"]["seeds"]]
                    for e in env_names])            # (n_env, n_seed)
                pooled_seeds = seed_mat.mean(axis=0)
                entry["mlp"] = {"mean": float(pooled_seeds.mean()),
                                "std": float(pooled_seeds.std()),
                                "per_seed": pooled_seeds.tolist(),
                                "per_env_mean": dict(zip(
                                    env_names, seed_mat.mean(axis=1).tolist())),
                                "train_r2_mean": float(np.mean([
                                    s["r2_train"]
                                    for e in env_names
                                    for s in envs[e]["contexts"][ctx][tname]["mlp"]["seeds"]]))}
                entry["linear_train_r2_mean"] = float(np.mean([
                    envs[e]["contexts"][ctx][tname]["linear"]["r2_train"]
                    for e in env_names]))
                pc[tname] = entry
            pooled[model][ctx] = pc
    return pooled


# ---------------------------------------------------------------- markdown/png
def fmt(x: float) -> str:
    return f"{x:.3f}"


def write_markdown(out_dir: Path, results: dict, pooled: dict, env_names: list[str],
                   model_names: list[str], args) -> None:
    lines = [
        "# E1: Nonlinear / temporal accessibility probe (cls0917-b)",
        "",
        f"{args.n_windows} windows/env; contexts: WINDOW (27-frame ctx) and WARM "
        "(full-episode forward) on the same t positions. R2 = held-out episode-level "
        "75/25, uniform average over non-constant columns of the 128-d padded target "
        "(= native dims here); literal 128-col values in JSON only. Velocity = "
        "pseudo-velocity (first difference; npz has no qvel).",
        "",
        "MLP: 256-hidden, AdamW, early stopping, 3 seeds (mean +- seed std).",
        "",
        "## (i) s_t, pooled over envs: linear vs MLP vs temporal",
        "",
        "| model | win linear | win MLP | win MLP std | win temporal | "
        "warm linear | warm MLP | warm MLP std | warm temporal |",
        "|---|" + "---|" * 8,
    ]
    for m in model_names:
        pw, pm = pooled[m]["window"]["s_t"], pooled[m]["warm"]["s_t"]
        lines.append(
            f"| {m} | {fmt(pw['linear']['mean'])} | {fmt(pw['mlp']['mean'])} "
            f"| {fmt(pw['mlp']['std'])} | {fmt(pw['temporal']['mean'])} "
            f"| {fmt(pm['linear']['mean'])} | {fmt(pm['mlp']['mean'])} "
            f"| {fmt(pm['mlp']['std'])} | {fmt(pm['temporal']['mean'])} |")

    lines += ["", "## (ii) Headline: linear -> MLP gain on s_t (pooled)",
              "", "| model | win gain (MLP - linear) | warm gain |",
              "|---|---|---|"]
    for m in model_names:
        pw, pm = pooled[m]["window"]["s_t"], pooled[m]["warm"]["s_t"]
        lines.append(f"| {m} | {fmt(pw['mlp']['mean'] - pw['linear']['mean'])} "
                     f"| {fmt(pm['mlp']['mean'] - pm['linear']['mean'])} |")
    lines += ["", "Controls: LeWM/GRU gains should be small; a large trace gain "
              "=> information hidden nonlinearly, not discarded.",
              "", "## (iii) Pseudo-velocity (pooled)", "",
              "| model | win linear | win MLP | warm linear | warm MLP |",
              "|---|" + "---|" * 4]
    for m in model_names:
        pw, pm = pooled[m]["window"]["vel"], pooled[m]["warm"]["vel"]
        lines.append(
            f"| {m} | {fmt(pw['linear']['mean'])} | {fmt(pw['mlp']['mean'])} "
            f"| {fmt(pm['linear']['mean'])} "
            f"| {fmt(pm['mlp']['mean'])} |")

    for tgt in ("s_t+5", "s_t+20"):
        lines += ["", f"## Futures: {tgt} (pooled)", "",
                  "| model | win linear | win MLP | warm linear | "
                  "warm MLP |", "|---|" + "---|" * 4]
        for m in model_names:
            pw, pm = pooled[m]["window"][tgt], pooled[m]["warm"][tgt]
            lines.append(
                f"| {m} | {fmt(pw['linear']['mean'])} | {fmt(pw['mlp']['mean'])} "
                f"| {fmt(pm['linear']['mean'])} "
                f"| {fmt(pm['mlp']['mean'])} |")

    lines += ["", "## Overfitting check (s_t, pooled): train vs test R2", "",
              "| model | win lin train/test | win MLP train/test | warm lin train/test "
              "| warm MLP train/test |", "|---|" + "---|" * 4]
    for m in model_names:
        pw, pm = pooled[m]["window"]["s_t"], pooled[m]["warm"]["s_t"]
        lines.append(
            f"| {m} | {fmt(pw['linear_train_r2_mean'])} / {fmt(pw['linear']['mean'])} "
            f"| {fmt(pw['mlp']['train_r2_mean'])} / {fmt(pw['mlp']['mean'])} "
            f"| {fmt(pm['linear_train_r2_mean'])} / {fmt(pm['linear']['mean'])} "
            f"| {fmt(pm['mlp']['train_r2_mean'])} / {fmt(pm['mlp']['mean'])} |")

    lines += ["", "## Per-env appendix: s_t R2 (linear / MLP / temporal)", ""]
    for ctx in ("window", "warm"):
        lines += [f"### {ctx}", "", "| model | " + " | ".join(
            f"{e} lin/MLP/temp" for e in env_names) + " |", "|---|" + "---|" * len(env_names)]
        for m in model_names:
            row = [m]
            for e in env_names:
                c = results[m][e]["contexts"][ctx]["s_t"]
                row.append(f"{fmt(c['linear']['r2_test'])} / "
                           f"{fmt(c['mlp']['mean'])} / {fmt(c['temporal']['r2_test'])}")
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Per-env appendix: velocity R2 (linear / MLP)", ""]
    for ctx in ("window", "warm"):
        lines += [f"### {ctx}", "", "| model | " + " | ".join(
            f"{e} lin/MLP" for e in env_names) + " |", "|---|" + "---|" * len(env_names)]
        for m in model_names:
            row = [m]
            for e in env_names:
                c = results[m][e]["contexts"][ctx]["vel"]
                row.append(f"{fmt(c['linear']['r2_test'])} / "
                           f"{fmt(c['mlp']['mean'])}")
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Per-env appendix: futures R2, linear / MLP (window ctx)", "",
              "| model | env | s_t+5 lin/MLP | s_t+20 lin/MLP |", "|---|---|---|---|"]
    for m in model_names:
        for e in env_names:
            c5 = results[m][e]["contexts"]["window"]["s_t+5"]
            c20 = results[m][e]["contexts"]["window"]["s_t+20"]
            lines.append(f"| {m} | {e} | {fmt(c5['linear']['r2_test'])} / "
                         f"{fmt(c5['mlp']['mean'])} | {fmt(c20['linear']['r2_test'])} / "
                         f"{fmt(c20['mlp']['mean'])} |")

    md_path = out_dir / "nonlinear_probe_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    try:
        make_png(out_dir, pooled, model_names)
    except Exception as exc:  # chart is auxiliary; never fail the run on it
        print(f"[e1] PNG skipped: {exc}", flush=True)


def make_png(out_dir: Path, pooled: dict, model_names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lin = [pooled[m]["window"]["s_t"]["linear"]["mean"] for m in model_names]
    mlp = [pooled[m]["window"]["s_t"]["mlp"]["mean"] for m in model_names]
    err = [pooled[m]["window"]["s_t"]["mlp"]["std"] for m in model_names]
    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, lin, 0.4, label="linear ridge")
    ax.bar(x + 0.2, mlp, 0.4, yerr=err, capsize=3, label="MLP (3 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_ylabel(r"held-out $R^2$ (s_t, pooled over envs)")
    ax.set_title("E1: linear vs MLP accessibility of s_t — WINDOW context "
                 "(non-constant 128-d target cols)")
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "nonlinear_probe_s_t_window.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

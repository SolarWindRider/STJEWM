"""S4 (stab0917_maskobs): Partial-observability / state-estimator test.

Question (stable-predictive-state hypothesis, S4/Exp3): is STJEWM's advantage carried by
a *state estimate* maintained in latent dynamics — robust to dropping the newest
observation because the 27-frame history carries the state — rather than by more accurate
one-step encoding?

Protocol (per model x env x 128 windows):
  1. Windows: contexts of CTX=27 frames [t-26..t], zero-padded to 128 obs / 56 action
     dims (probe.py convention), sampled with [t-26 .. t+20] inside one episode (rng 0).
  2. Conditions:
       clean        : untouched window (control).
       last25/last50: k obs dims zeroed on the LAST frame only.
       win25/win50  : the same k dims zeroed on EVERY frame of the 27-frame window
                      (history-redundancy control).
     k = max(1, round(frac * D_real)) over the REAL obs dims (cartpole 2, cheetah 9);
     3 fixed mask seeds per (env, frac), shared across models (paired comparison).
     Masking choice: training zero-pads obs to 128 dims, so zero is the learned
     'no observation' sentinel in model input space; we zero the selected real dims and
     never impute. Padded dims are always zero = permanently unobserved, so mask draws
     only cover real dims. Cartpole's 2-dim obs make 25% and 50% coincide (1/2 dims);
     the fraction granularity is meaningful for cheetah (2/9 vs 4/9).
  3. Latents: z = model(o, a)["emb"][:, -1] (position t). z_clean from the clean window,
     z_mask from each masked window. All models are prefix-causal, and both latents come
     from the same window-forward path, so the displacement is internally consistent
     without warm episode forwards.
  4. Metrics:
       rel displacement = ||z_mask - z_clean||_2 / (||z_clean||_2 + 1e-8), mean over all
       windows, averaged over the 3 mask seeds; cosine distance recorded as auxiliary
       (saturates near chance for small-norm latents — see M2 anti-saturation lesson).
       R2 drop: per-Delta ridge probes fit ONCE on CLEAN latents (z -> s_{t+Delta}),
       standardized, alpha=1.0, episode-level 75/25 split (rng 123); held-out clean R2
       is the reference, masked latents are scored with the SAME fixed probe on the SAME
       held-out windows, so drops are attributable to the input condition, not probe
       refitting. Delta in {1, 5, 10, 20}; state = qpos slice (probe.py ENV_PROBE).
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
from code.scripts.mech0917_multistep import (
    CTX,
    PAD_ACT,
    PAD_OBS,
    DELTAS,
    episode_split,
    forward_windows,
    load_env_data,
    pad_batch,
    sample_windows,
)

# ---------------------------------------------------------------- constants
HMAX = max(DELTAS)           # 20; windows need [t-26 .. t+20] in-episode
MASK_FRACS = {"25": 0.25, "50": 0.50}
N_MASK_SEEDS = 3
# (condition name, scope, frac tag)
CONDITIONS = [
    ("last25", "last", "25"),
    ("last50", "last", "50"),
    ("win25", "window", "25"),
    ("win50", "window", "50"),
]

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


# ---------------------------------------------------------------- masking
def build_mask_specs(env_name: str, d_real: int, mask_seed: int, env_idx: int):
    """Deterministic mask dim sets per (env, frac tag, seed idx), shared across models."""
    specs: dict[str, list[np.ndarray]] = {}
    for frac_idx, (tag, frac) in enumerate(MASK_FRACS.items()):
        k = max(1, int(round(frac * d_real)))
        dims = []
        for s in range(N_MASK_SEEDS):
            rng = np.random.default_rng([mask_seed, env_idx, frac_idx, s])
            dims.append(np.sort(rng.choice(d_real, size=k, replace=False)))
        specs[tag] = dims
    return specs


def apply_mask(seqs: list[np.ndarray], dims: np.ndarray, scope: str) -> list[np.ndarray]:
    """Zero `dims` on the last frame ('last') or every frame ('window') of each (T,D) seq."""
    out = []
    for s in seqs:
        c = s.copy()
        if scope == "last":
            c[-1, dims] = 0.0
        else:
            c[:, dims] = 0.0
        out.append(c)
    return out


def forward_last_latents(model, obs_seqs, act_seqs, device, batch: int) -> np.ndarray:
    """Window forward -> latent at the last position (t). Returns (n, D) float32."""
    outs = []
    for i in range(0, len(obs_seqs), batch):
        emb = forward_windows(model, obs_seqs[i : i + batch], act_seqs[i : i + batch], device)
        outs.append(emb[:, -1].float().cpu().numpy())
    return np.concatenate(outs, axis=0)


# ---------------------------------------------------------------- metrics
def rel_displacement(z_mask: np.ndarray, z_clean: np.ndarray) -> tuple[float, float]:
    """Mean relative L2 ||dz||/(||z_clean||+1e-8) and mean cosine distance (aux)."""
    num = np.linalg.norm(z_mask - z_clean, axis=1)
    den = np.linalg.norm(z_clean, axis=1) + 1e-8
    rel = float((num / den).mean())
    a = torch.from_numpy(z_mask)
    b = torch.from_numpy(z_clean)
    cos_d = float((1.0 - F.cosine_similarity(a, b, dim=-1)).mean().item())
    return rel, cos_d


# ---------------------------------------------------------------- model loading
def load_model(hint: str, ckpt_path: str, device: str):
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(hint, state_dim=PAD_OBS, action_dim=PAD_ACT,
                        ck_args=ck["args"], state_dict=ck["model"])
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------- per model x env
def run_env_model(model, env_name: str, pos: tuple[int, int], obs, act, episodes,
                  win_starts, win_ts, masks, args) -> dict:
    device = args.device
    n = len(win_ts)
    o_seqs = [obs[t - CTX + 1 : t + 1] for t in win_ts]
    a_seqs = [act[t - CTX + 1 : t + 1] for t in win_ts]

    # ---- clean latents (control)
    z_clean = forward_last_latents(model, o_seqs, a_seqs, device, args.batch)
    d_emb = z_clean.shape[1]

    # ---- masked latents per (condition, mask seed)
    z_masked: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in CONDITIONS}
    for name, scope, tag in CONDITIONS:
        for seed_idx in range(N_MASK_SEEDS):
            m_seqs = apply_mask(o_seqs, masks[tag][seed_idx], scope)
            z_masked[name].append(
                forward_last_latents(model, m_seqs, a_seqs, device, args.batch))

    # ---- probe targets s_{t+Delta} and fixed episode split
    p0, p1 = pos
    s_future = {delta: np.stack([obs[t + delta, p0:p1] for t in win_ts])
                for delta in DELTAS}
    train_mask = episode_split(win_starts, np.random.default_rng(123))
    tr, te = train_mask, ~train_mask

    res = {
        "emb_dim": int(d_emb),
        "n_windows": int(n),
        "n_windows_test": int(te.sum()),
        "mask_k_dims": {tag: int(masks[tag][0].size) for tag in MASK_FRACS},
        "probe_r2_clean": {},     # held-out clean R2 per Delta (fixed reference)
        "conditions": {},
    }

    # ---- fixed clean probes: fit ONCE per Delta on clean latents
    probes = {}
    for delta in DELTAS:
        sc = StandardScaler().fit(z_clean[tr])
        rg = Ridge(alpha=1.0).fit(sc.transform(z_clean[tr]), s_future[delta][tr])
        r2_clean = float(rg.score(sc.transform(z_clean[te]), s_future[delta][te]))
        probes[delta] = (sc, rg)
        res["probe_r2_clean"][str(delta)] = r2_clean

    # ---- per condition: displacement + R2 under the fixed clean probes
    for name, scope, tag in CONDITIONS:
        rels, cos_ds = [], []
        r2_masked = {str(d): [] for d in DELTAS}
        r2_drops = {str(d): [] for d in DELTAS}
        for seed_idx in range(N_MASK_SEEDS):
            zm = z_masked[name][seed_idx]
            rel, cos_d = rel_displacement(zm, z_clean)
            rels.append(rel)
            cos_ds.append(cos_d)
            for delta in DELTAS:
                sc, rg = probes[delta]
                r2 = float(r2_score(s_future[delta][te],
                                    rg.predict(sc.transform(zm[te])),
                                    multioutput="uniform_average"))
                r2_masked[str(delta)].append(r2)
                r2_drops[str(delta)].append(res["probe_r2_clean"][str(delta)] - r2)
        res["conditions"][name] = {
            "scope": scope,
            "frac": MASK_FRACS[tag],
            "rel_l2": float(np.mean(rels)),
            "rel_l2_per_seed": rels,
            "cos_dist_aux": float(np.mean(cos_ds)),
            "r2_masked": {d: float(np.mean(v)) for d, v in r2_masked.items()},
            "r2_drop": {d: float(np.mean(v)) for d, v in r2_drops.items()},
            "r2_drop_per_seed": {d: v for d, v in r2_drops.items()},
        }
    return res


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="S4: partial-observability / state-estimator test (obs masking)")
    ap.add_argument("--n-windows", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64, help="windows per forward batch")
    ap.add_argument("--seed", type=int, default=0, help="window sampling seed")
    ap.add_argument("--split-seed", type=int, default=123, help="episode split seed")
    ap.add_argument("--mask-seed", type=int, default=2026, help="base mask seed")
    ap.add_argument("--models", default="", help="comma filter on display names (debug)")
    ap.add_argument("--out-dir",
                    default="/home/lx/snn/results/mech_0917_b/maskobs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = ([m.strip() for m in args.models.split(",") if m.strip()]
            if args.models else [d for d, _, _ in MODELS])

    t_start = time.time()
    # per-env data + windows + mask specs (shared across models, paired)
    env_data = {}
    for env_idx, (env_name, cfg) in enumerate(ENVS.items()):
        obs, act, episodes = load_env_data(cfg["npz"], CTX + HMAX)
        starts, ends, ts = sample_windows(episodes, args.n_windows,
                                          np.random.default_rng(args.seed))
        masks = build_mask_specs(env_name, obs.shape[1], args.mask_seed, env_idx)
        env_data[env_name] = (obs, act, episodes, starts, ts, masks)
        ks = {tag: masks[tag][0].tolist() for tag in MASK_FRACS}
        print(f"[s4] {env_name}: d_real={obs.shape[1]} masks={ks}, "
              f"{len(ts)} windows", flush=True)

    results: dict = {}
    for disp, hint, ckpt in MODELS:
        if disp not in want:
            continue
        model = load_model(hint, ckpt, args.device)
        results[disp] = {}
        for env_name, cfg in ENVS.items():
            obs, act, episodes, starts, ts, masks = env_data[env_name]
            res = run_env_model(model, env_name, cfg["pos"], obs, act, episodes,
                                starts, ts, masks, args)
            results[disp][env_name] = res
            c = res["conditions"]
            print(f"[s4] {disp} x {env_name}: "
                  f"rel last25={c['last25']['rel_l2']:.4f} "
                  f"win25={c['win25']['rel_l2']:.4f} "
                  f"r2drop(d1,last25)={c['last25']['r2_drop']['1']:.4f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---------------- pooled (mean over envs)
    pooled: dict = {}
    for disp in results:
        envs = list(results[disp].keys())
        p: dict = {"n_envs": len(envs), "rel_l2": {}, "r2_drop": {},
                   "probe_r2_clean_mean": {}}
        for name, _, _ in CONDITIONS:
            p["rel_l2"][name] = float(np.mean(
                [results[disp][e]["conditions"][name]["rel_l2"] for e in envs]))
            p["r2_drop"][name] = {
                str(d): float(np.mean([results[disp][e]["conditions"][name]["r2_drop"][str(d)]
                                       for e in envs]))
                for d in DELTAS}
        p["probe_r2_clean_mean"] = {
            str(d): float(np.mean([results[disp][e]["probe_r2_clean"][str(d)]
                                   for e in envs]))
            for d in DELTAS}
        pooled[disp] = p

    payload = {
        "experiment": "S4 partial observability / state-estimator test (obs masking)",
        "protocol": {
            "context_frames": CTX,
            "n_windows": args.n_windows,
            "mask_fracs": MASK_FRACS,
            "n_mask_seeds": N_MASK_SEEDS,
            "mask_seed_base": args.mask_seed,
            "masking_choice": (
                "obs are zero-padded from D_real to 128 at train time, so zero is the "
                "learned 'no observation' sentinel in model input space; masked real "
                "dims are zeroed, never imputed. Mask draws cover real dims only "
                "(padded dims are always zero = permanently unobserved). k = "
                "max(1, round(frac*D_real)) per (env, frac); dim sets are fixed per "
                "(env, frac, seed) and shared across models (paired comparison). "
                "cartpole D_real=2 makes 25% and 50% coincide (1/2 dims); cheetah "
                "D_real=9 gives 2/9 vs 4/9."),
            "window": "win* = same k dims zeroed at EVERY frame of the 27-frame window "
                      "(history-redundancy control); last* = last frame only",
            "latents": "model(o,a)['emb'][:, -1] at position t from 27-frame window "
                       "forwards (all models prefix-causal; clean and masked share the "
                       "same forward path)",
            "metrics": {
                "rel_l2": "mean ||z_mask - z_clean|| / (||z_clean|| + 1e-8) over all "
                          "windows, averaged over mask seeds (primary)",
                "cos_dist_aux": "mean cosine distance (auxiliary; saturates for "
                                "small-norm latents)",
                "r2_drop": "probe_r2_clean - r2_masked; ridge probes (standardized, "
                           "alpha=1.0) fit ONCE on clean latents z -> s_{t+Delta}, "
                           "episode-level 75/25 split; masked latents scored with the "
                           "same fixed probe on the same held-out windows",
            },
            "deltas": DELTAS,
            "seeds": {"windows": args.seed, "split": args.split_seed},
            "checkpoints": {d: c for d, _, c in MODELS if d in results},
            "data": {k: v["npz"] for k, v in ENVS.items()},
        },
        "results": results,
        "pooled": pooled,
    }
    json_path = out_dir / "maskobs_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False))

    # ---------------- markdown
    def disp_table(getter) -> list[str]:
        header = "| model | " + " | ".join(n for n, _, _ in CONDITIONS) + " |"
        sep = "|---|" + "---|" * len(CONDITIONS)
        rows = [header, sep]
        for d in results:
            rows.append("| " + d + " | " +
                        " | ".join(f"{getter(d, n):.4f}" for n, _, _ in CONDITIONS) + " |")
        return rows

    lines = [
        "# S4: Partial observability / state-estimator test (obs masking)",
        "",
        f"{len(results)} models x {len(ENVS)} envs x {args.n_windows} windows "
        f"(27-frame contexts; 3 mask seeds per condition, shared across models).",
        "",
        "**Masking choice:** obs are zero-padded to 128 dims at train time, so zero is "
        "the learned 'no observation' sentinel; masked real dims are zeroed, never "
        "imputed (cartpole D_real=2 -> 25%/50% coincide at 1/2 dims; cheetah 2/9 vs "
        "4/9). `last*` masks the last frame only; `win*` masks the same dims at every "
        "frame of the window (history-redundancy control).",
        "",
        "Primary metrics: relative latent displacement ||z_mask - z_clean|| / "
        "(||z_clean|| + 1e-8); R2 drop vs clean probes (ridge fit ONCE on clean "
        "latents, standardized, alpha=1.0, episode 75/25 split, held-out). Cosine "
        "distance is auxiliary only (saturation risk).",
        "",
        "**Caveat:** low displacement is only meaningful for latents that carry state. "
        "Read displacement jointly with the clean-probe R2 headroom tables: "
        "mlp_baseline is fully collapsed (constant output, per-dim std = 0) and "
        "stacked_lif_trace is near-constant (clean R2 < 0), so their ~0 displacement "
        "reflects input-insensitivity, not state estimation; their R2 drops are noise "
        "on a degenerate base.",
    ]
    for env_name in ENVS:
        lines += ["", f"## {env_name}", "",
                  "### Relative latent displacement (mean over 3 mask seeds; lower = "
                  "latent unaffected by missing obs)", ""]
        lines += disp_table(lambda d, n: results[d][env_name]["conditions"][n]["rel_l2"])
        lines += ["", "### Clean probe R2 (held-out; headroom reference)", "",
                  "| model | " + " | ".join(f"d={d}" for d in DELTAS) + " |",
                  "|---|" + "---|" * len(DELTAS)]
        for d in results:
            lines.append("| " + d + " | " + " | ".join(
                f"{results[d][env_name]['probe_r2_clean'][str(dd)]:.4f}"
                for dd in DELTAS) + " |")
        for delta in DELTAS:
            lines += ["", f"### R2 drop vs clean probe, Delta={delta} "
                          "(positive = state decoding degraded)", ""]
            lines += disp_table(
                lambda d, n, dd=delta: results[d][env_name]["conditions"][n]["r2_drop"][str(dd)])
    lines += ["", "## Pooled (mean over envs)", "",
              "### Relative latent displacement", ""]
    lines += disp_table(lambda d, n: pooled[d]["rel_l2"][n])
    for delta in DELTAS:
        lines += ["", f"### R2 drop, Delta={delta} (pooled)", ""]
        lines += disp_table(lambda d, n, dd=delta: pooled[d]["r2_drop"][n][str(dd)])

    # ---------------- verdict block
    def get(disp, env, field):
        return results[disp][env]["conditions"][field]["rel_l2"]

    v = ["", "## Verdict (auto-computed)", ""]
    if "stjewm_trace_only" in results and "lewm_baseline_v2" in results:
        for env_name in ENVS:
            t25, l25 = get("stjewm_trace_only", env_name, "last25"), get("lewm_baseline_v2", env_name, "last25")
            t50, l50 = get("stjewm_trace_only", env_name, "last50"), get("lewm_baseline_v2", env_name, "last50")
            smaller = t25 < l25 and t50 < l50
            v.append(f"- `{env_name}`: last-frame displacement trace={t25:.4f}/{t50:.4f} "
                     f"vs lewm={l25:.4f}/{l50:.4f} "
                     f"(trace {'SMALLER' if smaller else 'NOT smaller'} at both fracs)")
        pt = pooled["stjewm_trace_only"]["rel_l2"]
        pl = pooled["lewm_baseline_v2"]["rel_l2"]
        v += ["", f"- Pooled: trace last25={pt['last25']:.4f} last50={pt['last50']:.4f} "
                  f"win25={pt['win25']:.4f} win50={pt['win50']:.4f}; lewm "
                  f"last25={pl['last25']:.4f} last50={pl['last50']:.4f} "
                  f"win25={pl['win25']:.4f} win50={pl['win50']:.4f}.",
              f"- Whole-window break ratio (win/last, pooled): trace "
              f"{pt['win25'] / max(pt['last25'], 1e-12):.2f}x (25%), "
              f"{pt['win50'] / max(pt['last50'], 1e-12):.2f}x (50%); lewm "
              f"{pl['win25'] / max(pl['last25'], 1e-12):.2f}x / "
              f"{pl['win50'] / max(pl['last50'], 1e-12):.2f}x."]
    lines += v
    (out_dir / "maskobs_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[s4] wrote {json_path} and maskobs_summary.md "
          f"({time.time() - t_start:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

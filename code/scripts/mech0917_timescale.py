"""M3: MultiCompStack per-layer timescale probe (mech0917).

For stjewm_trace_only, capture every MultiCompStack layer's hidden output
h^(l)(t) (l=1..4) with forward hooks on stack.cells[i], plus the gated trace
and the final readout emb, during full 27-frame window forwards. For each
representation r and lag Delta in {0,1,5,10,20}, ridge-probe r[t] ->
s_{t+Delta} (dataset state at the absolute position, qpos slice per probe.py
conventions), held-out multioutput R^2, split by episode 75/25.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m code.scripts.mech0917_timescale \
        --device cuda:0 --out results/mech_0917/timescale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")

from code.scripts.event_align import build_model

CKPT = "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt"
MODEL_HINT = "stjewm_trace_only"

# env -> (npz path, qpos slice as in probe.py ENV_PROBE)
ENVS: dict[str, tuple[str, tuple[int, int]]] = {
    "cartpole_2d": ("/home/lx/snn/data/dm_control/cartpole_250k.npz", (0, 2)),
    "cheetah": ("/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz", (0, 9)),
    "finger": ("/home/lx/snn/data/dm_control/3d_rollouts_250k/finger_250k.npz", (0, 3)),
}

DELTAS = [0, 1, 5, 10, 20]
LAYER_REPS = ["h1", "h2", "h3", "h4"]          # per-layer hidden h^(l), l=1..4
REF_REPS = ["trace", "emb"]                     # gated trace, final readout
REPS = LAYER_REPS + REF_REPS

WINDOW = 27          # 1 (history) + 25 (context) + 1 (goal)
STATE_DIM_PAD = 128
ACTION_DIM_PAD = 56


# ============================================================
# Data (mirrors code/data/loaders.py load_dmc + _npz_window_starts)
# ============================================================
def load_npz(path: str):
    d = np.load(path)
    obs = d["observations"][:, 0, :].astype(np.float32)
    act = d["actions"][:, 0, :].astype(np.float32)
    dones = np.asarray(d["dones"]).reshape(-1).astype(bool)
    if len(obs) != len(dones):
        raise ValueError(f"{path}: obs/dones length mismatch")
    return obs, act, dones


def episode_last_frame(dones: np.ndarray) -> np.ndarray:
    """ep_last[i] = last frame index of the episode containing frame i.

    Done flags mark the final frame of an episode; frames after the last done
    are treated as one trailing episode ending at the file end (same semantics
    as the loader's no-done-in-window validity rule).
    """
    n = len(dones)
    ep_last = np.empty(n, dtype=np.int64)
    nxt = n - 1
    for i in range(n - 1, -1, -1):
        if dones[i]:
            nxt = i
        ep_last[i] = nxt
    return ep_last


def valid_window_starts(dones: np.ndarray, window: int) -> np.ndarray:
    starts = np.arange(max(0, len(dones) - window + 1))
    counts = np.concatenate([[0], np.cumsum(dones)])
    return starts[(counts[starts + window] - counts[starts]) == 0]


def episode_id_of_start(dones: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Episode index of each window (= number of dones before the start)."""
    return np.concatenate([[0], np.cumsum(dones)[:-1]])[starts]


# ============================================================
# Model + hooks
# ============================================================
class CellHookStore:
    """Forward hooks on stack.cells[i]; cells return dicts, take 'spike'."""

    def __init__(self, cells):
        self.spikes: dict[int, torch.Tensor] = {}
        self.handles = []
        for i, cell in enumerate(cells):
            self.handles.append(cell.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, i):
        def hook(_module, _inp, out):
            t = out
            if isinstance(t, (tuple, list)):
                t = next(x for x in t if isinstance(x, torch.Tensor))
            if isinstance(t, dict):
                t = t["spike"]
            self.spikes[i] = t.detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def capture_representations(model, obs_b: torch.Tensor, act_b: torch.Tensor):
    """One forward pass with hooks; returns dict name -> (B, T, D) CPU tensors."""
    store = CellHookStore(model.stack.cells)
    try:
        with torch.no_grad():
            out = model(obs_b, act_b)
        spikes = dict(store.spikes)
    finally:
        store.remove()

    n_layers = len(model.stack.cells)
    missing = [i for i in range(n_layers) if i not in spikes]
    if missing:
        raise RuntimeError(f"hooks did not fire for cells {missing}")

    # Reconstruct per-layer hidden h^(l) exactly as MultiCompStack.forward:
    #   h = x_in ; for each layer: h = h + norm_l(post_mlp_l(spike_l))
    h = out["emb_pre_cell"] + out["act_emb"]
    reps: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for l in range(n_layers):
            r = model.stack.post_mlps[l](spikes[l])
            h = h + model.stack.norms[l](r)
            reps[f"h{l + 1}"] = h.detach().float().cpu()
    # Sanity: reconstruction must match the stack's own final hidden state.
    h_check = float((reps[f"h{n_layers}"] - out["h"].detach().float().cpu()).abs().max())
    reps["trace"] = out["trace"].detach().float().cpu()
    reps["emb"] = out["emb"].detach().float().cpu()
    return reps, h_check


# ============================================================
# Ridge probe
# ============================================================
def ridge_probe_r2(X: np.ndarray, Y: np.ndarray, ep_ids: np.ndarray,
                   seed: int = 0, train_frac: float = 0.75):
    """Ridge (alpha=1.0) on standardized features; episode-level 75/25 split;
    held-out mean multioutput R^2 (constant targets scored 0.0, probe.py rule)."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    eps = np.unique(ep_ids)
    perm = rng.permutation(len(eps))
    n_train_eps = max(1, int(round(train_frac * len(eps))))
    train_eps = set(eps[perm[:n_train_eps]].tolist())
    tr = np.array([e in train_eps for e in ep_ids])
    va = ~tr
    if tr.sum() < 10 or va.sum() < 10:
        raise RuntimeError(f"degenerate split: n_train={tr.sum()} n_val={va.sum()}")

    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xtr = (X[tr] - mu) / sd
    Xva = (X[va] - mu) / sd

    mdl = Ridge(alpha=1.0)
    mdl.fit(Xtr, Y[tr])
    pred = mdl.predict(Xva)
    yt = Y[va].astype(np.float64)
    r2s = []
    for d in range(yt.shape[1]):
        ss_res = float(((yt[:, d] - pred[:, d]) ** 2).sum())
        ss_tot = float(((yt[:, d] - yt[:, d].mean()) ** 2).sum())
        r2s.append(0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot)
    return float(np.mean(r2s)), int(tr.sum()), int(va.sum()), len(train_eps), len(eps) - n_train_eps


# ============================================================
# Per-env data collection
# ============================================================
def collect_env(model, env: str, device: str, n_windows: int, batch: int,
                seed: int = 0):
    """Returns reps[delta][name] -> (N, D) float64 samples, targets[delta],
    ep_ids[delta], plus bookkeeping."""
    path, (p0, p1) = ENVS[env]
    obs, act, dones = load_npz(path)
    starts_all = valid_window_starts(dones, WINDOW)
    rng = np.random.default_rng(seed)
    take = min(n_windows, len(starts_all))
    starts = np.sort(rng.choice(starts_all, size=take, replace=False))
    ep_last = episode_last_frame(dones)
    win_ep = episode_id_of_start(dones, starts)

    # ---- forward all windows in batches, gather per-step representations ----
    rep_batches: dict[str, list[torch.Tensor]] = {r: [] for r in REPS}
    h_checks = []
    for b0 in range(0, len(starts), batch):
        idx = starts[b0: b0 + batch]
        obs_w = torch.from_numpy(obs[idx[:, None] + np.arange(WINDOW)])          # (B, 27, obs_dim)
        act_w = torch.from_numpy(act[idx[:, None] + np.arange(WINDOW - 1)])  # (B, 26, A)
        B = obs_w.shape[0]
        obs_pad = torch.zeros(B, WINDOW, STATE_DIM_PAD)
        obs_pad[:, :, : obs_w.shape[-1]] = obs_w
        act_pad = torch.zeros(B, WINDOW, ACTION_DIM_PAD)
        act_pad[:, : act_w.shape[1], : act_w.shape[-1]] = act_w   # zero-pad last row
        reps, h_check = capture_representations(
            model, obs_pad.to(device), act_pad.to(device))
        h_checks.append(h_check)
        for r in REPS:
            rep_batches[r].append(reps[r])
    reps_win = {r: torch.cat(rep_batches[r], dim=0).numpy() for r in REPS}  # (W, 27, D)

    # ---- assemble per-Delta flattened samples (shared across reps) ----
    per_delta = {}
    for delta in DELTAS:
        t_max = np.minimum(WINDOW - 1, ep_last[starts] - starts - delta)
        if (t_max < 0).any():
            raise RuntimeError(f"{env} delta={delta}: window without target")
        Xs = {r: [] for r in REPS}
        Ys, Eps = [], []
        for b, p in enumerate(starts):
            n_b = int(t_max[b]) + 1
            for r in REPS:
                Xs[r].append(reps_win[r][b, :n_b])
            Ys.append(obs[p + delta: p + delta + n_b, p0:p1])
            Eps.append(np.full(n_b, win_ep[b]))
        entry = {
            "X": {r: np.concatenate(Xs[r]).astype(np.float64) for r in REPS},
            "Y": np.concatenate(Ys).astype(np.float64),
            "ep": np.concatenate(Eps),
            "n_windows": int(len(starts)),
            "n_used_windows": int((t_max >= 0).sum()),
        }
        per_delta[delta] = entry

    info = {
        "n_windows": int(len(starts)),
        "n_valid_starts": int(len(starts_all)),
        "h_reconstruction_max_abs_err": float(max(h_checks)),
        "obs_dim": int(obs.shape[1]),
        "action_dim": int(act.shape[1]),
        "probe_slice": [p0, p1],
        "episode_frames": int(np.flatnonzero(dones)[0] + 1) if dones.any() else -1,
    }
    return per_delta, info


# ============================================================
# Verdict logic
# ============================================================
def gradient_verdict(layer_mat: np.ndarray):
    """layer_mat: (4, n_deltas) mean R^2 for h1..h4 (averaged over envs)."""
    out = {"per_delta_argmax_layer": [], "per_delta_cross_layer_range": []}
    inc = dec = True
    for j in range(layer_mat.shape[1]):
        col = layer_mat[:, j]
        out["per_delta_argmax_layer"].append(LAYER_REPS[int(np.argmax(col))])
        out["per_delta_cross_layer_range"].append(round(float(col.max() - col.min()), 4))
        if not np.all(np.diff(col) > -1e-3):
            inc = False
        if not np.all(np.diff(col) < 1e-3):
            dec = False
    argmaxes = set(out["per_delta_argmax_layer"])
    if inc or dec:
        verdict = ("monotone_shallow_to_deep" if dec else
                   "monotone_deep_to_shallow")
    elif len(argmaxes) > 1:
        verdict = "division_of_labor"
    else:
        verdict = "shared_profile"
    out["verdict"] = verdict
    return out


# ============================================================
# Main
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="M3 per-layer timescale probe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-windows", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/lx/snn/results/mech_0917/timescale")
    args = ap.parse_args()

    if not Path(CKPT).exists():
        raise FileNotFoundError(f"checkpoint missing: {CKPT}")
    for env, (path, _) in ENVS.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{env} data missing: {path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {}) or {}
    model = build_model(MODEL_HINT, state_dim=STATE_DIM_PAD,
                        action_dim=ACTION_DIM_PAD, ck_args=ck_args,
                        state_dict=ck["model"])
    model.load_state_dict(ck["model"], strict=True)
    model.to(args.device).eval()
    print(f"[timescale] loaded {MODEL_HINT} readout_mode={model.readout_mode}",
          flush=True)

    P: dict[str, dict[str, dict[str, float]]] = {}   # env -> rep -> delta -> r2
    env_info: dict[str, dict] = {}
    sample_counts: dict[str, dict[int, dict]] = {}
    for env in ENVS:
        per_delta, info = collect_env(model, env, args.device,
                                      args.n_windows, args.batch, args.seed)
        env_info[env] = info
        P[env] = {}
        sample_counts[env] = {}
        for delta in DELTAS:
            entry = per_delta[delta]
            P[env][delta] = {}
            for r in REPS:
                r2, n_tr, n_va, n_eps_tr, n_eps_va = ridge_probe_r2(
                    entry["X"][r], entry["Y"], entry["ep"], seed=args.seed)
                P[env][delta][r] = round(r2, 6)
                sample_counts[env][delta] = {
                    "n_samples": int(entry["Y"].shape[0]),
                    "n_train": n_tr, "n_val": n_va,
                    "n_train_episodes": n_eps_tr, "n_val_episodes": n_eps_va,
                    "target_dim": int(entry["Y"].shape[1]),
                }
            print(f"[timescale] {env} d={delta}: "
                  f"{ {r: P[env][delta][r] for r in REPS} }", flush=True)

    # ---- mean matrix over envs ----
    P_mean = {delta: {r: round(float(np.mean([P[e][delta][r] for e in ENVS])), 6)
                      for r in REPS} for delta in DELTAS}
    layer_mean = np.array([[P_mean[d][r] for d in DELTAS] for r in LAYER_REPS])
    verdict = gradient_verdict(layer_mean)

    result = {
        "protocol": {
            "experiment": "M3 MultiCompStack per-layer timescale probe",
            "model": MODEL_HINT, "ckpt": CKPT,
            "readout_mode": str(model.readout_mode),
            "window_frames": WINDOW, "deltas": DELTAS,
            "representations": REPS,
            "rep_definition": {
                "h1": "stack layer-1 hidden h^(1) = x_in + norm(post_mlp(spike_1))",
                "h4": "stack layer-4 hidden (== forward output 'h')",
                "trace": "gated spike trace (GatedSpikeTrace output)",
                "emb": "final readout z_final ('emb' key)",
            },
            "probe": "sklearn Ridge alpha=1.0 on standardized features; "
                     "episode-level 75/25 split; held-out mean multioutput R^2",
            "target": "dataset state qpos slice at absolute position t+Delta "
                      "(probe.py ENV_PROBE pos slice)",
            "window_sampling": f"rng(numpy default_rng({args.seed})), {args.n_windows} "
                               "valid starts per env, uniform without replacement",
            "state_dim_pad": STATE_DIM_PAD, "action_dim_pad": ACTION_DIM_PAD,
            "n_windows_per_env": args.n_windows,
        },
        "env_info": env_info,
        "sample_counts": sample_counts,
        "P_per_env": P,
        "P_mean": P_mean,
        "verdict": verdict,
    }
    json_path = out_dir / "timescale_summary.json"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False))

    # ---- markdown ----
    lines = ["# M3 MultiCompStack per-layer timescale probe (stjewm_trace_only)", ""]
    lines.append(f"Model: `{MODEL_HINT}` | window={WINDOW} frames | "
                 f"{args.n_windows} windows/env | ridge alpha=1.0, episode-level 75/25, "
                 "held-out multioutput R^2 | targets: qpos slice at t+Delta")
    lines.append("")
    for env in ENVS:
        lines.append(f"## {env}  (n={env_info[env]['n_windows']} windows, "
                     f"h-check max|err|={env_info[env]['h_reconstruction_max_abs_err']:.2e})")
        lines.append("")
        lines.append("| rep | " + " | ".join(f"Δ={d}" for d in DELTAS) + " |")
        lines.append("|" + "---|" * (len(DELTAS) + 1))
        for r in REPS:
            lines.append(f"| {r} | " + " | ".join(f"{P[env][d][r]:.4f}" for d in DELTAS) + " |")
        lines.append("")
    lines.append("## Mean over envs")
    lines.append("")
    lines.append("| rep | " + " | ".join(f"Δ={d}" for d in DELTAS) + " |")
    lines.append("|" + "---|" * (len(DELTAS) + 1))
    for r in REPS:
        lines.append(f"| {r} | " + " | ".join(f"{P_mean[d][r]:.4f}" for d in DELTAS) + " |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    for j, d in enumerate(DELTAS):
        lines.append(f"- Δ={d}: argmax layer **{verdict['per_delta_argmax_layer'][j]}**, "
                     f"cross-layer (h1–h4) range "
                     f"{verdict['per_delta_cross_layer_range'][j]:.4f}")
    lines.append(f"- Gradient verdict: **{verdict['verdict']}**")
    lines.append("")
    (out_dir / "timescale_summary.md").write_text("\n".join(lines))

    # ---- heatmap ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.array([[P_mean[d][r] for d in DELTAS] for r in REPS])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(DELTAS)), [f"Δ={d}" for d in DELTAS])
    ax.set_yticks(range(len(REPS)), REPS)
    for i in range(len(REPS)):
        for j in range(len(DELTAS)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    color="white" if mat[i, j] < mat.max() * 0.6 else "black",
                    fontsize=9)
    ax.set_title("M3 timescale probe: held-out R² per (rep, Δ), "
                 "mean over envs (stjewm_trace_only)")
    fig.colorbar(im, ax=ax, label="held-out R²")
    fig.tight_layout()
    fig.savefig(out_dir / "timescale_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"[timescale] verdict={verdict['verdict']} -> {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""P0-4: dose-matched masking (last-frame vs scatter) + high-dose reference.

Conditions per (model, env, mask seed):
  A last25:   0.25*D elements zeroed on the LAST frame only (same dims as v1).
  B scatter25: the SAME TOTAL count (0.25*D elements) zeroed, positions drawn
              uniformly across all 27 frames (seeded per window+mask seed).
  C win25 (high-dose reference, v1): same 25% dims zeroed on EVERY frame.
Same input windows (SEED_WIN=0, 64/env), 3 mask seeds shared across models.
Metric: rel_l2 = ||z_mask - z_clean|| / (||z_clean|| + 1e-8).
Representation-displacement only (no closed-loop claim).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/home/lx/snn/code/scripts")

import mech0917_multistep as ms
import stab0917_rollout as S1

OUT = Path("/home/lx/audit_20261120")
DEVICE = "cuda:0"
EPS = 1e-8
N_WIN = 64
MASK_SEEDS = 3
CTX = 27
QUARTET = [("trace", "stjewm_trace_only", "stjewm"),
           ("membrane", "stjewm_membrane_readout", "stjewm"),
           ("lewm", "lewm_baseline_v2", "baseline"),
           ("gru", "gru_baseline", "baseline")]


def apply_mask(o_seqs, dims, mode, rng, frac=0.25):
    """o_seqs: list of (T, D_obs) arrays. Returns masked copies."""
    out = []
    D = o_seqs[0].shape[1]
    k_last = len(dims)  # total element count = 0.25*D by construction
    total = k_last  # same total element count for A and B
    for wi, o in enumerate(o_seqs):
        o = o.copy()
        if mode == "last25":
            o[-1, list(dims)] = 0.0
        elif mode == "scatter25":
            rr = np.random.default_rng(rng + wi)
            frames = rr.integers(0, CTX, size=total)
            cols = rr.integers(0, D, size=total)
            for f, c in zip(frames, cols):
                o[f, c] = 0.0
        elif mode == "win25":
            o[:, list(dims)] = 0.0
        out.append(o)
    return out


def main() -> int:
    env_data = {}
    for env_name, cfg in S1.ENVS.items():
        obs, act, episodes = ms.load_env_data(cfg["npz"], S1.CTX + S1.HMAX)
        starts, ends, ts = ms.sample_windows(episodes, N_WIN, np.random.default_rng(0))
        env_data[env_name] = (obs, act, episodes, starts, ends, ts)
    win_rows, summary = [], []
    for disp, hint, family in QUARTET:
        base = ("/data/lx/tmp/results/5m_5mpar" if family == "stjewm"
                else "/data/lx/tmp/results/5m")
        ck = Path(base) / "oodc_F1" / hint / "seed_0" / "final.pt"
        model = S1.load_model(hint, str(ck), DEVICE)
        for env_name, (obs, act, episodes, starts, ends, ts) in env_data.items():
            n = len(ts)
            ep_map = dict(episodes)
            spans = [(s, ep_map[s]) for s in sorted(set(starts.tolist()))]
            ep_emb = ms.forward_episodes(model, obs, act, spans, DEVICE, batch=32)
            for ms_seed in range(MASK_SEEDS):
                rr = np.random.default_rng(1000 + ms_seed)
                # per-window dims for last/scatter base selection
                dims_per_win = [rr.choice(obs.shape[1], size=max(1, int(round(0.25 * obs.shape[1]))),
                                          replace=False) for _ in range(n)]
                # do it per window (64 windows, cheap)
                acc = {"last25": np.zeros(n), "scatter25": np.zeros(n), "win25": np.zeros(n)}
                with torch.inference_mode():
                    o_seqs = [obs[t - CTX + 1: t + 1] for t in ts]
                    a_seqs = [act[t - CTX + 1: t + 1] for t in ts]
                    ctx_c = S1.forward_windows(model, o_seqs, a_seqs, DEVICE)[:, -1, :]
                    zc_n = ctx_c.cpu().numpy()
                    for wi in range(n):
                        dims = dims_per_win[wi]
                        s_ep, t = int(starts[wi]), int(ts[wi])
                        embs = ep_emb[s_ep]
                        eps_c = S1.EPS_FRAC * float(embs.norm(dim=-1).mean())
                        u = np.random.default_rng(2000 + ms_seed * 100 + wi).normal(size=(ctx_c.shape[1],))
                        u /= np.linalg.norm(u) + 1e-12
                        base_w = torch.from_numpy((zc_n[wi] + eps_c * u).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
                        aw = S1.pad_batch([a_seqs[wi]], S1.PAD_ACT).to(DEVICE)
                        in_clean = float(np.linalg.norm(o_seqs[wi]))
                        zclean_norm = float(np.linalg.norm(zc_n[wi]))
                        for mode in acc:
                            om = apply_mask([o_seqs[wi]], dims, mode, 3000 + ms_seed * 100 + wi)[0]
                            in_masked = float(np.linalg.norm(om))
                            zm = S1.forward_windows(model, [om], [a_seqs[wi]], DEVICE)[:, -1, :]
                            abs_disp = float(np.linalg.norm(zm.cpu().numpy()[0] - zc_n[wi]))
                            disp_val = abs_disp / (zclean_norm + EPS)
                            acc[mode][wi] = disp_val
                            win_rows.append({"model": disp, "training_seed": 0,
                                             "split": "oodc_F1", "checkpoint_path": str(ck),
                                             "environment": env_name,
                                             "window_id": wi, "episode_id": s_ep,
                                             "mask_seed": ms_seed,
                                             "n_zeroed_dims_selected": int(len(dims)),
                                             "input_l2_clean": f"{in_clean:.6g}",
                                             "input_l2_masked": f"{in_masked:.6g}",
                                             "clean_latent_norm": f"{zclean_norm:.6g}",
                                             "latent_abs_displacement": f"{abs_disp:.6g}",
                                             "protocol_version": "maskdose_v2",
                                             "planner_mode": "none", "horizon": 0,
                                             "perturbation_type": f"zeroing_{mode}",
                                             "perturbation_magnitude": f"0.25*{dims.size} elems" if mode != "win25" else f"0.25*{dims.size} elems/frame x27",
                                             "metric_name": "rel_l2", "metric_value": f"{disp_val:.6g}",
                                             "normalization": "||z_clean||+1e-8", "epsilon": EPS,
                                             "random_seed": 3000 + ms_seed * 100 + wi})
                for mode in acc:
                    v = acc[mode]
                    summary.append({"model": disp, "environment": env_name,
                                    "mask_seed_span": MASK_SEEDS, "condition": mode,
                                    "rel_l2_mean": f"{v.mean():.4g}",
                                    "rel_l2_median": f"{np.median(v):.4g}",
                                    "rel_l2_std": f"{v.std():.4g}",
                                    "rel_l2_q10": f"{np.quantile(v, 0.1):.4g}",
                                    "rel_l2_q90": f"{np.quantile(v, 0.9):.4g}",
                                    "dose_elems_last": int(dims_per_win[0].size),
                                    "dose_elems_total_win25": int(dims_per_win[0].size * CTX)})
            print(f"[done] {disp} {env_name}", flush=True)
    with open(OUT / "maskdose_per_window.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
        w.writeheader(); w.writerows(win_rows)
    with open(OUT / "maskdose_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"win rows {len(win_rows)}; summary {len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

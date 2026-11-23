"""P0-2: latent rollout error decomposition for trace/gru across train seeds.

E_lat(h) = mean_i ||zhat_ih - z*_ih||_2 / (||z*_ih||_2 + 1e-8), teacher-forced
recorded future actions, S1 windows (SEED_WIN=0, 128/env). Exports per-window
numerator, denominator, zhat/z* norms per h in {1,5,10,20} plus summary stats
(mean/median/std/quantiles), near-zero-denominator fractions and per-window
dominance. Schema follows the audit 5.1 field list.
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
HS = (1, 5, 10, 20)
MANIFEST = {r.get("model") + f"|s{r.get('train_seed')}": r
            for r in csv.DictReader(open(OUT / "checkpoint_manifest.csv"))}


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    env_data = {}
    for env_name, cfg in S1.ENVS.items():
        obs, act, episodes = ms.load_env_data(cfg["npz"], S1.CTX + S1.HMAX)
        starts, ends, ts = ms.sample_windows(episodes, 128, np.random.default_rng(0))
        env_data[env_name] = (obs, act, episodes, starts, ends, ts)
    rows = []
    summary = []
    for model, hint, family in [("trace", "stjewm_trace_only", "stjewm"),
                                ("gru", "gru_baseline", "baseline")]:
        for train_seed in (0, 1, 2):
            base = ("/data/lx/tmp/results/5m_5mpar" if family == "stjewm" else "/data/lx/tmp/results/5m") \
                if train_seed == 0 else f"/data/lx/tmp/results/5m_seed{train_seed}"
            ck = Path(base) / "cross_benchmark_F1" / hint / "seed_0" / "final.pt"
            ck_sha = MANIFEST.get(f"{hint}|s{train_seed}", {}).get("sha256", "")
            model_t = S1.load_model(hint, str(ck), DEVICE)
            for env_name, (obs, act, episodes, starts, ends, ts) in env_data.items():
                n = len(ts)
                ep_map = dict(episodes)
                spans = [(s, ep_map[s]) for s in sorted(set(starts.tolist()))]
                ep_emb = ms.forward_episodes(model_t, obs, act, spans, DEVICE, batch=32)
                D = ep_emb[starts[0]].shape[-1]
                zhat_acc = np.zeros((n, S1.HMAX, D), dtype=np.float32)
                for i in range(0, n, 64):
                    sl = slice(i, min(i + 64, n))
                    o_seqs = [obs[t - S1.CTX + 1: t + 1] for t in ts[sl]]
                    a_seqs = [act[t - S1.CTX + 1: t + 1] for t in ts[sl]]
                    fut = np.stack([act[t + 1: t + S1.HMAX + 1] for t in ts[sl]])
                    fut = np.pad(fut, ((0, 0), (0, 0), (0, S1.PAD_ACT - fut.shape[-1])))
                    with torch.inference_mode():
                        ctx = S1.forward_windows(model_t, o_seqs, a_seqs, DEVICE)
                        pred = S1.rollout(model_t, ctx, S1.pad_batch(a_seqs, S1.PAD_ACT).to(DEVICE),
                                          torch.from_numpy(fut).to(DEVICE)).cpu().numpy()
                    zhat_acc[sl] = pred
                for h in HS:
                    for wi in range(n):
                        s_ep, t = int(starts[wi]), int(ts[wi])
                        zt = ep_emb[s_ep][t + h - s_ep].numpy()
                        zh = zhat_acc[wi, h - 1]
                        num = float(np.linalg.norm(zh - zt))
                        den = float(np.linalg.norm(zt))
                        rows.append({
                            "model": model, "training_seed": train_seed,
                            "split": "cross_benchmark_F1", "checkpoint_path": str(ck),
                            "checkpoint_sha256": ck_sha, "environment": env_name,
                            "window_id": wi, "episode_id": s_ep,
                            "initial_state_id": f"ep{s_ep}", "goal_id": "n/a",
                            "protocol_version": "S1_teacher_forced_v1",
                            "planner_mode": "none", "horizon": h,
                            "perturbation_type": "none", "perturbation_magnitude": "",
                            "metric_name": "E_lat_component",
                            "metric_value": f"{num / (den + EPS):.6g}",
                            "numerator_l2": f"{num:.6g}",
                            "ref_latent_norm": f"{den:.6g}",
                            "pred_latent_norm": f"{float(np.linalg.norm(zh)):.6g}",
                            "normalization": "||z*||_2 + eps",
                            "epsilon": EPS, "random_seed": 0})
                # summary per (env, h)
                key = (model, train_seed, env_name)
                zt_all = None
                for h in HS:
                    nums, dens, zh_n = [], [], []
                    for wi in range(n):
                        s_ep, t = int(starts[wi]), int(ts[wi])
                        zt = ep_emb[s_ep][t + h - int(s_ep)].numpy()
                        zh = zhat_acc[wi, h - 1]
                        nums.append(np.linalg.norm(zh - zt))
                        dens.append(np.linalg.norm(zt))
                        zh_n.append(np.linalg.norm(zh))
                    nums, dens = np.array(nums), np.array(dens)
                    ratios = nums / (dens + EPS)
                    med_den = np.median(dens)
                    summary.append({
                        "model": model, "training_seed": train_seed,
                        "environment": env_name, "horizon": h,
                        "num_mean": f"{nums.mean():.4g}", "num_median": f"{np.median(nums):.4g}",
                        "num_std": f"{nums.std():.4g}", "num_q90": f"{np.quantile(nums, 0.9):.4g}",
                        "den_mean": f"{dens.mean():.4g}", "den_median": f"{med_den:.4g}",
                        "den_std": f"{dens.std():.4g}",
                        "ratio_mean": f"{ratios.mean():.4g}",
                        "ratio_median": f"{np.median(ratios):.4g}",
                        "ratio_std": f"{ratios.std():.4g}",
                        "ratio_q90": f"{np.quantile(ratios, 0.9):.4g}",
                        "frac_den_lt_1e-3": f"{float((dens < 1e-3).mean()):.4g}",
                        "frac_den_lt_1pct_median": f"{float((dens < 0.01 * med_den).mean()):.4g}",
                        "top1_window_share_of_num": f"{float(np.sort(nums)[-1] / nums.sum()):.4g}"})
                print(f"[done] {model} seed{train_seed} {env_name}", flush=True)
    with open(OUT / "latent_error_decomposition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(OUT / "latent_error_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"rows {len(rows)}; summary {len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

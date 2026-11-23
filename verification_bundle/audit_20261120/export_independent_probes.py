"""R2.3: independent linear and nonlinear probes on FROZEN latents.

Five checkpoints (original, lam0 continued-training control, lam 0.1/1/10),
identical windows (128/env, rng(0)), identical episode-level 75/25 split
(rng(123)), targets = padded state s_t standardized per-dim; R2 = uniform
average over NON-CONSTANT columns. Probes are fit from scratch per checkpoint:
linear = Ridge(alpha=1) on standardized latents; nonlinear = 2-layer MLP
(hidden 256, early stopping on a 15% slice of train episodes), 3 probe seeds.
Per-sample predictions saved to probe_predictions.npz. Leakage check: episode
ids of train and test sets must be disjoint.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/lx/snn")
sys.path.insert(0, "/home/lx/snn/code/scripts")

import mech0917_multistep as ms
import stab0917_rollout as S1
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

OUT = Path("/home/lx/audit_20261120")
DEV = "cuda:0"
CKPTS = {
    "orig": "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt",
    "control": "/data/lx/tmp/results/accessibility_20260917/oodc_F1/stjewm_trace_only_lam00/seed_0/final.pt",
    "lam0.1": "/data/lx/tmp/results/accessibility_20260917/oodc_F1/stjewm_trace_only_lam01/seed_0/final.pt",
    "lam1": "/data/lx/tmp/results/accessibility_20260917/oodc_F1/stjewm_trace_only_lam10/seed_0/final.pt",
    "lam10": "/data/lx/tmp/results/accessibility_20260917/oodc_F1/stjewm_trace_only_lam100/seed_0/final.pt",
}
PAD = 128


def mlp_fit_predict(xtr, ytr, xte, seed):
    torch.manual_seed(seed)
    xtr_t = torch.from_numpy(xtr.astype(np.float32)).to(DEV)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(DEV)
    xte_t = torch.from_numpy(xte.astype(np.float32)).to(DEV)
    net = nn.Sequential(nn.Linear(xtr.shape[1], 256), nn.GELU(), nn.Linear(256, ytr.shape[1])).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n = len(xtr_t); perm = rng.permutation(n); cut = int(0.85 * n)
    tr_i, va_i = perm[:cut], perm[cut:]
    best, best_state, patience = 1e9, None, 0
    for ep in range(300):
        net.train()
        for i in range(0, cut, 64):
            idx = torch.from_numpy(tr_i[i:i + 64]).to(DEV)
            opt.zero_grad()
            loss = ((net(xtr_t[idx]) - ytr_t[idx]) ** 2).mean()
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(((net(xtr_t[torch.from_numpy(va_i).to(DEV)]) - ytr_t[torch.from_numpy(va_i).to(DEV)]) ** 2).mean())
        if vl < best - 1e-6:
            best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 20:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return net(xte_t).cpu().numpy()


def main() -> int:
    data = {}
    for env_name, cfg in S1.ENVS.items():
        obs, act, episodes = ms.load_env_data(cfg["npz"], S1.CTX + 1)
        starts, ends, ts = ms.sample_windows(episodes, 128, np.random.default_rng(0))
        data[env_name] = (obs, act, episodes, starts, ts)
    rows, preds = [], {}
    for tag, ck in CKPTS.items():
        model = S1.load_model("stjewm_trace_only", ck, DEV)
        Z, Y, EP, EV = [], [], [], []
        for env_name, (obs, act, episodes, starts, ts) in data.items():
            with torch.inference_mode():
                o_seqs = [obs[t - S1.CTX + 1: t + 1] for t in ts]
                a_seqs = [act[t - S1.CTX + 1: t + 1] for t in ts]
                ctx = S1.forward_windows(model, o_seqs, a_seqs, DEV)[:, -1, :].cpu().numpy()
            s_t = np.zeros((len(ts), PAD), dtype=np.float32)
            for j, t in enumerate(ts):
                s_t[j, :obs.shape[1]] = obs[t]
            Z.append(ctx); Y.append(s_t)
            EP.extend(int(s) for s in starts); EV.extend([env_name] * len(ts))
        Z, Y = np.concatenate(Z), np.concatenate(Y)
        eps = np.array(EP)
        uniq = np.unique(eps)
        rng = np.random.default_rng(123)
        tr_eps = set(uniq[rng.permutation(len(uniq))[: int(0.75 * len(uniq))]].tolist())
        tr = np.array([e in tr_eps for e in eps]); te = ~tr
        assert not (set(eps[tr]) & set(eps[te])), "episode leakage"
        keep = Y.std(axis=0) > 1e-8  # non-constant columns
        sc = StandardScaler().fit(Z[tr])
        rg = Ridge(alpha=1.0).fit(sc.transform(Z[tr]), Y[tr])
        pl = rg.predict(sc.transform(Z[te]))
        r2_lin = float(r2_score(Y[te][:, keep], pl[:, keep], multioutput="uniform_average"))
        rows.append({"checkpoint": tag, "probe": "linear_ridge", "probe_seed": 0,
                     "heldout_r2": f"{r2_lin:.4f}", "n_train_windows": int(tr.sum()),
                     "n_test_windows": int(te.sum()), "n_train_episodes": len(tr_eps),
                     "n_test_episodes": int(len(uniq) - len(tr_eps)), "n_targets": int(keep.sum()),
                     "episode_leakage": 0})
        preds[f"{tag}__linear__y_true"] = Y[te][:, keep]
        preds[f"{tag}__linear__y_pred"] = pl[:, keep]
        preds[f"{tag}__test_episode"] = eps[te]
        preds[f"{tag}__test_env"] = np.array(EV)[te]
        mlp_scores = []
        for seed in (0, 1, 2):
            pm = mlp_fit_predict(sc.transform(Z[tr]), Y[tr], sc.transform(Z[te]), seed)
            r2 = float(r2_score(Y[te][:, keep], pm[:, keep], multioutput="uniform_average"))
            mlp_scores.append(r2)
            rows.append({"checkpoint": tag, "probe": "mlp_2layer", "probe_seed": seed,
                         "heldout_r2": f"{r2:.4f}", "n_train_windows": int(tr.sum()),
                         "n_test_windows": int(te.sum()), "n_train_episodes": len(tr_eps),
                         "n_test_episodes": int(len(uniq) - len(tr_eps)), "n_targets": int(keep.sum()),
                         "episode_leakage": 0})
            if seed == 0:
                preds[f"{tag}__mlp__y_true"] = Y[te][:, keep]
                preds[f"{tag}__mlp__y_pred"] = pm[:, keep]
        print(f"[probe] {tag}: linear={r2_lin:.4f} mlp={np.mean(mlp_scores):.4f}±{np.std(mlp_scores):.4f}", flush=True)
    with open(OUT / "probe_control_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    np.savez_compressed(OUT / "probe_predictions.npz", **preds)
    print("rows:", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

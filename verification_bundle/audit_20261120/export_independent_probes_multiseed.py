"""Independent frozen-latent probes (linear ridge + 3-seed MLP) for lam arms at training seeds 0/1/2."""
import csv, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/lx/snn"); sys.path.insert(0, "/home/lx/snn/code/scripts")
import mech0917_multistep as ms, stab0917_rollout as S1
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
sys.path.insert(0, "/home/lx/audit_20261120")
from export_independent_probes import mlp_fit_predict, PAD, DEV
OUT = Path("/home/lx/audit_20261120")
CK = {}
for s in (0, 1, 2):
    if s == 0:
        CK["s0_orig"] = "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt"
        R = "/data/lx/tmp/results/accessibility_20260917/oodc_F1"
        CK.update({"s0_control": f"{R}/stjewm_trace_only_lam00/seed_0/final.pt", "s0_lam0.1": f"{R}/stjewm_trace_only_lam01/seed_0/final.pt",
                   "s0_lam1": f"{R}/stjewm_trace_only_lam10/seed_0/final.pt", "s0_lam10": f"{R}/stjewm_trace_only_lam100/seed_0/final.pt"})
    else:
        CK[f"s{s}_orig"] = f"/home/lx/snn/results/5m_seed{s}/oodc_F1/stjewm_trace_only/seed_0/final.pt"
        R = f"/data/lx/tmp/results/accessibility_20261123/seed{s}/oodc_F1"
        CK.update({f"s{s}_control": f"{R}/stjewm_trace_only_lam00/seed_0/final.pt", f"s{s}_lam0.1": f"{R}/stjewm_trace_only_lam01/seed_0/final.pt",
                   f"s{s}_lam1": f"{R}/stjewm_trace_only_lam10/seed_0/final.pt", f"s{s}_lam10": f"{R}/stjewm_trace_only_lam100/seed_0/final.pt"})
data = {}
for env_name, cfg in S1.ENVS.items():
    obs, act, episodes = ms.load_env_data(cfg["npz"], S1.CTX + 1)
    starts, ends, ts = ms.sample_windows(episodes, 128, np.random.default_rng(0))
    data[env_name] = (obs, act, episodes, starts, ts)
rows = []
for tag, ck in CK.items():
    if not Path(ck).exists(): print("missing", tag); continue
    model = S1.load_model("stjewm_trace_only", ck, DEV)
    Z, Y, EP = [], [], []
    for env_name, (obs, act, episodes, starts, ts) in data.items():
        with torch.inference_mode():
            o_seqs = [obs[t - S1.CTX + 1: t + 1] for t in ts]; a_seqs = [act[t - S1.CTX + 1: t + 1] for t in ts]
            Z.append(S1.forward_windows(model, o_seqs, a_seqs, DEV)[:, -1, :].cpu().numpy())
        s_t = np.zeros((len(ts), PAD), dtype=np.float32)
        for j, t in enumerate(ts): s_t[j, :obs.shape[1]] = obs[t]
        Y.append(s_t); EP.extend(int(s) for s in starts)
    Z, Y, eps = np.concatenate(Z), np.concatenate(Y), np.array(EP)
    uniq = np.unique(eps); rng = np.random.default_rng(123)
    tr_eps = set(uniq[rng.permutation(len(uniq))[: int(0.75 * len(uniq))]].tolist())
    tr = np.array([e in tr_eps for e in eps]); te = ~tr
    keep = Y.std(axis=0) > 1e-8
    sc = StandardScaler().fit(Z[tr])
    pl = Ridge(alpha=1.0).fit(sc.transform(Z[tr]), Y[tr]).predict(sc.transform(Z[te]))
    r2l = float(r2_score(Y[te][:, keep], pl[:, keep], multioutput="uniform_average"))
    m = [float(r2_score(Y[te][:, keep], mlp_fit_predict(sc.transform(Z[tr]), Y[tr], sc.transform(Z[te]), sd)[:, keep], multioutput="uniform_average")) for sd in (0, 1, 2)]
    rows.append([tag, f"{r2l:.4f}", f"{np.mean(m):.4f}", f"{np.std(m):.4f}", int(te.sum())])
    print(f"[probe] {tag}: linear={r2l:.4f} mlp={np.mean(m):.4f}±{np.std(m):.4f}", flush=True)
with open(OUT / "probe_control_comparison_multiseed.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["checkpoint", "linear_ridge_r2", "mlp_r2_mean", "mlp_r2_std", "n_test_windows"]); w.writerows(rows)

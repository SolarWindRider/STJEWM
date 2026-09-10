#!/usr/bin/env python3
"""Collect 200-step random-policy trajectories for paper figures (Fig.2d/4b)
plus SIGReg-sweep div/resp for Fig.5a.

Outputs JSON files under /data/lx/tmp/results/fig_data/.
"""
import sys, json, numpy as np, torch

sys.path.insert(0, "/home/lx/snn")
from code.eval.closed_loop import make_env
from code.train.train import build_model

DEV = "cuda:0"
OUT = "/data/lx/tmp/results/fig_data"

TRAJ_CKPTS = [
    ("mlp",  "MLP",           "/data/lx/tmp/results/5m/oodc_F1/mlp_baseline/seed_0/final.pt"),
    ("gru",  "GRU",           "/data/lx/tmp/results/5m/oodc_F1/gru_baseline/seed_0/final.pt"),
    ("lewm", "LeWM",          "/data/lx/tmp/results/5m/oodc_F1/lewm_baseline_v2/seed_0/final.pt"),
    ("stjewm_trace", "ST-JEWM", "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/seed_0/final.pt"),
]
SIGREG_CKPTS = [
    ("0.09",   "/data/lx/tmp/results/5m_sigreg_sweep/cross_benchmark_F1/stjewm_trace_only_sig0.09/seed_0/final.pt"),
    ("0.01",   "/data/lx/tmp/results/5m_sigreg_sweep/cross_benchmark_F1/stjewm_trace_only_sig0.01/seed_0/final.pt"),
    ("0.001",  "/data/lx/tmp/results/5m_sigreg_sweep/cross_benchmark_F1/stjewm_trace_only_sig0.001/seed_0/final.pt"),
    ("0.0",    "/data/lx/tmp/results/5m_sigreg_sweep/cross_benchmark_F1/stjewm_trace_only_sig0.0/seed_0/final.pt"),
]


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    isz = a.get("image_size", 0)
    sd = ck.get("model", {})
    for k, v in sd.items():
        if "position_embeddings" in k and hasattr(v, "shape") and v.ndim == 3:
            n_patches = int(v.shape[1]) - 1
            isz = int(round((n_patches ** 0.5) * 14)) if n_patches > 0 else 84
            break
    model = build_model(
        a.get("model", "stjewm"), obs_dim=a.get("pad_obs_to", 128),
        action_dim=a.get("action_dim", 56), n_layers=a.get("n_layers", 4),
        readout_mode=a.get("readout_mode", "hidden_leak"),
        embed_dim=a.get("embed_dim"), hidden_dim=a.get("hidden_dim"),
        mlp_hidden=a.get("mlp_hidden"), mlp_layers=a.get("mlp_layers"),
        image_size=isz,
    )
    model.load_state_dict(ck["model"])  # strict
    return model.to(device).eval(), a


def rollout(model, env, device, n_steps=200, seed=0):
    """Single-frame encode per step; returns ||dz|| and ||d_state|| series."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    a0 = np.zeros(56, dtype=np.float32)
    prev_z, prev_s = None, None
    dz, ds = [], []
    with torch.no_grad():
        for t in range(n_steps):
            st = np.asarray(env.get_state(), dtype=np.float32)
            a = rng.uniform(env.spec.action_low, env.spec.action_high).astype(np.float32)
            ap = np.zeros(56, dtype=np.float32); ap[:len(a)] = a
            v = np.zeros(128, dtype=np.float32); v[:len(st)] = st
            x = torch.from_numpy(v).float().reshape(1, 1, -1).to(device)
            at = torch.from_numpy(ap).float().reshape(1, 1, -1).to(device)
            out = model(x, at)
            emb = out["emb"] if isinstance(out, dict) else out
            z = emb[0, -1].cpu().numpy()
            o, r, done, _ = env.step(a)
            st2 = np.asarray(env.get_state(), dtype=np.float32)
            if prev_z is not None:
                dz.append(float(np.linalg.norm(z - prev_z)))
                ds.append(float(np.linalg.norm(st2 - prev_s)))
            prev_z, prev_s = z, st2
            if done:
                env.reset(seed=seed + t + 1)
    return dz, ds


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    # (a) trajectories for Fig.2d / Fig.4b
    for key, label, ck in TRAJ_CKPTS:
        ck = ck.replace("/seed_0/seed_0/", "/seed_0/")
        if not ckpt_ok(ck):
            print("skip (no ckpt):", key, ck); continue
        model, _ = load_model(ck, DEV)
        env = make_env("cartpole")
        dz, ds = rollout(model, env, DEV)
        json.dump({"key": key, "label": label, "dz": dz, "ds": ds},
                  open(f"{OUT}/traj_{key}.json", "w"))
        print(f"traj {key}: steps={len(dz)} mean|dz|={np.mean(dz):.4f} mean|ds|={np.mean(ds):.4f}", flush=True)
    # (b) SIGReg sweep div/resp (Fig.5a)
    for tag, ck in SIGREG_CKPTS:
        if not ckpt_ok(ck):
            print("skip (no ckpt):", tag); continue
        model, _ = load_model(ck, DEV)
        env = make_env("cartpole")
        dz, ds = rollout(model, env, DEV)
        L = np.array(dz); S = np.array(ds)
        json.dump({"lambda": tag, "resp": float(L.mean() / max(S.mean(), 1e-9)),
                   "mean_dz": float(L.mean())},
                  open(f"{OUT}/sigreg_{tag}.json", "w"))
        print(f"sigreg {tag}: resp={L.mean()/max(S.mean(),1e-9):.3f} mean_dz={L.mean():.4f}", flush=True)


def ckpt_ok(p):
    import os
    return os.path.exists(p)


if __name__ == "__main__":
    main()

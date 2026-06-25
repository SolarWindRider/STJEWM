"""Jaco arm closed-loop sim eval."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import mujoco

sys.path.insert(0, "/home/lx/LeWM")
sys.path.insert(0, "/home/lx/snn/code")
sys.path.insert(0, "/home/lx/snn/code/scripts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

XML = '/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/third_party/kinova/jaco_arm.xml'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--ep-len", type=int, default=30)
    p.add_argument("--out", required=True)
    return p.parse_args()

def load_model(ckpt_path, state_dim, action_dim):
    from lewm_stjewm_v4 import STJEWMv4
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    saved_args = ck.get("args", {})
    n_layers = saved_args.get("n_layers", 4)
    model = STJEWMv4(d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
                     state_dim=state_dim, cell_n_layers=n_layers, n_d=3, trace_beta=0.9, freeze_encoder=True)
    model.load_state_dict(sd, strict=False)
    return model.to(DEVICE).eval()

@torch.no_grad()
def closed_loop_sim(model, m, d, n_episodes, ep_len):
    joint_ranges = []
    for i in range(m.njnt):
        r = m.jnt_range[i] if m.jnt_range.shape[0] > i else None
        joint_ranges.append((i, (r[0], r[1]) if r is not None and r[0] < r[1] else (0, 0)))
    ctrl_low = np.array([-0.05] * m.nq)
    ctrl_high = np.array([0.05] * m.nq)
    rng = np.random.default_rng(3072)
    errors, cosines, nan_eps = [], [], 0
    for ep in range(n_episodes):
        mujoco.mj_resetData(m, d)
        for i, (lo, hi) in joint_ranges:
            qs = m.jnt_qposadr[i]
            if lo != hi:
                d.qpos[qs] = rng.uniform(lo, hi)
            else:
                d.qpos[qs] = 0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        if not np.isfinite(d.qpos).all():
            nan_eps += 1
            continue
        ep_err, ep_cos = [], []
        for t in range(ep_len):
            action = rng.uniform(ctrl_low, ctrl_high).astype(np.float32)
            state_np = d.qpos[:].copy().astype(np.float32)
            state_t = torch.from_numpy(state_np).to(DEVICE).unsqueeze(0).unsqueeze(0)
            action_t = torch.from_numpy(action).to(DEVICE).unsqueeze(0).unsqueeze(0)
            out = model(state_t, action_t)
            pred_emb = out["emb"][:, 0]
            # Apply predicted action (as delta qpos since no actuators)
            d.qpos[:] += action
            for i, (lo, hi) in joint_ranges:
                if lo != hi:
                    qs = m.jnt_qposadr[i]
                    d.qpos[qs] = np.clip(d.qpos[qs], lo, hi)
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            if not np.isfinite(d.qpos).all():
                nan_eps += 1
                break
            actual_nxt = d.qpos[:].copy().astype(np.float32)
            nxt_t = torch.from_numpy(actual_nxt).to(DEVICE).unsqueeze(0).unsqueeze(0)
            zero_act = torch.zeros_like(action_t)
            out_nxt = model(nxt_t, zero_act)
            tgt_emb = out_nxt["emb"][:, 0]
            ep_err.append(F.mse_loss(pred_emb, tgt_emb).item())
            ep_cos.append(F.cosine_similarity(pred_emb, tgt_emb, dim=-1).item())
        if ep_err:
            errors.append(np.mean(ep_err))
            cosines.append(np.mean(ep_cos))
    return np.mean(errors), np.std(errors), np.mean(cosines), nan_eps

def main():
    args = parse_args()
    print(f"[jaco-closed-loop] ckpt={args.ckpt}", flush=True)
    m = mujoco.MjModel.from_xml_path(XML)
    state_dim = m.nq
    action_dim = m.nq
    model = load_model(args.ckpt, state_dim, action_dim)
    d = mujoco.MjData(m)
    print(f"  state={state_dim}, act={action_dim}, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    err, err_std, cos, nan_eps = closed_loop_sim(model, m, d, args.n_episodes, args.ep_len)
    print(f"  Closed-loop MSE: {err:.4f} +/- {err_std:.4f}, cos: {cos:.4f}, nan_eps: {nan_eps}", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"env": "jaco_arm", "ckpt": args.ckpt, "n_episodes": args.n_episodes, "ep_len": args.ep_len, "n_nan_eps": nan_eps,
                   "closed_loop_mse": err, "closed_loop_std": err_std, "closed_loop_cos": cos}, f, indent=2)
    print(f"  Saved {out_path}", flush=True)

if __name__ == "__main__":
    main()

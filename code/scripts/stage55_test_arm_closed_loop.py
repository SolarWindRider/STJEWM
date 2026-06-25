"""Test arm closed-loop sim eval."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import mujoco

sys.path.insert(0, "/home/lx/LeWM")
sys.path.insert(0, "/home/lx/snn/code")
sys.path.insert(0, "/home/lx/snn/code/scripts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
XML = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/mjcf/test_assets/robot_arm.xml"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--ep-len", type=int, default=30)
    p.add_argument("--out", required=True)
    return p.parse_args()

def load_model(ckpt_path, sd, ad):
    from lewm_stjewm_v4 import STJEWMv4
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = STJEWMv4(d_hid=192, embed_dim=192, action_dim=ad, action_emb_dim=192,
                     state_dim=sd, cell_n_layers=4, n_d=3, trace_beta=0.9, freeze_encoder=True)
    model.load_state_dict(ck.get("model", ck), strict=False)
    return model.to(DEVICE).eval()

@torch.no_grad()
def cl(model, m, d, n_ep, l):
    cl, ch = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    rng = np.random.default_rng(3072)
    errs, coss, nans = [], [], 0
    for ep in range(n_ep):
        mujoco.mj_resetData(m, d)
        for i in range(m.njnt):
            r = m.jnt_range[i] if m.jnt_range.shape[0] > i else None
            if r is not None and r[0] < r[1]:
                d.qpos[m.jnt_qposadr[i]] = rng.uniform(r[0]*0.3, r[1]*0.3)
            elif r is not None and r[0] == r[1]:
                d.qpos[m.jnt_qposadr[i]] = 0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        if not np.isfinite(d.qpos).all():
            nans += 1
            continue
        ee, ec = [], []
        for t in range(l):
            a = rng.uniform(cl, ch).astype(np.float32)
            s = d.qpos[:].copy().astype(np.float32)
            st = torch.from_numpy(s).to(DEVICE).unsqueeze(0).unsqueeze(0)
            at = torch.from_numpy(a).to(DEVICE).unsqueeze(0).unsqueeze(0)
            pe = model(st, at)["emb"][:, 0]
            d.ctrl[:] = a
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all():
                nans += 1
                break
            n = d.qpos[:].copy().astype(np.float32)
            nt = torch.from_numpy(n).to(DEVICE).unsqueeze(0).unsqueeze(0)
            te = model(nt, torch.zeros_like(at))["emb"][:, 0]
            ee.append(F.mse_loss(pe, te).item())
            ec.append(F.cosine_similarity(pe, te, dim=-1).item())
        if ee:
            errs.append(np.mean(ee))
            coss.append(np.mean(ec))
    return np.mean(errs), np.mean(coss), nans

def main():
    args = parse_args()
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    model = load_model(args.ckpt, m.nq, m.nu)
    print(f"[test-arm-cl] state={m.nq} act={m.nu} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    err, cos, nans = cl(model, m, d, args.n_episodes, args.ep_len)
    print(f"  MSE: {err:.4f}, cos: {cos:.4f}, nans: {nans}", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"env": "test_arm", "ckpt": args.ckpt, "n_episodes": args.n_episodes, "ep_len": args.ep_len, "n_nan_eps": nans,
                   "closed_loop_mse": err, "closed_loop_cos": cos}, f, indent=2)
    print(f"  Saved {out_path}", flush=True)

if __name__ == "__main__":
    main()

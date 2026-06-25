"""v5 3D world model REAL mujoco closed-loop simulation.

For each 3D env:
1. Initialize env with random qpos (in joint range)
2. Take random action
3. Compare model prediction to actual next_qpos
4. Report "v5 prediction error" per env

This is the "real env" closed-loop test, not just next-step on offline data.
"""
import argparse
import json
import os
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
XML_BASE = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite/"

ENVS = {
    'manipulator':   ('manipulator.xml',   5,  'bring_ball'),
    'quadruped':     ('quadruped.xml',    12,  'walk'),
    'humanoid':      ('humanoid.xml',     21,  'walk'),
    'dog':           ('dog.xml',          38,  'walk'),
    'humanoid_CMU':  ('humanoid_CMU.xml', 56,  'walk'),
    'stacker':       ('stacker.xml',       5,  'stack_4'),
    'finger':        ('finger.xml',        2,  'turn_easy'),
    'ball_in_cup':   ('ball_in_cup.xml',   2,  'catch'),
    'walker':        ('walker.xml',        6,  'walk'),
    'cheetah':       ('cheetah.xml',       6,  'run'),
    'hopper':        ('hopper.xml',        4,  'hop'),
    'fish':          ('fish.xml',          5,  'upright'),
}



def get_safe_qpos_init(m, env_name, rng):
    nq = m.nq
    qpos = np.zeros(nq)
    for i in range(m.njnt):
        jt = m.jnt_type[i]
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            qpos_start = m.jnt_qposadr[i]
            if env_name == 'manipulator':
                qpos[qpos_start:qpos_start+7] = [0, 0, 0, 1, 0, 0, 0]
            else:
                qpos[qpos_start:qpos_start+3] = [0, 0, 1.0]
                qpos[qpos_start+3:qpos_start+7] = [1, 0, 0, 0]
        else:
            qpos_start = m.jnt_qposadr[i]
            r = m.jnt_range[i] if m.jnt_range is not None and m.jnt_range.shape[0] > i else None
            if r is not None and r[0] < r[1]:
                center = (r[0] + r[1]) / 2
                half = (r[1] - r[0]) / 2 * 0.5
                qpos[qpos_start] = rng.uniform(center - half, center + half)
    return qpos


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--ep-len", type=int, default=50)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_model(ckpt_path, env_name):
    from lewm_stjewm_v4 import STJEWMv4
    xml_path = XML_BASE + ENVS[env_name][0]
    m_test = mujoco.MjModel.from_xml_path(xml_path)
    obs_dim = m_test.nq
    action_dim = ENVS[env_name][1]
    model = STJEWMv4(
        d_hid=192, embed_dim=192,
        action_dim=action_dim, action_emb_dim=192,
        state_dim=obs_dim, cell_n_layers=4, n_d=3,
        trace_beta=0.9, freeze_encoder=True,
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    model.load_state_dict(sd, strict=False)
    model = model.to(DEVICE).eval()
    return model, m_test, obs_dim, action_dim


@torch.no_grad()
def closed_loop_sim(model, env_name, n_episodes, ep_len):
    """Real mujoco env, compare model prediction vs actual next_qpos."""
    xml_file, nu, task = ENVS[env_name]
    m = mujoco.MjModel.from_xml_path(XML_BASE + xml_file)
    d = mujoco.MjData(m)
    ctrl_low = m.actuator_ctrlrange[:, 0]
    ctrl_high = m.actuator_ctrlrange[:, 1]

    rng = np.random.default_rng(3072)

    errors = []
    cosines = []
    nan_eps = 0
    for ep in range(n_episodes):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = get_safe_qpos_init(m, env_name, rng)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        # Stabilize
        for _ in range(5):
            d.ctrl[:] = 0
            mujoco.mj_step(m, d)

        if not np.isfinite(d.qpos).all():
            nan_eps += 1
            continue

        ep_err = []
        ep_cos = []
        for t in range(ep_len):
            # Random action
            action = rng.uniform(ctrl_low, ctrl_high).astype(np.float32)
            state_np = d.qpos[:].copy().astype(np.float32)

            # Model prediction
            state_t = torch.from_numpy(state_np).to(DEVICE).unsqueeze(0).unsqueeze(0)
            action_t = torch.from_numpy(action).to(DEVICE).unsqueeze(0).unsqueeze(0)
            out = model(state_t, action_t)
            pred_emb = out["emb"][:, 0]  # (1, 192)

            # Real mujoco step
            d.ctrl[:] = action
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all():
                nan_eps += 1
                break
            actual_nxt = d.qpos[:].copy().astype(np.float32)

            # Encode actual next state
            nxt_t = torch.from_numpy(actual_nxt).to(DEVICE).unsqueeze(0).unsqueeze(0)
            zero_act = torch.zeros_like(action_t)
            out_nxt = model(nxt_t, zero_act)
            tgt_emb = out_nxt["emb"][:, 0]

            err = F.mse_loss(pred_emb, tgt_emb).item()
            cos = F.cosine_similarity(pred_emb, tgt_emb, dim=-1).item()
            ep_err.append(err)
            ep_cos.append(cos)

        if ep_err:
            errors.append(np.mean(ep_err))
            cosines.append(np.mean(ep_cos))

    return np.mean(errors), np.std(errors), np.mean(cosines), nan_eps


def main():
    args = parse_args()
    print(f"[v5-closed-loop] env={args.env}, ckpt={args.ckpt}", flush=True)

    model, m_test, obs_dim, action_dim = load_model(args.ckpt, args.env)
    print(f"  state_dim={obs_dim}, action_dim={action_dim}, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    err, err_std, cos, nan_eps = closed_loop_sim(model, args.env, args.n_episodes, args.ep_len)
    print(f"  Closed-loop MSE: {err:.4f} +/- {err_std:.4f}, cos: {cos:.4f}, nan_eps: {nan_eps}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "env": args.env,
        "ckpt": args.ckpt,
        "n_episodes": args.n_episodes,
        "ep_len": args.ep_len,
        "n_nan_eps": nan_eps,
        "closed_loop_mse": err,
        "closed_loop_std": err_std,
        "closed_loop_cos": cos,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

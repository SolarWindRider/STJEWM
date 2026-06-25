"""Model-Predictive Control (MPC) for v5 reach task.

At each step, find the action that minimizes
||predict(state_with_target, action) - ideal_next_state||^2
where ideal_next_state is computed from the goal.

This is gradient-based action selection, more powerful than CEM.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, "/home/lx/LeWM")
sys.path.insert(0, "/home/lx/snn/code")
sys.path.insert(0, "/home/lx/snn/code/scripts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

XML = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite/manipulator.xml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--lr-action", type=float, default=0.1)
    p.add_argument("--n-action-iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_v5(ckpt_path, state_dim, action_dim):
    from lewm_stjewm_v4 import STJEWMv4
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    saved_args = ck.get("args", {})
    n_layers = saved_args.get("n_layers", 4)
    model = STJEWMv4(
        d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
        state_dim=state_dim, cell_n_layers=n_layers, n_d=3, trace_beta=0.9,
        freeze_encoder=True,
    )
    model.load_state_dict(sd, strict=False)
    return model.to(DEVICE).eval()


@torch.no_grad()
def encode_state(model, state, action_dim):
    s = torch.from_numpy(state.astype(np.float32)).reshape(1, 1, -1).to(DEVICE)
    a = torch.zeros(1, 1, action_dim, device=DEVICE)
    out = model(s, a)
    return out["emb"][0, 0]


def find_action_via_gradient(model, state, ideal_next_emb, action_dim, lr=0.1, n_iters=20):
    """Find action that minimizes ||predict(state, action) - ideal_next_emb||^2."""
    action = torch.zeros(action_dim, requires_grad=True, device=DEVICE)
    optim = torch.optim.Adam([action], lr=lr)
    state_t = torch.from_numpy(state.astype(np.float32)).reshape(1, 1, -1).to(DEVICE)
    for it in range(n_iters):
        optim.zero_grad()
        a_t = action.unsqueeze(0).unsqueeze(0)  # (1, 1, A)
        out = model(state_t, a_t)
        pred_emb = out["emb"][:, 0]  # (1, D)
        loss = ((pred_emb - ideal_next_emb.unsqueeze(0)) ** 2).sum()
        loss.backward()
        optim.step()
        # Clamp to action range
        with torch.no_grad():
            action.clamp_(-1, 1)
    return action.detach().cpu().numpy()


def get_fingertip_pos(m, d):
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    return d.xpos[fid, :3].copy() if fid >= 0 else None


def sample_qpos_for_target(m, d, target_pos, rng, n_samples=200):
    arm_joints = []
    for i in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if n in ['arm_shoulder', 'arm_elbow', 'arm_wrist', 'thumb', 'thumbtip', 'finger', 'fingertip']:
            arm_joints.append(i)
    best_qpos = None
    best_dist = float('inf')
    for _ in range(n_samples):
        for j in arm_joints:
            qs = m.jnt_qposadr[j]
            r = m.jnt_range[j]
            if r[0] < r[1]:
                d.qpos[qs] = rng.uniform(r[0], r[1])
        mujoco.mj_forward(m, d)
        fp = get_fingertip_pos(m, d)
        if fp is None:
            continue
        dist = float(np.linalg.norm(fp[:2] - target_pos[:2]))
        if dist < best_dist:
            best_dist = dist
            best_qpos = d.qpos[:].copy()
        if dist < 0.01:
            return best_qpos
    return best_qpos if best_dist < 0.1 else None


def main():
    args = parse_args()
    print(f"[v5-mpc-reach] ckpt={args.ckpt}", flush=True)

    m = mujoco.MjModel.from_xml_path(XML)
    state_dim = m.nq + 3
    action_dim = m.nu

    model = load_v5(args.ckpt, state_dim, action_dim)
    print(f"  state_dim={state_dim}, action_dim={action_dim}, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    rng = np.random.default_rng(args.seed)
    d = mujoco.MjData(m)
    results = []

    for ep in range(args.n_episodes):
        mujoco.mj_resetData(m, d)
        for j in range(m.njnt):
            jt = m.jnt_type[j]
            qs = m.jnt_qposadr[j]
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
            if jt == mujoco.mjtJoint.mjJNT_FREE:
                if 'ball' in jname or 'peg' in jname:
                    d.qpos[qs:qs+7] = [0, 0, 0, 1, 0, 0, 0]
                else:
                    d.qpos[qs:qs+3] = [0, 0, 0]
                    d.qpos[qs+3:qs+7] = [1, 0, 0, 0]
            else:
                r = m.jnt_range[j]
                if r[0] < r[1]:
                    d.qpos[qs] = rng.uniform(r[0] * 0.3, r[1] * 0.3)
        d.qvel[:] = 0
        target_pos = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0.05, 0.25)])
        target_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_ball")
        m.geom_pos[target_id] = target_pos
        goal_qpos = sample_qpos_for_target(m, d, target_pos, rng, n_samples=200)
        if goal_qpos is None:
            continue

        # Re-init
        for j in range(m.njnt):
            jt = m.jnt_type[j]
            qs = m.jnt_qposadr[j]
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
            if jt == mujoco.mjtJoint.mjJNT_FREE:
                if 'ball' in jname or 'peg' in jname:
                    d.qpos[qs:qs+7] = [0, 0, 0, 1, 0, 0, 0]
                else:
                    d.qpos[qs:qs+3] = [0, 0, 0]
                    d.qpos[qs+3:qs+7] = [1, 0, 0, 0]
            else:
                r = m.jnt_range[j]
                if r[0] < r[1]:
                    d.qpos[qs] = rng.uniform(r[0] * 0.3, r[1] * 0.3)
        d.qvel[:] = 0
        m.geom_pos[target_id] = target_pos
        mujoco.mj_forward(m, d)
        for _ in range(3):
            d.ctrl[:] = 0
            mujoco.mj_step(m, d)

        init_qpos = d.qpos[:].copy()
        init_state = np.concatenate([init_qpos, target_pos]).astype(np.float32)
        goal_state = np.concatenate([goal_qpos, target_pos]).astype(np.float32)

        init_dist = float(np.linalg.norm(init_qpos - goal_qpos))
        # MPC: at each step, find action that minimizes ||predict(state, action) - ideal_emb||
        # ideal_emb is from interpolating between init and goal embeddings
        init_emb = encode_state(model, init_state, action_dim)
        goal_emb = encode_state(model, goal_state, action_dim)

        for step in range(args.max_steps):
            cur_state = np.concatenate([d.qpos[:].copy(), target_pos]).astype(np.float32)
            cur_emb = encode_state(model, cur_state, action_dim)
            # Linear interpolation target
            progress = (step + 1) / args.max_steps
            ideal_next_emb = init_emb * (1 - progress) + goal_emb * progress

            # Find action via gradient
            action = find_action_via_gradient(model, cur_state, ideal_next_emb, action_dim,
                                            lr=args.lr_action, n_iters=args.n_action_iters)
            d.ctrl[:] = action
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all():
                break

        final_qpos = d.qpos[:].copy()
        final_dist = float(np.linalg.norm(final_qpos - goal_qpos))
        improved = final_dist < init_dist
        success = final_dist < 0.1
        results.append({"ep": ep, "init_dist": init_dist, "final_dist": final_dist, "improved": improved, "success": success})

    if not results:
        print("No valid episodes")
        return

    n_imp = sum(r["improved"] for r in results)
    n_succ = sum(r["success"] for r in results)
    final = {
        "env": "manipulator_mpc_reach",
        "n_episodes": len(results),
        "n_improved": n_imp,
        "n_success": n_succ,
        "improved_pct": n_imp / len(results) * 100,
        "success_pct": n_succ / len(results) * 100,
        "mean_init_qpos_dist": float(np.mean([r["init_dist"] for r in results])),
        "mean_final_qpos_dist": float(np.mean([r["final_dist"] for r in results])),
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nFINAL: improved={final['improved_pct']:.1f}%, success={final['success_pct']:.1f}%", flush=True)
    print(f"  init={final['mean_init_qpos_dist']:.3f} -> final={final['mean_final_qpos_dist']:.3f}", flush=True)
    print(f"  Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

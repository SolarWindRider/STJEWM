"""v5 Manipulator target-conditioned planning (honest bench).

State = (qpos 14D, target_ball_xyz 3D) = 17D
Test: can the model use CEM to plan qpos movements that bring the ball to the target?

We can't directly test "ball at target" because the model doesn't have ball_pos in state.
Instead, we test:
1. Generate goal qpos via IK (find qpos whose fingertip is at target)
2. Encode goal state = (goal_qpos, target) - this is the model's "goal"
3. CEM from init state (random qpos, target) to goal state
4. Execute the CEM plan in real mujoco
5. Measure: does the arm move TOWARD the goal qpos?

Honest metric: delta_qpos = ||final_qpos - goal_qpos|| vs ||init_qpos - goal_qpos||
If the model works, delta_qpos should be smaller than init_qpos - goal_qpos.
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
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--replan-every", type=int, default=3)
    p.add_argument("--cem-samples", type=int, default=64)
    p.add_argument("--cem-elites", type=int, default=8)
    p.add_argument("--cem-iters", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=50)
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


@torch.no_grad()
def cem_plan(model, init_emb, goal_emb, horizon, action_dim, cem_samples, cem_elites, cem_iters, history_size=3):
    mean = torch.zeros(horizon, action_dim, device=DEVICE)
    var = torch.ones(horizon, action_dim, device=DEVICE)

    def cost_of_actions(actions_flat):
        K = actions_flat.shape[0]
        h = init_emb.expand(K, -1, -1).contiguous()
        for t in range(horizon):
            avail = horizon - t
            if avail >= history_size:
                a_t = actions_flat[:, t:t + history_size]
            else:
                a_t_partial = actions_flat[:, t:]
                pad = torch.zeros(K, history_size - avail, action_dim, device=DEVICE)
                a_t = torch.cat([a_t_partial, pad], dim=1)
            h_in = h[:, -history_size:]
            nxt = model.predict(h_in, a_t)[:, -1]
            h = torch.cat([h[:, 1:], nxt.unsqueeze(1)], dim=1)
        z_final = h[:, -1]
        return ((z_final - goal_emb) ** 2).sum(-1)

    for it in range(cem_iters):
        candidates = mean.unsqueeze(0) + var.sqrt().unsqueeze(0) * torch.randn(cem_samples, horizon, action_dim, device=DEVICE)
        costs = cost_of_actions(candidates)
        topk = torch.topk(costs, cem_elites, largest=False).indices
        elites = candidates[topk]
        mean = elites.mean(dim=0)
        var = elites.var(dim=0).clamp_min(1e-4)
    candidates = mean.unsqueeze(0) + var.sqrt().unsqueeze(0) * torch.randn(cem_samples, horizon, action_dim, device=DEVICE)
    costs = cost_of_actions(candidates)
    return candidates[costs.argmin()]


def get_fingertip_pos(m, d):
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    if fid < 0:
        return None
    return d.xpos[fid, :3].copy()


def sample_qpos_for_target(m, d, target_pos, rng, n_samples=200):
    """Find qpos whose fingertip is at target_pos."""
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
    print(f"[v5-target-plan] ckpt={args.ckpt}", flush=True)

    m = mujoco.MjModel.from_xml_path(XML)
    state_dim = m.nq + 3  # 14 + 3
    action_dim = m.nu
    print(f"  state_dim={state_dim}, action_dim={action_dim}", flush=True)

    model = load_v5(args.ckpt, state_dim, action_dim)
    print(f"  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    rng = np.random.default_rng(args.seed)
    d = mujoco.MjData(m)
    HISTORY_SIZE = 3
    results = []

    for ep in range(args.n_episodes):
        mujoco.mj_resetData(m, d)
        # Random arm qpos
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

        # Set target
        target_pos = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0.05, 0.25)])
        target_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_ball")
        m.geom_pos[target_id] = target_pos

        # Find goal qpos (IK)
        goal_qpos = sample_qpos_for_target(m, d, target_pos, rng, n_samples=200)
        if goal_qpos is None:
            continue

        # Reset to init
        d.qpos[:] = m.qpos0  # not what we want
        # Actually re-init
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

        init_dist_qpos = float(np.linalg.norm(init_qpos - goal_qpos))

        # Encode and CEM
        init_emb = encode_state(model, init_state, action_dim)
        goal_emb = encode_state(model, goal_state, action_dim)
        history_emb = init_emb.unsqueeze(0).unsqueeze(0).expand(1, HISTORY_SIZE, -1).contiguous()

        for step in range(args.max_steps):
            if step % args.replan_every == 0:
                cur_emb = encode_state(model, np.concatenate([d.qpos[:].copy(), target_pos]).astype(np.float32), action_dim)
                history_emb = torch.cat([history_emb[:, 1:], cur_emb.unsqueeze(0).unsqueeze(0)], dim=1)
                best_actions = cem_plan(
                    model, history_emb, goal_emb,
                    args.horizon, action_dim,
                    args.cem_samples, args.cem_elites, args.cem_iters,
                    history_size=HISTORY_SIZE,
                )
            action = best_actions[step % args.horizon].cpu().numpy()
            action = np.clip(action, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            d.ctrl[:] = action
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all():
                break

        final_qpos = d.qpos[:].copy()
        final_dist_qpos = float(np.linalg.norm(final_qpos - goal_qpos))
        # Success if the model moved closer to goal
        improved = final_dist_qpos < init_dist_qpos
        success = final_dist_qpos < 0.1  # within 0.1 in qpos space

        results.append({
            "ep": ep,
            "init_qpos_dist": init_dist_qpos,
            "final_qpos_dist": final_dist_qpos,
            "improved": improved,
            "success": success,
        })
        if (ep + 1) % 5 == 0:
            n_imp = sum(r["improved"] for r in results)
            n_succ = sum(r["success"] for r in results)
            mean_init = np.mean([r["init_qpos_dist"] for r in results])
            mean_final = np.mean([r["final_qpos_dist"] for r in results])
            print(f"  ep {ep+1}: improved={n_imp}/{len(results)}={n_imp/len(results)*100:.1f}%, success(<0.1)={n_succ}/{len(results)}, init={mean_init:.3f}, final={mean_final:.3f}", flush=True)

    if not results:
        print("No valid episodes")
        return

    n_imp = sum(r["improved"] for r in results)
    n_succ = sum(r["success"] for r in results)
    final = {
        "env": "manipulator_target_plan",
        "n_episodes": len(results),
        "n_improved": n_imp,
        "n_success": n_succ,
        "improved_pct": n_imp / len(results) * 100,
        "success_pct": n_succ / len(results) * 100,
        "mean_init_qpos_dist": float(np.mean([r["init_qpos_dist"] for r in results])),
        "mean_final_qpos_dist": float(np.mean([r["final_qpos_dist"] for r in results])),
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nFINAL: improved={final['improved_pct']:.1f}%, success={final['success_pct']:.1f}%", flush=True)
    print(f"  init_qpos_dist={final['mean_init_qpos_dist']:.3f} -> final_qpos_dist={final['mean_final_qpos_dist']:.3f}", flush=True)
    print(f"  Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

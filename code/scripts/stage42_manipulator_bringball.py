"""v5 Manipulator "reach target qpos" real mujoco bench eval (v2).

The model was trained on (qpos, action) -> next_qpos dynamics.
This eval asks: can the model use CEM to plan from a RANDOM init to a TARGET qpos?

The target qpos is chosen by:
1. Sample 10 random arm qpos configurations
2. Use one as the goal (fingertip at some position P)
3. Use a DIFFERENT one as init (fingertip at a different position P')
4. Verify init fingertip is > 5cm from target
5. Ask model to plan init -> goal
6. Measure final fingertip distance to target

Bench: 20 episodes, success: final fingertip within 5cm of target.

This is the "real" 3D robotic arm bench, comparable to the Reacher eval.
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
FINGERTIP_THRESH = 0.05  # 5cm success threshold
INIT_MIN_DIST = 0.08  # Require init fingertip > 8cm from target


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


def sample_random_qpos(m, d, rng, arm_joints):
    """Sample a random valid arm qpos and return (qpos, fingertip_pos)."""
    for j in arm_joints:
        qs = m.jnt_qposadr[j]
        r = m.jnt_range[j]
        if r[0] < r[1]:
            d.qpos[qs] = rng.uniform(r[0], r[1])
    mujoco.mj_forward(m, d)
    return d.qpos[:].copy(), get_fingertip_pos(m, d)


def find_init_goal_pair(m, d, rng, n_samples=20, min_init_dist=INIT_MIN_DIST):
    """Sample two distinct qpos configurations, return (init_qpos, goal_qpos, init_fp, goal_fp).

    Requires that init_fp is at least min_init_dist away from goal_fp.
    """
    arm_joint_names = ['arm_shoulder', 'arm_elbow', 'arm_wrist', 'thumb', 'thumbtip', 'finger', 'fingertip']
    arm_joints = [i for i in range(m.njnt) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) in arm_joint_names]

    # Sample N random configurations
    configs = []
    for _ in range(n_samples):
        qpos, fp = sample_random_qpos(m, d, rng, arm_joints)
        if fp is not None:
            configs.append((qpos, fp))

    if len(configs) < 2:
        return None

    # Find pair with max init-goal distance
    best = None
    best_dist = 0
    for i, (q1, fp1) in enumerate(configs):
        for j, (q2, fp2) in enumerate(configs):
            if i == j:
                continue
            dist = float(np.linalg.norm(fp1[:2] - fp2[:2]))
            if dist > best_dist:
                best_dist = dist
                best = (q1, q2, fp1, fp2)

    if best is None or best_dist < min_init_dist:
        return None
    return best


def main():
    args = parse_args()
    print(f"[v5-reach-qpos-v2] ckpt={args.ckpt}", flush=True)

    m = mujoco.MjModel.from_xml_path(XML)
    state_dim = m.nq  # 14D qpos
    action_dim = m.nu
    print(f"  state_dim={state_dim}, action_dim={action_dim}", flush=True)

    model = load_v5(args.ckpt, state_dim, action_dim)
    print(f"  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    rng = np.random.default_rng(args.seed)
    d = mujoco.MjData(m)

    HISTORY_SIZE = 3
    results = []

    for ep in range(args.n_episodes):
        # Find init/goal pair with sufficient distance
        pair = find_init_goal_pair(m, d, rng, n_samples=30, min_init_dist=INIT_MIN_DIST)
        if pair is None:
            continue
        init_qpos, goal_qpos, init_fp, goal_fp = pair
        init_dist = float(np.linalg.norm(init_fp[:2] - goal_fp[:2]))

        # Reset env to init
        d.qpos[:] = init_qpos
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

        # Encode initial and goal
        init_emb = encode_state(model, init_qpos, action_dim)
        goal_emb = encode_state(model, goal_qpos, action_dim)

        # Initialize history
        history_emb = init_emb.unsqueeze(0).unsqueeze(0).expand(1, HISTORY_SIZE, -1).contiguous()

        # Run CEM episode
        for step in range(args.max_steps):
            if step % args.replan_every == 0:
                cur_emb = encode_state(model, d.qpos[:], action_dim)
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

        # Final fingertip position
        final_fp = get_fingertip_pos(m, d)
        if final_fp is None:
            continue
        final_dist = float(np.linalg.norm(final_fp[:2] - goal_fp[:2]))
        success = final_dist < FINGERTIP_THRESH

        results.append({
            "ep": ep,
            "init_dist": init_dist,
            "final_dist": final_dist,
            "success": bool(success),
        })

        if (ep + 1) % 5 == 0 or ep == 0:
            n_succ = sum(r["success"] for r in results)
            mean_init = np.mean([r["init_dist"] for r in results])
            mean_final = np.mean([r["final_dist"] for r in results])
            print(f"  ep {ep+1}: SR={n_succ}/{len(results)}={n_succ/len(results)*100:.1f}%, init={mean_init:.3f}, final={mean_final:.3f}", flush=True)

    n_succ = sum(r["success"] for r in results)
    final = {
        "env": "manipulator_reach_qpos",
        "n_episodes": len(results),
        "n_success": n_succ,
        "success_rate_pct": n_succ / len(results) * 100 if results else 0,
        "mean_init_dist": float(np.mean([r["init_dist"] for r in results])) if results else 0,
        "mean_final_dist": float(np.mean([r["final_dist"] for r in results])) if results else 0,
        "threshold": FINGERTIP_THRESH,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nFINAL: SR={final['success_rate_pct']:.1f}% ({n_succ}/{len(results)})", flush=True)
    print(f"  init_dist={final['mean_init_dist']:.3f} -> final_dist={final['mean_final_dist']:.3f}", flush=True)
    print(f"  Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

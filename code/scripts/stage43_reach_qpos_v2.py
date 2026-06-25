"""v5 Reach qpos task (v2: generic, works for all envs).

Generic CEM-based reach task for any env with arm configuration:
- Sample 2 distinct arm qpos configurations
- Find IK for "target" (we use a random valid qpos)
- Ask model to plan init -> goal
- Measure final position error

Bench: 20 episodes, success: final body position within threshold.
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

XML_BASE = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite/"

ENVS = {
    'manipulator':   ('manipulator.xml',   5,  'bring_ball'),
    'stacker':       ('stacker.xml',       5,  'stack_4'),
    'finger':        ('finger.xml',        2,  'turn_easy'),
    'ball_in_cup':   ('ball_in_cup.xml',   2,  'catch'),
}

FINGERTIP_THRESH = 0.05
INIT_MIN_DIST = 0.08


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env", required=True)
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


def get_arm_tip_pos(m, d, env_name):
    """Get arm tip position depending on env."""
    if env_name == 'manipulator':
        body_name = 'fingertip'
    elif env_name == 'stacker':
        body_name = 'fingertip'  # also has fingertip
    elif env_name == 'finger':
        body_name = 'fingertip'
    elif env_name == 'ball_in_cup':
        body_name = 'ball'  # for ball_in_cup, track the ball
    else:
        body_name = 'fingertip'
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        # try other names
        for n in ['fingertip', 'hand', 'tip', 'ball']:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid >= 0:
                return d.xpos[bid, :3].copy()
        return None
    return d.xpos[bid, :3].copy()


def get_arm_joints(m, env_name):
    """Get arm joint indices for random sampling."""
    if env_name == 'manipulator':
        names = ['arm_shoulder', 'arm_elbow', 'arm_wrist', 'thumb', 'thumbtip', 'finger', 'fingertip']
    elif env_name == 'stacker':
        names = ['arm_shoulder', 'arm_elbow', 'arm_wrist', 'thumb', 'thumbtip', 'finger', 'fingertip']
    elif env_name == 'finger':
        names = ['proximal', 'distal', 'hinge']
    elif env_name == 'ball_in_cup':
        # ball_in_cup has cup_x, cup_z (controlled) and ball_x, ball_z (free)
        # The model was trained on all 4 qpos, so we need to sample all
        names = ['cup_x', 'cup_z', 'ball_x', 'ball_z']
    else:
        names = []
    return [i for i in range(m.njnt) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) in names]


def sample_qpos(m, d, arm_joints, rng):
    for j in arm_joints:
        qs = m.jnt_qposadr[j]
        r = m.jnt_range[j]
        if r[0] < r[1]:
            d.qpos[qs] = rng.uniform(r[0], r[1])
    mujoco.mj_forward(m, d)


def find_init_goal_pair(m, d, rng, env_name, n_samples=20, min_dist=INIT_MIN_DIST):
    arm_joints = get_arm_joints(m, env_name)
    if not arm_joints:
        return None
    configs = []
    for _ in range(n_samples):
        sample_qpos(m, d, arm_joints, rng)
        fp = get_arm_tip_pos(m, d, env_name)
        if fp is not None:
            configs.append((d.qpos[:].copy(), fp))
    if len(configs) < 2:
        return None
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
    if best is None or best_dist < min_dist:
        return None
    return best


def main():
    args = parse_args()
    print(f"[v5-reach-v3] env={args.env}, ckpt={args.ckpt}", flush=True)

    if args.env not in ENVS:
        print(f"  Skip {args.env}: not in ENVS list")
        return

    xml_file, _, _ = ENVS[args.env]
    m = mujoco.MjModel.from_xml_path(XML_BASE + xml_file)
    state_dim = m.nq
    action_dim = m.nu
    print(f"  state_dim={state_dim}, action_dim={action_dim}", flush=True)

    model = load_v5(args.ckpt, state_dim, action_dim)
    print(f"  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    rng = np.random.default_rng(args.seed)
    d = mujoco.MjData(m)

    HISTORY_SIZE = 3
    results = []

    for ep in range(args.n_episodes):
        pair = find_init_goal_pair(m, d, rng, args.env, n_samples=30, min_dist=INIT_MIN_DIST)
        if pair is None:
            continue
        init_qpos, goal_qpos, init_fp, goal_fp = pair
        init_dist = float(np.linalg.norm(init_fp[:2] - goal_fp[:2]))

        d.qpos[:] = init_qpos
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

        init_emb = encode_state(model, init_qpos, action_dim)
        goal_emb = encode_state(model, goal_qpos, action_dim)
        history_emb = init_emb.unsqueeze(0).unsqueeze(0).expand(1, HISTORY_SIZE, -1).contiguous()

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

        final_fp = get_arm_tip_pos(m, d, args.env)
        if final_fp is None:
            continue
        final_dist = float(np.linalg.norm(final_fp[:2] - goal_fp[:2]))
        success = final_dist < FINGERTIP_THRESH

        results.append({
            "ep": ep, "init_dist": init_dist, "final_dist": final_dist, "success": bool(success),
        })

    if not results:
        print("  No valid episodes (couldn't find IK solutions)")
        return

    n_succ = sum(r["success"] for r in results)
    final = {
        "env": f"{args.env}_reach_qpos",
        "n_episodes": len(results),
        "n_success": n_succ,
        "success_rate_pct": n_succ / len(results) * 100,
        "mean_init_dist": float(np.mean([r["init_dist"] for r in results])),
        "mean_final_dist": float(np.mean([r["final_dist"] for r in results])),
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

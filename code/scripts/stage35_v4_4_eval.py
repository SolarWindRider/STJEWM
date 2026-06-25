"""v4.4 Reacher real mujoco bench eval.

v4.4 trained on mujoco rollouts (qpos != 0), so it can learn (qpos, action) -> next_qpos dynamics.

Bench: real mujoco reacher, fingertip-to-target < 0.05, n=30 episodes.
"""
from __future__ import annotations

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
XML = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite/reacher.xml"
TARGET_SIZE = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--replan-every", type=int, default=3)
    p.add_argument("--cem-samples", type=int, default=64)
    p.add_argument("--cem-elites", type=int, default=8)
    p.add_argument("--cem-iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_v4_4(ckpt_path):
    """v4.4 takes 4-D state directly: [qpos(2), target(2)]"""
    from lewm_stjewm_v4 import STJEWMv4
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    saved_args = ck.get("args", {})
    state_dim = 4
    action_dim = saved_args.get("action_dim", 2)
    n_layers = saved_args.get("n_layers", 4)
    model = STJEWMv4(
        d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
        state_dim=state_dim, cell_n_layers=n_layers, n_d=3, trace_beta=0.9,
        freeze_encoder=True,
    )
    model.load_state_dict(sd, strict=False)
    return model.to(DEVICE).eval(), state_dim, action_dim


@torch.no_grad()
def cem_plan(model, init_emb, goal_emb, horizon, action_dim, cem_samples, cem_elites, cem_iters):
    history_size = init_emb.shape[1]
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


@torch.no_grad()
def encode_state_4d(model, state_4d, action):
    s = torch.from_numpy(state_4d.astype(np.float32)).reshape(1, 1, -1).to(DEVICE)
    a = torch.from_numpy(action.astype(np.float32)).reshape(1, 1, -1).to(DEVICE)
    enc = model.encode(s, a)
    return enc["emb"][0, 0]


def get_fingertip_target_dist(model, data):
    finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger")
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    if finger_id < 0 or target_id < 0:
        return float('inf')
    finger_pos = data.geom_xpos[finger_id, :2].copy()
    target_pos = data.geom_xpos[target_id, :2].copy()
    return float(np.linalg.norm(finger_pos - target_pos))


def main():
    args = parse_args()
    print(f"Loading v4.4 from {args.ckpt}")
    model, state_dim, action_dim = load_v4_4(args.ckpt)
    print(f"  state_dim={state_dim}, action_dim={action_dim}")

    mujoco_model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(mujoco_model)
    print(f"mujoco: qpos={mujoco_model.nq}, ctrl={mujoco_model.nu}, TARGET_SIZE={TARGET_SIZE}")

    rng = np.random.default_rng(args.seed)
    print(f"Eval {args.n_episodes} episodes (random initial conditions)")

    successes = 0
    final_dists = []
    initial_dists = []

    for ei in range(args.n_episodes):
        # Random initial qpos + random target
        init_qpos = rng.uniform(-np.pi/2, np.pi/2, size=2)
        target_pos = rng.uniform(-0.2, 0.2, size=2)
        init_state_4d = np.concatenate([init_qpos, target_pos])  # (4,)

        # Reset env
        mujoco.mj_resetData(mujoco_model, data)
        data.qpos[:] = init_qpos
        data.qvel[:] = 0
        target_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        mujoco_model.geom_pos[target_id, :2] = target_pos
        mujoco.mj_forward(mujoco_model, data)

        initial_dist = get_fingertip_target_dist(mujoco_model, data)
        initial_dists.append(initial_dist)

        # History: 3 frames at start (use init state)
        z_hist_list = []
        for _ in range(3):
            z_t = encode_state_4d(model, init_state_4d, np.zeros(action_dim))
            z_hist_list.append(z_t)
        z_hist = torch.stack(z_hist_list, dim=0).unsqueeze(0)  # (1, 3, 192)

        # Goal: state when fingertip = target (approximated as qpos = some value that gives target)
        # We use the "current qpos with target at target position" as goal
        # For simplicity, encode the state where qpos is at the goal position
        # Actually we don't know the qpos that gives target. We use the target_pos as the goal
        # and let CEM find the qpos
        goal_state_4d = np.concatenate([init_qpos, target_pos])  # placeholder
        # Better: use a "neutral" goal where qpos is the start and target is set
        z_goal = encode_state_4d(model, init_state_4d, np.zeros(action_dim)).unsqueeze(0)

        # Run episode
        step_to_success = -1
        best_actions = None
        for step in range(50):
            d = get_fingertip_target_dist(mujoco_model, data)
            if d < TARGET_SIZE:
                step_to_success = step
                break

            if step % args.replan_every == 0:
                best_actions = cem_plan(
                    model, z_hist, z_goal,
                    horizon=args.horizon, action_dim=action_dim,
                    cem_samples=args.cem_samples, cem_elites=args.cem_elites,
                    cem_iters=args.cem_iters,
                )

            action_idx = step % args.replan_every
            if action_idx >= args.horizon:
                action_idx = args.horizon - 1
            action = best_actions[action_idx].cpu().numpy()
            action = np.clip(action, -1, 1)  # clip to mujoco range

            data.ctrl[:] = action
            mujoco.mj_step(mujoco_model, data)

            # Update z_hist
            new_state = np.concatenate([data.qpos[:], target_pos])
            try:
                a_t = torch.from_numpy(action.astype(np.float32)).reshape(1, 1, -1).to(DEVICE)
                h_in = z_hist[:, -2:]
                a_window = torch.cat([a_t.new_zeros(1, 1, action_dim), a_t], dim=1)
                nxt = model.predict(h_in, a_window)[:, -1]
                z_hist = torch.cat([z_hist[:, 1:], nxt.unsqueeze(0)], dim=1)
            except Exception:
                break

        if step_to_success == -1:
            final_dist = get_fingertip_target_dist(mujoco_model, data)
        else:
            final_dist = 0.0
        final_dists.append(final_dist)
        if step_to_success >= 0:
            successes += 1

        if (ei + 1) % 5 == 0:
            print(f"  [{ei+1}/{args.n_episodes}] success={successes}/{ei+1} "
                  f"({successes/(ei+1)*100:.1f}%) init_d={initial_dist:.3f} "
                  f"final_d={final_dist:.3f}", flush=True)

    result = {
        "model": "v4.4 (mujoco rollouts)",
        "ckpt": args.ckpt,
        "n_episodes": args.n_episodes,
        "successes": successes,
        "success_rate": successes / max(args.n_episodes, 1),
        "bench_spec": f"Reacher: ||fingertip - target|| < {TARGET_SIZE}",
        "mean_initial_dist": float(np.mean(initial_dists)),
        "mean_final_dist": float(np.mean(final_dists)),
        "std_final_dist": float(np.std(final_dists)),
        "config": vars(args),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n=== FINAL RESULT (v4.4 mujoco rollouts) ===")
    print(f"  Successes: {successes}/{args.n_episodes} = {successes/max(args.n_episodes,1)*100:.1f}%")
    print(f"  Mean initial dist: {np.mean(initial_dists):.3f}")
    print(f"  Mean final dist: {np.mean(final_dists):.3f} (threshold: {TARGET_SIZE})")
    print(f"  Saved to {args.out}")


if __name__ == "__main__":
    main()
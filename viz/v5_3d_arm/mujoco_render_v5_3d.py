"""v5 3D manipulator render — bringball task (5D arm + ball + target).

This script:
  1. Loads v5 3D SNN ckpt (state_dim=17, action_dim=5)
  2. Sets up mujoco manipulator env (5D arm)
  3. Plans with CEM receding-horizon (mirrors stage50 strategy)
  4. Records: arm qpos trajectory, fingertip/ball/target positions,
     actions, AND spike raster (per-step)
  5. Renders 4-row GIF: 3D view (top-down projection), qpos, action, spike raster

Output:
  viz/v5_3d_arm/output/v5_3d_best_improved.gif
  viz/v5_3d_arm/output/v5_3d_failed.gif
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import mujoco

sys.path.insert(0, "/home/lx/snn/code")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
XML_PATH = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite/manipulator.xml"
TARGET_SIZE = 0.05  # 5cm — same as reacher threshold
mj_model_local = None
mj_data_local = None


def load_v5(ckpt_path):
    from lewm_stjewm_v4 import STJEWMv4
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    state_dim = sd["state_projector.proj.0.weight"].shape[1]
    action_dim = sd["action_encoder.proj.weight"].shape[1]
    n_layers = ck.get("args", {}).get("n_layers", 4)
    model = STJEWMv4(
        d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
        state_dim=state_dim, cell_n_layers=n_layers, n_d=3, trace_beta=0.9,
        freeze_encoder=True,
    )
    model.load_state_dict(sd, strict=False)
    return model.to(DEVICE).eval(), state_dim, action_dim


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


def get_fingertip_pos(mj_model, mj_data):
    fid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    return mj_data.xpos[fid, :3].copy() if fid >= 0 else None


def get_ball_pos(mj_model, mj_data):
    bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    return mj_data.geom_xpos[bid, :3].copy() if bid >= 0 else None


def get_target_pos(mj_model):
    tid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "target_ball")
    return mj_model.geom_pos[tid, :3].copy() if tid >= 0 else None


def find_close_init_goal(mj_model, mj_data, rng, target_pos, max_init_dist=0.5):
    """Find two qpos configs that put fingertip near target, with qpos distance < max_dist."""
    arm_joints = []
    for i in range(mj_model.njnt):
        n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if n and n.startswith(('arm_', 'thumb', 'finger')):
            arm_joints.append(i)
    # Try with multiple tolerance levels (loose to strict) to find close pair
    candidates = []
    for ft_tol in [0.20, 0.15, 0.10, 0.05]:
        for _ in range(2000):
            for j in arm_joints:
                qs = mj_model.jnt_qposadr[j]
                r = mj_model.jnt_range[j]
                if r[0] < r[1]:
                    mj_data.qpos[qs] = rng.uniform(r[0] * 0.3, r[1] * 0.3)
            mujoco.mj_forward(mj_model, mj_data)
            fp = get_fingertip_pos(mj_model, mj_data)
            if fp is None:
                continue
            ft_dist = float(np.linalg.norm(fp[:2] - target_pos[:2]))
            if ft_dist < ft_tol:
                candidates.append(mj_data.qpos[:].copy())
            if len(candidates) >= 50:
                break
        if len(candidates) >= 2:
            break
    if len(candidates) < 2:
        return None
    best = None
    best_dist = float('inf')
    for i in range(min(10, len(candidates))):
        for j in range(i + 1, min(10, len(candidates))):
            d = float(np.linalg.norm(candidates[i] - candidates[j]))
            if d < best_dist and d < max_init_dist:
                best_dist = d
                best = (candidates[i], candidates[j])
    return best

def get_state_vec(mj_data, target_pos, state_dim):
    """Build state vec matching the ckpt's state_dim.
    state_dim=14 -> [qpos(14)] (no target)
    state_dim=17 -> [qpos(14), target(3)] (with target)
    """
    if state_dim == 14:
        return mj_data.qpos[:14].astype(np.float32)
    elif state_dim == 17:
        return np.concatenate([mj_data.qpos[:], target_pos]).astype(np.float32)
    else:
        qpos = mj_data.qpos[:state_dim - len(target_pos)]
        return np.concatenate([qpos, target_pos]).astype(np.float32)


@torch.no_grad()
def run_episode(model, mj_model, mj_data, state_dim, action_dim, init_qpos, goal_qpos,
                target_pos, args):
    """Run one CEM-controlled mujoco episode, recording state at each step."""
    # Reset env to init
    mj_data.qpos[:] = init_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    # 3-frame z_hist from init state
    init_state = get_state_vec(mj_data, target_pos, state_dim)
    z_hist_list = [encode_state(model, init_state, action_dim) for _ in range(3)]
    z_hist = torch.stack(z_hist_list, dim=0).unsqueeze(0)  # (1, 3, 192)

    # Goal state = goal qpos + same target
    mj_data.qpos[:] = goal_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)
    goal_state = get_state_vec(mj_data, target_pos, state_dim)
    z_goal = encode_state(model, goal_state, action_dim).unsqueeze(0)

    # Reset to init
    mj_data.qpos[:] = init_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    init_fingertip = get_fingertip_pos(mj_model, mj_data) if get_fingertip_pos(mj_model, mj_data) is not None else np.zeros(3)
    init_ball = get_ball_pos(mj_model, mj_data) if get_ball_pos(mj_model, mj_data) is not None else np.zeros(3)
    init_qpos_dist = float(np.linalg.norm(init_qpos - goal_qpos))

    history = {
        "qpos": [], "fingertip": [], "ball": [], "target": target_pos.copy(),
        "dist_qpos": [], "dist_fingertip_ball": [], "dist_ball_target": [],
        "spike": [], "spike_layers": [], "actions": [],
        "init_qpos_dist": init_qpos_dist,
        "init_fingertip": init_fingertip.copy(),
        "init_ball": init_ball.copy(),
    }
    best_actions = None
    state_buf = []
    action_buf = []

    for step in range(args.max_steps):
        # Record current state
        cur_qpos = mj_data.qpos[:].copy()
        cur_fingertip = get_fingertip_pos(mj_model, mj_data) if get_fingertip_pos(mj_model, mj_data) is not None else np.zeros(3)
        cur_ball = get_ball_pos(mj_model, mj_data) if get_ball_pos(mj_model, mj_data) is not None else np.zeros(3)
        dist_qpos = float(np.linalg.norm(cur_qpos - goal_qpos))
        dist_fb = float(np.linalg.norm(cur_fingertip - cur_ball)) if (get_fingertip_pos(mj_model, mj_data) is not None and get_ball_pos(mj_model, mj_data) is not None) else 0
        dist_bt = float(np.linalg.norm(cur_ball - target_pos)) if get_ball_pos(mj_model, mj_data) is not None else 0
        history["qpos"].append(cur_qpos)
        history["fingertip"].append(cur_fingertip)
        history["ball"].append(cur_ball)
        history["dist_qpos"].append(dist_qpos)
        history["dist_fingertip_ball"].append(dist_fb)
        history["dist_ball_target"].append(dist_bt)

        # Re-plan
        if step % args.replan_every == 0:
            best_actions = cem_plan(model, z_hist, z_goal, args.horizon, action_dim,
                                    args.cem_samples, args.cem_elites, args.cem_iters)
        action_idx = step % args.replan_every
        if action_idx >= args.horizon:
            action_idx = args.horizon - 1
        action = best_actions[action_idx].cpu().numpy()
        action = np.clip(action, -1, 1)
        history["actions"].append(action)

        # Step env
        mj_data.ctrl[:] = action
        mujoco.mj_step(mj_model, mj_data)

        # Update z_hist via model.predict
        new_state = get_state_vec(mj_data, target_pos, state_dim)
        a_t = torch.from_numpy(action.astype(np.float32)).reshape(1, 1, -1).to(DEVICE)
        h_in = z_hist[:, -2:]
        a_window = torch.cat([a_t.new_zeros(1, 1, action_dim), a_t], dim=1)
        nxt = model.predict(h_in, a_window)[:, -1]
        z_hist = torch.cat([z_hist[:, 1:], nxt.unsqueeze(0)], dim=1)

        # Record spike via model.forward
        state_buf.append(torch.from_numpy(new_state).float().reshape(1, 1, -1).to(DEVICE))
        action_buf.append(a_t)
        if len(state_buf) > 3:
            state_buf = state_buf[-3:]
            action_buf = action_buf[-3:]
        while len(state_buf) < 3:
            state_buf.insert(0, state_buf[0])
            action_buf.insert(0, torch.zeros_like(action_buf[0]))
        s_window = torch.cat(state_buf[-3:], dim=1)
        a_window_fwd = torch.cat(action_buf[-3:], dim=1)
        fwd_out = model(s_window, a_window_fwd)
        last_spike = fwd_out["spike"][0, -1].cpu().numpy()
        last_spike_layers = torch.stack(
            [sl[0, -1] for sl in fwd_out["spike_layers"]], dim=0
        ).cpu().numpy()
        history["spike"].append(last_spike)
        history["spike_layers"].append(last_spike_layers)

    # Final state
    cur_qpos = mj_data.qpos[:].copy()
    cur_fingertip = get_fingertip_pos(mj_model, mj_data) if get_fingertip_pos(mj_model, mj_data) is not None else np.zeros(3)
    cur_ball = get_ball_pos(mj_model, mj_data) if get_ball_pos(mj_model, mj_data) is not None else np.zeros(3)
    history["qpos"].append(cur_qpos)
    history["fingertip"].append(cur_fingertip)
    history["ball"].append(cur_ball)
    history["dist_qpos"].append(float(np.linalg.norm(cur_qpos - goal_qpos)))
    history["dist_fingertip_ball"].append(float(np.linalg.norm(cur_fingertip - cur_ball)))
    history["dist_ball_target"].append(float(np.linalg.norm(cur_ball - target_pos)))
    history["improved"] = history["dist_qpos"][-1] < history["dist_qpos"][0]
    return history


def render_arm_3d(ax, mj_model, mj_data, title=""):
    """Render real 3D manipulator arm skeleton + ball + target.

    Arm chain (5 links from base to fingertip):
      upper_arm (0,0,0.4) → middle_arm (z+0.18) → lower_arm (z+0.15)
      → hand (z+0.12) → thumb + finger + thumbtip + fingertip (z+0.03, dx=±0.04)
    """
    ax.clear()
    body_names = ['upper_arm', 'middle_arm', 'lower_arm', 'hand',
                  'thumb', 'thumbtip', 'finger', 'fingertip',
                  'ball', 'target_ball']
    body_xpos = {}
    for name in body_names:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        body_xpos[name] = mj_data.xpos[bid, :3].copy() if bid >= 0 else None

    # === Arm skeleton (x-z plane, y is depth out of page) ===
    arm_base = np.array([0, 0, 0.4])
    arm_chain = [body_xpos[n] for n in ['upper_arm', 'middle_arm', 'lower_arm', 'hand']
                 if body_xpos[n] is not None]
    if len(arm_chain) > 0:
        ax.plot([arm_base[0], arm_chain[0][0]], [arm_base[2], arm_chain[0][2]],
                '-', color='#666666', lw=3, alpha=0.7)
        for i in range(len(arm_chain) - 1):
            ax.plot([arm_chain[i][0], arm_chain[i+1][0]],
                    [arm_chain[i][2], arm_chain[i+1][2]],
                    '-', color='#0072B2', lw=5, solid_capstyle='round',
                    label='Arm link' if i == 0 else '')
    for p in [arm_base] + arm_chain:
        ax.plot(p[0], p[2], 'o', color='#0072B2', markersize=8)
    # Thumbs + fingers
    for name, color, label in [('thumb', '#009E73', 'Thumb'),
                                ('finger', '#CC79A7', 'Finger')]:
        p = body_xpos[name]
        if p is not None:
            ax.plot(p[0], p[2], 's', color=color, markersize=8, label=label)
    for name, color in [('thumbtip', '#009E73'), ('fingertip', '#CC79A7')]:
        p = body_xpos[name]
        if p is not None:
            ax.plot(p[0], p[2], 'o', color=color, markersize=10)
    fp = body_xpos['fingertip']
    if fp is not None:
        ax.plot(fp[0], fp[2], 'o', color='#CC79A7', markersize=14,
                markeredgecolor='black', markeredgewidth=1.5,
                label='Fingertip (end-effector)')
    # Ball + target
    ball = body_xpos['ball']
    target = body_xpos['target_ball']
    if ball is not None:
        ax.plot(ball[0], ball[2], 's', color='#FF9500', markersize=14, label='Ball')
    if target is not None:
        ax.plot(target[0], target[2], '*', color='#D55E00', markersize=18, label='Target')
        theta = np.linspace(0, 2 * np.pi, 50)
        ax.plot(target[0] + TARGET_SIZE * np.cos(theta),
                target[2] + TARGET_SIZE * np.sin(theta),
                '--', color='#D55E00', lw=1.0, alpha=0.6, label='Reach (5cm)')
    ax.axhline(0, color='#999999', lw=0.5, alpha=0.4)
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.05, 0.95)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("z (height, m)")
    ax.set_title(title, fontsize=9)
    ax.legend(loc='upper right', fontsize=6, framealpha=0.8, ncol=2)


def build_episode_gif(ep, output_path, title, mj_model, mj_data, fps=6, dpi=100):
    n_frames = len(ep["qpos"])
    T = len(ep["dist_qpos"])
    # Local copies for animation closure
    global mj_model_local, mj_data_local
    mj_model_local = mj_model
    mj_data_local = mujoco.MjData(mj_model)
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1.4, 0.7, 0.7, 1.0], hspace=0.4, wspace=0.3)

    ax_view = fig.add_subplot(gs[0, :])
    ax_qpos = fig.add_subplot(gs[1, :])
    ax_act = fig.add_subplot(gs[2, 0])
    ax_dist = fig.add_subplot(gs[2, 1])
    ax_spike = fig.add_subplot(gs[3, :])

    fig.suptitle(f"{title}  |  improved={ep['improved']}  |  "
                 f"init_qpos_dist={ep['init_qpos_dist']:.3f}  |  "
                 f"final_qpos_dist={ep['dist_qpos'][-1]:.3f}", fontsize=11)

    # === Static plots: qpos, action, dist, spike (base) ===
    t = np.arange(T)
    # Qpos — show first 5 dims (arm 5D)
    qpos = np.array(ep["qpos"])
    if qpos.shape[0] > 0:
        for i in range(min(5, qpos.shape[1])):
            ax_qpos.plot(t, qpos[:, i], label=f'qpos[{i}]', lw=1.0, alpha=0.8)
    goal_qpos = qpos[0]  # not actually shown since goal is at index t=start
    ax_qpos.set_xlabel("step")
    ax_qpos.set_ylabel("Joint angle")
    ax_qpos.set_title("R1a  qpos (first 5 joints)")
    ax_qpos.legend(loc='upper right', fontsize=7, ncol=3)
    ax_qpos.grid(True, alpha=0.3)

    t_act = np.arange(len(ep["actions"]))
    if len(ep["actions"]) > 0:
        actions = np.array(ep["actions"])
        for i in range(actions.shape[1]):
            ax_act.plot(t_act, actions[:, i], '-o', markersize=2, label=f'act[{i}]')
    ax_act.set_xlabel("step")
    ax_act.set_ylabel("Action value")
    ax_act.set_title("R3a  Action stream (5D)")
    ax_act.legend(loc='upper right', fontsize=7, ncol=3)
    ax_act.grid(True, alpha=0.3)
    ax_act.axhline(0, color='black', lw=0.5)

    ax_dist.plot(t, ep["dist_qpos"], '-o', color='#0072B2', markersize=2, label='qpos dist')
    ax_dist.plot(t, ep["dist_fingertip_ball"], '-s', color='#D55E00', markersize=2, label='finger-ball')
    ax_dist.plot(t, ep["dist_ball_target"], '-^', color='#009E73', markersize=2, label='ball-target')
    ax_dist.set_xlabel("step")
    ax_dist.set_ylabel("Distance")
    ax_dist.set_title("R3b  Multi-distance over time")
    ax_dist.legend(loc='upper right', fontsize=7)
    ax_dist.grid(True, alpha=0.3)

    # Spike raster (base — original neuron order, NO sort)
    spike_data = ep.get("spike", [])
    if len(spike_data) > 0:
        spike_arr = np.array(spike_data)
        T_spike = spike_arr.shape[0]
        spike_display = spike_arr.T  # (192, T_spike)
        ax_spike.imshow(spike_display, aspect='auto', cmap='gray_r',
                        interpolation='nearest', vmin=0, vmax=1,
                        extent=[0, T_spike - 1, 0, 192], origin='lower')
        sparsity = 1.0 - spike_arr.mean()
        ax_spike.set_xlabel("Time step t")
        ax_spike.set_ylabel("Neuron index (0-191)")
        ax_spike.set_title(f"R5  SNN spike raster (final layer, ORIGINAL order, sp={sparsity:.2f}, mean FR={spike_arr.mean():.2f})")
        ax_spike.grid(True, alpha=0.3)

    # Initial 3D arm render (frame 0 state)
    mj_data_local.qpos[:] = qpos[0]
    mujoco.mj_forward(mj_model_local, mj_data_local)
    render_arm_3d(ax_view, mj_model_local, mj_data_local,
                   title=f"R1  3D arm at t=0  "
                         f"(init qpos_dist={ep['init_qpos_dist']:.3f})")

    # Pre-add cursors
    cursor_color = '#0072B2'
    cursor_kw = dict(color=cursor_color, lw=1.0, alpha=0.7)
    qpos_cursor = ax_qpos.axvline(0, **cursor_kw)
    act_cursor = ax_act.axvline(0, **cursor_kw)
    dist_cursor = ax_dist.axvline(0, **cursor_kw)
    # 3D arm rendered by render_arm_3d (no separate cursor markers)

    def update(frame):
        # Re-run mujoco forward kinematics to get current 3D body positions
        mj_data_local.qpos[:] = qpos[frame]
        mujoco.mj_forward(mj_model_local, mj_data_local)
        render_arm_3d(ax_view, mj_model_local, mj_data_local,
                       title=f"R1  3D arm at t={frame}  "
                             f"(qpos_dist={ep['dist_qpos'][frame]:.2f}, "
                             f"finger-ball={ep['dist_fingertip_ball'][frame]:.2f})")
        qpos_cursor.set_xdata([frame, frame])
        act_cursor.set_xdata([frame, frame])
        dist_cursor.set_xdata([frame, frame])
        # Spike raster (unsorted, ORIGINAL order)
        if len(spike_data) > 0:
            width = min(frame + 1, spike_arr.shape[0])
            ax_spike.clear()
            ax_spike.imshow(spike_display[:, :width], aspect='auto', cmap='gray_r',
                            interpolation='nearest', vmin=0, vmax=1,
                            extent=[0, max(width, 1), 0, 192], origin='lower')
            ax_spike.axvline(frame, color=cursor_color, lw=1.0, alpha=0.7)
            ax_spike.set_xlim(-0.5, max(spike_arr.shape[0], 1) - 0.5)
            ax_spike.set_ylim(0, 192)
            ax_spike.set_xlabel("Time step t")
            ax_spike.set_ylabel("Neuron index (0-191)")
            sparsity = 1.0 - spike_arr.mean()
            ax_spike.set_title(
                f"R5  SNN spike raster (final layer, ORIGINAL order, sp={sparsity:.2f}, t={frame})"
            )
            ax_spike.grid(True, alpha=0.3)
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False, repeat=True,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer='pillow', fps=fps, dpi=dpi)
    plt.close()
    print(f"  saved {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/lx/snn/results/stage44_train/v5/manipulator_1M_5ep/step30000.pt",
                    help="v5 3D ckpt path")
    ap.add_argument("--n-episodes", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--replan-every", type=int, default=2)
    ap.add_argument("--cem-samples", type=int, default=64)
    ap.add_argument("--cem-elites", type=int, default=8)
    ap.add_argument("--cem-iters", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--max-init-dist", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/home/lx/snn/viz/v5_3d_arm/output")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=90)
    args = ap.parse_args()

    print(f"[load] {args.ckpt}")
    model, state_dim, action_dim = load_v5(args.ckpt)
    print(f"  state_dim={state_dim}, action_dim={action_dim}")

    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data = mujoco.MjData(mj_model)
    print(f"mujoco manipulator: nq={mj_model.nq} nu={mj_model.nu} njnt={mj_model.njnt}")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[run] rolling out {args.n_episodes} episodes (stage50-style plan, improved metric)...")
    results = []
    for ep_i in range(args.n_episodes):
        # Sample target
        target_pos = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0.05, 0.25)])
        target_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "target_ball")
        mj_model.geom_pos[target_id] = target_pos
        pair = find_close_init_goal(mj_model, mj_data, rng, target_pos, max_init_dist=args.max_init_dist)
        if pair is None:
            print(f"  ep {ep_i + 1}: no close pair found, skipping")
            continue
        init_qpos, goal_qpos = pair
        init_dist = float(np.linalg.norm(init_qpos - goal_qpos))
        # Reset to init before run
        mj_data.qpos[:] = init_qpos
        mj_data.qvel[:] = 0
        mujoco.mj_forward(mj_model, mj_data)
        # Run episode
        history = run_episode(model, mj_model, mj_data, state_dim, action_dim,
                              init_qpos, goal_qpos, target_pos, args)
        history["ep_idx"] = ep_i
        history["init_qpos_dist"] = init_dist
        results.append(history)
        print(f"  ep {ep_i + 1}: init_qpos_dist={init_dist:.3f} "
              f"final_qpos_dist={history['dist_qpos'][-1]:.3f} "
              f"finger-ball={history['dist_fingertip_ball'][-1]:.3f} "
              f"ball-target={history['dist_ball_target'][-1]:.3f} "
              f"improved={history['improved']}")

    n_improved = sum(1 for h in results if h["improved"])
    print(f"\nFinal: {n_improved}/{len(results)} improved ({n_improved / max(len(results), 1) * 100:.1f}%)")

    # Find best improved (most reduction in qpos_dist) and worst
    if len(results) > 0:
        best = min(results, key=lambda h: h["dist_qpos"][-1])
        worst = max(results, key=lambda h: h["dist_qpos"][-1])
        build_episode_gif(best, out_dir / "v5_3d_best_improved.gif",
                          f"v5 3D BEST (qpos {best['init_qpos_dist']:.2f} -> {best['dist_qpos'][-1]:.2f}, "
                          f"improved={best['improved']})",
                          mj_model, mj_data,
                          fps=args.fps, dpi=args.dpi)
        if worst is not best:
            build_episode_gif(worst, out_dir / "v5_3d_failed.gif",
                              f"v5 3D WORST (qpos {worst['init_qpos_dist']:.2f} -> {worst['dist_qpos'][-1]:.2f}, "
                              f"improved={worst['improved']})",
                              mj_model, mj_data,
                              fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()

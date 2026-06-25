"""3D quadruped render — true 3D matplotlib plot.

Quadruped is a true 3D robot:
  - 4 legs (4 × 3 joints = 12 actuated)
  - Body + abdomen (4 passive joints)
  - All joints have axis "0 0 1" or "0 1 0" → real 3D motion

Renders:
  - 3D body skeleton (4 leg chains, body, abdomen)
  - 3D ball + target (sphere geom)
  - Multi-view (front + side) so y dimension is visible
  - Spike raster (unsorted) + qpos + action + distance streams

Output:
  viz/v5_quadruped_3d/output/v5_quadruped_3d_best.gif (mp4 if ffmpeg)
  viz/v5_quadruped_3d/output/v5_quadruped_3d_top3.png
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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import mujoco

sys.path.insert(0, "/home/lx/snn/code")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
XML_PATH = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite/quadruped.xml"
TARGET_SIZE = 0.10  # 10cm — quadruped is bigger than reacher/manipulator


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


def get_state_vec(mj_data, target_pos, state_dim):
    if state_dim == mj_data.qpos.shape[0]:
        return mj_data.qpos[:].astype(np.float32)
    elif state_dim == mj_data.qpos.shape[0] + 3:
        return np.concatenate([mj_data.qpos[:], target_pos]).astype(np.float32)
    else:
        # fallback
        return np.concatenate([mj_data.qpos[:], target_pos]).astype(np.float32)[:state_dim]


def render_quadruped_3d(ax, mj_model, mj_data, view='iso', title=''):
    """Render quadruped 3D skeleton in true 3D matplotlib.

    view='iso'  : default 3D view (azim=-60, elev=30)
    view='top'  : top-down 3D (azim=-90, elev=89)
    view='side' : side view 3D (azim=0, elev=5)
    """
    ax.clear()
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(0, 0.6)
    ax.set_xlabel('x (forward, m)')
    ax.set_ylabel('y (left, m)')
    ax.set_zlabel('z (up, m)')
    ax.set_title(title, fontsize=9)
    if view == 'iso':
        ax.view_init(elev=30, azim=-60)
    elif view == 'top':
        ax.view_init(elev=89, azim=-90)
    elif view == 'side':
        ax.view_init(elev=5, azim=0)
    else:
        ax.view_init(elev=30, azim=-60)

    # Get all body positions
    body_names = []
    for i in range(mj_model.nbody):
        n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        if n and not n.startswith('world'):
            body_names.append(n)
    body_xpos = {}
    for n in body_names:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
        body_xpos[n] = mj_data.xpos[bid, :3].copy()

    # === Body chain (torso + 4 legs) ===
    # Quadruped body structure: torso → abdomen → ...
    # Leg chains: torso → [front_left, front_right, back_left, back_right]
    # Find leg bodies
    leg_patterns = ['front_left_', 'front_right_', 'back_left_', 'back_right_',
                    '1_', '2_', '3_', '4_']
    leg_bodies = [n for n in body_names if any(p in n for p in leg_patterns) and n not in ['world']]

    # Define 4 leg chains explicitly (real quadruped has 4 legs)
    leg_chains = [
        ['hip_front_left',  'knee_front_left',  'ankle_front_left',  'toe_front_left'],
        ['hip_front_right', 'knee_front_right', 'ankle_front_right', 'toe_front_right'],
        ['hip_back_left',   'knee_back_left',   'ankle_back_left',   'toe_back_left'],
        ['hip_back_right',  'knee_back_right',  'ankle_back_right',  'toe_back_right'],
    ]
    leg_labels = ['Front-Left', 'Front-Right', 'Back-Left', 'Back-Right']
    leg_colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']

    torso_pos = body_xpos.get('torso')
    if torso_pos is None:
        torso_pos = body_xpos.get('root')
    if torso_pos is not None:
        for leg, label, color in zip(leg_chains, leg_labels, leg_colors):
            points = [torso_pos] + [body_xpos[n] for n in leg if n in body_xpos]
            for i in range(len(points) - 1):
                ax.plot([points[i][0], points[i+1][0]],
                        [points[i][1], points[i+1][1]],
                        [points[i][2], points[i+1][2]],
                        '-', color=color, lw=3, solid_capstyle='round', alpha=0.85,
                        label=label if i == 0 else '')
            for p in points[1:]:
                ax.plot([p[0]], [p[1]], [p[2]], 'o', color=color, markersize=6,
                        markeredgecolor='black', markeredgewidth=0.5)

    # === Torso (draw once at root position) ===
    if 'torso' in body_xpos:
        p = body_xpos['torso']
        ax.plot([p[0]], [p[1]], [p[2]], '^', color='black', markersize=10, label='Torso')
    elif 'root' in body_xpos:
        p = body_xpos['root']
        ax.plot([p[0]], [p[1]], [p[2]], '^', color='black', markersize=10)

    # === Target (3D sphere marker) ===
    tid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    if tid >= 0:
        target_pos = mj_model.geom_pos[tid, :3]
        # 3D sphere mesh for visual
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 15), np.linspace(0, np.pi, 10))
        r = TARGET_SIZE
        x = target_pos[0] + r * np.outer(np.cos(u), np.sin(v))
        y = target_pos[1] + r * np.outer(np.sin(u), np.sin(v))
        z = target_pos[2] + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color='#D55E00', alpha=0.4, label='Target')
        ax.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], '*', color='#D55E00', markersize=15)

    # === Ground plane (grid) ===
    gx, gy = np.meshgrid(np.linspace(-0.8, 0.8, 9), np.linspace(-0.5, 0.5, 5))
    gz = np.zeros_like(gx)
    ax.plot_wireframe(gx, gy, gz, color='#999999', alpha=0.2)

    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)


@torch.no_grad()
def run_episode(model, mj_model, mj_data, state_dim, action_dim, init_qpos, goal_qpos,
                target_pos, args):
    """Run CEM-controlled mujoco episode with quadruped."""
    mj_data.qpos[:] = init_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    init_state = get_state_vec(mj_data, target_pos, state_dim)
    z_hist_list = [encode_state(model, init_state, action_dim) for _ in range(3)]
    z_hist = torch.stack(z_hist_list, dim=0).unsqueeze(0)

    mj_data.qpos[:] = goal_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)
    goal_state = get_state_vec(mj_data, target_pos, state_dim)
    z_goal = encode_state(model, goal_state, action_dim).unsqueeze(0)

    mj_data.qpos[:] = init_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    init_qpos_dist = float(np.linalg.norm(init_qpos - goal_qpos))

    history = {
        "qpos": [], "target": target_pos.copy(),
        "dist_qpos": [], "spike": [], "spike_layers": [],
        "actions": [], "init_qpos_dist": init_qpos_dist,
    }
    best_actions = None
    state_buf = []
    action_buf = []

    for step in range(args.max_steps):
        cur_qpos = mj_data.qpos[:].copy()
        dist_qpos = float(np.linalg.norm(cur_qpos - goal_qpos))
        history["qpos"].append(cur_qpos)
        history["dist_qpos"].append(dist_qpos)

        if step % args.replan_every == 0:
            best_actions = cem_plan(model, z_hist, z_goal, args.horizon, action_dim,
                                    args.cem_samples, args.cem_elites, args.cem_iters)
        action_idx = step % args.replan_every
        if action_idx >= args.horizon:
            action_idx = args.horizon - 1
        action = best_actions[action_idx].cpu().numpy()
        action = np.clip(action, -1, 1)
        history["actions"].append(action)

        mj_data.ctrl[:] = action
        mujoco.mj_step(mj_model, mj_data)

        new_state = get_state_vec(mj_data, target_pos, state_dim)
        a_t = torch.from_numpy(action.astype(np.float32)).reshape(1, 1, -1).to(DEVICE)
        h_in = z_hist[:, -2:]
        a_window = torch.cat([a_t.new_zeros(1, 1, action_dim), a_t], dim=1)
        nxt = model.predict(h_in, a_window)[:, -1]
        z_hist = torch.cat([z_hist[:, 1:], nxt.unsqueeze(0)], dim=1)

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

    cur_qpos = mj_data.qpos[:].copy()
    history["qpos"].append(cur_qpos)
    history["dist_qpos"].append(float(np.linalg.norm(cur_qpos - goal_qpos)))
    history["improved"] = history["dist_qpos"][-1] < history["dist_qpos"][0]
    return history


def find_close_init_goal(mj_model, mj_data, rng, target_pos, max_init_dist=1.5):
    """Find two qpos configs that put a foot/toe near target (real 3D reach)."""
    arm_joints = list(range(mj_model.njnt))
    # Pick a foot body — try 'toe_front_left' first
    foot_body = 'toe_front_left'
    foot_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, foot_body)
    if foot_bid < 0:
        foot_body = 'torso'
        foot_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, foot_body)
    candidates = []
    for ft_tol in [0.25, 0.20, 0.15, 0.10]:
        for _ in range(2000):
            for j in arm_joints:
                # Skip FREE joints (range [0,0]) and zero-range joints
                jt = mj_model.jnt_type[j]
                if jt == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                qs = mj_model.jnt_qposadr[j]
                r = mj_model.jnt_range[j]
                if r[0] < r[1]:
                    mj_data.qpos[qs] = rng.uniform(r[0], r[1])
            mujoco.mj_forward(mj_model, mj_data)
            foot_pos = mj_data.xpos[foot_bid, :3]
            # 3D distance
            d = float(np.linalg.norm(foot_pos - target_pos))
            if d < ft_tol:
                candidates.append(mj_data.qpos[:].copy())
            if len(candidates) >= 50:
                break
        if len(candidates) >= 2:
            break
    if len(candidates) < 2:
        return None
    best = None
    best_dist = float('inf')
    n_nq = mj_model.nq
    # For 30-D qpos, L2 dist > 0.5 always. Use per-dim max instead.
    use_max_per_dim = n_nq > 6
    for i in range(min(15, len(candidates))):
        for j in range(i + 1, min(15, len(candidates))):
            d = float(np.linalg.norm(candidates[i] - candidates[j]))
            if use_max_per_dim:
                d_metric = float(np.max(np.abs(candidates[i] - candidates[j])))
            else:
                d_metric = d
            if d_metric < best_dist and d_metric < max_init_dist:
                best_dist = d_metric
                best = (candidates[i], candidates[j])
    return best


def build_episode_gif(ep, output_path, title, mj_model, mj_data, fps=4, dpi=100):
    n_frames = len(ep["qpos"])
    T = len(ep["dist_qpos"])
    # 2-panel layout: 3D (2-view side by side) + lower row = qpos + action + spike
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.6, 0.7, 1.0], hspace=0.35, wspace=0.25)

    # 3D views (side + iso) for depth perception
    ax_3d_side = fig.add_subplot(gs[0, 0], projection='3d')
    ax_3d_iso = fig.add_subplot(gs[0, 1], projection='3d')
    ax_qpos = fig.add_subplot(gs[1, :])
    ax_action = fig.add_subplot(gs[2, 0])
    ax_spike = fig.add_subplot(gs[2, 1])

    fig.suptitle(f"{title}  |  improved={ep['improved']}  |  "
                 f"init_qpos_dist={ep['init_qpos_dist']:.3f}  |  "
                 f"final_qpos_dist={ep['dist_qpos'][-1]:.3f}", fontsize=11)

    # Static plot: qpos (first 8 dims)
    t = np.arange(T)
    qpos = np.array(ep["qpos"])
    if qpos.shape[0] > 0:
        for i in range(min(8, qpos.shape[1])):
            ax_qpos.plot(t, qpos[:, i], label=f'qpos[{i}]', lw=1.0, alpha=0.8)
    ax_qpos.set_xlabel("step")
    ax_qpos.set_ylabel("Joint angle")
    ax_qpos.set_title(f"R1a  qpos trajectory (first 8 of {qpos.shape[1] if qpos.shape[0] > 0 else '?'} joints)")
    ax_qpos.legend(loc='upper right', fontsize=7, ncol=4)
    ax_qpos.grid(True, alpha=0.3)

    # Static plot: action
    if len(ep["actions"]) > 0:
        actions = np.array(ep["actions"])
        t_act = np.arange(len(actions))
        for i in range(actions.shape[1]):
            ax_action.plot(t_act, actions[:, i], '-o', markersize=2, label=f'act[{i}]')
    ax_action.set_xlabel("step")
    ax_action.set_ylabel("Action value")
    ax_action.set_title(f"R2  Action stream ({actions.shape[1] if len(ep['actions']) > 0 else '?'}D)")
    ax_action.legend(loc='upper right', fontsize=6, ncol=4)
    ax_action.grid(True, alpha=0.3)
    ax_action.axhline(0, color='black', lw=0.5)

    # Spike raster (base, original order, NO sort)
    spike_data = ep.get("spike", [])
    if len(spike_data) > 0:
        spike_arr = np.array(spike_data)
        T_spike = spike_arr.shape[0]
        spike_display = spike_arr.T
        ax_spike.imshow(spike_display, aspect='auto', cmap='gray_r',
                        interpolation='nearest', vmin=0, vmax=1,
                        extent=[0, T_spike - 1, 0, 192], origin='lower')
        sparsity = 1.0 - spike_arr.mean()
        ax_spike.set_xlabel("Time step t")
        ax_spike.set_ylabel("Neuron index (0-191)")
        ax_spike.set_title(f"R3  SNN spike raster (final layer, ORIGINAL order, sp={sparsity:.2f}, FR={spike_arr.mean():.2f})")
        ax_spike.grid(True, alpha=0.3)

    # Init render
    mj_data_local = mujoco.MjData(mj_model)
    mj_data_local.qpos[:] = qpos[0]
    mujoco.mj_forward(mj_model, mj_data_local)
    render_quadruped_3d(ax_3d_side, mj_model, mj_data_local, view='side',
                          title=f"R0  SIDE view at t=0  (init_qpos_dist={ep['init_qpos_dist']:.3f})")
    render_quadruped_3d(ax_3d_iso, mj_model, mj_data_local, view='iso',
                          title="R0  ISO (3D) view at t=0")

    def update(frame):
        mj_data_local.qpos[:] = qpos[frame]
        mujoco.mj_forward(mj_model, mj_data_local)
        # Side view (xz plane, looking along -y) — shows up-down
        render_quadruped_3d(ax_3d_side, mj_model, mj_data_local, view='side',
                              title=f"R0a  SIDE view t={frame}  (qpos_dist={ep['dist_qpos'][frame]:.2f})")
        # ISO view (true 3D) — shows y dimension
        render_quadruped_3d(ax_3d_iso, mj_model, mj_data_local, view='iso',
                              title=f"R0b  ISO (3D) view t={frame}  (qpos_dist={ep['dist_qpos'][frame]:.2f})")
        # Spike update
        if len(spike_data) > 0:
            width = min(frame + 1, spike_arr.shape[0])
            ax_spike.clear()
            ax_spike.imshow(spike_display[:, :width], aspect='auto', cmap='gray_r',
                            interpolation='nearest', vmin=0, vmax=1,
                            extent=[0, max(width, 1), 0, 192], origin='lower')
            ax_spike.axvline(frame, color='#0072B2', lw=1.0, alpha=0.7)
            ax_spike.set_xlim(-0.5, max(spike_arr.shape[0], 1) - 0.5)
            ax_spike.set_ylim(0, 192)
            ax_spike.set_xlabel("Time step t")
            ax_spike.set_ylabel("Neuron index (0-191)")
            sparsity = 1.0 - spike_arr.mean()
            ax_spike.set_title(
                f"R3  SNN spike raster (sp={sparsity:.2f}, t={frame})")
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
    ap.add_argument("--ckpt", default="/home/lx/snn/results/stage34_train/v5/quadruped/final.pt",
                    help="v5 quadruped ckpt path")
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--replan-every", type=int, default=2)
    ap.add_argument("--cem-samples", type=int, default=64)
    ap.add_argument("--cem-elites", type=int, default=8)
    ap.add_argument("--cem-iters", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--max-init-dist", type=float, default=1.5,
                    help="Max per-dim qpos distance (0.5 for manipulator, 1.0 for quadruped)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/home/lx/snn/viz/v5_quadruped_3d/output")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=80)
    args = ap.parse_args()

    print(f"[load] {args.ckpt}")
    model, state_dim, action_dim = load_v5(args.ckpt)
    print(f"  state_dim={state_dim}, action_dim={action_dim}")

    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data = mujoco.MjData(mj_model)
    print(f"mujoco quadruped: nq={mj_model.nq} nu={mj_model.nu} njnt={mj_model.njnt} nbody={mj_model.nbody}")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[run] rolling out {args.n_episodes} episodes...")
    results = []
    # Pre-defined 4 toe positions (quadruped toes are roughly fixed; only qpos changes leg angles)
    toe_positions = [
        np.array([0.74, 0.65, 0.22]),   # toe_front_left
        np.array([0.58, -0.78, 0.16]),  # toe_front_right
        np.array([-0.53, 0.73, 0.18]),  # toe_back_left
        np.array([-0.68, -0.57, 0.19]), # toe_back_right
    ]
    for ep_i in range(args.n_episodes):
        # Pick a random toe and perturb
        base_toe = toe_positions[ep_i % 4].copy()
        target_pos = base_toe + rng.normal(0, 0.05, size=3)
        target_pos[2] = max(0.0, target_pos[2])  # z >= 0
        # Find close init-goal pair (using torso distance as proxy)
        print(f"  ep {ep_i + 1}: target = {target_pos}")
        pair = find_close_init_goal(mj_model, mj_data, rng, target_pos, max_init_dist=args.max_init_dist)
        if pair is None:
            print(f"  ep {ep_i + 1}: no close pair, skipping")
            continue
        init_qpos, goal_qpos = pair
        init_dist = float(np.linalg.norm(init_qpos - goal_qpos))
        mj_data.qpos[:] = init_qpos
        mj_data.qvel[:] = 0
        mujoco.mj_forward(mj_model, mj_data)
        history = run_episode(model, mj_model, mj_data, state_dim, action_dim,
                              init_qpos, goal_qpos, target_pos, args)
        history["ep_idx"] = ep_i
        results.append(history)
        print(f"  ep {ep_i + 1}: init_qpos_dist={init_dist:.3f} "
              f"final_qpos_dist={history['dist_qpos'][-1]:.3f} "
              f"improved={history['improved']}")

    n_improved = sum(1 for h in results if h["improved"])
    print(f"\nFinal: {n_improved}/{len(results)} improved ({n_improved / max(len(results), 1) * 100:.1f}%)")

    if len(results) > 0:
        best = min(results, key=lambda h: h["dist_qpos"][-1])
        worst = max(results, key=lambda h: h["dist_qpos"][-1])
        build_episode_gif(best, out_dir / "v5_quadruped_3d_best.gif",
                          title=f"v5 quadruped BEST (qpos {best['init_qpos_dist']:.2f} -> {best['dist_qpos'][-1]:.2f}, "
                                f"improved={best['improved']})",
                          mj_model=mj_model, mj_data=mj_data,
                          fps=args.fps, dpi=args.dpi)
        if worst is not best:
            build_episode_gif(worst, out_dir / "v5_quadruped_3d_failed.gif",
                              title=f"v5 quadruped WORST (qpos {worst['init_qpos_dist']:.2f} -> {worst['dist_qpos'][-1]:.2f}, "
                                    f"improved={worst['improved']})",
                              mj_model=mj_model, mj_data=mj_data,
                              fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()

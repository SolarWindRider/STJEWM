"""Multi-panel GIF renderer for STJEWM.

3 panels per frame:
  1. TOP: Env state (3D mujoco for DMC, 2D for PushT/TwoRoom)
  2. MIDDLE: Spike raster (ALL 192 neurons over time, with frame markers)
  3. BOTTOM: Action values heatmap (with frame markers)

Adds visual indicators:
  - Title bar with SUCCESS/FAILURE label (red border for failure, green for success)
  - Goal indicator (green dotted line) in env panel
  - Current step marked in spike/action panels
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def _ensure_agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_env_panel(ax, state, goal_state, env_id, env=None, current_step=None):
    """Render the current env state. Goal state shown as a marker (green for block, cyan for agent)."""
    if env is not None and hasattr(env, "_model") and hasattr(env, "_data"):
        # 3D mujoco render
        try:
            os.environ.setdefault("MUJOCO_GL", "egl")
            from code.core.viz.render_3d import render_env_frame
            import mujoco as _mj
            nq = env._model.nq
            if state.shape[0] >= nq:
                env._data.qpos[:nq] = state[:nq]
                env._data.qvel[:] = 0
                _mj.mj_forward(env._model, env._data)
            # Set goal by saving current, then putting goal in, rendering, restoring
            saved_qpos = env._data.qpos[:].copy()
            if goal_state.shape[0] >= nq:
                env._data.qpos[:nq] = goal_state[:nq]
                env._data.qvel[:] = 0
                _mj.mj_forward(env._model, env._data)
            img_goal = render_env_frame(env._model, env._data, width=320, height=240, camera_distance=2.5)
            # Restore
            env._data.qpos[:nq] = saved_qpos
            env._data.qvel[:] = 0
            _mj.mj_forward(env._model, env._data)
            img_current = render_env_frame(env._model, env._data, width=320, height=240, camera_distance=2.5)
            # Side-by-side composite
            ax.clear()
            ax.imshow(np.hstack([img_current, img_goal]), aspect='auto')
            ax.set_xticks([80, 400])
            ax.set_xticklabels(['current', 'goal'])
            ax.set_yticks([])
            ax.set_title(f"Env: {env_id} (3D mujoco render) | LEFT: current step, RIGHT: goal", fontsize=9)
            return
        except Exception as e:
            pass

    import matplotlib.patches as patches
    if "tworoom" in env_id.lower():
        ax.clear()
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_aspect('equal')
        ax.set_title(f"Env: {env_id} (TwoRoom) | current (blue) / goal (cyan)")
        ax.add_patch(patches.Rectangle((0, 0), 2, 5, fill=True, facecolor='lightyellow', edgecolor='black'))
        ax.add_patch(patches.Rectangle((3, 0), 2, 5, fill=True, facecolor='lightblue', edgecolor='black'))
        ax.add_patch(patches.Rectangle((1.9, 2), 1.2, 1, fill=True, facecolor='gray'))
        if len(state) >= 2:
            ax.plot(state[0], state[1], 'bo', markersize=15, label='agent (current)')
        if goal_state is not None and len(goal_state) >= 2:
            ax.plot(goal_state[0], goal_state[1], 'c*', markersize=20, label='agent (goal)')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    elif "pusht" in env_id.lower():
        # PushT state layout: [agent_x, agent_y, block_x, block_y, block_angle, block_vx, block_vy]
        ax.clear()
        # Determine plot bounds from trajectory
        all_x = [state[0], state[2]]
        all_y = [state[1], state[3]]
        if goal_state is not None:
            all_x += [goal_state[0], goal_state[2]]
            all_y += [goal_state[1], goal_state[3]]
        margin = 50
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.set_aspect('equal')
        ax.set_title(f"Env: {env_id} (PushT) | current (blue/orange) / goal (cyan/green)")
        # agent (current)
        ax.plot(state[0], state[1], 'bo', markersize=12, label='agent (current)')
        # block (current) - draw as rotated rectangle
        from matplotlib.transforms import Affine2D
        block_x, block_y, block_angle = state[2], state[3], state[4]
        block_size = 30  # rough size
        rect = patches.Rectangle((block_x - block_size/2, block_y - block_size/2),
                                  block_size, block_size, angle=np.degrees(block_angle),
                                  fill=True, facecolor='orange', edgecolor='darkred', label='block (current)')
        ax.add_patch(rect)
        # goal
        if goal_state is not None:
            ax.plot(goal_state[0], goal_state[1], 'c*', markersize=20, label='agent (goal)')
            gblock = patches.Rectangle((goal_state[2] - block_size/2, goal_state[3] - block_size/2),
                                       block_size, block_size, fill=False, edgecolor='green', linewidth=2,
                                       linestyle='--', label='block (goal)')
            ax.add_patch(gblock)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    else:
        ax.clear()
        ax.bar(range(len(state)), state, label='current', alpha=0.7)
        if goal_state is not None and len(goal_state) == len(state):
            ax.bar(range(len(goal_state)), goal_state, label='goal', alpha=0.5)
        ax.set_title(f"Env: {env_id} (state)")
        ax.set_xlabel("state dim")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


def render_spike_panel(ax, spike_history, current_spike, max_steps=30, current_step=None, total_steps=None):
    """Spike raster showing ALL neurons over time. Right edge = current step."""
    ax.clear()
    n_neurons = current_spike.shape[0]

    # Build the full raster (T x N)
    if spike_history:
        history_arr = np.stack(spike_history, axis=0)  # (T, n_neurons)
    else:
        history_arr = np.zeros((0, n_neurons), dtype=np.float32)
    full_arr = np.vstack([history_arr, current_spike[None, :]])  # (T+1, n_neurons)
    T = full_arr.shape[0]

    im = ax.imshow(full_arr.T, aspect='auto', cmap='gray_r',
                   vmin=0, vmax=1, interpolation='nearest', extent=[0, T, 0, n_neurons])
    # Mark current step with red vertical line
    ax.axvline(x=T - 0.5, color='red', linewidth=1.5, linestyle='--', label='current')
    ax.set_xlabel("time step (most recent on right)")
    ax.set_ylabel(f"neuron idx (0-{n_neurons-1})")
    sparsity = 1.0 - current_spike.mean()
    n_firing = int(current_spike.sum())
    title = f"Spike raster (sparsity={sparsity:.2f}, {n_firing}/{n_neurons} firing this step)"
    if total_steps is not None and current_step is not None:
        title = f"[{current_step+1}/{total_steps}] " + title
    ax.set_title(title, fontsize=9)
    if T < 10:
        ax.set_xticks(range(T))


def render_action_panel(ax, action_history, current_action, max_steps=30, current_step=None, total_steps=None):
    """Action values as heatmap. Right edge = current step."""
    ax.clear()
    A = current_action.shape[0]

    history = list(action_history) + [current_action]
    if len(history) == 0:
        return
    history_arr = np.stack(history, axis=0)
    T = history_arr.shape[0]

    vmax = max(np.abs(history_arr).max(), 0.01)
    im = ax.imshow(history_arr.T, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest', extent=[0, T, 0, A])
    ax.axvline(x=T - 0.5, color='red', linewidth=1.5, linestyle='--', label='current')
    ax.set_xlabel("time step (most recent on right)")
    ax.set_ylabel(f"action dim (0-{A-1})")
    title = f"Action values (range [{history_arr.min():.2f}, {history_arr.max():.2f}])"
    if total_steps is not None and current_step is not None:
        title = f"[{current_step+1}/{total_steps}] " + title
    ax.set_title(title, fontsize=9)
    if T < 10:
        ax.set_xticks(range(T))


def render_stjewm_gif(
    traj: dict,
    env,
    output_path: str,
    fps: int = 6,
    dpi: int = 100,
    max_steps_display: int = 30,
    is_success: bool = True,
    cos_dist: float = 0.0,
    threshold: float = 0.1,
):
    """Render a multi-panel GIF showing env state + spike raster + action.

    Args:
        traj: dict with 'states', 'actions', 'spikes', 'init_state', 'goal_state'
        env: env (for mujoco rendering)
        output_path: where to save the .gif
        is_success: if True, add green border; if False, add red border
        cos_dist: final cos_dist (for annotation)
        threshold: success threshold (typically 0.1)
    """
    plt = _ensure_agg()
    from PIL import Image
    import imageio.v2 as imageio

    states = traj["states"]
    actions = traj["actions"]
    spikes = traj.get("spikes", None)
    goal_state = traj.get("goal_state", None)
    init_state = traj.get("init_state", None)

    n_states = len(states)
    n_actions = len(actions)
    n_spikes = len(spikes) if spikes is not None else 0

    if n_states == 0:
        return

    if spikes is None or len(spikes) == 0:
        from code.core.viz.render_2d import render_state_trajectory
        render_state_trajectory(traj, env, output_path, fps=fps, dpi=dpi)
        return

    # Pad spikes: states has init + 1 per action
    if n_spikes < n_states:
        pad_spike = np.zeros_like(spikes[0]) if spikes else np.zeros(192, dtype=np.float32)
        padded_spikes = [pad_spike] + list(spikes) + [pad_spike] * (n_states - n_spikes - 1)
    else:
        padded_spikes = spikes[:n_states]

    env_id = getattr(env, "spec", None)
    env_id = env_id.env_id if env_id is not None else "env"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        frame_paths = []
        for frame_idx in range(n_states):
            current_state = states[frame_idx]
            current_spike = padded_spikes[frame_idx]
            if frame_idx == 0:
                current_action = np.zeros_like(actions[0]) if n_actions > 0 else np.zeros(2, dtype=np.float32)
            else:
                current_action = actions[frame_idx - 1] if frame_idx - 1 < n_actions else np.zeros_like(actions[0])

            # Histories (last max_steps frames)
            spike_history = list(padded_spikes[max(0, frame_idx - max_steps_display + 1) : frame_idx + 1])
            action_history = list(actions[max(0, frame_idx - max_steps_display) : frame_idx])

            fig, axes = plt.subplots(3, 1, figsize=(8, 9), dpi=dpi,
                                      gridspec_kw={"height_ratios": [2, 2, 1.5]})

            render_env_panel(axes[0], current_state, goal_state, env_id, env=env)
            render_spike_panel(axes[1], spike_history, current_spike, max_steps=max_steps_display)
            render_action_panel(axes[2], action_history, current_action, max_steps=max_steps_display)
            # Color-coded title
            color = 'green' if is_success else 'red'
            status = "✓ SUCCESS" if is_success else "✗ FAILURE"
            fig.suptitle(
                f"{status}  |  cos_dist={cos_dist:.3f} (threshold {threshold})  |  Frame {frame_idx+1}/{n_states}",
                fontsize=12, fontweight='bold', color=color,
            )

            p = td / f"frame_{frame_idx:04d}.png"
            plt.tight_layout()
            plt.savefig(p, dpi=dpi, bbox_inches='tight')
            frame_paths.append(str(p))
            plt.close(fig)

        if not frame_paths:
            return

        with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
            for p in frame_paths:
                writer.append_data(imageio.imread(p))

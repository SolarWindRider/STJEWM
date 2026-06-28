"""2D rendering: matplotlib-based plots of state trajectories, Reacher joints, etc."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np


def _ensure_agg():
    """Use headless backend for matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_reacher_gif(
    traj: dict,
    env,                 # ReacherEnv
    output_path: str,
    fps: int = 6,
    dpi: int = 100,
) -> None:
    """Render a 2D Reacher trajectory as a GIF.

    Each frame shows: the arm (line from origin to fingertip), the target,
    and a trajectory trace.
    """
    plt = _ensure_agg()
    import matplotlib.patches as patches

    states = np.asarray(traj["states"])  # (T+1, 4) = [qpos(2), target(2)]
    actions = np.asarray(traj["actions"])

    if states.ndim != 2 or states.shape[1] != 4:
        # Fallback: just render the qpos trajectory
        render_state_trajectory(traj, env, output_path, fps=fps, dpi=dpi)
        return

    # Build frames
    import mujoco
    n_frames = len(states)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    frames = []
    qpos_history = [states[0, :2]]
    for t in range(1, n_frames):
        qpos_history.append(states[t, :2])
    for frame_idx in range(n_frames):
        ax.clear()
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Reacher (frame {frame_idx}/{n_frames-1})\n"
                     f"cos_dist={traj['cos_dist']:.3f}, env_success={traj['env_success']}")
        # Arm: 2 joints, link lengths 0.1 m each (DMC reacher)
        link_len = 0.1
        q0, q1 = states[frame_idx, 0], states[frame_idx, 1]
        # Forward kinematics
        p0 = (0.0, 0.0)
        p1 = (link_len * np.cos(q0), link_len * np.sin(q0))
        p2 = (p1[0] + link_len * np.cos(q0 + q1), p1[1] + link_len * np.sin(q0 + q1))
        # Draw arm
        ax.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], "b-o", linewidth=2, markersize=6)
        # Draw target
        target = states[frame_idx, 2:4]
        ax.plot(target[0], target[1], "r*", markersize=15, label="target")
        # Draw goal
        goal = traj["goal_state"][2:4]
        ax.plot(goal[0], goal[1], "g+", markersize=15, label="goal")
        # Draw trajectory trace
        if len(qpos_history) > 0 and frame_idx > 0:
            history_q = np.array(qpos_history[: frame_idx + 1])
            # Compute fingertip history
            trace_x, trace_y = [], []
            for q in history_q:
                x = link_len * np.cos(q[0]) + link_len * np.cos(q[0] + q[1])
                y = link_len * np.sin(q[0]) + link_len * np.sin(q[0] + q[1])
                trace_x.append(x)
                trace_y.append(y)
            ax.plot(trace_x, trace_y, "b--", alpha=0.5, linewidth=1)
        ax.legend(loc="upper right", fontsize=8)
        plt.savefig(f"/tmp/_frame_{os.getpid()}.png", dpi=dpi)
        frames.append(f"/tmp/_frame_{os.getpid()}.png")
        plt.pause(0.001)

    # Save as GIF
    import imageio.v2 as imageio
    images = [imageio.imread(f) for f in frames]
    # All frames have the same name; use unique tempdir
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        frame_paths = []
        for i, img in enumerate(images):
            p = Path(td) / f"frame_{i:04d}.png"
            from PIL import Image
            Image.fromarray(img).save(p)
            frame_paths.append(str(p))
        with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
            for p in frame_paths:
                writer.append_data(imageio.imread(p))
    # Cleanup the single-frame file
    for f in set(frames):
        try:
            os.remove(f)
        except Exception:
            pass


def render_state_trajectory(
    traj: dict,
    env,
    output_path: str,
    fps: int = 6,
    dpi: int = 100,
) -> None:
    """Generic fallback: plot state components over time as a GIF.

    Each frame shows the current state values as a bar chart.
    """
    plt = _ensure_agg()
    states = np.asarray(traj["states"])
    n_frames = len(states)
    if n_frames == 0:
        return
    state_dim = states.shape[1]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    images = []
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        for frame_idx in range(n_frames):
            ax.clear()
            ax.bar(range(state_dim), states[frame_idx])
            ax.set_title(f"{env.spec.env_id} (frame {frame_idx}/{n_frames-1})")
            ax.set_xlabel("state dim")
            ax.set_ylabel("value")
            p = Path(td) / f"frame_{frame_idx:04d}.png"
            plt.savefig(p, dpi=dpi)
            images.append(str(p))
        import imageio.v2 as imageio
        with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
            for p in images:
                writer.append_data(imageio.imread(p))

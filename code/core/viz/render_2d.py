"""2D rendering: matplotlib-based plots of state trajectories, Reacher joints, etc."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _ensure_agg():
    """Use headless backend for matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_reacher_gif(
    traj: dict,
    env,
    output_path: str,
    fps: int = 6,
    dpi: int = 100,
) -> None:
    """Render distinct Reacher frames without intermediate image files."""
    plt = _ensure_agg()
    import imageio.v2 as imageio

    states = np.asarray(traj["states"])
    if states.ndim != 2 or states.shape[1] != 4:
        render_state_trajectory(traj, env, output_path, fps=fps, dpi=dpi)
        return

    n_frames = len(states)
    link_len = 0.1
    q0, q1 = states[:, 0], states[:, 1]
    joint_x, joint_y = link_len * np.cos(q0), link_len * np.sin(q0)
    tip_x = joint_x + link_len * np.cos(q0 + q1)
    tip_y = joint_y + link_len * np.sin(q0 + q1)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    try:
        with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
            for frame_idx in range(n_frames):
                ax.clear()
                ax.set_xlim(-0.3, 0.3)
                ax.set_ylim(-0.3, 0.3)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)
                ax.set_title(f"Reacher (frame {frame_idx}/{n_frames-1})\n"
                             f"cos_dist={traj['cos_dist']:.3f}, env_success={traj['env_success']}")
                ax.plot(
                    [0.0, joint_x[frame_idx], tip_x[frame_idx]],
                    [0.0, joint_y[frame_idx], tip_y[frame_idx]],
                    "b-o", linewidth=2, markersize=6,
                )
                target = states[frame_idx, 2:4]
                ax.plot(target[0], target[1], "r*", markersize=15, label="target")
                goal = traj["goal_state"][2:4]
                ax.plot(goal[0], goal[1], "g+", markersize=15, label="goal")
                if frame_idx > 0:
                    ax.plot(tip_x[:frame_idx + 1], tip_y[:frame_idx + 1],
                            "b--", alpha=0.5, linewidth=1)
                ax.legend(loc="upper right", fontsize=8)
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba()).copy())
    finally:
        plt.close(fig)


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

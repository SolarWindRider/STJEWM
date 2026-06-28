"""3D rendering: mujoco-based rendering of arm/manipulator trajectories."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _ensure_egl():
    os.environ.setdefault("MUJOCO_GL", "egl")


def render_manipulator_gif(
    traj: dict,
    env,                 # env with mujoco model
    output_path: str,
    fps: int = 6,
    dpi: int = 100,
    camera_distance: float = 1.5,
) -> None:
    """Render a Manipulator (DMC XML) trajectory as a 3D GIF.

    The env should have a mujoco model with body 'fingertip', 'ball', 'target_ball'.
    We render via off-screen mujoco and save frames.
    """
    _ensure_egl()
    import mujoco
    import numpy as np

    # We need to reconstruct the env's mujoco model from scratch (since the
    # trajectory is just states+actions, not a full env history)
    # For simplicity, we use the env's existing model and re-apply the trajectory
    # state by state.
    model = getattr(env, "_model", None)
    data = getattr(env, "_data", None)
    if model is None:
        # Fallback: use a 2D plot
        from code.core.viz.render_2d import render_state_trajectory
        render_state_trajectory(traj, env, output_path, fps=fps, dpi=dpi)
        return

    # Render each state
    import tempfile
    from PIL import Image
    import imageio.v2 as imageio

    n_frames = len(traj["states"])
    with tempfile.TemporaryDirectory() as td:
        frame_paths = []
        for frame_idx in range(n_frames):
            # Try to set the env to this state (best-effort)
            try:
                state = traj["states"][frame_idx]
                if isinstance(env, type(env)) and state.shape[0] >= 4:
                    env._data.qpos[:] = state[: env._model.nq]
                    mujoco.mj_forward(env._model, env._data)
            except Exception:
                pass
            try:
                img = render_env_frame(model, data, camera_distance=camera_distance)
                p = Path(td) / f"frame_{frame_idx:04d}.png"
                Image.fromarray(img).save(p)
                frame_paths.append(str(p))
            except Exception:
                continue

        if not frame_paths:
            return
        with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
            for p in frame_paths:
                writer.append_data(imageio.imread(p))


def render_env_frame(model, data, width: int = 320, height: int = 240, camera_distance: float = 1.5) -> "np.ndarray":
    """Render a single mujoco frame. Returns RGB array."""
    import mujoco
    import numpy as np
    try:
        renderer = mujoco.Renderer(model, width=width, height=height)
        # Default camera
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance = camera_distance
        cam.elevation = -20
        cam.azimuth = 45
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        renderer.close()
        return img
    except Exception:
        # Fallback: create a basic RGB
        return np.full((height, width, 3), 200, dtype=np.uint8)

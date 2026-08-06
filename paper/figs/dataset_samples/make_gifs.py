"""Generate animated GIFs — full motion over 50 random-action steps per env.

Static PNGs in the parent directory stay untouched; GIFs go to `gifs/<env>.gif`.

Camera strategy per frame:
  - quadruped: orbit close-up (same as the static figure) with per-frame
    lookat tracking of the body centroid; rangefinder beams hidden.
  - other envs: prefer a camera whose name contains track/lookat/overview,
    else the first trackcom/track/targetbody camera (auto-follows the body),
    else camera 0 — so moving subjects stay in frame.

Usage:
    MUJOCO_GL=egl python make_gifs.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_samples as g  # noqa: E402  (reuses _build_env / ENV_META)

OUT_DIR = Path(__file__).resolve().parent / "gifs"
N_STEPS = 50       # same random-action horizon as the static samples
FPS = 10           # frames per second
FRAME_H, FRAME_W = 180, 240   # 4:3; smaller than the static 320x240 panels


def _pick_cam_id(env) -> int:
    """Prefer track/lookat/overview cameras, then any auto-following camera."""
    model = env._model
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name and ("track" in name.lower() or "lookat" in name.lower()
                     or "overview" in name.lower()):
            return i
    for i in range(model.ncam):
        if model.cam(i).mode in (mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
                                 mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
                                 mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY):
            return i
    return 0


def _quad_cam(env) -> mujoco.MjvCamera:
    """Orbit close-up aimed at the body centroid (decorative ball excluded)."""
    model, data = env._model, env._data
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(1, model.nbody)]
    body = data.xpos[1:][np.array([n != "ball" for n in names])]
    centroid = body.mean(axis=0)
    radius = float(np.max(np.linalg.norm(body - centroid, axis=1)))
    cam.lookat[:] = centroid
    cam.distance = max(1.5, min(3.5, 2.2 * radius))
    cam.azimuth = 45.0
    cam.elevation = -20.0
    return cam


def _render_frame(env, renderer) -> np.ndarray:
    """Render the current scene to an RGB frame."""
    env_id = str(getattr(env.spec, "env_id", ""))
    if "quadruped" in env_id:
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        renderer.update_scene(env._data, camera=_quad_cam(env), scene_option=opt)
    else:
        renderer.update_scene(env._data, camera=_pick_cam_id(env))
    return renderer.render()


def _render_2d_frame(state: np.ndarray, panel_fn, title: str) -> np.ndarray:
    """Render a 2D (PushT / TwoRoom) state via the matplotlib panel fn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(FRAME_W / 60, FRAME_H / 60), dpi=60)
    panel_fn(state, ax, title)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


_2D_PANELS = {
    "pusht": g._render_pusht_panel,
    "tworoom": g._render_tworoom_panel,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for env_kind, _meta in g.ENV_META.items():
        env = g._build_env(env_kind)
        rng = np.random.default_rng(42)  # same seed as static samples
        env.reset(seed=0)
        if hasattr(env, "_model"):
            renderer = mujoco.Renderer(env._model, height=FRAME_H, width=FRAME_W)
            frames = [_render_frame(env, renderer)]
            for _ in range(N_STEPS):
                a = rng.uniform(env.spec.action_low, env.spec.action_high).astype(np.float32)
                _obs, _r, done, _info = env.step(a)
                frames.append(_render_frame(env, renderer))
                if done:
                    break
            renderer.close()
        else:
            panel_fn = _2D_PANELS[env_kind]
            frames = [_render_2d_frame(env.get_state(), panel_fn, env_kind)]
            for _ in range(N_STEPS):
                a = rng.uniform(env.spec.action_low, env.spec.action_high).astype(np.float32)
                _obs, _r, done, _info = env.step(a)
                frames.append(_render_2d_frame(env.get_state(), panel_fn, env_kind))
                if done:
                    break
        out = OUT_DIR / f"{env_kind}.gif"
        frames = [Image.fromarray(f) for f in frames]
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=1000 // FPS, loop=0)
        print(f"[gif] {env_kind}: {len(frames)} frames -> {out} "
              f"({out.stat().st_size // 1024} KB)", flush=True)
        env.close()


if __name__ == "__main__":
    main()

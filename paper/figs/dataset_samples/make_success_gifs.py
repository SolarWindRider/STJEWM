"""Generate SUCCESS GIFs — episodes where the trained model + CEM closed-loop
control actually completes the task (env.check_success == True).

The random-policy GIFs (make_gifs.py) never succeed by construction; these
use the 5M-aligned checkpoints whose closed-loop env-SR is high, and replay
the recorded mujoco states frame-by-frame so the animation is an exact
reproduction of the real episode.

Static PNGs and `gifs/<env>.gif` are untouched; outputs go to
`gifs/success/<env>.gif`.

Targets (env-SR from the 5M closed-loop evals, results/5m_5mpar):
  cartpole / cheetah / finger : cross_benchmark_F2 / stjewm_no_trace   (1.00)
  pendulum                    : oodc_F1F3 / stjewm_spike_only         (0.60)
  hopper                      : generalist_16env / stjewm_trace_only  (0.40)
  pusht                       : expert demonstration (PushT h5)

ball_in_cup is deliberately excluded: its 250k rollout data is degenerate
(obs std ~0.01; the cup/ball barely move), so no meaningful catch animation
can be produced from it.

Usage:
    MUJOCO_GL=egl python make_success_gifs.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be on sys.path BEFORE `import torch`: torch internally imports the
# stdlib `code` module; if that resolves to stdlib code.py (a non-package) it
# is cached in sys.modules and the later `from code.core.cem import ...` fails
# with "code is not a package". Inserting the repo first makes `code` resolve
# to the project package (harmless: __init__.py is docstring-only).
REPO = Path("/home/lx/snn")
sys.path.insert(0, str(REPO))

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import mujoco
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from code.core.cem import CEM                       # noqa: E402
from code.core.encode import encode_obs, encode_history  # noqa: E402
from code.data.loaders import load_dmc              # noqa: E402
from code.eval.closed_loop import _set_env_state, _PadObsWrapper  # noqa: E402
from make_gifs import _pick_cam_id, _render_frame   # noqa: E402  (camera policy)

OUT_DIR = Path(__file__).resolve().parent / "gifs" / "success"
PAD_OBS, ACTION_DIM = 128, 56
HORIZON, BUDGET = 5, 50
N_EPISODES_TRIED = 60   # canonical goals are harder than dataset goals
PENDULUM_MAX_ANGLE = 0.25  # rad (~14 deg): pendulum must reach near-vertical
FPS = 10
FRAME_H, FRAME_W = 180, 240
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# env_kind -> (ckpt split, model dir, readout mode, canonical success goal or
# None, min init-goal distance)
# Canonical goals make the success visually obvious (pole upright / pendulum
# straight up / ball in cup); None = sample the goal from the dataset.
TARGETS = {
    "cartpole":    ("cross_benchmark_F2", "stjewm_no_trace", "no_trace",
                     np.array([0.0, 0.0]), 0.0),         # cart at 0, pole vertical
    "cheetah":     ("cross_benchmark_F2", "stjewm_no_trace", "no_trace", None, 0.0),
    "finger":      ("cross_benchmark_F2", "stjewm_no_trace", "no_trace", None, 0.0),
    "pendulum":    ("oodc_F1F3",          "stjewm_spike_only", "spike_only",
                     np.array([-1.0, 0.0]), 0.0),        # cos(pi)=-1: straight UP (DMC theta=0 is hanging down)
    "hopper":      ("generalist_16env",   "stjewm_trace_only", "trace_only", None, 0.0),
}

NPZ = {
    "cartpole":    REPO / "data/dm_control/cartpole_250k.npz",
    "pendulum":    REPO / "data/dm_control/pendulum_250k.npz",
    "ball_in_cup": REPO / "data/dm_control/3d_rollouts_250k/ball_in_cup_250k.npz",
    "cheetah":     REPO / "data/dm_control/3d_rollouts_250k/cheetah_250k.npz",
    "finger":      REPO / "data/dm_control/3d_rollouts_250k/finger_250k.npz",
    "hopper":      REPO / "data/dm_control/3d_rollouts_250k/hopper_250k.npz",
}


def build_model(ckpt_path: Path, readout_mode: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from code.stjewm import STJEWM
    # State checkpoints were trained with image_size=84 (train.py maps 0 -> 84),
    # so the frozen ViT's position embeddings expect 1 + (84/14)^2 = 37 tokens.
    image_size = int(ck.get("args", {}).get("image_size") or 84)
    model = STJEWM(
        d_hid=192, embed_dim=192, action_dim=ACTION_DIM, action_emb_dim=192,
        state_dim=PAD_OBS, cell_n_layers=4, n_d=3,
        trace_beta=0.9, freeze_encoder=True, image_size=image_size,
        readout_mode=readout_mode,
    ).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def run_episode(env, ds, model, rng, cem, goal_override=None,
                min_init_dist: float = 0.0) -> tuple[list[np.ndarray], bool]:
    """One closed-loop episode (same protocol as closed_loop.py); records the
    real mujoco state (qpos/qvel) after every env step.

    goal_override: canonical success state (e.g. pole upright) instead of the
    dataset-sampled goal. min_init_dist > 0: keep resampling the init window
    until it is at least this far (native state space) from the goal, so the
    animation shows real motion instead of a near-goal start."""
    item = ds[int(rng.integers(len(ds)))]
    init_state_np = np.asarray(item["init_state"])
    if min_init_dist > 0:
        goal_probe = (np.asarray(goal_override, dtype=np.float32)
                      if goal_override is not None
                      else np.asarray(item["goal_state"]))
        if goal_probe.shape[-1] < PAD_OBS:
            goal_probe = np.concatenate(
                [goal_probe, np.zeros(PAD_OBS - goal_probe.shape[-1], np.float32)])
        for _ in range(500):
            d = float(np.linalg.norm(init_state_np - goal_probe))
            if d >= min_init_dist:
                break
            item = ds[int(rng.integers(len(ds)))]
            init_state_np = np.asarray(item["init_state"])
    goal_state_np = (np.asarray(goal_override, dtype=np.float32)
                     if goal_override is not None
                     else np.asarray(item["goal_state"]))
    if goal_state_np.shape[-1] < PAD_OBS:  # canonical goals are native-dim
        goal_state_np = np.concatenate(
            [goal_state_np, np.zeros(PAD_OBS - goal_state_np.shape[-1], np.float32)])
    env.reset(seed=0)
    _set_env_state(env, init_state_np)
    base = env._base if hasattr(env, "_base") else env  # unwrap pad wrapper
    data = base._data

    z_history = encode_history(model, [torch.from_numpy(init_state_np).float()] * 1,
                               ACTION_DIM, DEVICE)
    z_init = z_history[-1]
    z_goal = encode_obs(model, torch.from_numpy(goal_state_np).float(), ACTION_DIM, DEVICE)

    states = [data.qpos.copy(), data.qvel.copy()]
    taken = 0
    done = False
    while taken < BUDGET and not done:
        seq = cem.plan(z_init, z_goal)  # (H, A)
        for a_idx in range(min(HORIZON, BUDGET - taken)):
            a = seq[a_idx].cpu().numpy().astype(np.float32)[: env.spec.action_dim]
            a = np.clip(a, env.spec.action_low, env.spec.action_high)
            _obs, _r, done, _info = env.step(a)
            states.append(data.qpos.copy())
            states.append(data.qvel.copy())
            taken += 1
            if done:
                break
        if done or taken >= BUDGET:
            break
        with torch.no_grad():
            a_window = seq[:1].unsqueeze(0)
            nxt = model.predict(z_history.unsqueeze(0), a_window)
            z_history = torch.cat([z_history[1:], nxt[0:1, -1]], dim=0)
            z_init = z_history[-1]

    final_state_np = env.get_state()
    ok = bool(env.check_success(final_state_np, goal_state_np)[0])
    if ok and goal_override is not None and getattr(env, "_expand_pendulum", False):
        # Stricter than the env tol (0.5 rad): the pendulum must actually swing
        # to near-vertical so the animation reads as a successful swing-up.
        ca, sa = final_state_np[0], final_state_np[1]
        cg, sg = goal_state_np[0], goal_state_np[1]
        ang = float(np.arccos(np.clip(ca * cg + sa * sg, -1.0, 1.0)))
        ok = ang < PENDULUM_MAX_ANGLE
    return states, ok


def render_states(env, states) -> list[np.ndarray]:
    """Frame-by-frame replay of the recorded (qpos, qvel) sequence."""
    model, data = env._model, env._data
    renderer = mujoco.Renderer(model, height=FRAME_H, width=FRAME_W)
    frames = []
    for i in range(0, len(states), 2):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = states[i]
        data.qvel[:] = states[i + 1]
        mujoco.mj_forward(model, data)
        frames.append(_render_frame(env, renderer))
    renderer.close()
    return frames


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for env_kind, (split, model_dir, readout, goal, min_d) in TARGETS.items():
        ckpt = REPO / f"results/5m_5mpar/{split}/{model_dir}/seed_0/final.pt"
        if not ckpt.exists():
            print(f"[skip] {env_kind}: missing {ckpt}", flush=True)
            continue
        model = build_model(ckpt, readout)
        base = make_dmc_env(env_kind)
        env = _PadObsWrapper(base, PAD_OBS)
        ds = load_dmc(str(NPZ[env_kind]), history_size=1, goal_offset=25,
                      pad_obs_to=PAD_OBS, env_id=env.spec.env_id)
        cem = CEM(model, action_dim=ACTION_DIM, horizon=HORIZON,
                  n_samples=300, n_elites=30, n_iters=10,
                  history_size=1, device=DEVICE)
        rng = np.random.default_rng(7)
        states = success = None
        for trial in range(N_EPISODES_TRIED):
            states, success = run_episode(env, ds, model, rng, cem, goal, min_d)
            if success:
                print(f"[ok]   {env_kind}: success on trial {trial} "
                      f"({len(states) // 2} steps)", flush=True)
                break
        if not success:
            print(f"[fail] {env_kind}: no success in {N_EPISODES_TRIED} trials "
                  f"(model env-SR was expected >= 0.4)", flush=True)
            base.close()
            continue
        frames = render_states(base, states)
        out = OUT_DIR / f"{env_kind}.gif"
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:],
                     duration=1000 // FPS, loop=0)
        print(f"[gif]  {env_kind}: {len(imgs)} frames -> {out} "
              f"({out.stat().st_size // 1024} KB)", flush=True)
        base.close()
    render_pusht_success()


def make_dmc_env(env_kind: str):
    from code.core.envs.dmc_env import make_dmc_env as _m
    return _m(env_kind)


def render_pusht_success():
    """PushT expert demonstration (all episodes are successful by construction)."""
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_gifs import _render_2d_frame
    with h5py.File("/home/lx/LeWM/data/pusht_expert_train.h5", "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        states = f["state"][:]
        # longest episode: most motion to show
        i = int(np.argmax(ep_len))
        s0, n = int(ep_offset[i]), int(ep_len[i])
        traj = states[s0 : s0 + n]
    from generate_samples import _render_pusht_panel
    frames = [_render_2d_frame(t, _render_pusht_panel, "pusht") for t in traj]
    out = OUT_DIR / "pusht.gif"
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=1000 // FPS, loop=0)
    print(f"[gif]  pusht: {len(imgs)} frames (expert) -> {out} "
          f"({out.stat().st_size // 1024} KB)", flush=True)


if __name__ == "__main__":
    main()

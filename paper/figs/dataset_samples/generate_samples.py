"""Generate per-env sample observation figures for the report.

For each of the 16 envs used in the generalist_16env.json config:
  - Reset with seed=0, take 50 random steps, capture the state
  - Render a 3-panel figure:
      Top:    physical scene (mujoco offscreen render for DMC/Reacher;
              2D scatter for PushT/TwoRoom)
      Middle: bar chart of the obs vector with per-dim labels
      Bottom: caption (task description, obs_dim, action_dim, sample
              numerical state)

Outputs:
  paper/figs/dataset_samples/<env>.png             (per-env, 800x600)
  paper/figs/dataset_samples/all_envs_overview.png (4x4 grid, 1600x1600)
  paper/figs/dataset_samples/obs_samples.json      (numerical obs dumps)
"""
from __future__ import annotations

import json
import os
# Must be set BEFORE importing mujoco / mujoco-using packages
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
from pathlib import Path

# Ensure EGL is used before any mujoco import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
# Register the CJK font so Chinese captions render
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path("/home/lx/snn")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import mujoco  # must be imported before any module that imports it transitively
from matplotlib.gridspec import GridSpec

from code.core.envs import (
    make_dmc_env,
    make_reacher_env,
    PushTEnv,
    TwoRoomEnv,
    DMC_ENVS,
)


# ============================================================
# Per-env metadata
# ============================================================
# Chinese 1-line task description + per-dim labels.
# Order matches configs/generalist_16env.json:
#   13 standard DMC (no manipulator) + reacher + pusht + tworoom = 16

ENV_META: dict[str, dict] = {
    # ------ 13 standard DMC envs ------
    "cartpole": {
        "label_zh": "Cartpole — 推车平衡倒立摆",
        "task_zh": "在水平滑轨上推车,顶部通过铰链连接一根竖直摆杆;"
                   "agent 沿 x 方向施加力使摆杆保持直立。",
        "kind": "dmc",
        "dim_labels": ["cart_x", "pole_angle"],
    },
    "pendulum": {
        "label_zh": "Pendulum — 钟摆起摆",
        "task_zh": "单自由度的钟摆从下垂姿态起摆;"
                   "agent 通过施加力矩使其摆至并稳定在竖直向上。",
        "kind": "dmc",
        "dim_labels": ["cos(theta)", "sin(theta)"],
    },
    "finger": {
        "label_zh": "Finger — 双指拨动旋拧",
        "task_zh": "两只手指夹持一个可旋转的指轮,"
                   "agent 通过 2D 力控制指尖位置,目标是把指轮旋到指定姿态。",
        "kind": "dmc",
        "dim_labels": [f"qpos[{i}]" for i in range(3)],
    },
    "ball_in_cup": {
        "label_zh": "Ball-in-Cup — 杯接小球",
        "task_zh": "杯子挂在一根水平铰链摆杆末端,小球用一根绳挂在支架上;"
                   "agent 通过 2D 力使杯子摆动接住小球。",
        "kind": "dmc",
        "dim_labels": [f"qpos[{i}]" for i in range(4)],
    },
    "cheetah": {
        "label_zh": "Cheetah — 平面奔跑(2D)",
        "task_zh": "2D 半猎豹,6 个关节;"
                   "agent 通过 6 维关节扭矩让模型沿 x 方向奔跑,目标速度最大化。",
        "kind": "dmc",
        "dim_labels": [f"qpos[{i}]" for i in range(9)],
    },
    "walker": {
        "label_zh": "Walker — 2D 双足行走",
        "task_zh": "2D 双足行走器,6 个关节;"
                   "agent 通过 6 维扭矩控制其稳定直立并前进。",
        "kind": "dmc",
        "dim_labels": [f"qpos[{i}]" for i in range(9)],
    },
    "hopper": {
        "label_zh": "Hopper — 单腿跳跃",
        "task_zh": "2D 单腿跳跃器,4 个关节;"
                   "agent 通过 4 维扭矩控制单腿保持平衡并向前跳跃。",
        "kind": "dmc",
        "dim_labels": [f"qpos[{i}]" for i in range(7)],
    },
    "quadruped": {
        "label_zh": "Quadruped — 四足行走(2D)",
        "task_zh": "2D 四足机器人,每条腿 3 个关节,共 12 个关节;"
                   "agent 通过 12 维扭矩控制其稳定直立并向 x 方向行走。",
        "kind": "dmc",
        "dim_labels": [f"qpos[{i}]" for i in range(30)],
    },
    "humanoid": {
        "label_zh": "Humanoid — 简化人形(2D)",
        "task_zh": "2D 简化人形,21 个关节;"
                   "agent 通过 21 维扭矩让模型保持直立并稳定向前行走。",
        "kind": "dmc",
        "dim_labels": [f"qpos[{i}]" for i in range(28)],
    },
    "humanoid_cmu": {
        "label_zh": "Humanoid CMU — CMU 姿态人形(3D)",
        "task_zh": "3D 高自由度人形,56 个关节;"
                   "agent 通过 56 维扭矩控制其保持直立并稳定行走。",
        "kind": "dmc",
        "dim_labels": [f"qpos[{i}]" for i in range(63)],
    },
    "dog": {
        "label_zh": "Dog — 三维四足狗",
        "task_zh": "3D 高自由度四足狗,38 个关节;"
                   "agent 通过 38 维扭矩控制其协调四肢完成三足/四足步态行走。",
        "kind": "dmc",
        "dim_labels": [f"qpos[{i}]" for i in range(87)],
    },
    "fish": {
        "label_zh": "Fish — 鱼形游泳",
        "task_zh": "2D 简化鱼形,5 个关节;"
                   "agent 通过 5 维扭矩让鱼形躯体摆动推进向前游动。",
        "kind": "dmc",
        "dim_labels": [f"qpos[{i}]" for i in range(14)],
    },
    "stacker": {
        "label_zh": "Stacker — 立方体堆叠",
        "task_zh": "2D 平面上的抓取机械臂,堆叠三个立方体;"
                   "agent 通过 5 维力/扭矩控制末端到达指定目标位置堆叠。",
        "kind": "dmc",
        "dim_labels": [f"qpos[{i}]" for i in range(20)],
    },
    # ------ reacher (mujoco direct) ------
    "reacher": {
        "label_zh": "Reacher — 机械臂触靶(2D)",
        "task_zh": "2D 双关节机械臂,目标是一个圆形目标;"
                   "agent 通过 2 维扭矩使末端到达目标位置。",
        "kind": "dmc",
        "dim_labels": ["shoulder_q", "elbow_q", "target_x", "target_y"],
    },
    # ------ 2 swm envs ------
    "pusht": {
        "label_zh": "PushT — 推送方块到目标",
        "task_zh": "二维平面上,agent 是一个圆盘,任务是推动一个 T 形方块到达目标位置/姿态;"
                   "obs 同时包含 agent 和方块的位置/速度/姿态。",
        "kind": "swm_pusht",
        "dim_labels": ["agent_x", "agent_y", "agent_vx", "agent_vy",
                       "block_x", "block_y", "block_angle"],
    },
    "tworoom": {
        "label_zh": "TwoRoom — 双房间导航(T-Maze)",
        "task_zh": "二维平面被中间墙分割成两个房间,agent 需要穿过中间的通道门到达目标点;"
                   "obs 包含 agent 位置、目标位置及环境内部状态(门状态等)。",
        "kind": "swm_tworoom",
        "dim_labels": ["agent_x", "agent_y", "target_x", "target_y",
                       "door?", "v?", "f?", "g?", "h?", "i?"],
    },
}


def _build_env(env_kind: str):
    """Build the env for the given short name."""
    if env_kind == "reacher":
        return make_reacher_env()
    if env_kind == "pusht":
        return PushTEnv()
    if env_kind == "tworoom":
        return TwoRoomEnv()
    return make_dmc_env(env_kind)


def _capture_state(env, env_kind: str, rng: np.random.Generator, n_steps: int = 50):
    """Reset, take n_steps random actions, capture obs/state and mujoco state.

    Quadruped is captured at the reset pose (n_steps=0): after 50 random
    steps it falls over and the splayed limbs no longer read as four legs.
    """
    if env_kind == "quadruped":
        n_steps = 0
    obs = env.reset(seed=0)
    for _ in range(n_steps):
        a = rng.uniform(env.spec.action_low, env.spec.action_high).astype(np.float32)
        obs, _r, done, _info = env.step(a)
        if done:
            obs = env.reset(seed=0)
    state = env.get_state()
    return state, obs


# ============================================================
# Rendering helpers
# ============================================================

def _render_dmc_panel(env, ax, title: str):
    """Render the DMC/Reacher env via mujoco offscreen render into ax."""
    model = env._model
    data = env._data
    H, W = 240, 320
    renderer = mujoco.Renderer(model, height=H, width=W)
    # Quadruped: the DMC "global" camera (fallback cam 0) sits at (-10, 10, 10)
    # — far too far to make out the four legs. Use an orbit close-up aimed at
    # the body centroid instead.
    env_id = str(getattr(env.spec, "env_id", ""))
    if "quadruped" in env_id:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Body centroid + extent (skip world body 0 and decorative bodies such
        # as the scene ball); distance adapts so the whole robot fits in frame.
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(1, model.nbody)]
        body = data.xpos[1:][np.array([n != "ball" for n in names])]
        centroid = body.mean(axis=0)
        radius = float(np.max(np.linalg.norm(body - centroid, axis=1)))
        cam.lookat[:] = centroid
        cam.distance = max(1.5, min(3.5, 2.2 * radius))
        cam.azimuth = 45.0   # degrees: side-front quarter view
        cam.elevation = -20.0  # degrees: slight top-down (MuJoCo sign: negative = above lookat)
        renderer.update_scene(data, camera=cam)
    else:
        # Pick a camera. Prefer "overview" / tracking camera when available.
        cam_id = 0
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name and ("overview" in name.lower() or "track" in name.lower()):
                cam_id = i
                break
        renderer.update_scene(data, camera=cam_id)
    img = renderer.render()  # (H, W, 3) uint8
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    renderer.close()


def _render_pusht_panel(state: np.ndarray, ax, title: str):
    """2D scatter for PushT: agent (dot), block (oriented T), target (ring)."""
    agent_x, agent_y, _avx, _avy, block_x, block_y, block_angle = state
    ax.set_xlim(-300, 500)
    ax.set_ylim(-300, 500)
    ax.set_aspect("equal")
    L, W_ = 30, 30
    corners = np.array([[-L, -W_], [L, -W_], [L, W_], [-L, W_]])
    cos_a, sin_a = np.cos(block_angle), np.sin(block_angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = corners @ rot.T + np.array([block_x, block_y])
    block_poly = plt.Polygon(rotated, closed=True, fill=True,
                             facecolor="#7BAE3F", edgecolor="black", lw=1.5)
    ax.add_patch(block_poly)
    agent = plt.Circle((agent_x, agent_y), 12, color="#1F77B4", zorder=5)
    ax.add_patch(agent)
    tgt_x, tgt_y, tgt_a = 200, 100, 0.0
    cos_a, sin_a = np.cos(tgt_a), np.sin(tgt_a)
    rotated = corners @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T
    rotated = rotated + np.array([tgt_x, tgt_y])
    tgt_poly = plt.Polygon(rotated, closed=True, fill=False,
                           edgecolor="red", lw=1.5, ls="--")
    ax.add_patch(tgt_poly)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)


def _render_tworoom_panel(state: np.ndarray, ax, title: str):
    """2D scatter for TwoRoom: agent, target, walls/door."""
    ax.set_xlim(-50, 350)
    ax.set_ylim(-50, 350)
    ax.set_aspect("equal")
    wall = plt.Rectangle((148, 0), 4, 100, color="black")
    wall2 = plt.Rectangle((148, 200), 4, 100, color="black")
    ax.add_patch(wall)
    ax.add_patch(wall2)
    ax.text(150, 150, "door", ha="center", va="center",
            fontsize=7, color="gray")
    ax.plot(state[0], state[1], "o", color="#1F77B4",
            markersize=12, label="agent")
    ax.plot(state[2], state[3], "*", color="red",
            markersize=14, label="target")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)


def _render_state_bars(state: np.ndarray, dim_labels: list[str], ax, obs_dim: int):
    """Bar chart of state values, with labels."""
    n = len(state)
    colors = ["#1F77B4" if v >= 0 else "#D62728" for v in state]
    x = np.arange(n)
    ax.bar(x, state, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    if n <= 14:
        ax.set_xticklabels(dim_labels, rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticklabels([f"[{i}]" for i in range(n)],
                           rotation=45, ha="right", fontsize=6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("value", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(f"obs vector (dim={obs_dim})", fontsize=9)


def _format_state(state: np.ndarray) -> str:
    if len(state) > 12:
        head = ", ".join(f"{v:+.3f}" for v in state[:6])
        tail = ", ".join(f"{v:+.3f}" for v in state[-4:])
        return f"[{head}, …, {tail}]"
    return "[" + ", ".join(f"{v:+.3f}" for v in state) + "]"


def _render_one_env(env_kind: str, meta: dict, out_path: Path):
    """Render the 3-panel figure for one env."""
    env = _build_env(env_kind)
    rng = np.random.default_rng(42)
    state, _obs = _capture_state(env, env_kind, rng, n_steps=50)
    obs_dim = env.spec.obs_dim
    action_dim = env.spec.action_dim

    fig = plt.figure(figsize=(8, 6), dpi=100)
    gs = GridSpec(3, 1, height_ratios=[3, 2, 1.4], hspace=0.55)
    ax_scene = fig.add_subplot(gs[0])
    ax_bars = fig.add_subplot(gs[1])
    ax_caption = fig.add_subplot(gs[2])

    title = meta["label_zh"]
    if meta["kind"] == "swm_pusht":
        _render_pusht_panel(state, ax_scene, title)
    elif meta["kind"] == "swm_tworoom":
        _render_tworoom_panel(state, ax_scene, title)
    else:
        _render_dmc_panel(env, ax_scene, title)

    _render_state_bars(state, meta["dim_labels"], ax_bars, obs_dim)

    ax_caption.axis("off")
    state_str = _format_state(state)
    caption = (
        f"{meta['task_zh']}\n"
        f"obs_dim={obs_dim}, action_dim={action_dim}   |   "
        f"sample state (after 50 random steps from seed=0):\n"
        f"state = {state_str}"
    )
    ax_caption.text(0.0, 1.0, caption, ha="left", va="top", fontsize=8,
                    wrap=True, transform=ax_caption.transAxes)

    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    env.close()
    return state, obs_dim, action_dim


def _render_overview(samples: dict, out_path: Path):
    """4x4 grid of small panels (one per env, top-only view)."""
    env_names = list(ENV_META.keys())
    assert len(env_names) == 16, f"expected 16 envs, got {len(env_names)}"
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=100)
    fig.suptitle(
        "16-env overview: sample state for each env (seed=0, 50 random steps)",
        fontsize=14,
    )
    for idx, env_kind in enumerate(env_names):
        ax = axes[idx // 4, idx % 4]
        meta = ENV_META[env_kind]
        env = _build_env(env_kind)
        rng = np.random.default_rng(42)
        state, _obs = _capture_state(env, env_kind, rng, n_steps=50)
        title = f"{env_kind}\nobs={env.spec.obs_dim}, act={env.spec.action_dim}"
        if meta["kind"] == "swm_pusht":
            _render_pusht_panel(state, ax, title)
        elif meta["kind"] == "swm_tworoom":
            _render_tworoom_panel(state, ax, title)
        else:
            _render_dmc_panel(env, ax, title)
        env.close()
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    out_dir = REPO_ROOT / "paper" / "figs" / "dataset_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    assert len(ENV_META) == 16, f"expected 16 envs, got {len(ENV_META)}"

    samples: dict[str, dict] = {}
    for env_kind, meta in ENV_META.items():
        out_path = out_dir / f"{env_kind}.png"
        print(f"[generate] {env_kind} -> {out_path}", flush=True)
        state, obs_dim, action_dim = _render_one_env(env_kind, meta, out_path)
        samples[env_kind] = {
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "state": state.tolist(),
            "dim_labels": meta["dim_labels"],
            "task_zh": meta["task_zh"],
            "label_zh": meta["label_zh"],
        }

    overview_path = out_dir / "all_envs_overview.png"
    print(f"[generate] overview -> {overview_path}", flush=True)
    _render_overview(samples, overview_path)

    json_path = out_dir / "obs_samples.json"
    with open(json_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"[generate] obs_samples.json -> {json_path}", flush=True)

    print("\n=== Summary ===")
    for env_kind, s in samples.items():
        size = (out_dir / f"{env_kind}.png").stat().st_size
        print(f"  {env_kind:20s} obs_dim={s['obs_dim']:3d} "
              f"act_dim={s['action_dim']:3d}  png={size//1024} KB")
    print(f"  overview png: "
          f"{(out_dir / 'all_envs_overview.png').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
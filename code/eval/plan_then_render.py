"""Plan + closed-loop + render: produces a GIF showing the agent's trajectory.

Runs the canonical observed-state closed-loop evaluator with a recording
environment wrapper. Rendering does not replace observations with model
predictions or suppress simulator/encoder failures.
Usage:

    python -m code.eval.plan_then_render --env reacher --ckpt .../final.pt --data .../reacher.npz --out /tmp/out.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")

from code.core.envs import (
    ReacherEnv, OGBCubeEnv, OGBenchSceneEnv, BaseEnv, DMCStateEnv,
)
from code.eval.closed_loop import (
    make_env as closed_loop_make_env,
    eval_closed_loop,
)


class _RecordingEnv(BaseEnv):
    """Record successful native transitions without changing control inputs."""

    def __init__(self, env: BaseEnv):
        super().__init__()
        self._base = env
        self.spec = env.spec
        self._physical = env
        while hasattr(self._physical, "_base"):
            self._physical = self._physical._base
        self.states = []
        self.actions = []

    def _physical_state(self):
        if isinstance(self._physical, DMCStateEnv):
            state = DMCStateEnv.get_state(self._physical)
        else:
            state = self._physical.get_state()
        return np.asarray(state, dtype=np.float32).copy()

    def reset(self, seed=None, **kwargs):
        self.states.clear()
        self.actions.clear()
        return self._base.reset(seed=seed, **kwargs)

    def step(self, action):
        if not self.states:
            self.states.append(self._physical_state())
        result = self._base.step(action)
        self.actions.append(np.asarray(action).copy())
        self.states.append(self._physical_state())
        return result

    def get_state(self):
        return self._base.get_state()

    def check_success(self, state, goal_state):
        return self._base.check_success(state, goal_state)


def plan_and_run_single_episode(
    model,
    env: BaseEnv,
    data_path: str,
    cem_samples: int = 300,
    cem_elites: int = 30,
    cem_iters: int = 10,
    horizon: int = 5,
    eval_budget: int = 50,
    goal_offset: int = 25,
    history_size: int = 3,
    device: str = "cuda",
    seed: int = 42,
    pad_obs_to: int | None = None,
    model_action_dim: int | None = None,
) -> Dict[str, Any]:
    """Render one canonical evaluation episode for the requested protocol seed."""
    recording = _RecordingEnv(env)
    result = eval_closed_loop(
        model.to(device).eval(), recording, data_path,
        n_episodes=1, n_seeds=1, seed_start=seed,
        cem_samples=cem_samples, cem_elites=cem_elites, cem_iters=cem_iters,
        horizon=horizon, eval_budget=eval_budget, goal_offset=goal_offset,
        history_size=history_size, device=device,
        pad_obs_to=pad_obs_to, model_action_dim=model_action_dim,
    )
    episode = result.per_episode[0]
    return {
        **episode,
        "states": recording.states,
        "actions": recording.actions,
        "init_state": np.asarray(episode["init_state"], dtype=np.float32),
        "goal_state": np.asarray(episode["goal_state"], dtype=np.float32),
        "protocol": result.protocol,
    }


def render_gif(traj: dict, env: BaseEnv, output_path: str, fps: int = 6, dpi: int = 100) -> None:
    """Render the trajectory to a GIF. Dispatches to the right renderer per env."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(env, ReacherEnv):
        from code.core.viz.render_2d import render_reacher_gif
        render_reacher_gif(traj, env, str(output_path), fps=fps, dpi=dpi)
    elif isinstance(env, (OGBCubeEnv, OGBenchSceneEnv)):
        from code.core.viz.render_3d import render_manipulator_gif
        render_manipulator_gif(traj, env, str(output_path), fps=fps, dpi=dpi)
    else:
        from code.core.viz.render_2d import render_state_trajectory
        render_state_trajectory(traj, env, str(output_path), fps=fps, dpi=dpi)


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="Output .gif path")
    # SAME default CEM params as closed_loop, NOT the old (50/10/3) values
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-elites", type=int, default=30)
    p.add_argument("--cem-iters", type=int, default=10)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=42, help="Canonical evaluation seed index")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = closed_loop_make_env(args.env, args.data)
    try:
        from code.scripts.event_align import build_model

        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ck_args = ck["args"]
        state_dim = ck_args.get("pad_obs_to") or ck_args.get("state_dim") or env.spec.obs_dim
        action_dim = ck_args.get("action_dim") or env.spec.action_dim
        state_dict = {key.replace("_orig_mod.", ""): value for key, value in ck["model"].items()}
        model = build_model(ck_args["model"], state_dim, action_dim, ck_args, state_dict=state_dict)
        model.load_state_dict(state_dict, strict=True)
        traj = plan_and_run_single_episode(
            model, env, args.data,
            cem_samples=args.cem_samples, cem_elites=args.cem_elites, cem_iters=args.cem_iters,
            horizon=args.horizon, eval_budget=args.eval_budget,
            goal_offset=args.goal_offset, history_size=args.history_size,
            device=device, seed=args.seed,
            pad_obs_to=state_dim, model_action_dim=action_dim,
        )
        print(f"[plan_render] cos_dist={traj['cos_dist']:.3f}, env_success={traj['env_success']}, phys_dist={traj['phys_dist']:.3f}", flush=True)
        render_gif(traj, env, args.out, fps=args.fps, dpi=args.dpi)
        print(f"[plan_render] saved {args.out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

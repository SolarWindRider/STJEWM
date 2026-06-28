"""Plan + closed-loop + render: produces a GIF showing the agent's trajectory.

CRITICAL: this module uses the EXACT same closed-loop code as
code.eval.closed_loop.eval_closed_loop, so the GIF and the experimental
eval compute the same trajectory. The only addition is rendering the
trajectory as a GIF.
Usage:

    python -m code.eval.plan_then_render --env reacher --ckpt .../final.pt --data .../reacher.npz --out /tmp/out.gif
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")

# Register stable_worldmodel envs
import stable_worldmodel  # noqa: F401

from code.core.cem import CEM
from code.core.encode import encode_history, encode_obs
from code.core.envs import (
    ReacherEnv, OGBCubeEnv, OGBenchSceneEnv, BaseEnv,
)
from code.eval.closed_loop import (
    make_env as closed_loop_make_env,
    _infer_env_kind,
    _set_env_state,
)


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
) -> Dict[str, Any]:
    """Run one episode of closed-loop CEM planning. THIS IS THE EXACT SAME
    loop body as code.eval.closed_loop.eval_closed_loop (one episode slice).

    Returns the trajectory dict for rendering.
    """
    model = model.to(device).eval()
    action_dim = env.spec.action_dim
    action_low = env.spec.action_low
    action_high = env.spec.action_high

    # Load dataset (same logic as closed_loop)
    from code.data import load_dataset
    env_kind = _infer_env_kind(env)
    if data_path == "(none)" or data_path is None:
        if env_kind in ("ogb_cube", "ogb_scene"):
            ds = load_dataset(env_kind + "_env", n_episodes=20, max_steps_per_ep=50,
                              history_size=history_size, goal_offset=goal_offset, seed=seed)
        else:
            ds = None
    else:
        ds = load_dataset(env_kind, path=data_path, history_size=history_size, goal_offset=goal_offset)

    # Sample init/goal (same as closed_loop)
    if ds is not None:
        rng = np.random.default_rng(seed)
        ep_idx = int(rng.choice(len(ds)))
        item = ds[ep_idx]
        init_state_np = item["init_state"].numpy() if hasattr(item["init_state"], "numpy") else np.asarray(item["init_state"])
        goal_state_np = item["goal_state"].numpy() if hasattr(item["goal_state"], "numpy") else np.asarray(item["goal_state"])
    else:
        env.reset(seed=seed)
        init_state_np = env.get_state()
        goal_state_np = init_state_np.copy()

    # Set env to init (best-effort, same as closed_loop)
    try:
        _set_env_state(env, init_state_np)
    except Exception:
        pass

    # Build CEM with SAME defaults as closed_loop
    cem = CEM(model, action_dim=action_dim, horizon=horizon,
              n_samples=cem_samples, n_elites=cem_elites, n_iters=cem_iters,
              history_size=history_size, device=device)

    # History and goal
    history_states = [init_state_np.copy() for _ in range(history_size)]
    try:
        z_history = encode_history(model, [torch.from_numpy(s).float() for s in history_states],
                                   action_dim, device)
    except Exception:
        z_history = encode_obs(model, torch.from_numpy(init_state_np).float(), action_dim, device).unsqueeze(0).repeat(history_size, 1)
    z_goal = encode_obs(model, torch.from_numpy(goal_state_np).float(), action_dim, device)

    # === EXACT SAME LOOP AS code.eval.closed_loop.eval_closed_loop ===
    states: List[np.ndarray] = [init_state_np.copy()]
    actions_taken: List[np.ndarray] = []
    t_start = time.time()
    # Match closed_loop's seed setting for determinism
    torch.manual_seed(seed)
    np.random.seed(seed)
    z_init = z_history[-1]
    actions_count = 0
    done = False
    while actions_count < eval_budget:
        seq = cem.plan(z_init, z_goal)
        for a_idx in range(min(horizon, eval_budget - actions_count)):
            action = seq[a_idx].cpu().numpy().astype(np.float32)
            action = np.clip(action, action_low, action_high)
            try:
                _obs, _r, done, _info = env.step(action)
            except Exception:
                done = True
            actions_taken.append(action)
            try:
                s = env.get_state()
            except Exception:
                s = states[-1]
            states.append(s)
            actions_count += 1
            if done:
                break
        if done or actions_count >= eval_budget:
            break
        # Roll history forward (same as closed_loop)
        try:
            with torch.no_grad():
                a_window = seq[:history_size].unsqueeze(0)
                nxt = model.predict(z_history.unsqueeze(0), a_window)
                z_history = torch.cat([z_history[1:], nxt[0:1, -1]], dim=0)
                z_init = z_history[-1]
        except Exception:
            pass
    plan_time = time.time() - t_start

    # Compute metrics (same as closed_loop)
    final_state = states[-1]
    z_final = encode_obs(model, torch.from_numpy(final_state).float(), action_dim, device)
    cos = torch.nn.functional.cosine_similarity(z_final.unsqueeze(0), z_goal.unsqueeze(0))
    cos_dist = float((1.0 - cos.item()) / 2.0)
    env_success, phys_dist = env.check_success(final_state, goal_state_np)

    return {
        "states": states,
        "actions": actions_taken,
        "init_state": init_state_np,
        "goal_state": goal_state_np,
        "cos_dist": cos_dist,
        "env_success": env_success,
        "phys_dist": phys_dist,
        "plan_time_sec": plan_time,
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the SAME make_env as closed_loop
    env = closed_loop_make_env(args.env, args.data)

    # Build model (same as closed_loop)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    state_dim = env.spec.obs_dim
    action_dim = env.spec.action_dim
    if ck_args.get("model", "stjewm") == "lewm_baseline":
        from code.lewm_transformer_baseline import LeWMTransformerBaseline
        model = LeWMTransformerBaseline(state_dim=state_dim, action_dim=action_dim, embed_dim=192,
                                         num_layers=ck_args.get("n_layers", 6))
    else:
        from code.stjewm import STJEWM
        n_layers = ck_args.get("n_layers", 4)
        model = STJEWM(
            d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
            state_dim=state_dim, cell_n_layers=n_layers, n_d=3,
            trace_beta=0.9, freeze_encoder=True,
        )
    model.load_state_dict(ck["model"])

    # Plan + run (SAME algorithm as closed_loop)
    traj = plan_and_run_single_episode(
        model, env, args.data,
        cem_samples=args.cem_samples, cem_elites=args.cem_elites, cem_iters=args.cem_iters,
        horizon=args.horizon, eval_budget=args.eval_budget,
        goal_offset=args.goal_offset, history_size=args.history_size,
        device=device, seed=args.seed,
    )
    print(f"[plan_render] cos_dist={traj['cos_dist']:.3f}, env_success={traj['env_success']}, phys_dist={traj['phys_dist']:.3f}", flush=True)

    # Render
    render_gif(traj, env, args.out, fps=args.fps, dpi=args.dpi)
    print(f"[plan_render] saved {args.out}")


if __name__ == "__main__":
    main()

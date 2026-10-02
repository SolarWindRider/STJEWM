#!/usr/bin/env python
"""B4 probe-derived timed trace intervention on trained checkpoints.

Uses matched probe-derived timed planning windows, not events detected on the
controlled trajectory. The intervention zeros the STJEWM gated trace inside
CEM predictions; current observations are always re-encoded after execution.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from code.core.envs.dmc_env import DMCPixelEnv
from code.eval.closed_loop import eval_closed_loop, make_env
from code.data.loaders import DATA_LOADER_PROTOCOL_VERSION
from code.train.train import build_model

ROOT = Path("/home/lx/snn")
ENV_CONFIG = {
    "cartpole_2d": ("cartpole", ROOT / "data/dm_control/cartpole_250k.npz"),
    "cheetah": ("cheetah", ROOT / "data/dm_control/3d_rollouts_250k/cheetah_250k.npz"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=ENV_CONFIG)
    p.add_argument("--model", default="stjewm_trace_only")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pixel", action="store_true")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-elites", type=int, default=30)
    p.add_argument("--cem-iters", type=int, default=10)
    p.add_argument("--probe-steps", type=int, default=99)
    p.add_argument("--mad-k", type=float, default=1.0)
    p.add_argument("--half-w", type=int, default=2)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def probe_rollout(env, n_steps=99, seed=0):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    observations = [np.asarray(env.get_state(), dtype=np.float32).copy()]
    valid_transitions = []
    resets = 0
    previous_reset = False
    for _ in range(n_steps):
        action = rng.uniform(env.spec.action_low, env.spec.action_high).astype(np.float32)
        _, _, done, _ = env.step(action)
        observations.append(np.asarray(env.get_state(), dtype=np.float32).copy())
        valid_transitions.append(not previous_reset)
        previous_reset = bool(done)
        if done:
            resets += 1
            env.reset(seed=seed + resets)
    return np.stack(observations), np.asarray(valid_transitions), resets


def detect_event_steps(observations, valid_transitions, mad_k=1.0,
                       eval_budget=50, horizon=5, half_w=2):
    """Classify motion windows only at eligible controlled-episode plan times."""
    d_obs = np.linalg.norm(np.diff(observations, axis=0), axis=1)
    valid = np.asarray(valid_transitions, dtype=bool) & np.isfinite(d_obs)
    support_end = min(len(d_obs), eval_budget)
    eligible_steps = []
    motion_scores = []
    for step in range(0, support_end, horizon):
        start, stop = max(0, step - half_w), min(support_end, step + half_w + 1)
        local_valid = valid[start:stop]
        if local_valid.any():
            eligible_steps.append(step)
            motion_scores.append(float(np.mean(d_obs[start:stop][local_valid])))
    if not eligible_steps:
        raise ValueError("The probe has no valid motion windows at eligible planning times")
    eligible_steps = np.asarray(eligible_steps)
    motion_scores = np.asarray(motion_scores)
    median = float(np.median(motion_scores))
    mad = float(np.median(np.abs(motion_scores - median))) + 1e-9
    threshold = median + mad_k * mad
    events = eligible_steps[motion_scores > threshold]
    non_events = eligible_steps[motion_scores < median]
    selection = {
        "support_end_exclusive": support_end,
        "eligible_planning_steps": eligible_steps.tolist(),
        "motion_window_scores": motion_scores.tolist(),
        "median_motion_window_score": median,
        "mad_motion_window_score": mad,
        "event_threshold": threshold,
        "score_definition": "mean L2 observation increment in valid probe transitions within +/-half_w frames, clipped to evaluation support",
        "event_rule": "score > median + mad_k * MAD over eligible planning windows",
        "non_event_rule": "score < eligible-window median",
    }
    return events, non_events, selection


def build_window_sets(event_idx, non_event_idx, eligible_steps, max_windows=12):
    """Select nonempty, exposure-matched schedules from eligible plan starts."""
    n = min(len(event_idx), len(non_event_idx), max_windows)
    if n == 0:
        raise ValueError("Eligible probe windows cannot supply matched event/non-event schedules")
    event = set(map(int, np.random.default_rng(0).choice(event_idx, size=n, replace=False)))
    non_event = set(map(int, np.random.default_rng(1).choice(non_event_idx, size=n, replace=False)))
    random = set(map(int, np.random.default_rng(2).choice(eligible_steps, size=n, replace=False)))
    return {"event": event, "non_event": non_event, "random": random}


@contextmanager
def zero_stjewm_trace(model):
    """Zero the gated trace for exactly the surrounding predict call."""
    if not hasattr(model, "gated_trace"):
        raise TypeError("The trace intervention requires an STJEWM gated_trace readout")
    gated_trace = model.gated_trace
    original = gated_trace.forward

    def zero_forward(spike, context):
        return torch.zeros_like(spike)

    gated_trace.forward = zero_forward
    try:
        yield
    finally:
        gated_trace.forward = original


def step_hook(step_set):
    def hook(model, ctx_emb, ctx_act, env_step):
        if env_step in step_set:
            with zero_stjewm_trace(model):
                return model.predict(ctx_emb, ctx_act)
        return model.predict(ctx_emb, ctx_act)
    return hook


def cem_hook(model, ctx_emb, ctx_act):
    with zero_stjewm_trace(model):
        return model.predict(ctx_emb, ctx_act)


def build_ckpt_model(ckpt_path, device, pixel=False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {}) or {}
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    if pixel:
        model = build_model(
            args.get("model", "stjewm"), args.get("pad_obs_to", 21168),
            args.get("action_dim", 56), args.get("n_layers", 4),
            args.get("readout_mode", "hidden_leak"),
            embed_dim=args.get("embed_dim", 192), image_size=args.get("image_size", 84),
        )
    else:
        from code.scripts.event_align import build_model as build_state_model
        model = build_state_model(
            args["model"], state_dict["state_projector.proj.0.weight"].shape[1],
            args["action_dim"], args, state_dict=state_dict,
        )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), args, ckpt.get("data_protocol_version")


def pixel_eval(model, env_name, args, step_predict_hook=None, cem_predict_hook=None):
    """Live-pixel observed-state MPC with native controls and physical metrics."""
    import mujoco
    from code.core.cem import CEM, make_native_action_predict_hook

    env_kind, _ = ENV_CONFIG[env_name]
    env = DMCPixelEnv(
        env_kind, image_size=int(getattr(args, "image_size", 84)),
        max_episode_steps=args.eval_budget,
    )
    current_env_step = 0

    def predict_for_step(model, ctx_emb, ctx_act):
        if step_predict_hook is not None:
            return step_predict_hook(model, ctx_emb, ctx_act, current_env_step)
        if cem_predict_hook is not None:
            return cem_predict_hook(model, ctx_emb, ctx_act)
        return model.predict(ctx_emb, ctx_act)

    cem = CEM(
        model, action_dim=env.spec.action_dim, horizon=args.horizon,
        n_samples=args.cem_samples, n_elites=args.cem_elites, n_iters=args.cem_iters,
        history_size=1, device=args.device,
        predict_hook=make_native_action_predict_hook(
            model.action_dim, env.spec.action_low, env.spec.action_high,
            args.device, predict_for_step,
        ),
    )

    @torch.no_grad()
    def encode(pixel):
        x = torch.from_numpy(pixel).float().unsqueeze(0).unsqueeze(0).to(args.device)
        action = torch.zeros(1, 1, model.action_dim, device=args.device)
        return model(x, action)["emb"][0, 0]

    goal_state = np.zeros(env._nq, dtype=np.float32)
    episodes = []
    t0 = time.time()
    try:
        env.reset(seed=0)
        env._data.qpos[:env._nq] = goal_state
        env._data.qvel[:] = 0.0
        mujoco.mj_forward(env._model, env._data)
        z_goal = encode(env._render())
        for seed in range(args.n_seeds):
            for ep in range(args.n_episodes):
                episode_seed = seed * 1000 + ep
                torch.manual_seed(episode_seed)
                np.random.seed(episode_seed)
                obs = env.reset(seed=episode_seed)
                init_state = env.get_state()
                actions_taken = 0
                planning_calls = 0
                done = False
                while actions_taken < args.eval_budget:
                    current_env_step = actions_taken
                    seq = cem.plan(encode(obs["pixel"]), z_goal)
                    if not torch.isfinite(seq).all():
                        raise FloatingPointError("CEM returned non-finite pixel-evaluation actions")
                    planning_calls += 1
                    for candidate in seq[:min(args.horizon, args.eval_budget - actions_taken)].cpu().numpy():
                        action = np.clip(candidate, env.spec.action_low, env.spec.action_high)
                        obs, _, done, _ = env.step(action.astype(np.float32, copy=False))
                        actions_taken += 1
                        if done:
                            break
                    if done:
                        break
                final_state = env.get_state()
                success, phys_dist = env.check_success(final_state, goal_state)
                z_final = encode(obs["pixel"])
                cos_dist = float((1.0 - torch.nn.functional.cosine_similarity(
                    z_final.unsqueeze(0), z_goal.unsqueeze(0)).item()) / 2.0)
                if not np.isfinite([cos_dist, phys_dist]).all():
                    raise FloatingPointError("Pixel evaluation produced non-finite terminal metrics")
                episodes.append({
                    "seed": seed, "episode_idx": ep, "initialization_seed": episode_seed,
                    "init_state": init_state.tolist(), "goal_state": goal_state.tolist(),
                    "final_state": final_state.tolist(), "actions_taken": actions_taken,
                    "planning_calls": planning_calls, "env_success": bool(success),
                    "phys_dist": float(phys_dist), "cos_dist": cos_dist,
                    "lewm_success": cos_dist < 0.1,
                })
    finally:
        env.close()
    per_seed = [{
        "success_rate_env": float(np.mean([e["env_success"] for e in episodes if e["seed"] == seed])),
        "success_rate_lewm": float(np.mean([e["lewm_success"] for e in episodes if e["seed"] == seed])),
    } for seed in range(args.n_seeds)]
    return {
        "n_episodes": len(episodes), "n_seeds": args.n_seeds,
        "success_rate_env": float(np.mean([s["success_rate_env"] for s in per_seed])),
        "success_rate_env_std": float(np.std([s["success_rate_env"] for s in per_seed])),
        "success_rate_lewm": float(np.mean([s["success_rate_lewm"] for s in per_seed])),
        "success_rate_lewm_std": float(np.std([s["success_rate_lewm"] for s in per_seed])),
        "mean_cos_dist": float(np.mean([e["cos_dist"] for e in episodes])),
        "mean_phys_dist": float(np.mean([e["phys_dist"] for e in episodes])),
        "per_seed": per_seed, "per_episode": episodes,
        "physical_success_tolerance": env._success_tol,
        "physical_state_dim": env._nq,
        "goal_definition": "zero qpos, rendered in the actual simulator",
        "latent_representation": "forward.emb, one observed frame, zero action",
        "wall_time_sec": time.time() - t0,
    }


def result_dict(result):
    return result if isinstance(result, dict) else asdict(result)




def main():
    args = parse_args()
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(args.ckpt)
    if Path(args.out).exists():
        raise FileExistsError(f"Archive the existing result before rerunning: {args.out}")
    device = args.device
    args.device = device
    model, ck_args, checkpoint_data_protocol = build_ckpt_model(args.ckpt, device, pixel=args.pixel)
    args.image_size = ck_args.get("image_size", 84)

    env_kind, data_path = ENV_CONFIG[args.env]
    probe_env = make_env(env_kind, str(data_path))
    try:
        observations, valid_transitions, resets = probe_rollout(probe_env, args.probe_steps)
    finally:
        probe_env.close()
    event_idx, non_event_idx, selection = detect_event_steps(
        observations, valid_transitions, args.mad_k,
        args.eval_budget, args.horizon, args.half_w,
    )
    windows = build_window_sets(
        event_idx, non_event_idx, selection["eligible_planning_steps"], args.max_windows,
    )

    all_steps = set(range(0, args.eval_budget, args.horizon))
    modes = {
        "baseline": (None, None, set()),
        "event_window": (step_hook(windows["event"]), None, windows["event"]),
        "non_event_window": (step_hook(windows["non_event"]), None, windows["non_event"]),
        "random_window": (step_hook(windows["random"]), None, windows["random"]),
        "ablate_all": (step_hook(all_steps), None, all_steps),
        "cem_rollout_ablation": (None, cem_hook, all_steps),
    }
    results = {}
    for name, (step_predict_hook, cem_predict_hook, scheduled_steps) in modes.items():
        print(f"[B4] {args.env}/{args.model}/{name}", flush=True)
        if args.pixel:
            raw = pixel_eval(model, args.env, args, step_predict_hook, cem_predict_hook)
        else:
            env = make_env(env_kind, str(data_path))
            try:
                raw = eval_closed_loop(
                    model, env, str(data_path), n_episodes=args.n_episodes,
                    n_seeds=args.n_seeds, cem_samples=args.cem_samples,
                    cem_elites=args.cem_elites, cem_iters=args.cem_iters,
                    horizon=args.horizon, eval_budget=args.eval_budget,
                    goal_offset=25, history_size=1, device=device,
                    pad_obs_to=ck_args.get("pad_obs_to", 128),
                    model_action_dim=ck_args.get("action_dim", 56),
                    step_predict_hook=step_predict_hook,
                    cem_predict_hook=cem_predict_hook,
                )
            finally:
                env.close()
        rd = result_dict(raw)
        rd.update({"env": args.env, "model": args.model, "ablation_mode": name,
                   "horizon": args.horizon, "eval_budget": args.eval_budget,
                   "n_scheduled_plan_steps": len(scheduled_steps),
                   "n_intervened_plans": sum(
                       len(set(range(0, e["actions_taken"], args.horizon)) & scheduled_steps)
                       for e in rd["per_episode"]
                   )})
        results[name] = rd
        print(f"[B4] env_sr={rd['success_rate_env']:.3f} cos={rd['mean_cos_dist']:.6f}", flush=True)

    base = results["baseline"]
    drops = {}
    for name in modes:
        if name == "baseline":
            continue
        mode = results[name]
        drops[name] = {
            "env_sr_drop_pp": (base["success_rate_env"] - mode["success_rate_env"]) * 100.0,
            "lewm_sr_drop_pp": (base["success_rate_lewm"] - mode["success_rate_lewm"]) * 100.0,
            "cos_dist_increase": mode["mean_cos_dist"] - base["mean_cos_dist"],
        }
    out = {
        "env": args.env, "model": args.model, "modality": "pixel" if args.pixel else "state",
        "ckpt": str(Path(args.ckpt).resolve()),
        "checkpoint_data_protocol_version": checkpoint_data_protocol,
        "data_loader_protocol": DATA_LOADER_PROTOCOL_VERSION,
        "protocol_version": 2,
        "latent_representation": "forward.emb, one observed frame, zero action",
        "protocol": {"cem_samples": args.cem_samples, "cem_elites": args.cem_elites,
                     "cem_iters": args.cem_iters, "horizon": args.horizon,
                     "eval_budget": args.eval_budget, "history_size": 1,
                     "n_episodes": args.n_episodes, "n_seeds": args.n_seeds},
        "intervention_definition": "gated trace zeroed inside CEM predictions at probe-derived timed replanning starts",
        "window_definition": "separate random-policy probe; motion classified at eligible replanning starts within evaluation support; matched event/non-event/random exposure",
        "window_step_unit": "environment step at start of replanning",
        "causal_scope": "timed probe-derived intervention, not online controlled-trajectory event matching",
        "all_timed_and_cem_wide_equivalent": True,
        "probe": {"n_event_steps": int(len(event_idx)),
                  "n_non_event_steps": int(len(non_event_idx)),
                  "selection": selection, "mad_k": args.mad_k,
                  "half_w": args.half_w, "max_windows": args.max_windows,
                  "n_probe_steps": args.probe_steps, "n_resets": resets,
                  "observations": observations.tolist(),
                  "valid_transitions": valid_transitions.tolist(),
                  "full_d_obs_norm": np.linalg.norm(np.diff(observations, axis=0), axis=1).tolist()},
        "windows": {"event_n_steps": len(windows["event"]),
                    "non_event_n_steps": len(windows["non_event"]),
                    "random_n_steps": len(windows["random"]),
                    "event_set": sorted(windows["event"]),
                    "non_event_set": sorted(windows["non_event"]),
                    "random_set": sorted(windows["random"])},
        "results": results, "drops": drops,
        "drops_pp": {"event": drops["event_window"]["env_sr_drop_pp"],
                     "non_event": drops["non_event_window"]["env_sr_drop_pp"],
                     "random": drops["random_window"]["env_sr_drop_pp"],
                     "ablate_all": drops["ablate_all"]["env_sr_drop_pp"],
                     "cem_rollout": drops["cem_rollout_ablation"]["env_sr_drop_pp"]},
        "probe_event_drop_exceeds_controls": bool(
            drops["event_window"]["env_sr_drop_pp"] > drops["non_event_window"]["env_sr_drop_pp"]
            and drops["event_window"]["env_sr_drop_pp"] > drops["random_window"]["env_sr_drop_pp"]),
        "cem_wide_drop_exceeds_all_timed": bool(
            drops["cem_rollout_ablation"]["env_sr_drop_pp"] > drops["ablate_all"]["env_sr_drop_pp"]),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(f"[B4] saved {path}")


if __name__ == "__main__":
    main()

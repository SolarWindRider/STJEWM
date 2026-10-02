"""Evaluate a strictly loaded, audited checkpoint on Event-Window reward.

This is a latent self-goal CEM heuristic, not a reward-optimising predictor or a
supervised categorical prediction head. Scores are the actual full-episode
rewards returned by the native task (20 scored windows per default episode).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from code.core.cem import CEM, make_native_action_predict_hook
from code.core.encode import encode_obs
from code.core.envs.event_window import make_event_window
from code.eval.closed_loop import _PadObsWrapper
from code.train.train import build_model
from code.scripts.audited_results import TrainingAudit, require, write_new_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--cem-samples", type=int, default=100)
    parser.add_argument("--cem-elites", type=int, default=10)
    parser.add_argument("--cem-iters", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--eval-budget", type=int, default=200)
    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def evaluate(args):
    require(min(args.n_episodes, args.n_seeds, args.cem_samples, args.cem_iters, args.horizon) > 0,
            "Evaluation budgets must be positive")
    require(2 <= args.cem_elites <= args.cem_samples, "CEM requires 2 <= elites <= samples")
    audit = TrainingAudit(args.training_manifest)
    audit.protect_output(args.out)
    provenance = audit.provenance(args.ckpt)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ck["args"]
    native = make_event_window()
    require(args.eval_budget >= native.spec.max_episode_steps, "Reward claims require all 20 scored windows")
    state_dim, action_dim = saved["pad_obs_to"], saved["action_dim"]
    require(state_dim >= native.spec.obs_dim and action_dim >= native.spec.action_dim, "Checkpoint cannot represent this task")
    env = _PadObsWrapper(native, state_dim) if state_dim > native.spec.obs_dim else native
    model = build_model(
        saved["model"], state_dim, action_dim, saved["n_layers"], saved.get("readout_mode", "hidden_leak"),
        embed_dim=saved.get("embed_dim", 192), image_size=saved.get("image_size", 84),
        hidden_dim=saved.get("hidden_dim"), mlp_hidden=saved.get("mlp_hidden"),
        mlp_layers=saved.get("mlp_layers"), stacked_lif_layers=saved.get("stacked_lif_layers"),
        stacked_lif_din=saved.get("stacked_lif_din"),
    )
    model.load_state_dict(ck["model"], strict=True)
    model.to(args.device).eval()
    del ck
    native_action_dim = native.spec.action_dim
    cem = CEM(model=model, action_dim=native_action_dim, n_samples=args.cem_samples,
              n_elites=args.cem_elites, n_iters=args.cem_iters, horizon=args.horizon,
              history_size=args.history_size, device=args.device,
              predict_hook=make_native_action_predict_hook(action_dim, native.spec.action_low,
                                                           native.spec.action_high, args.device))
    per_seed, per_episode = [], []
    started = time.monotonic()
    try:
        for seed in range(args.n_seeds):
            episodes = []
            for episode in range(args.n_episodes):
                episode_seed = seed * 10000 + episode
                torch.manual_seed(episode_seed)
                env.reset(seed=episode_seed)
                actions_taken, reward, done = 0, 0.0, False
                while not done and actions_taken < args.eval_budget:
                    with torch.no_grad():
                        state = torch.as_tensor(env.get_state(), dtype=torch.float32, device=args.device)[None]
                        latent = encode_obs(model, state, action_dim, args.device)
                        sequence = cem.plan(latent, latent)
                    for action in sequence[:min(args.horizon, args.eval_budget - actions_taken)]:
                        native_action = np.clip(action.cpu().numpy(), native.spec.action_low, native.spec.action_high).astype(np.float32)
                        _, value, done, _ = env.step(native_action)
                        reward += float(value)
                        actions_taken += 1
                        if done:
                            break
                require(done and actions_taken == native.spec.max_episode_steps, "Incomplete native reward episode")
                episodes.append({"seed": seed, "episode_idx": episode, "episode_seed": episode_seed,
                                 "ep_reward": reward, "n_windows": native.cfg.n_windows, "n_actions": actions_taken})
            per_episode.extend(episodes)
            per_seed.append({"seed": seed, "n": len(episodes),
                             "mean_reward": float(np.mean([row["ep_reward"] for row in episodes]))})
    finally:
        env.close()
    require(audit.checkpoint(args.ckpt)["sha256"] == provenance["checkpoint_sha256"], "Checkpoint changed during evaluation")
    means = [row["mean_reward"] for row in per_seed]
    result = {"status": "completed", "env_id": native.spec.env_id, "protocol_version": 2,
              "measurement_object": "forward.emb", "weights_loaded_strict": True,
              "planner_objective": "squared_l2_self_goal_not_reward_optimization",
              "n_episodes": args.n_episodes, "n_seeds": args.n_seeds,
              "mean_reward": float(np.mean(means)),
              "mean_reward_std": float(np.std(means, ddof=1)) if len(means) > 1 else None,
              "per_seed": per_seed, "per_episode": per_episode, "repair_provenance": provenance,
              "wall_time_sec": time.monotonic() - started}
    write_new_json(args.out, result)
    print(f"Event-Window reward={result['mean_reward']:.3f}; seeds={args.n_seeds}; output={args.out}")
    return result


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()

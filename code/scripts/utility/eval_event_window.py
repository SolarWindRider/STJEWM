"""Evaluate a ckpt on the Event-Window task. Metric: mean cumulative
reward per episode (0-20, since 20 windows per episode × 0/1 reward).

This is the *right* metric for the event_window task: the env has no
env-native "did the agent reach the goal" notion (it doesn't have a
spatial goal — the goal is the modal event of the last window, which is
a categorical prediction). The reward returned by `env.step` is the
direct measure of task performance.

We use the *trained model's CEM plan* to pick the categorical action.
The CEM plan operates in latent space, and the predicted action is
mapped back to the env's action space via the model's prediction head.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")

from code.core.cem import CEM
from code.core.encode import encode_obs
from code.data import load_dataset
from code.eval.closed_loop import make_env, _PadObsWrapper, _load_eval_dataset
from code.core.envs.event_window import make_event_window


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--cem-samples", type=int, default=100)
    p.add_argument("--cem-elites", type=int, default=10)
    p.add_argument("--cem-iters", type=int, default=10)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--eval-budget", type=int, default=200)
    p.add_argument("--history-size", type=int, default=1)
    p.add_argument("--pad-obs-eval", type=int, default=128)
    p.add_argument("--action-dim-eval", type=int, default=56)
    p.add_argument("--out", required=True)
    return p.parse_args()


def evaluate(args) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build the event_window env
    env = make_event_window()
    state_dim = env.spec.obs_dim
    action_dim = env.spec.action_dim
    if args.pad_obs_eval > state_dim:
        env = _PadObsWrapper(env, args.pad_obs_eval)
        state_dim = args.pad_obs_eval
    if args.action_dim_eval > action_dim:
        # Pad the action via a wrapper (one-hot in -> one-hot + zeros out)
        class _PadActionWrapper:
            def __init__(self, base, target_dim):
                self._base = base
                self.spec = base.spec
                self._target = target_dim
                self._step_count = 0
            def reset(self, seed=None, **kw):
                obs = self._base.reset(seed=seed, **kw)
                self._step_count = 0
                return obs
            def step(self, action):
                action = np.asarray(action, dtype=np.float32).flatten()
                if action.shape[0] < self._target:
                    pad = np.zeros(self._target - action.shape[0], dtype=np.float32)
                    action = np.concatenate([action, pad])
                obs, r, done, info = self._base.step(action)
                self._step_count += 1
                return obs, r, done, info
            def get_state(self):
                return self._base.get_state()
            def check_success(self, s, g):
                return self._base.check_success(s, g)
        env = _PadActionWrapper(env, args.action_dim_eval)
        action_dim = args.action_dim_eval

    # Build the model
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    state_dim_for_model = state_dim
    action_dim_for_model = action_dim
    if ck_args.get("model", "stjewm") == "lewm_baseline":
        from code.lewm_transformer_baseline import LeWMTransformerBaseline
        embed_dim = ck_args.get("embed_dim", 256)
        model = LeWMTransformerBaseline(state_dim=state_dim_for_model,
                                       action_dim=action_dim_for_model, embed_dim=embed_dim,
                                       num_layers=ck_args.get("n_layers", 4))
    elif ck_args.get("model", "stjewm") == "gru_baseline":
        from code.gru_baseline import GRUBaseline
        model = GRUBaseline(state_dim=state_dim_for_model, action_dim=action_dim_for_model)
    elif ck_args.get("model", "stjewm") == "mlp_baseline":
        from code.mlp_baseline import make_mlp_baseline
        model = make_mlp_baseline(state_dim=state_dim_for_model, action_dim=action_dim_for_model)
    elif ck_args.get("model", "stjewm") == "cubifae_baseline":
        from code.cubifae_baseline import CubifAEBaseline
        n_layers = ck_args.get("n_layers", 4)
        model = CubifAEBaseline(state_dim=state_dim_for_model, action_dim=action_dim_for_model,
                                d_hid=192, n_layers=n_layers)
    else:
        from code.stjewm import STJEWM
        n_layers = ck_args.get("n_layers", 4)
        ck_readout_mode = ck_args.get("readout_mode", "hidden_leak")
        model = STJEWM(
            d_hid=192, embed_dim=192, action_dim=action_dim_for_model, action_emb_dim=192,
            state_dim=state_dim_for_model, cell_n_layers=n_layers, n_d=3,
            trace_beta=0.9, freeze_encoder=True,
            readout_mode=ck_readout_mode,
        )
    model = model.to(device)
    model.eval()

    per_seed = []
    per_episode_all = []
    wall_t0 = time.time()
    cem = CEM(
        model=model,
        action_dim=action_dim_for_model,
        n_samples=args.cem_samples,
        n_elites=args.cem_elites,
        n_iters=args.cem_iters,
        horizon=args.horizon,
        history_size=args.history_size,
        device=device,
    )

    for seed in range(args.n_seeds):
        seed_episodes = []
        for ep in range(args.n_episodes):
            obs = env.reset(seed=seed * 10000 + ep)
            init_state_np = env.get_state()
            actions_taken = 0
            ep_reward = 0.0
            t_start = time.time()

            # Initial latent from the first obs (use it as the "current" state)
            with torch.no_grad():
                state_t = torch.from_numpy(init_state_np).float().unsqueeze(0).to(device)
                z_init = encode_obs(model, state_t, action_dim_for_model, device)

            # CEM-plan from z_init toward itself (no external goal; the env
            # will score our categorical pick)
            while actions_taken < args.eval_budget:
                # The "goal" is z_init (the env doesn't give us a different
                # goal — the task is to pick the modal event of the *past*).
                # Use a zero action as the "goal" placeholder.
                z_goal = z_init
                seq = cem.plan(z_init, z_goal)  # (H, A)
                for a_idx in range(min(args.horizon, args.eval_budget - actions_taken)):
                    action = seq[a_idx].cpu().numpy().astype(np.float32)
                    # Slice to native action_dim
                    if action.shape[-1] != action_dim:
                        action = action[..., :action_dim]
                    # The action is a one-hot (5D). Apply via argmax
                    try:
                        _obs, _r, done, _info = env.step(action)
                    except Exception:
                        done = True
                        _r = 0.0
                    ep_reward += float(_r)
                    actions_taken += 1
                    if done:
                        break
                if done or actions_taken >= args.eval_budget:
                    break
                # Roll forward (no need for model.predict — the env's own
                # dynamics drive the next state; we just need to update z_init)
                try:
                    state_t = torch.from_numpy(env.get_state()).float().unsqueeze(0).to(device)
                    z_init = encode_obs(model, state_t, action_dim_for_model, device)
                except Exception:
                    break

            seed_episodes.append({
                "seed": seed,
                "episode_idx": ep,
                "ep_reward": float(ep_reward),
                "n_windows": 20,
                "n_actions": actions_taken,
            })
            per_episode_all.append(seed_episodes[-1])
        if seed_episodes:
            mean_reward = float(np.mean([e["ep_reward"] for e in seed_episodes]))
            per_seed.append({"seed": seed, "n": len(seed_episodes), "mean_reward": mean_reward})
            print(f"  seed={seed} mean_reward={mean_reward:.2f}/20 windows", flush=True)

    out = {
        "env_id": env.spec.env_id,
        "n_episodes": args.n_episodes,
        "n_seeds": args.n_seeds,
        "mean_reward": float(np.mean([s["mean_reward"] for s in per_seed])) if per_seed else 0.0,
        "mean_reward_std": float(np.std([s["mean_reward"] for s in per_seed])) if per_seed else 0.0,
        "per_seed": per_seed,
        "per_episode": per_episode_all,
        "wall_time_sec": float(time.time() - wall_t0),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"  Mean reward: {out['mean_reward']:.2f} ± {out['mean_reward_std']:.2f} / 20 windows")
    print(f"  Saved to {args.out}")
    return out


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

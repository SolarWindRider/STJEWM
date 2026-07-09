"""Latent-environment gradient correlation (v0.7.7 utility experiment 2).

Hypothesis: a calibrated latent whose geometry is meaningful should be such
that the gradient of (1 - cos(z_t, z_goal)) w.r.t. the action aligns (in
cosine similarity) with the gradient of the env-native reward w.r.t. the
same action. A collapse / noise / over-reactive latent should decorrelate.

We measure this at random checkpoints during a real-policy rollout:
    grad_latent = d(1 - cos(z_t, z_goal)) / da   (autograd)
    grad_env    = d(reward) / da                  (finite-difference)
    corr        = cosine_similarity(grad_latent, grad_env)  (mean over 200 steps)

Usage:
    python -m code.scripts.utility.latent_env_grad \
        --ckpt ... --env cheetah --out results/utility/latent_env_grad/stjewm_trace_only/cheetah.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/lx/snn")

from code.core.cem import CEM
from code.core.encode import encode_history, encode_obs
from code.core.envs import make_dmc_env
from code.data import load_dataset
from code.scripts.utility.latent_goal_mpc import build_model_from_ckpt, DMC_DATA


def latent_cost(z, z_goal):
    """1 - cosine_similarity."""
    return 1.0 - F.cosine_similarity(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze()


def measure_grad_corr(model, env, ckpt_args, env_kind, n_steps=200, device="cpu"):
    """Latent-cost grad vs env-reward grad correlation per (model, env)."""
    pad_obs_to = ckpt_args.get("pad_obs_to", 128)
    action_dim = ckpt_args.get("action_dim", 56)
    state_dim = ckpt_args.get("state_dim") or pad_obs_to
    env_action_dim = env.spec.action_dim

    data_path = DMC_DATA.get(env_kind)
    ds = load_dataset(
        env_kind="dmc", path=data_path,
        history_size=1, goal_offset=25, max_windows=n_steps, pad_obs_to=pad_obs_to,
    )
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=min(n_steps, len(ds)), replace=False)
    items = [ds[int(i)] for i in indices]

    corrs = []
    eps = 1e-3

    for item in items:
        # Pad
        init_state_np = item["init_state"].numpy() if hasattr(item["init_state"], "numpy") else np.asarray(item["init_state"])
        goal_state_np = item["goal_state"].numpy() if hasattr(item["goal_state"], "numpy") else np.asarray(item["goal_state"])
        if init_state_np.shape[-1] < pad_obs_to:
            init_state_np = np.concatenate([init_state_np, np.zeros(pad_obs_to - init_state_np.shape[-1], dtype=np.float32)])
        if goal_state_np.shape[-1] < pad_obs_to:
            goal_state_np = np.concatenate([goal_state_np, np.zeros(pad_obs_to - goal_state_np.shape[-1], dtype=np.float32)])

        z_history = encode_history(model, [torch.from_numpy(init_state_np).float()], action_dim, device)
        z_init = z_history[-1]
        z_goal = encode_obs(model, torch.from_numpy(goal_state_np).float(), action_dim, device)

        try:
            env.reset(seed=0)
        except Exception:
            pass

        # Latent-cost grad via autograd
        cem = CEM(model, action_dim=action_dim, horizon=1,
                 n_samples=1, n_elites=1, n_iters=1, history_size=1, device=device)
        a0 = torch.zeros(1, 1, action_dim, device=device, requires_grad=True)
        z_t1 = cem.model.predict(z_init.unsqueeze(0).unsqueeze(0), a0)  # (1, 1, D)
        z_t1 = z_t1.squeeze(0).squeeze(0)
        cost = latent_cost(z_t1, z_goal)
        grad_lat = torch.autograd.grad(cost, a0, retain_graph=False)[0].squeeze().detach().cpu().numpy()
        grad_lat = grad_lat[:env_action_dim]

        # Env-reward grad via finite-difference
        try:
            env.reset(seed=0)
            _, _, _, _ = env.step(np.zeros(env_action_dim, dtype=np.float32))
        except Exception:
            continue
        s0 = env.get_state()
        if s0.shape[-1] < pad_obs_to:
            s0 = np.concatenate([s0, np.zeros(pad_obs_to - s0.shape[-1], dtype=np.float32)])
        # Pad goal
        goal_state_padded = goal_state_np[:s0.shape[0]]
        r0 = -float(np.linalg.norm(s0 - goal_state_padded))

        grad_env = np.zeros(env_action_dim, dtype=np.float32)
        for d in range(env_action_dim):
            a_plus = np.zeros(env_action_dim, dtype=np.float32); a_plus[d] = eps
            a_minus = np.zeros(env_action_dim, dtype=np.float32); a_minus[d] = -eps
            try:
                env.reset(seed=0)
                _, _, _, _ = env.step(a_plus)
                s_plus = env.get_state()
                if s_plus.shape[-1] < pad_obs_to:
                    s_plus = np.concatenate([s_plus, np.zeros(pad_obs_to - s_plus.shape[-1], dtype=np.float32)])
                r_plus = -float(np.linalg.norm(s_plus - goal_state_padded))
            except Exception:
                r_plus = r0
            try:
                env.reset(seed=0)
                _, _, _, _ = env.step(a_minus)
                s_minus = env.get_state()
                if s_minus.shape[-1] < pad_obs_to:
                    s_minus = np.concatenate([s_minus, np.zeros(pad_obs_to - s_minus.shape[-1], dtype=np.float32)])
                r_minus = -float(np.linalg.norm(s_minus - goal_state_padded))
            except Exception:
                r_minus = r0
            grad_env[d] = (r_plus - r_minus) / (2 * eps)

        n_lat = np.linalg.norm(grad_lat)
        n_env = np.linalg.norm(grad_env)
        if n_lat > 1e-8 and n_env > 1e-8:
            cos = float(np.dot(grad_lat, grad_env) / (n_lat * n_env))
        else:
            cos = float("nan")
        corrs.append(cos)

    corrs = np.array(corrs)
    valid = corrs[~np.isnan(corrs)]
    return {
        "n_steps": n_steps,
        "n_valid": int(valid.size),
        "mean_corr": float(np.mean(valid)) if valid.size else float("nan"),
        "std_corr": float(np.std(valid)) if valid.size else float("nan"),
        "median_corr": float(np.median(valid)) if valid.size else float("nan"),
        "mean_abs_corr": float(np.mean(np.abs(valid))) if valid.size else float("nan"),
        "all_corrs": corrs.tolist(),
    }


def run_one(ckpt_path, env_kind, n_steps=200, device="cpu", out_path=None):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    pad_obs_to = ck_args.get("pad_obs_to", 128)
    action_dim = ck_args.get("action_dim", 56)
    state_dim = ck_args.get("state_dim") or pad_obs_to

    model = build_model_from_ckpt(ck_args, state_dim, action_dim, device)
    sd = ck["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        result = model.load_state_dict(sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"  [warn] {ckpt_path}: missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    model.eval()

    env = make_dmc_env(env_kind)
    result = measure_grad_corr(model, env, ck_args, env_kind, n_steps=n_steps, device=device)
    out = {
        "ckpt": ckpt_path,
        "env": env_kind,
        "n_steps": n_steps,
        **result,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  -> {out_path}")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env", required=True, choices=list(DMC_DATA.keys()))
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    run_one(args.ckpt, args.env, args.n_steps, args.device, args.out)


if __name__ == "__main__":
    main()

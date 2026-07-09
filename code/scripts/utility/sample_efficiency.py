"""Frozen-encoder sample efficiency (v0.7.7 utility experiment 3).

Hypothesis: a calibrated latent should be usable by a tiny linear policy
even with little data. A collapse / noise / over-reactive latent should
need more data to reach the same env-SR.

For each (model, env, data_fraction), freeze the world-model encoder,
then train a tiny linear policy pi(z_t) = a_t on the data using a
behavior-cloning loss (MSE between predicted action and dataset action).

Metric: env-SR after training at each data fraction.

Usage:
    python -m code.scripts.utility.sample_efficiency \
        --ckpt ... --env cheetah --out results/utility/sample_efficiency/stjewm_trace_only/cheetah.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/lx/snn")

from code.core.cem import CEM
from code.core.encode import encode_history, encode_obs
from code.core.envs import make_dmc_env
from code.data import load_dataset
from code.scripts.utility.latent_goal_mpc import build_model_from_ckpt, DMC_DATA


def train_linear_policy(model, z_dataset, a_dataset, n_epochs=20, batch_size=128, lr=1e-3, device="cpu"):
    """Train a tiny linear policy pi(z) = W z + b.

    z_dataset: (N, D) tensor of encoded latents.
    a_dataset: (N, A) tensor of dataset actions.
    """
    D = z_dataset.shape[1]
    A = a_dataset.shape[1]
    pi = nn.Linear(D, A).to(device)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    n = z_dataset.shape[0]
    for ep in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            z = z_dataset[idx]
            a = a_dataset[idx]
            a_pred = pi(z)
            loss = F.mse_loss(a_pred, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return pi


def measure_efficiency(model, env, ckpt_args, env_kind, data_fractions=(0.01, 0.05, 0.1, 0.25, 1.0), n_steps=50, device="cpu"):
    pad_obs_to = ckpt_args.get("pad_obs_to", 128)
    action_dim = ckpt_args.get("action_dim", 56)
    state_dim = ckpt_args.get("state_dim") or pad_obs_to
    env_action_dim = env.spec.action_dim

    data_path = DMC_DATA.get(env_kind)
    ds_full = load_dataset(
        env_kind="dmc", path=data_path,
        history_size=1, goal_offset=25, max_windows=10000, pad_obs_to=pad_obs_to,
    )
    n_total = len(ds_full)
    rng = np.random.default_rng(0)

    # Pre-encode all latents (cost = O(N) once)
    print(f"  pre-encoding {n_total} latents...", flush=True)
    z_list, a_list = [], []
    for i in range(n_total):
        item = ds_full[i]
        s_np = item["init_state"].numpy() if hasattr(item["init_state"], "numpy") else np.asarray(item["init_state"])
        if s_np.shape[-1] < pad_obs_to:
            s_np = np.concatenate([s_np, np.zeros(pad_obs_to - s_np.shape[-1], dtype=np.float32)])
        z = encode_obs(model, torch.from_numpy(s_np).float(), action_dim, device)
        z_list.append(z.detach().cpu())
        a_list.append(item["action"][0, :env_action_dim].clone() if hasattr(item["action"], "clone") else torch.as_tensor(item["action"][0, :env_action_dim]))
    z_all = torch.stack(z_list, dim=0)  # (N, D)
    a_all = torch.stack(a_list, dim=0)  # (N, env_action_dim)

    results = {}
    for frac in data_fractions:
        n_train = max(1, int(n_total * frac))
        idx = rng.choice(n_total, size=n_train, replace=False)
        z_train = z_all[idx].to(device)
        a_train = a_all[idx].to(device)

        t0 = time.time()
        pi = train_linear_policy(model, z_train, a_train, n_epochs=20, device=device)
        train_time = time.time() - t0

        # Eval: roll out the learned policy in the env, measure env-native success
        env_successes = []
        cem = CEM(model, action_dim=action_dim, horizon=5, n_samples=10, n_elites=2,
                 n_iters=2, history_size=1, device=device)
        # Use a sample of init states (not the training set)
        eval_indices = rng.choice(n_total, size=min(n_steps, n_total), replace=False)
        cos_dist_terminals = []
        for ei in eval_indices:
            item = ds_full[int(ei)]
            init_state_np = item["init_state"].numpy() if hasattr(item["init_state"], "numpy") else np.asarray(item["init_state"])
            goal_state_np = item["goal_state"].numpy() if hasattr(item["goal_state"], "numpy") else np.asarray(item["goal_state"])
            if init_state_np.shape[-1] < pad_obs_to:
                init_state_np = np.concatenate([init_state_np, np.zeros(pad_obs_to - init_state_np.shape[-1], dtype=np.float32)])
            if goal_state_np.shape[-1] < pad_obs_to:
                goal_state_np = np.concatenate([goal_state_np, np.zeros(pad_obs_to - goal_state_np.shape[-1], dtype=np.float32)])
            z_goal = encode_obs(model, torch.from_numpy(goal_state_np).float(), action_dim, device)
            try:
                env.reset(seed=int(ei))
            except Exception:
                pass
            # Roll out the linear policy
            z_t = encode_obs(model, torch.from_numpy(init_state_np).float(), action_dim, device)
            done = False
            t = 0
            while not done and t < 50:
                with torch.no_grad():
                    a_pi = pi(z_t.unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)
                a_pi = np.clip(a_pi, env.spec.action_low, env.spec.action_high)
                try:
                    _obs, _r, done, _info = env.step(a_pi)
                except Exception:
                    done = True
                t += 1
            try:
                final = env.get_state()
            except Exception:
                final = init_state_np
            if final.shape[-1] < pad_obs_to:
                final = np.concatenate([final, np.zeros(pad_obs_to - final.shape[-1], dtype=np.float32)])
            z_final = encode_obs(model, torch.from_numpy(final).float(), action_dim, device)
            cos = F.cosine_similarity(z_final.unsqueeze(0), z_goal.unsqueeze(0))
            cos_dist_terminals.append(float((1.0 - cos.item()) / 2.0))
            try:
                ok, _ = env.check_success(final, goal_state_np[:final.shape[0]])
            except Exception:
                ok = False
            env_successes.append(1.0 if ok else 0.0)

        results[f"{frac:.3f}"] = {
            "data_fraction": frac,
            "n_train": n_train,
            "env_success": float(np.mean(env_successes)),
            "env_success_std": float(np.std(env_successes)),
            "mean_cos_dist_terminal": float(np.mean(cos_dist_terminals)),
            "mean_cos_dist_terminal_std": float(np.std(cos_dist_terminals)),
            "train_time_sec": train_time,
        }

    return {
        "n_steps": n_steps,
        "n_total": n_total,
        "per_fraction": results,
    }


def run_one(ckpt_path, env_kind, n_steps=50, data_fractions=(0.01, 0.05, 0.1, 0.25, 1.0), device="cpu", out_path=None):
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
    res = measure_efficiency(model, env, ck_args, env_kind, data_fractions, n_steps, device)
    out = {
        "ckpt": ckpt_path,
        "env": env_kind,
        "n_steps": n_steps,
        "data_fractions": list(data_fractions),
        **res,
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
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--fractions", type=str, default="0.01,0.05,0.1,0.25,1.0")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    fractions = tuple(float(f) for f in args.fractions.split(","))
    run_one(args.ckpt, args.env, args.n_steps, fractions, args.device, args.out)


if __name__ == "__main__":
    main()

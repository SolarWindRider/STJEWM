"""Frozen-encoder sample efficiency (v0.7.7 utility experiment 3).

For each (model, env, data_fraction), freeze the world-model encoder and
behavior-clone a tiny linear policy pi(z_t) = a_t on a fraction of the
windows. The held-out evaluation windows are drawn once, before any
fraction is seen, and no training index may overlap them.

Each fraction is evaluated by rolling out the learned policy in the native
environment from the held-out init states, re-encoding the observed state
every step. Success is env-native (closed_loop's reacher override applies);
native physical distance is recorded alongside the terminal cosine distance.

Source-only producer: requires the final consolidated training repair
manifest and writes a status/protocol/provenance-stamped JSON to a fresh
output path, never overwriting.

Usage:
    python -m code.scripts.utility.sample_efficiency \
        --ckpt ... --model stjewm_trace_only --training-manifest <final.json> \
        --env cheetah --out <fresh>/stjewm_trace_only/cheetah.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/lx/snn")

from code.core.encode import encode_obs
from code.core.envs import make_dmc_env
from code.data import load_dataset
from code.eval.closed_loop import _set_env_state
from code.scripts.audited_results import TrainingAudit, heldout_native_success, require
from code.scripts.utility.latent_goal_mpc import DMC_DATA


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
    require(pad_obs_to >= env.spec.obs_dim and action_dim >= env.spec.action_dim, "Checkpoint cannot represent this task")
    env_action_dim = env.spec.action_dim

    data_path = DMC_DATA.get(env_kind)
    ds_full = load_dataset(
        env_kind="dmc", path=data_path,
        history_size=1, goal_offset=25, max_windows=10000, pad_obs_to=pad_obs_to,
    )
    n_total = len(ds_full)
    rng = np.random.default_rng(0)

    require(0 < n_steps < n_total, "Need positive held-out coverage and at least one remaining training window")
    require(data_fractions and all(0 < frac <= 1 for frac in data_fractions), "Fractions must be in (0, 1]")
    require(len({f"{frac:.3f}" for frac in data_fractions}) == len(data_fractions), "Duplicate fraction output keys")
    # Draw the holdout once. Fractions refer to the remaining training pool.
    order = rng.permutation(n_total)
    eval_indices = order[:n_steps].tolist()
    training_pool = order[n_steps:]

    def observed_latent():
        state = torch.as_tensor(env.get_state(), dtype=torch.float32)
        state = F.pad(state, (0, pad_obs_to - state.numel()))
        return encode_obs(model, state, action_dim, device)

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
    training_by_fraction = {}
    for frac in data_fractions:
        n_train = max(1, int(len(training_pool) * frac))
        idx = training_pool[:n_train]
        training_by_fraction[f"{frac:.3f}"] = sorted(int(value) for value in idx)
        z_train = z_all[idx].to(device)
        a_train = a_all[idx].to(device)

        t0 = time.time()
        torch.manual_seed(0)
        pi = train_linear_policy(model, z_train, a_train, n_epochs=20, device=device)
        train_time = time.time() - t0

        # Eval: roll out the learned policy on the fixed held-out init states
        env_successes = []
        # Use the fixed held-out sample drawn before the fraction loop
        cos_dist_terminals = []
        phys_dists = []
        for ei in eval_indices:
            item = ds_full[int(ei)]
            init_state_np = item["init_state"].numpy() if hasattr(item["init_state"], "numpy") else np.asarray(item["init_state"])
            goal_state_np = item["goal_state"].numpy() if hasattr(item["goal_state"], "numpy") else np.asarray(item["goal_state"])
            if init_state_np.shape[-1] < pad_obs_to:
                init_state_np = np.concatenate([init_state_np, np.zeros(pad_obs_to - init_state_np.shape[-1], dtype=np.float32)])
            if goal_state_np.shape[-1] < pad_obs_to:
                goal_state_np = np.concatenate([goal_state_np, np.zeros(pad_obs_to - goal_state_np.shape[-1], dtype=np.float32)])
            z_goal = encode_obs(model, torch.from_numpy(goal_state_np).float(), action_dim, device)
            env.reset(seed=int(ei))
            native_init = init_state_np[:env.spec.obs_dim]
            native_goal = goal_state_np[:env.spec.obs_dim]
            _set_env_state(env, native_init)
            require(np.allclose(env.get_state(), native_init, rtol=1e-5, atol=1e-4), "Failed to restore held-out state")
            # Roll out the linear policy, re-encoding each observed state so the
            # policy actually reacts to the simulator trajectory.
            z_t = observed_latent()
            done = False
            t = 0
            while not done and t < 50:
                with torch.no_grad():
                    a_pi = pi(z_t.unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)
                a_pi = np.clip(a_pi, env.spec.action_low, env.spec.action_high)
                _obs, _r, done, _info = env.step(a_pi)
                z_t = observed_latent()
                t += 1
            final = env.get_state()
            z_final = observed_latent()
            cos = F.cosine_similarity(z_final.unsqueeze(0), z_goal.unsqueeze(0))
            cos_dist_terminals.append(float((1.0 - cos.item()) / 2.0))
            ok, phys = heldout_native_success(env_kind, env, final, native_goal)
            env_successes.append(1.0 if ok else 0.0)
            require(np.isfinite(phys), "Non-finite physical distance in held-out rollout")
            phys_dists.append(float(phys))

        results[f"{frac:.3f}"] = {
            "data_fraction": frac,
            "n_train": n_train,
            "n_eval": len(eval_indices),
            "training_pool_size": len(training_pool),
            "env_success": float(np.mean(env_successes)),
            "env_success_std": float(np.std(env_successes)),
            "mean_phys_dist": float(np.mean(phys_dists)),
            "phys_dist_std": float(np.std(phys_dists)),
            "mean_cos_dist_terminal": float(np.mean(cos_dist_terminals)),
            "mean_cos_dist_terminal_std": float(np.std(cos_dist_terminals)),
            "train_time_sec": train_time,
        }

    return {
        "n_steps": n_steps,
        "n_total": n_total,
        "training_pool_size": len(training_pool),
        "fraction_denominator": "nonheldout_training_windows",
        "heldout_indices": [int(value) for value in eval_indices],
        "training_indices_by_fraction": training_by_fraction,
        "per_fraction": results,
    }


def run_one(ckpt_path, model_name, env_kind, n_steps=50, data_fractions=(0.01, 0.05, 0.1, 0.25, 1.0), device="cpu", out_path=None,
            training_manifest=None):
    ckpt_path = str(Path(ckpt_path).resolve())
    require(training_manifest is not None, "A final training manifest is required")
    require(Path(ckpt_path).parent.parent.name == model_name, "Checkpoint/model identity mismatch")
    audit = TrainingAudit(training_manifest)
    provenance = audit.provenance(ckpt_path)
    if out_path is not None:
        audit.protect_output(out_path)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    pad_obs_to = ck_args.get("pad_obs_to", 128)
    action_dim = ck_args.get("action_dim", 56)
    state_dim = ck_args.get("state_dim") or pad_obs_to

    from code.scripts.utility.latent_goal_mpc import build_model_from_ckpt
    model = build_model_from_ckpt(ck_args, state_dim, action_dim, device)
    sd = ck["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    env = make_dmc_env(env_kind)
    try:
        res = measure_efficiency(model, env, ck_args, env_kind, data_fractions, n_steps, device)
    finally:
        env.close()
    require(audit.checkpoint(ckpt_path)["sha256"] == provenance["checkpoint_sha256"], "Checkpoint changed during evaluation")
    out = {
        "status": "completed",
        "protocol_version": 2,
        "measurement_object": "forward.emb",
        "weights_loaded_strict": True,
        "restore": "state_vector_exact_dmc",
        "model": model_name,
        "ckpt": ckpt_path,
        "env": env_kind,
        "n_steps": n_steps,
        "data_fractions": list(data_fractions),
        **res,
        "repair_provenance": provenance,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if out_path:
        out_path = Path(out_path)
        require(not out_path.exists(), f"Refusing existing sample-efficiency output: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("x") as f:
            json.dump(out, f, indent=2, allow_nan=False)
        print(f"  -> {out_path}")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--training-manifest", required=True)
    p.add_argument("--env", required=True, choices=list(DMC_DATA.keys()))
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--fractions", type=str, default="0.01,0.05,0.1,0.25,1.0")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    fractions = tuple(float(f) for f in args.fractions.split(","))
    run_one(args.ckpt, args.model, args.env, args.n_steps, fractions, args.device, args.out, args.training_manifest)


if __name__ == "__main__":
    main()

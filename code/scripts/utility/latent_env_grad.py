"""Compare action gradients at identical restored offline DMC states.

The latent objective is 1-cosine(predicted readout, goal readout). The physical
objective is negative native goal-state L2 distance, NOT environment reward.
Their signed cosine is undefined when either gradient has zero norm.
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

from code.core.encode import encode_history, encode_obs
from code.core.envs import make_dmc_env
from code.data import load_dataset
from code.eval.closed_loop import _set_env_state
from code.scripts.audited_results import TrainingAudit, require
from code.scripts.utility.latent_goal_mpc import build_model_from_ckpt, DMC_DATA



def latent_cost(z, z_goal):
    """1 - cosine_similarity."""
    return 1.0 - F.cosine_similarity(z.unsqueeze(0), z_goal.unsqueeze(0)).squeeze()


def measure_grad_corr(model, env, ckpt_args, env_kind, n_steps=200, device="cpu"):
    """Latent-cost gradient versus negative native goal-distance gradient."""
    pad_obs_to = ckpt_args.get("pad_obs_to", 128)
    action_dim = ckpt_args.get("action_dim", 56)
    env_action_dim = env.spec.action_dim
    require(pad_obs_to >= env.spec.obs_dim and action_dim >= env_action_dim, "Checkpoint cannot represent native controls/state")

    data_path = DMC_DATA.get(env_kind)
    ds = load_dataset(
        env_kind="dmc", path=data_path,
        history_size=1, goal_offset=25, max_windows=n_steps, pad_obs_to=pad_obs_to,
    )
    require(0 < n_steps <= len(ds), "Insufficient legal windows for the requested gradient budget")
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=n_steps, replace=False)

    corrs = []
    eps = 1e-3

    for index in indices:
        item = ds[int(index)]
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

        # Only native controls vary; padded model controls stay exactly zero.
        a0 = torch.zeros(1, 1, env_action_dim, device=device, requires_grad=True)
        a_model = F.pad(a0, (0, action_dim - env_action_dim))
        z_t1 = model.predict(z_init.unsqueeze(0).unsqueeze(0), a_model)
        z_t1 = z_t1.squeeze(0).squeeze(0)
        cost = latent_cost(z_t1, z_goal)
        grad_lat = torch.autograd.grad(cost, a0, retain_graph=False)[0].reshape(-1).detach().cpu().numpy()

        native_init = init_state_np[:env.spec.obs_dim]
        native_goal = goal_state_np[:env.spec.obs_dim]

        def physical_objective(action):
            env.reset(seed=0)
            _set_env_state(env, native_init)
            require(np.allclose(env.get_state(), native_init, rtol=1e-5, atol=1e-4), "Failed to restore derivative state")
            env.step(action)
            return -float(np.linalg.norm(env.get_state() - native_goal))

        grad_env = np.zeros(env_action_dim, dtype=np.float32)
        for d in range(env_action_dim):
            a_plus = np.zeros(env_action_dim, dtype=np.float32); a_plus[d] = eps
            a_minus = np.zeros(env_action_dim, dtype=np.float32); a_minus[d] = -eps
            r_plus = physical_objective(a_plus)
            r_minus = physical_objective(a_minus)
            grad_env[d] = (r_plus - r_minus) / (2 * eps)
        require(np.isfinite(grad_lat).all() and np.isfinite(grad_env).all(), "Non-finite action gradient")

        n_lat = np.linalg.norm(grad_lat)
        n_env = np.linalg.norm(grad_env)
        if n_lat > 1e-8 and n_env > 1e-8:
            corrs.append(float(np.dot(grad_lat, grad_env) / (n_lat * n_env)))
        else:
            # A zero gradient is mathematically undefined, not a zero correlation.
            corrs.append(None)

    valid = [float(value) for value in corrs if value is not None]
    require(len(corrs) == n_steps, "Gradient rollout dropped a sampled window")
    return {
        "n_steps": n_steps,
        "n_valid": len(valid),
        "n_undefined": len(corrs) - len(valid),
        "mean_corr": float(np.mean(valid)) if valid else None,
        "std_corr": float(np.std(valid)) if valid else None,
        "median_corr": float(np.median(valid)) if valid else None,
        "mean_abs_corr": float(np.mean(np.abs(valid))) if valid else None,
        "all_corrs": corrs,
    }


def run_one(ckpt_path, model_name, env_kind, n_steps=200, device="cpu", out_path=None, training_manifest=None):
    ckpt_path = str(Path(ckpt_path).resolve())
    require(training_manifest is not None, "A training manifest is required for source-only producer output")
    audit = TrainingAudit(training_manifest)
    require(Path(ckpt_path).parent.parent.name == model_name, "Checkpoint/model identity mismatch")
    if out_path is not None:
        audit.protect_output(out_path)
    provenance = audit.provenance(ckpt_path)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    pad_obs_to = ck_args.get("pad_obs_to", 128)
    action_dim = ck_args.get("action_dim", 56)
    state_dim = ck_args.get("state_dim") or pad_obs_to

    model = build_model_from_ckpt(ck_args, state_dim, action_dim, device)
    sd = ck["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    env = make_dmc_env(env_kind)
    try:
        result = measure_grad_corr(model, env, ck_args, env_kind, n_steps=n_steps, device=device)
    finally:
        env.close()
    require(audit.checkpoint(ckpt_path)["sha256"] == provenance["checkpoint_sha256"], "Checkpoint changed during evaluation")
    require(result["n_valid"] + result["n_undefined"] == n_steps, "Gradient rollout coverage mismatch")
    if result["n_undefined"]:
        print(f"  [warn] {result['n_undefined']} undefined gradient correlations (zero gradient); reported explicitly")
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
        **result,
        "latent_objective": "1-cosine(predicted_forward_emb, goal_forward_emb)",
        "physical_objective": "negative native goal-state L2 distance",
        "repair_provenance": provenance,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if out_path:
        out_path = Path(out_path)
        require(not out_path.exists(), f"Refusing existing gradient output: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("x") as handle:
            handle.write(json.dumps(out, indent=2, allow_nan=False) + "\n")
        print(f"  -> {out_path}")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--training-manifest", required=True)
    p.add_argument("--env", required=True, choices=list(DMC_DATA.keys()))
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    run_one(args.ckpt, args.model, args.env, args.n_steps, args.device, args.out, args.training_manifest)


if __name__ == "__main__":
    main()

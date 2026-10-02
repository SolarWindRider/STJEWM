"""Seeded, checkpoint-backed state diagnostics at explicitly named interfaces.

All models are evaluated on the same single-frame observation/action protocol.
The primary representation is forward()["emb"] (the model readout); the
observation-side forward()["emb_pre_cell"] is recorded separately. These are
not interchangeable with persistent internal trace state. Reset boundaries are
excluded from first-difference metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from code.scripts.event_align import ENV_KIND_MAP, build_model

PROTOCOL_VERSION = 2


def extract_representations(model, state: torch.Tensor, action: torch.Tensor):
    """Use the same explicit forward fields for every architecture."""
    out = model(state, action)
    return out, {"readout": out["emb"], "observation_embedding": out["emb_pre_cell"]}


def trajectory_metrics(observations, latents, episode_ids):
    obs = np.asarray(observations, dtype=np.float64)
    z = np.asarray(latents, dtype=np.float64)
    if not np.isfinite(obs).all() or not np.isfinite(z).all():
        raise ValueError("Non-finite observations or representations in diagnostic rollout")
    valid = np.diff(episode_ids) == 0
    d_obs = np.linalg.norm(np.diff(obs, axis=0), axis=1)[valid]
    d_z = np.linalg.norm(np.diff(z, axis=0), axis=1)[valid]
    if len(d_obs) < 2:
        raise ValueError("At least two within-episode transitions are required")
    xc, yc = d_obs - d_obs.mean(), d_z - d_z.mean()
    denom = np.linalg.norm(xc) * np.linalg.norm(yc)
    rho = float(np.dot(xc, yc) / denom) if denom > 0 else None
    std = z.std(axis=0)
    return {
        "divergence": float(std.mean()),
        "responsiveness": float(d_z.mean() / d_obs.mean()) if d_obs.mean() > 0 else None,
        "event_rho": rho,
        "per_dim_std_max": float(std.max()),
        "per_dim_std_min": float(std.min()),
        "mean_norm_latent": float(np.linalg.norm(z, axis=1).mean()),
        "mean_d_obs": float(d_obs.mean()),
        "mean_d_lat": float(d_z.mean()),
        "n_transitions": int(valid.sum()),
    }


@torch.inference_mode()
def collect_state_rollout(model, env, state_dim, action_dim, *, n_steps=200,
                          n_resets=2, seed=0, device="cpu"):
    if n_resets < 1 or n_steps < 3 * n_resets:
        raise ValueError("Each requested rollout segment requires at least three observations")
    rng = np.random.default_rng(seed)
    observations, actions, episodes = [], [], []
    episode, previous_segment, done = -1, -1, True
    for t in range(n_steps):
        segment = t * n_resets // n_steps
        if done or segment != previous_segment:
            episode += 1
            env.reset(seed=seed + episode)
            previous_segment = segment
        obs = np.asarray(env.get_state(), dtype=np.float32)
        if len(obs) > state_dim:
            raise ValueError(f"Observation dim {len(obs)} exceeds checkpoint dim {state_dim}")
        padded_obs = np.zeros(state_dim, dtype=np.float32)
        padded_obs[:len(obs)] = obs
        action = rng.uniform(env.spec.action_low, env.spec.action_high).astype(np.float32)
        if len(action) > action_dim:
            raise ValueError(f"Action dim {len(action)} exceeds checkpoint dim {action_dim}")
        padded_action = np.zeros(action_dim, dtype=np.float32)
        padded_action[:len(action)] = action
        observations.append(padded_obs)
        actions.append(padded_action)
        episodes.append(episode)
        _, _, done, _ = env.step(action)
    obs_arr = np.stack(observations)
    episode_arr = np.asarray(episodes, dtype=np.int32)
    action_arr = np.stack(actions)
    out, fields = extract_representations(
        model, torch.from_numpy(obs_arr[:, None]).to(device),
        torch.from_numpy(action_arr[:, None]).to(device),
    )
    representations = {name: value[:, 0].float().cpu().numpy() for name, value in fields.items()}
    arrays = {
        "obs_arr": obs_arr,
        "action_arr": action_arr,
        "episode_id": episode_arr,
        "lat_arr": representations["readout"],
        "embedding_arr": representations["observation_embedding"],
    }
    metrics = {
        name: trajectory_metrics(obs_arr, values, episode_arr)
        for name, values in representations.items()
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "skipped": False,
        "measurement_object": "forward.emb",
        "context_steps": 1,
        "observation_embedding_object": "forward.emb_pre_cell",
        "seed": seed,
        "n_steps": n_steps,
        "n_episodes": episode + 1,
        "n_resets": episode,
        "requested_segments": n_resets,
        "representations": metrics,
        **metrics["readout"],
    }
    if "spike" in out:
        rate_arr = out["spike"][:, 0].float().mean(dim=-1).cpu().numpy().astype(np.float64)
        arrays["spike_rate_arr"] = rate_arr
        valid = np.diff(episode_arr) == 0
        x = np.linalg.norm(np.diff(obs_arr.astype(np.float64), axis=0), axis=1)[valid]
        y = rate_arr[1:][valid]
        xc, yc = x - x.mean(), y - y.mean()
        denom = np.linalg.norm(xc) * np.linalg.norm(yc)
        result["corr_obs_rate"] = float(np.dot(xc, yc) / denom) if denom > 0 else None
    else:
        result["corr_obs_rate"] = None
    return result, arrays


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--env", required=True, choices=sorted(ENV_KIND_MAP))
    p.add_argument("--out", required=True)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--n-resets", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--pad-obs-to", type=int)
    p.add_argument("--action-dim-eval", type=int)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ck.get("args", {})
    from code.eval.closed_loop import make_env
    env = make_env(ENV_KIND_MAP[args.env], data_path=None)
    try:
        state_dim = args.pad_obs_to or saved.get("pad_obs_to") or env.spec.obs_dim
        action_dim = args.action_dim_eval or saved.get("action_dim") or env.spec.action_dim
        model = build_model(args.model, state_dim, action_dim, saved, state_dict=ck["model"])
        model.load_state_dict(ck["model"], strict=True)
        model.to(args.device).eval()
        result, arrays = collect_state_rollout(
            model, env, state_dim, action_dim, n_steps=args.n_steps,
            n_resets=args.n_resets, seed=args.seed, device=args.device,
        )
    finally:
        env.close()
    result.update(env=args.env, model=args.model, ckpt=args.ckpt, weights_loaded_strict=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    np.savez(path.with_suffix(".npz"), **arrays)
    print(f"[latent_rollout] {args.env}/{args.model}: "
          f"div={result['divergence']:.6g} resp={result['responsiveness']} "
          f"rho={result['event_rho']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

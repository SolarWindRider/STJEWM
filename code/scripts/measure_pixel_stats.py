#!/usr/bin/env python3
"""Pixel-mode div/resp latent stats for all 5m_pixel checkpoints.

resp = mean||Δlatent|| / mean||Δphysical state|| over 200 random-policy steps;
div  = mean per-dim std of the latent. One JSON per (split, model, env).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")

from code.core.envs.dmc_env import DMCPixelEnv  # noqa: E402
from code.train.train import build_model  # noqa: E402

ENVS = ["cartpole", "cheetah", "ball_in_cup", "finger"]


def measure_pixel(ckpt_path, env_kind, image_size, n_steps, device):
    import mujoco  # noqa: F401
    env = DMCPixelEnv(env_kind, image_size=image_size, success_tol=0.1,
                      max_episode_steps=n_steps + 10)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ck.get("args", {})
    model = build_model(
        saved.get("model", "stjewm"), obs_dim=saved.get("pad_obs_to", 21168),
        action_dim=saved.get("action_dim", 56), n_layers=saved.get("n_layers", 4),
        readout_mode=saved.get("readout_mode", "hidden_leak"),
        embed_dim=saved.get("embed_dim"), image_size=saved.get("image_size", image_size),
    )
    model.load_state_dict(ck["model"])  # strict
    model.to(device).eval()

    a_low = np.array(env.spec.action_low, dtype=np.float32)
    a_high = np.array(env.spec.action_high, dtype=np.float32)
    obs = env.reset(seed=0)
    state = np.asarray(env.get_state(), dtype=np.float32)
    obs_traj, lat_traj, st_traj = [], [], [state]
    rng = np.random.default_rng(0)

    def encode(pixel, act):
        x = torch.from_numpy(np.asarray(pixel)).float().reshape(1, 1, 3, image_size, image_size).to(device)
        a = torch.from_numpy(act).float().reshape(1, 1, -1).to(device)
        with torch.no_grad():
            out = model(x, a)
        emb = out["emb"] if isinstance(out, dict) else out
        return emb[0, -1].cpu().numpy()

    act0 = np.zeros(saved.get("action_dim", 56), dtype=np.float32)
    lat_traj.append(encode(obs["pixel"], act0))
    obs_traj.append(obs["pixel"])
    for t in range(n_steps):
        a = rng.uniform(a_low, a_high).astype(np.float32)
        a_pad = np.zeros(saved.get("action_dim", 56), dtype=np.float32)
        a_pad[: len(a)] = a
        obs, _, done, _ = env.step(a)
        state = np.asarray(env.get_state(), dtype=np.float32)
        st_traj.append(state)
        obs_traj.append(obs["pixel"])
        lat_traj.append(encode(obs["pixel"], a_pad))
        if done:
            obs = env.reset(seed=seed_offset(t))
            state = np.asarray(env.get_state(), dtype=np.float32)
            st_traj[-1] = state
            lat_traj[-1] = encode(obs["pixel"], act0)
            obs_traj[-1] = obs["pixel"]
    st_arr = np.stack(st_traj)
    lat_arr = np.stack(lat_traj)
    d_st = np.linalg.norm(np.diff(st_arr, axis=0), axis=1)
    d_lat = np.linalg.norm(np.diff(lat_arr, axis=0), axis=1)
    resp = float(d_lat.mean() / d_st.mean()) if d_st.mean() > 1e-9 else 0.0
    div = float(lat_arr.std(axis=0).mean())
    env.close()
    return {"env": env_kind, "n_steps": n_steps,
            "responsiveness": resp, "divergence": div,
            "latent_std_max": float(lat_arr.std(axis=0).max()),
            "latent_std_min": float(lat_arr.std(axis=0).min())}


def seed_offset(t):
    return t + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("/data/lx/tmp/results/5m_pixel"))
    p.add_argument("--out", type=Path, default=Path("/data/lx/tmp/results/5m_pixel_stats"))
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image-size", type=int, default=84)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for split_dir in sorted(args.results.iterdir()):
        if not split_dir.is_dir():
            continue
        for model_dir in sorted(split_dir.iterdir()):
            ck = model_dir / "seed_0" / "final.pt"
            if not ck.exists():
                continue
            for env in ENVS:
                out = args.out / split_dir.name / model_dir.name / f"latent_stats_{env}.json"
                if out.exists():
                    continue
                tasks.append((split_dir.name, model_dir.name, env, ck, out))
    print(f"[pixel_stats] tasks: {len(tasks)}", flush=True)
    errs = 0
    for i, (split, model, env, ck, out) in enumerate(tasks):
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = measure_pixel(str(ck), env, args.image_size, args.n_steps, args.device)
            out.write_text(json.dumps(r, indent=2))
            print(f"  [{i+1}/{len(tasks)}] {split}/{model}/{env} "
                  f"resp={r['responsiveness']:.3f} div={r['divergence']:.4f}", flush=True)
        except Exception as e:
            errs += 1
            out.write_text(json.dumps({"skipped": True, "reason": str(e)[:200]}, indent=2))
            print(f"  ERR {split}/{model}/{env}: {e}", flush=True)
    print(f"[pixel_stats] done, {errs} errors", flush=True)


if __name__ == "__main__":
    main()

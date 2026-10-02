#!/usr/bin/env python3
"""Pixel readout diagnostics with physical-state normalization and explicit interfaces."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from code.core.envs.dmc_env import DMCPixelEnv
from code.scripts.latent_rollout import PROTOCOL_VERSION, extract_representations, trajectory_metrics
from code.train.train import build_model

ENVS = ["cartpole", "cheetah", "ball_in_cup", "finger"]


@torch.inference_mode()
def measure_pixel(ckpt_path, env_kind, image_size, n_steps, device, *, seed=0, out_npz=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ck.get("args", {})
    action_dim = saved.get("action_dim") or 56
    image_size = saved.get("image_size") or image_size
    model = build_model(
        saved.get("model", "stjewm"), obs_dim=saved.get("pad_obs_to") or 3 * image_size ** 2,
        action_dim=action_dim, n_layers=saved.get("n_layers", 4),
        readout_mode=saved.get("readout_mode", "hidden_leak"),
        embed_dim=saved.get("embed_dim"), image_size=image_size,
        hidden_dim=saved.get("hidden_dim"), mlp_hidden=saved.get("mlp_hidden"),
        mlp_layers=saved.get("mlp_layers"),
    )
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    env = DMCPixelEnv(env_kind, image_size=image_size,
                      max_episode_steps=n_steps + 10)
    rng = np.random.default_rng(seed)
    states, actions, episodes, pixels = [], [], [], []
    episode, previous_segment, done = -1, -1, True
    try:
        for t in range(n_steps):
            segment = t * 2 // n_steps
            if done or segment != previous_segment:
                episode += 1
                obs = env.reset(seed=seed + episode)
                previous_segment = segment
            native_action = rng.uniform(env.spec.action_low, env.spec.action_high).astype(np.float32)
            action = np.zeros(action_dim, dtype=np.float32)
            action[:len(native_action)] = native_action
            states.append(np.asarray(env.get_state(), dtype=np.float32))
            pixels.append(np.asarray(obs["pixel"], dtype=np.float32))
            actions.append(action)
            episodes.append(episode)
            obs, _, done, _ = env.step(native_action)
    finally:
        env.close()
    state_arr = np.stack(states)
    episode_arr = np.asarray(episodes, dtype=np.int32)
    action_arr = np.stack(actions)
    representations = {"readout": [], "observation_embedding": []}
    # Frames are independent context-length-one samples, not one recurrent sequence.
    for start in range(0, n_steps, 64):
        stop = min(start + 64, n_steps)
        _, fields = extract_representations(
            model, torch.from_numpy(np.stack(pixels[start:stop])[:, None]).to(device),
            torch.from_numpy(action_arr[start:stop, None]).to(device),
        )
        for name, value in fields.items():
            representations[name].append(value[:, 0].float().cpu().numpy())
    representations = {name: np.concatenate(values) for name, values in representations.items()}
    arrays = {
        "obs_arr": state_arr,
        "action_arr": action_arr,
        "episode_id": episode_arr,
        "lat_arr": representations["readout"],
        "embedding_arr": representations["observation_embedding"],
    }
    metrics = {
        name: trajectory_metrics(state_arr, values, episode_arr)
        for name, values in representations.items()
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "skipped": False,
        "ckpt": str(ckpt_path),
        "weights_loaded_strict": True,
        "env": env_kind,
        "modality": "pixel",
        "normalization_observation": "physical_state",
        "measurement_object": "forward.emb",
        "observation_embedding_object": "forward.emb_pre_cell",
        "context_steps": 1,
        "seed": seed,
        "n_steps": n_steps,
        "n_episodes": episode + 1,
        "representations": metrics,
        **metrics["readout"],
    }
    if out_npz is not None:
        Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz, **arrays)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("/data/lx/tmp/results/5m_pixel"))
    p.add_argument("--out", type=Path, default=Path("/data/lx/tmp/results/5m_pixel_stats"))
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", nargs="+")
    p.add_argument("--models", nargs="+")
    p.add_argument("--envs", nargs="+", default=ENVS)
    args = p.parse_args()
    tasks = []
    for split_dir in sorted(args.results.iterdir()):
        if not split_dir.is_dir() or (args.splits and split_dir.name not in args.splits):
            continue
        for model_dir in sorted(split_dir.iterdir()):
            if args.models and model_dir.name not in args.models:
                continue
            ck = model_dir / "seed_0" / "final.pt"
            if not ck.exists():
                continue
            for env in args.envs:
                out = args.out / split_dir.name / model_dir.name / f"latent_stats_{env}.json"
                if out.exists():
                    previous = json.loads(out.read_text())
                    if previous.get("protocol_version") == PROTOCOL_VERSION:
                        continue
                    raise RuntimeError(f"Archive obsolete diagnostics before rerunning: {out}")
                tasks.append((split_dir.name, model_dir.name, env, ck, out))
    print(f"[pixel_stats] tasks: {len(tasks)}", flush=True)
    failures = []
    for i, (split, model, env, ck, out) in enumerate(tasks):
        try:
            r = measure_pixel(str(ck), env, args.image_size, args.n_steps, args.device,
                              seed=args.seed, out_npz=out.with_suffix(".npz"))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(r, indent=2, allow_nan=False))
            print(f"[{i+1}/{len(tasks)}] {split}/{model}/{env} "
                  f"resp={r['responsiveness']} div={r['divergence']:.6g}", flush=True)
        except Exception as exc:
            failures.append(f"{split}/{model}/{env}: {exc}")
            print(f"ERR {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("Pixel diagnostics failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()

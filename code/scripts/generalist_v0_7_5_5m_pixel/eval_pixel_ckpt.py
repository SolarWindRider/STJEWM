#!/usr/bin/env python3
"""Closed-loop pixel evaluation for the 13 planned DMC environments.

Protocol: single-frame model(x, zero_action)["emb"] latents; CEM terminal
squared-L2 cost, history 1, horizon 5, replanning after every executed action;
50 actions per episode and one MuJoCo integration step per action. Fixed
physical goals are rendered and scored in the same native state coordinates.
The reported scalar latent distance is (1 - cosine(final_z, goal_z)) / 2.

The repaired planner searches native controls, clamps them to environment
bounds, then zero-pads to model width for prediction. This corrects the old
phantom-action protocol and requires rerunning every planned cell. Outputs
must be written to a fresh directory, never over existing evaluation files.

The original runtime attempted 13 environments including the canonical
``humanoid_CMU`` key, which failed only because the constructor never received
the native ``humanoid_cmu`` spelling. Results keep the canonical key while
routing construction through the native registry name.
"""
import sys
from pathlib import Path

ROOT = Path("/home/lx/snn")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, "/home/lx/LeWM")

# Force-import the user `code` package BEFORE numpy/torch/etc, because
# those packages transitively import the stdlib `code` module (via codeop)
# which then shadows the user package (Python caches it as a module,
# not a package, so `code.train` becomes unfindable).
import code as _code_pkg  # noqa: F401
from code.train.train import build_model
from code.scripts.audited_results import TrainingAudit, sha256
from code.core.cem import CEM, make_native_action_predict_hook
from code.core.envs.dmc_env import DMCPixelEnv  # noqa: F401

import argparse
import json
import traceback

import numpy as np
import torch

# The original runtime planned 13 environments; humanoid_CMU keeps its
# canonical result key and constructs through the native registry name.
DMC_ENVS = [
    "cartpole", "pendulum", "finger", "ball_in_cup", "cheetah",
    "walker", "hopper", "quadruped", "humanoid", "humanoid_CMU",
    "dog", "fish", "stacker",
]
DMC_ENV_KIND = {key: key.lower() for key in DMC_ENVS}


def make_goal_state_for(env_kind: str):
    """Use a fixed goal for each env: standing / upright / centered."""
    return {
        "cartpole": np.array([0.0, 0.0], dtype=np.float32),
        "pendulum": np.array([1.0, 0.0], dtype=np.float32),
        "finger": np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "ball_in_cup": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "cheetah": np.zeros(9, dtype=np.float32),
        "walker": np.zeros(9, dtype=np.float32),
        "hopper": np.zeros(7, dtype=np.float32),
        "quadruped": np.zeros(30, dtype=np.float32),
        "humanoid": np.zeros(28, dtype=np.float32),
        "humanoid_cmu": np.zeros(63, dtype=np.float32),
        "dog": np.zeros(87, dtype=np.float32),
        "fish": np.zeros(14, dtype=np.float32),
        "stacker": np.zeros(20, dtype=np.float32),
    }[env_kind]


def restore_goal(env, env_kind: str) -> np.ndarray:
    """Restore the physical target used both for rendering and goal scoring."""
    import mujoco

    target = make_goal_state_for(env_kind)
    if env._expand_pendulum:
        env._data.qpos[0] = float(np.arctan2(target[1], target[0]))
    else:
        env._data.qpos[:env._nq] = target[:env._nq]
    env._data.qvel[:] = 0.0
    # Zero quaternion coordinates are not a valid physical orientation.
    mujoco.mj_normalizeQuat(env._model, env._data.qpos)
    mujoco.mj_forward(env._model, env._data)
    return env.get_state()


def encode_obs(model, obs_pixel_np, device="cpu"):
    """Measure the model's full-forward output, not its pre-cell encoder.

    Derive the zero-action width from the loaded model, never the environment:
    generalist checkpoints were trained with 56 action channels even when the
    environment has fewer controls. Keep this same output space for current,
    goal and final observations and for the planner's predicted latents.
    """
    pixel = np.asarray(obs_pixel_np)
    if pixel.ndim != 3 or pixel.shape[0] != 3:
        raise ValueError(f"Expected a CHW RGB observation, got {pixel.shape}")
    x = torch.as_tensor(pixel, dtype=torch.float32, device=device)[None, None]
    a = x.new_zeros(1, 1, model.action_dim)
    with torch.no_grad():
        out = model(x, a)
    return out["emb"][0, -1].reshape(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--training-manifest", required=True)
    p.add_argument("--image_size", type=int, default=84)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--elites", type=int, default=30)
    p.add_argument("--cem_iters", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--envs", help="Comma-separated subset of the 13 planned environments")
    args = p.parse_args()
    if min(args.n_episodes, args.horizon, args.samples, args.cem_iters) < 1:
        p.error("episodes, horizon, samples and CEM iterations must be positive")
    if not 1 < args.elites <= args.samples:
        p.error("CEM requires 2 <= elites <= samples")
    selected_envs = DMC_ENVS if args.envs is None else args.envs.split(",")
    if (not selected_envs or len(set(selected_envs)) != len(selected_envs)
            or any(env not in DMC_ENVS for env in selected_envs)):
        p.error(f"--envs must contain unique names from {DMC_ENVS}")
    args.ckpt = str(Path(args.ckpt).resolve())
    audit = TrainingAudit(args.training_manifest)
    repair_provenance = audit.provenance(args.ckpt)

    results_per_env = {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [out_dir / "eval_summary.json"] + [
        out_dir / f"eval_{env}.json" for env in DMC_ENVS
    ]
    if any(path.exists() for path in output_paths):
        p.error(f"Refusing to overwrite evaluation data in {out_dir}; choose a fresh --out_dir")

    # Load ckpt
    print(f"[eval_pixel] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    training_protocol_version = ckpt.get("training_protocol_version")
    required_protocol = audit.payload["required_checkpoint_metadata"]["training_protocol_version"]
    if training_protocol_version != required_protocol:
        raise ValueError(
            f"Checkpoint training protocol {training_protocol_version!r} does not match "
            f"manifest protocol {required_protocol!r}"
        )
    if ckpt.get("training_provenance", {}).get("protocol_version") != training_protocol_version:
        raise ValueError("Checkpoint training provenance does not confirm its training protocol")
    saved_args = ckpt["args"]
    model_kind = saved_args["model"]
    obs_dim = saved_args["pad_obs_to"]
    action_dim = saved_args["action_dim"]
    n_layers = saved_args.get("n_layers", 4)
    embed_dim = saved_args.get("embed_dim", 192)
    readout_mode = saved_args.get("readout_mode", "hidden_leak")
    image_size = saved_args.get("image_size", args.image_size)
    print(f"  model={model_kind} obs_dim={obs_dim} action_dim={action_dim} "
          f"n_layers={n_layers} embed={embed_dim} readout={readout_mode} image_size={image_size}")
    # build_model imported at top
    # Build the model
    model = build_model(
        model_kind, obs_dim, action_dim, n_layers, readout_mode,
        embed_dim=embed_dim, image_size=image_size,
        hidden_dim=saved_args.get("hidden_dim"),
        mlp_hidden=saved_args.get("mlp_hidden"),
        mlp_layers=saved_args.get("mlp_layers"),
        stacked_lif_layers=saved_args.get("stacked_lif_layers"),
        stacked_lif_din=saved_args.get("stacked_lif_din"),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    del ckpt
    if sha256(args.ckpt) != repair_provenance["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after training-audit admission")
    model.to(args.device).eval()

    for env_kind in selected_envs:
        print(f"[eval_pixel] === {env_kind} ===", flush=True)
        env_kind_native = DMC_ENV_KIND[env_kind]
        env = None
        try:
            env = DMCPixelEnv(env_kind_native, image_size=image_size,
                               max_episode_steps=50)
            action_dim_env = env.spec.action_dim
            if action_dim_env > model.action_dim:
                raise ValueError(
                    f"{env_kind} needs {action_dim_env} actions, checkpoint has {model.action_dim}"
                )
            predict_native = make_native_action_predict_hook(
                model.action_dim, env.spec.action_low, env.spec.action_high, args.device,
            )
            cem = CEM(model, action_dim=action_dim_env, horizon=args.horizon,
                      n_samples=args.samples, n_elites=args.elites,
                      n_iters=args.cem_iters, history_size=1, device=args.device,
                      predict_hook=predict_native)
            env.reset(seed=0)
            goal_state = restore_goal(env, env_kind_native)
            goal_qpos = env._data.qpos.copy()
            goal_z = encode_obs(model, env._render(), args.device)

            success_count = 0
            lewm_cos_dists = []
            phys_dists = []
            per_episode = []
            for ep in range(args.n_episodes):
                torch.manual_seed(ep)
                obs = env.reset(seed=ep)
                for t in range(50):
                    cur_z = encode_obs(model, obs["pixel"], args.device)  # (D,)
                    with torch.no_grad():
                        action_seq = cem.plan(cur_z, goal_z)  # (H, A_native)
                    action = np.clip(
                        action_seq[0].cpu().numpy(),
                        env.spec.action_low, env.spec.action_high,
                    )
                    obs, r, done, _ = env.step(action)
                    if done:
                        break
                # Final-state latent for LeWM-SR
                final_z = encode_obs(model, obs["pixel"], args.device)  # (D,)
                cos_sim = torch.nn.functional.cosine_similarity(
                    final_z.unsqueeze(0), goal_z.unsqueeze(0), dim=-1,
                ).item()
                lewm_cos_dist = float((1.0 - cos_sim) / 2.0)
                lewm_cos_dists.append(lewm_cos_dist)
                # Check env success
                state = env.get_state()
                suc, phys = env.check_success(state, goal_state)
                if suc:
                    success_count += 1
                phys_dists.append(phys)
                per_episode.append({
                    "episode_idx": ep,
                    "env_success": bool(suc),
                    "phys_dist": float(phys),
                    "cos_dist": lewm_cos_dist,
                    "actions_taken": t + 1,
                })
            env_sr = success_count / args.n_episodes
            mean_phys = sum(phys_dists) / len(phys_dists)
            mean_lewm_cos = sum(lewm_cos_dists) / len(lewm_cos_dists)
            lewm_succ_005 = sum(d < 0.05 for d in lewm_cos_dists) / len(lewm_cos_dists)
            lewm_succ_001 = sum(d < 0.01 for d in lewm_cos_dists) / len(lewm_cos_dists)
            lewm_succ = sum(d < 0.1 for d in lewm_cos_dists) / len(lewm_cos_dists)
            results_per_env[env_kind] = {
                "env_id": f"mujoco/{env_kind_native}_pixel",
                "display_env_kind": env_kind,
                "n_episodes": args.n_episodes,
                "n_seeds": 1,
                "cem_samples": args.samples,
                "cem_elites": args.elites,
                "cem_iters": args.cem_iters,
                "horizon": args.horizon,
                "history_size": 1,
                "eval_budget": 50,
                "replan_every": 1,
                "frame_skip": 1,
                "protocol_version": 2,
                "measurement_object": "forward.emb",
                "goal_state": goal_state.tolist(),
                "goal_qpos": goal_qpos.tolist(),
                "success_threshold": env._success_tol,
                "episode_seeds": list(range(args.n_episodes)),
                "planner_seeds": list(range(args.n_episodes)),
                "model_action_dim": model.action_dim,
                "env_action_dim": action_dim_env,
                "success_rate_env": float(env_sr),
                "mean_cos_dist": float(mean_lewm_cos),
                "mean_phys_dist": float(mean_phys),
                "success_rate_lewm": float(lewm_succ),
                "success_rate_lewm_005": float(lewm_succ_005),
                "success_rate_lewm_001": float(lewm_succ_001),
                "per_episode": per_episode,
            }
            print(f"  {env_kind}: env_sr={env_sr:.3f} lewm_cos={mean_lewm_cos:.4f} "
                  f"lewm_sr@0.05={lewm_succ_005:.3f} phys={mean_phys:.4f}")

        except Exception as e:
            print(f"  {env_kind}: ERROR {e}")
            traceback.print_exc()
            results_per_env[env_kind] = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        finally:
            if env is not None:
                env.close()

    # Save summary
    summary = {
        "ckpt": args.ckpt,
        "protocol_version": 2,
        "training_protocol_version": training_protocol_version,
        "repair_provenance": repair_provenance,
        "image_size": image_size,
        "model_kind": model_kind,
        "obs_dim": obs_dim,
        "results_per_env": results_per_env,
        "evaluated_envs": selected_envs,
        "protocol": {
            "name": "pixel_static_qpos_native_action_cem",
            "observation": "RGB CHW float32 in [0,1], one frame; no extra normalization",
            "latent": "model(pixel[None,None], zeros[1,1,model.action_dim])['emb'][0,-1]",
            "prediction": "model.predict(latent[batch,1,D], actions[batch,1,model.action_dim])",
            "planner_cost": "sum_D((predicted_terminal_z - goal_z)**2)",
            "latent_distance": "(1 - cosine_similarity(final_z, goal_z)) / 2",
            "physical_distance": "qpos RMS; pendulum angular distance in radians",
            "goal": "fixed native state, pendulum cos/sin converted to angle and quaternions normalized; identical rendered/scored target",
            "history_size": 1,
            "horizon": args.horizon,
            "eval_budget": 50,
            "replan_every": 1,
            "frame_skip": 1,
            "cem_iters": args.cem_iters,
            "candidate_actions": "native controls, bounded identically for prediction/execution; zero-padded only for model.predict",
            "episode_seeds": list(range(args.n_episodes)),
            "planner_seeds": list(range(args.n_episodes)),
        },
    }
    out_path = out_dir / "eval_summary.json"
    with out_path.open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(f"[eval_pixel] Saved {out_path}")

    # Save per-env JSONs
    for env_kind, r in results_per_env.items():
        if "error" in r:
            continue
        per_env_path = out_dir / f"eval_{env_kind}.json"
        with per_env_path.open("x") as handle:
            json.dump(r, handle, indent=2, allow_nan=False)
    print(f"[eval_pixel] DONE for {args.ckpt}")
    failed_envs = [env for env, result in results_per_env.items() if "error" in result]
    if failed_envs:
        raise SystemExit(f"[eval_pixel] Failed planned cells: {', '.join(failed_envs)}")


if __name__ == "__main__":
    main()

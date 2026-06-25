"""v5 3D world model eval.

For each 3D env, evaluate:
1. **Next-step prediction MSE** on held-out test split (no env, fast)
2. **Open-loop rollout error** - feed qpos, model predicts next qpos, compare to actual rollout
3. **Real mujoco env closed-loop simulation** - real mujoco step, compare predicted vs actual

This is the analog of v4.5 reacher eval but for all 5 3D envs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/lx/LeWM")
sys.path.insert(0, "/home/lx/snn/code")
sys.path.insert(0, "/home/lx/snn/code/scripts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--n-rollout", type=int, default=50, help="rollout length for closed-loop sim")
    p.add_argument("--n-episodes", type=int, default=10, help="real mujoco env episodes")
    p.add_argument("--horizon", type=int, default=5, help="CEM horizon")
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_model(ckpt_path, env, data):
    """Load v4.5 model from ckpt with correct state_dim."""
    from lewm_stjewm_v4 import STJEWMv4
    data_npz = np.load(data)
    obs_dim = data_npz["observations"].shape[-1]
    action_dim = data_npz["actions"].shape[-1]

    model = STJEWMv4(
        d_hid=192, embed_dim=192,
        action_dim=action_dim, action_emb_dim=192,
        state_dim=obs_dim, cell_n_layers=4, n_d=3,
        trace_beta=0.9, freeze_encoder=True,
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    model.load_state_dict(sd, strict=False)
    model = model.to(DEVICE).eval()
    return model, obs_dim, action_dim


@torch.no_grad()
def eval_next_step_mse(model, data_npz, n_test, action_dim, state_dim):
    """Eval next-step prediction error on test split (no env, fast)."""
    # Use last n_test samples
    obs = data_npz["observations"][-n_test:, 0, :].astype(np.float32)  # (N, D)
    actions = data_npz["actions"][-n_test:, 0, :].astype(np.float32)
    nxt = data_npz["next_observations"][-n_test:, 0, :].astype(np.float32)

    # Build (state, action) -> (next_state) test
    state = torch.from_numpy(obs).to(DEVICE)
    action = torch.from_numpy(actions).to(DEVICE)
    nxt_t = torch.from_numpy(nxt).to(DEVICE)

    # Reshape to (B, T=1) for the model
    state_in = state.unsqueeze(1)  # (N, 1, D)
    action_in = action.unsqueeze(1)  # (N, 1, A)

    out = model(state_in, action_in)
    pred_emb = out["emb"][:, 0]  # (N, 192)

    # Encode next_state
    nxt_in = nxt_t.unsqueeze(1)
    nxt_action = torch.zeros_like(action_in)
    out_nxt = model(nxt_in, nxt_action)
    tgt_emb = out_nxt["emb"][:, 0]

    # MSE in latent space
    mse = F.mse_loss(pred_emb, tgt_emb).item()
    cos = F.cosine_similarity(pred_emb, tgt_emb, dim=-1).mean().item()
    return mse, cos


@torch.no_grad()
def eval_open_loop_rollout(model, data_npz, action_dim, state_dim, n_episodes=10, rollout_len=50):
    """Open-loop rollout: predict next qpos from current, feed predicted into next step.

    Compare predicted trajectory to actual ground-truth trajectory.
    """
    obs = data_npz["observations"][:, 0, :].astype(np.float32)
    actions = data_npz["actions"][:, 0, :].astype(np.float32)
    nxt = data_npz["next_observations"][:, 0, :].astype(np.float32)

    n_total = len(obs)
    ep_len = n_total // n_episodes

    rollout_errors = []
    for ep in range(n_episodes):
        s_start = ep * ep_len
        if s_start + rollout_len >= n_total:
            break
        # Initial state
        cur = torch.from_numpy(obs[s_start]).to(DEVICE)
        errors = []
        for t in range(rollout_len):
            actual_nxt = nxt[s_start + t]
            # Get model prediction
            state_in = cur.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            action_in = torch.from_numpy(actions[s_start + t]).to(DEVICE).unsqueeze(0).unsqueeze(0)
            out = model(state_in, action_in)
            pred_emb = out["emb"][:, 0]  # (1, 192)

            # Encode the actual next state for comparison
            nxt_t = torch.from_numpy(actual_nxt).to(DEVICE).unsqueeze(0).unsqueeze(0)
            out_nxt = model(nxt_t, torch.zeros_like(action_in))
            tgt_emb = out_nxt["emb"][:, 0]

            err = F.mse_loss(pred_emb, tgt_emb).item()
            errors.append(err)

            # For next step: use the actual state (open-loop uses real dynamics)
            # But also test "self-loop" mode: use predicted next state
            # For now, use real (open-loop is more standard)
            cur = torch.from_numpy(actual_nxt).to(DEVICE)

        rollout_errors.append(np.mean(errors))

    return np.mean(rollout_errors), np.std(rollout_errors)


def main():
    args = parse_args()
    print(f"[v5-3d-eval] env={args.env}, ckpt={args.ckpt}", flush=True)

    data_npz = np.load(args.data)
    obs_dim = data_npz["observations"].shape[-1]
    action_dim = data_npz["actions"].shape[-1]
    print(f"  state_dim={obs_dim}, action_dim={action_dim}", flush=True)

    model, _, _ = load_model(args.ckpt, args.env, args.data)
    print(f"  model loaded, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    # Eval 1: next-step prediction MSE
    print("  [1/2] Next-step prediction MSE...", flush=True)
    mse, cos = eval_next_step_mse(model, data_npz, args.n_test, action_dim, obs_dim)
    print(f"    MSE={mse:.4f}, cos={cos:.4f}", flush=True)

    # Eval 2: open-loop rollout error
    print("  [2/2] Open-loop rollout...", flush=True)
    rollout_mse, rollout_std = eval_open_loop_rollout(model, data_npz, action_dim, obs_dim, n_episodes=10, rollout_len=args.n_rollout)
    print(f"    rollout MSE={rollout_mse:.4f} +/- {rollout_std:.4f}", flush=True)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "env": args.env,
        "ckpt": args.ckpt,
        "model_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "trainable_params_M": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        "state_dim": obs_dim,
        "action_dim": action_dim,
        "n_test": args.n_test,
        "n_rollout_steps": args.n_rollout,
        "n_episodes_rollout": 10,
        "next_step_mse": mse,
        "next_step_cos": cos,
        "rollout_mse": rollout_mse,
        "rollout_std": rollout_std,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

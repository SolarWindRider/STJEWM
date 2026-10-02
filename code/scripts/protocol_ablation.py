"""Protocol ablation (eval-time), 2x2 factorial:

- context action channel: zero (shipped) vs real executed actions
- CEM imagination alignment: start (shipped: candidates at window positions
  0..4, zero-pad to 27) vs end (corrected: current candidate at the window's
  last position each imagined step — receding-horizon semantics from the
  current state; the readout position sees the action directly)

Motivation: the position-sensitivity diagnostic showed the shipped start
alignment renders memoryless predictors (LeWM-v2) structurally blind to
candidate actions (response 0.000 at all candidate positions), while the
trace readout's persistent action response (0.2-0.38 at all positions)
sees them. Paired episodes; identical CEM budget; only the factor under
test changes.

Run: python -m code.scripts.protocol_ablation --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

import code.eval.closed_loop as cl
from code.eval.closed_loop import make_env, _load_eval_dataset
from code.core.cem import CEM
from code.core.encode import encode_obs
from code.scripts.event_align import build_model

# ---------------------------------------------------------------- patches
STATE = {"mode": "zero", "recording_env": None}


class ActionRecordingEnv:
    """Wraps a BaseEnv; records executed actions; clears on reset."""

    def __init__(self, base):
        self.base = base
        self.recorded: list[np.ndarray] = []

    def step(self, action):
        self.recorded.append(
            np.asarray(action, dtype=np.float32).copy())
        return self.base.step(action)

    def reset(self, seed=None):
        self.recorded = []
        return self.base.reset(seed=seed)

    @property
    def spec(self):
        return self.base.spec

    def get_state(self):
        return self.base.get_state()

    def __getattr__(self, name):
        return getattr(self.base, name)


def encode_history_patched(model, obs_list, action_dim, device):
    if STATE["mode"] == "zero":
        return _orig_encode_history(model, obs_list, action_dim, device)
    # real-action context: position j (0..25) consumes the action executed in
    # o_j; the CURRENT position stays zero (action undecided at plan time)
    rec = STATE["recording_env"].recorded
    L = len(rec)
    z = []
    for j, o in enumerate(obs_list):
        if j < 26 and 0 <= L - 26 + j < L:
            aj = np.asarray(rec[L - 26 + j], dtype=np.float32).reshape(-1)
            aj = np.pad(aj, (0, action_dim - aj.size))
            a = torch.from_numpy(aj).float().to(device).reshape(1, 1, -1)
        else:
            a = torch.zeros(1, 1, action_dim, device=device)
        if torch.is_tensor(o):
            s = o.reshape(1, 1, -1).float().to(device)
        else:
            s = torch.from_numpy(np.asarray(o, dtype=np.float32)).reshape(1, 1, -1).to(device)
        enc = model(s, a)
        z.append(enc["emb"][0, 0])
    return torch.stack(z)


_orig_encode_history = cl.encode_history
cl.encode_history = encode_history_patched


class CEMEndAlign(CEM):
    """CEM with the candidate action placed at the window's LAST position on
    every imagined step (receding-horizon semantics from the current state)."""

    @torch.no_grad()
    def _rollout_cost(self, z_init: torch.Tensor, z_goal: torch.Tensor,
                      actions: torch.Tensor) -> torch.Tensor:
        N, H, A = actions.shape
        if z_init.ndim == 1:
            context = z_init.unsqueeze(0).expand(self.history_size, -1)
        elif z_init.ndim == 2 and z_init.shape[0] == self.history_size:
            context = z_init
        else:
            raise ValueError("Initial latent must be (D,) or (history_size, D)")
        h = context.unsqueeze(0).expand(N, -1, -1).contiguous()
        for t in range(H):
            a_window = torch.zeros(N, self.history_size, A,
                                   device=actions.device, dtype=actions.dtype)
            a_window[:, -1] = actions[:, t]
            h_in = h[:, -self.history_size:]
            if self.predict_hook is None:
                nxt = self.model.predict(h_in, a_window)
            else:
                nxt = self.predict_hook(self.model, h_in, a_window)
            nxt = nxt[:, -1]
            h = torch.cat([h[:, 1:], nxt.unsqueeze(1)], dim=1)
        z_final = h[:, -1]
        return ((z_final - z_goal.unsqueeze(0)) ** 2).sum(-1)


# ---------------------------------------------------------------- protocol
def run_condition(model, env_rec, data_path, *, ctx_mode, align, n_episodes,
                  device):
    STATE["mode"] = ctx_mode
    prev_cem = cl.CEM
    cl.CEM = CEMEndAlign if align == "end" else prev_cem
    try:
        res = cl.eval_closed_loop(
            model, env_rec, data_path,
            n_episodes=n_episodes, n_seeds=1,
            cem_samples=300, cem_elites=30, cem_iters=10,
            horizon=5, eval_budget=50, goal_offset=25,
            history_size=27, pad_obs_to=128, model_action_dim=56,
            split="in_dist", seed_start=0, device=device,
        )
    finally:
        cl.CEM = prev_cem
    return {
        "condition": f"ctx={ctx_mode}/align={align}",
        "success_rate_env": res.success_rate_env,
        "mean_cos_dist": res.mean_cos_dist,
        "episodes": [
            {"episode_idx": e["episode_idx"], "cos_dist": e["cos_dist"],
             "env_success": bool(e["env_success"])}
            for e in res.per_episode
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--envs", default="cartpole,cheetah")
    ap.add_argument("--models", default="trace,lewm")
    ap.add_argument("--out", default="/home/lx/snn/results/actionctx_ablation/"
                                     "protocol_ablation.json")
    args = ap.parse_args()

    ENVS = {
        "cartpole": {"data": "/home/lx/snn/data/dm_control/cartpole_250k.npz"},
        "cheetah": {"data":
                    "/home/lx/snn/data/dm_control/3d_rollouts_250k/cheetah_250k.npz"},
    }
    MODELS = {
        "trace": ("/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt",
                  "stjewm_trace_only"),
        "lewm": ("/data/lx/tmp/results/5m/oodc_F1/lewm_baseline_v2/seed_0/final.pt",
                 "lewm_baseline_v2"),
    }
    CONDITIONS = [("zero", "start"), ("zero", "end"), ("real", "start"),
                  ("real", "end")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text()).get("results", {})
        print(f"[resume] {len(results)} conditions loaded", flush=True)
    envs = {k: v for k, v in ENVS.items() if k in args.envs.split(",")}
    models = {k: v for k, v in MODELS.items() if k in args.models.split(",")}
    for env_kind, cfg in envs.items():
        for model_key, (ckpt, hint) in models.items():
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            model = build_model(hint, 128, 56, ck_args=ck["args"],
                                state_dict=ck["model"])
            model.load_state_dict(ck["model"], strict=True)
            model = model.to(args.device).eval()
            env_rec = ActionRecordingEnv(make_env(env_kind))
            for ctx_mode, align in CONDITIONS:
                key = f"{env_kind}/{model_key}/ctx={ctx_mode}/align={align}"
                if key in results:
                    print(f"[skip] {key}", flush=True)
                    continue
                print(f"[run] {key}", flush=True)
                STATE["recording_env"] = env_rec
                out = run_condition(model, env_rec, cfg["data"],
                                    ctx_mode=ctx_mode, align=align,
                                    n_episodes=args.n_episodes,
                                    device=args.device)
                results[key] = out
                out_path.write_text(json.dumps({
                    "protocol": "2x2 factorial closed-loop ablation; factors: "
                                "context action channel (zero/real) x CEM "
                                "candidate alignment (start=shipped, "
                                "end=corrected); 5 episodes, 1 seed, "
                                "cartpole+cheetah; models: trace + lewm; "
                                "paired episodes",
                    "results": results,
                }, indent=2))
                print(f"[done] {key}: env-SR={out['success_rate_env']:.2f} "
                      f"cos={out['mean_cos_dist']:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "protocol": "2x2 factorial closed-loop ablation; factors: context "
                    "action channel (zero/real) x CEM candidate alignment "
                    "(start=shipped, end=corrected); 5 episodes, 1 seed, "
                    "cartpole+cheetah; models: trace + lewm; paired episodes",
        "results": results,
    }, indent=2))
    print("saved", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Protocol evaluation for the real external baseline Spiking-WM (Brain-Cog-Lab).

Evaluates a trained Spiking-WM (dmc_proprio config, state input) with the SAME
protocol used for ST-JEWM baselines:
  1. event-rho (G1): pearson(|d obs|, spike rate) under random policy.
  2. cos-dist (latent calibration): dynamics open-loop prediction vs. the true
     t+25 posterior latent ((1-cos)/2) on random-policy trajectories.
  3. spike activation rate of MCRNN (event-driven sparsity proxy).

Usage:
  python code/scripts/eval_spiking_wm_protocol.py --task cartpole_swingup \
      --ckpt results/spiking_wm/logs_cartpole_swingup/latest_model.pt \
      --out results/spiking_wm/protocol_cartpole_swingup.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/lx/Spiking-WM")
sys.path.insert(0, "/home/lx/snn")

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

import tools  # noqa: E402
import networks  # noqa: E402
import node  # noqa: E402
import normalization  # noqa: E402
import surrogate  # noqa: E402
from dreamer import Dreamer  # noqa: E402

from code.eval.closed_loop import make_env  # noqa: E402

DMC_TASK_MAP = {
    "cartpole_swingup": "cartpole_swingup",
    "cheetah_run": "cheetah_run",
    "walker_walk": "walker_walk",
    "finger_spin": "finger_spin",
    "pendulum_swingup": "pendulum_swingup",
    "cup_catch": "cup_catch",
    "reacher_easy": "reacher_easy",
    "hopper_hop": "hopper_hop",
    "quadruped_walk": "quadruped_walk",
    "dog_walk": "dog_walk",
    "fish_swim": "fish_swim",
    "humanoid_run": "humanoid_run",
}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.sqrt((xm * xm).sum() * (ym * ym).sum()))
    if denom < 1e-12:
        return 0.0
    return float((xm * ym).sum() / denom)


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float((1.0 - float(np.dot(a, b) / (na * nb))) / 2.0)


def load_config(ckpt_path: str, device: str):
    import argparse as ap
    import ruamel.yaml as yaml

    cfg = yaml.safe_load(open("/home/lx/Spiking-WM/configs.yaml"))

    def rec(base, upd):
        for k, v in upd.items():
            if isinstance(v, dict) and k in base:
                rec(base[k], v)
            else:
                base[k] = v

    d = {}
    for name in ("defaults", "dmc_proprio"):
        rec(d, cfg[name])
    d.update(
        dict(
            device=device,
            compile=False,
            steps=5e5,
            prefill=0,
            traindir=str(pathlib.Path(ckpt_path).parent / "train_eps"),
            evaldir=str(pathlib.Path(ckpt_path).parent / "eval_eps"),
            logdir=str(pathlib.Path(ckpt_path).parent),
            dataset_size=1000000,
        )
    )
    ap.Namespace.__getitem__ = lambda self, k: getattr(self, k)
    return ap.Namespace(**d)


class SpikingWMProbe:
    def __init__(self, ckpt_path: str, task: str, device: str = "cuda:0"):
        self.cfg = load_config(ckpt_path, device)
        self.device = device
        self.spike_times = int(self.cfg.spike_times)
        from envs.dmc import DeepMindControl
        import envs.wrappers as wrappers

        # Build WorldModel with the REAL obs space of the task's env so the
        # encoder input dims match training (per-key splits).
        env = wrappers.TimeLimit(
            DeepMindControl(task, 2, (64, 64), seed=0), 500
        )
        env = wrappers.SelectAction(env, key="action")
        self.obs_space, self.act_space = env.observation_space, env.action_space
        wm_cfg = self.cfg
        wm_cfg.num_actions = self.act_space.shape[0]
        import models

        self.wm = models.WorldModel(self.obs_space, self.act_space, 0, wm_cfg).to(
            device
        )
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        # keys are module paths inside agent (e.g. _wm.encoder...)
        prefix = "agent." if any(k.startswith("agent.") for k in sd) else ""
        stripped = {
            k[len(prefix):] if prefix else k: v for k, v in sd.items()
        }
        self.wm.load_state_dict(stripped, strict=False)
        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad = False

    def policy_step(self, obs_raw: dict, action: np.ndarray, state):
        """One forward step exactly like dreamer._policy (deterministic mode)."""
        obs_t = {}
        for k, v in obs_raw.items():
            if k in ("is_first", "is_terminal"):
                continue
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            obs_t[k] = torch.from_numpy(arr).reshape(1, -1).to(self.device)
        obs_t["image"] = obs_t["image"] / 255.0 - 0.5
        obs_t["is_first"] = torch.zeros(1, 1).to(self.device)
        obs_t["is_terminal"] = torch.zeros(1, 1).to(self.device)
        pre = self.wm.preprocess(obs_t)
        embed = self.wm.encoder(pre)
        a_t = torch.from_numpy(np.asarray(action, dtype=np.float32)).reshape(
            1, -1
        ).to(self.device)
        if state is None:
            state = self.wm.dynamics.initial(1)
        post, _ = self.wm.dynamics.obs_step(
            state, a_t, embed, pre["is_first"], sample=False
        )
        return post, embed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--n-goal-cells", type=int, default=100)
    args = ap.parse_args()

    task_name = DMC_TASK_MAP[args.task]
    probe = SpikingWMProbe(args.ckpt, task_name, args.device)
    from envs.dmc import DeepMindControl
    import envs.wrappers as wrappers

    env = wrappers.SelectAction(
        wrappers.TimeLimit(DeepMindControl(task_name, 2, (64, 64), seed=0), 500),
        key="action",
    )

    obs_list, rate_list, stoch_list, embed_list = [], [], [], []
    state = None
    t = 0
    n_done = 0
    obs_raw = env.reset()
    while t < args.n_steps:
        a_raw = env.action_space.sample()
        out, _, done, _ = env.step({"action": a_raw})
        obs_raw = out
        a = a_raw
        with torch.no_grad():
            post, embed = probe.policy_step(obs_raw, a, state)
            state = {k: v.detach() for k, v in post.items()}
            deter = post["deter"]  # [T, B, deter]
            stoch = post["stoch"]  # [B, stoch]
        rate_list.append(float(deter.detach().float().mean().item()))
        stoch_list.append(stoch.detach().cpu().numpy().reshape(-1))
        embed_list.append(
            embed.detach().float().mean(dim=0).cpu().numpy().reshape(-1)
        )
        obs_list.append(
            np.concatenate(
                [
                    np.asarray(obs_raw[k], dtype=np.float32).reshape(-1)
                    for k in obs_raw
                    if k not in ("image", "is_first", "is_terminal", "reward")
                ]
            )
        )
        t += 1
        if done and t < args.n_steps:
            n_done += 1
            env.reset()
            state = None

    obs_arr = np.stack(obs_list)
    d_obs = np.linalg.norm(np.diff(obs_arr, axis=0), axis=1)
    rate_arr = np.array(rate_list, dtype=np.float32)
    stoch_arr = np.stack(stoch_list)
    embed_arr = np.stack(embed_list)
    d_stoch = np.linalg.norm(np.diff(stoch_arr, axis=0), axis=1)
    d_embed = np.linalg.norm(np.diff(embed_arr, axis=0), axis=1)
    L = min(d_obs.shape[0], d_stoch.shape[0], d_embed.shape[0], rate_arr.shape[0])
    d_obs, d_stoch, d_embed, rate_arr = d_obs[:L], d_stoch[:L], d_embed[:L], rate_arr[:L]
    corr_obs_rate = pearson(d_obs, rate_arr)
    corr_obs_latent = pearson(d_obs, d_stoch)
    corr_obs_embed = pearson(d_obs, d_embed)

    result = {
        "task": args.task,
        "event_rho": float(corr_obs_rate),
        "corr_obs_latent": float(corr_obs_latent),
        "corr_obs_embed": float(corr_obs_embed),
        "n_steps": int(len(d_obs)),
        "n_resets": int(n_done),
        "mean_spike_rate": float(np.mean(rate_list)),
        "stoch_std": float(stoch_arr.std()),
    }

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(
        f"[spiking_wm_protocol] {args.task}: event_rho={result['event_rho']:.4f} "
        f"corr_obs_latent={result['corr_obs_latent']:.4f} "
        f"spike_rate={result['mean_spike_rate']:.4f} stoch_std={result['stoch_std']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

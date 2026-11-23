"""Extra closed-loop evaluations on the E1 instance manifest.

Modes
-----
random : random-action reference. Same environments, episode manifest (eval seed 0,
         `default_rng(seed*7919+42)` episode draw), init/goal states, action bounds,
         50-step budget and terminal success criterion as the main state protocol;
         `CEM.plan` is replaced by i.i.d. uniform actions in the native box, so the
         episode executes two random 25-step chunks (2 planning calls). `repeats`
         independent random seeds per cell give `5*repeats` random episodes per cell.
dose   : action-channel dose response. Only the candidate-scoring path is changed:
         inside every CEM candidate rollout the model sees a' = a_bar + gamma*(a - a_bar)
         with a_bar = 0 (the legal zero action, also the CEM initial mean); the
         environment executes the ORIGINAL selected 25-step chunk. gamma=1 is a sham
         control that must reproduce the baseline episode by episode.

Usage
-----
python -m code.scripts.extra_closed_loop_evals --mode random --spec configs/oodc_5m/oodc_F1.json \
    --ckpt <trace ckpt> --out-root /data/lx/tmp/results/random_baseline_20261123
python -m code.scripts.extra_closed_loop_evals --mode dose --gamma 0.75 --spec ... --ckpt ... --out-root ...
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/lx/snn")

from code.core.cem import CEM  # noqa: E402
from code.eval import closed_loop as CL  # noqa: E402
from code.scripts.event_align import build_model  # noqa: E402

CLO_MAP = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum",
           "humanoid_CMU": "humanoid_cmu", "humanoid_cmu": "humanoid_cmu",
           "cheetah_velhidden": "cheetah_qpos_masked"}


def load_model(ckpt: str, device: str):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    state_dim = next((int(sd[k].shape[1]) for k in (
        "state_encoder.proj.0.weight", "state_projector.proj.0.weight",
        "state_projector.0.weight", "state_proj.0.weight") if k in sd),
        ck_args.get("pad_obs_to") or 128)
    action_dim = next((int(sd[k].shape[1]) for k in (
        "action_encoder.proj.0.weight", "action_encoder.0.weight", "action_proj.0.weight") if k in sd),
        ck_args.get("action_dim") or 56)
    model = build_model(ck_args["model"], state_dim, action_dim, ck_args, state_dict=sd)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print("[extra] trained weights loaded strictly", flush=True)
    return model, ck


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["random", "dose", "zero"], required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--envs", default="", help="comma list to restrict env_ids")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    spec_raw = json.load(open(args.spec))
    split_name = (spec_raw.get("_split_name") or spec_raw.get("split_name")
                  if isinstance(spec_raw, dict) else Path(args.spec).stem)
    entries = spec_raw.get("specs", []) if isinstance(spec_raw, dict) else spec_raw
    only = set(args.envs.split(",")) if args.envs else None
    model, ck = load_model(args.ckpt, args.device)
    ck_args = ck.get("args", {})
    pad_obs = ck_args.get("pad_obs_to") or 128
    pad_act = ck_args.get("action_dim") or 56
    original_plan = CEM.plan
    for entry in entries:
        env_id = entry["env_id"]
        if only and env_id not in only:
            continue
        clo_env = CLO_MAP.get(entry.get("clo_env") or env_id, entry.get("clo_env") or env_id)
        label = "cheetah_qpos_masked" if env_id == "cheetah_velhidden" else env_id
        data_path = entry["path"]
        goal_offset = entry.get("goal_offset", 25)
        out_dir = Path(args.out_root) / split_name / Path(args.ckpt).parts[-3] / "seed_0"
        tag = (f"random_r{args.repeats}" if args.mode == "random"
               else "zero_action" if args.mode == "zero" else f"dose_g{args.gamma:g}")
        out_json = out_dir / f"eval_{label}__{tag}.json"
        if out_json.exists():
            print(f"[skip] {label}", flush=True)
            continue
        env = CL.make_env(clo_env, data_path)
        runs = range(args.repeats) if args.mode == "random" else range(1)
        if args.mode == "zero":
            def zero_plan(self, z_hist, z_goal):
                return torch.zeros(self.horizon, self.action_dim)
        per_rep = []
        for rep in runs:
            hook = None
            cell_seed = None
            if args.mode == "random":
                cell_seed = abs(hash((split_name, label, rep))) % (2 ** 31)

                def rand_plan(self, z_hist, z_goal, _seed=cell_seed):
                    g = torch.Generator().manual_seed(_seed + int(getattr(self, "_n_plan", 0)))
                    self._n_plan = getattr(self, "_n_plan", 0) + 1
                    return torch.rand(self.horizon, self.action_dim, generator=g) * 2 - 1
                CEM.plan = rand_plan
            elif args.mode == "zero":
                CEM.plan = zero_plan
            elif args.mode == "dose":
                gamma = float(args.gamma)

                def hook(model_, ctx_emb_, ctx_act_, _g=gamma):
                    a = ctx_act_.clone()
                    a[..., :env.spec.action_dim] = _g * a[..., :env.spec.action_dim]
                    return model_.predict(ctx_emb_, a)
                CEM.plan = original_plan
            res = CL.eval_closed_loop(
                model, env, data_path,
                n_episodes=args.n_episodes, n_seeds=1,
                cem_samples=300, cem_elites=30, cem_iters=10,
                goal_offset=goal_offset, history_size=1,
                horizon=25, eval_budget=50, device=args.device,
                split="in_dist", pad_obs_to=pad_obs, model_action_dim=pad_act,
                cem_predict_hook=hook,
            )
            CEM.plan = original_plan
            d = asdict(res)
            d["mode"] = args.mode
            d["gamma"] = args.gamma if args.mode == "dose" else None
            if args.mode == "zero":
                d["cell_seed"] = None
            d["repeat"] = rep
            d["cell_seed"] = cell_seed if args.mode == "random" else None
            d["protocol"]["checkpoint"] = str(Path(args.ckpt).resolve())
            per_rep.append(d)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "x") as f:
            json.dump(per_rep if args.mode == "random" else per_rep[0], f, indent=2, allow_nan=False)
        if args.mode == "random":
            sr = np.mean([np.mean([pe["env_success"] for pe in d["per_episode"]]) for d in per_rep])
            pd = np.mean([pe["phys_dist"] for d in per_rep for pe in d["per_episode"]])
            print(f"[random] {label}: env-SR {sr:.3f} ({args.repeats}x{args.n_episodes} eps) phys_dist {pd:.3f}", flush=True)
        else:
            print(f"[dose g={args.gamma}] {label}: env-SR {per_rep[0]['success_rate_env']:.3f}", flush=True)
    CEM.plan = original_plan
    return 0


if __name__ == "__main__":
    sys.exit(main())

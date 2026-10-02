"""Measure checkpoint-backed protocol-2 readout and pre-cell diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from code.eval.closed_loop import make_env
from code.scripts.audited_results import TrainingAudit, require, write_new_json
from code.scripts.event_align import ENV_KIND_MAP, build_model
from code.scripts.latent_rollout import collect_state_rollout


def measure_one(ckpt_path, env_kind, data_path=None, *, n_steps=200, seed=0, device="cpu"):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    env = make_env(ENV_KIND_MAP[env_kind], data_path)
    try:
        state_dim = saved.get("pad_obs_to") or env.spec.obs_dim
        action_dim = saved.get("action_dim") or env.spec.action_dim
        model_name = Path(ckpt_path).parent.parent.name
        model = build_model(model_name, state_dim, action_dim, saved, state_dict=checkpoint["model"])
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
        result, _ = collect_state_rollout(model, env, state_dim, action_dim,
                                          n_steps=n_steps, n_resets=2, seed=seed, device=device)
        result.update(model=model_name, env=env_kind, ckpt=str(Path(ckpt_path).resolve()), weights_loaded_strict=True)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--env", required=True, choices=sorted(ENV_KIND_MAP))
    parser.add_argument("--data")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = TrainingAudit(args.training_manifest)
    audit.protect_output(args.out)
    provenance = audit.provenance(args.ckpt)
    result = measure_one(args.ckpt, args.env, args.data, n_steps=args.n_steps, seed=args.seed, device=args.device)
    require(audit.checkpoint(args.ckpt)["sha256"] == provenance["checkpoint_sha256"], "Checkpoint changed during measurement")
    result["repair_provenance"] = provenance
    write_new_json(args.out, result)
    print(f"[measure_latent_stats] {args.env}: event_rho={result['event_rho']}; output={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

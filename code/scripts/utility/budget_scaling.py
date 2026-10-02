"""Evaluate audited data-budget checkpoints into a new, explicit result root.

Training is owned by the consolidated training repair orchestrator. This consumer
never substitutes the 1x checkpoint for a requested 0.5x/2x cell and never reuses
unversioned historical diagnostics.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from code.scripts.audited_results import (
    ROOT, TrainingAudit, load_json, metric, require, sha256, summarize,
    validate_closed_loop, validate_diagnostic, write_new_json,
)

DMC_ENVS = ("cheetah", "walker", "cartpole_2d", "pendulum_2d", "finger", "ball_in_cup")
CLO_ENVS = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum", "humanoid_CMU": "humanoid_cmu"}


def native_env_id(env):
    return {"pusht": "swm/PushT-v1", "tworoom": "swm/TwoRoom-v1"}.get(env, "mujoco/" + CLO_ENVS.get(env, env))


def eval_entries(path):
    entries = [entry for entry in load_json(path) if not entry.get("extra_flags")]
    require(len({entry["env_id"] for entry in entries}) == len(entries), "Duplicate evaluation environments")
    return entries


def evaluate_checkpoint(checkpoint, model, out_root, audit, entries, diagnostic_envs,
                        *, seed=0, episodes=3, steps=200, device="cpu"):
    checkpoint, out_root = Path(checkpoint).resolve(), Path(out_root).resolve()
    require(episodes > 0 and steps >= 6, "Invalid evaluation budget")
    require(set(diagnostic_envs) <= {entry["env_id"] for entry in entries}, "Missing diagnostic environment in eval spec")
    provenance = audit.provenance(checkpoint)
    source_names = ("code/scripts/event_align.py", "code/scripts/latent_rollout.py", "code/eval/closed_loop.py")
    source_hashes = {name: sha256(ROOT / name) for name in source_names}
    provenance["evaluation_source_sha256"] = source_hashes
    provenance["budget"] = {"episodes": episodes, "steps": steps, "seed": seed}
    audit.protect_output(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    rows = []

    def produce(command, path, validator):
        require(not path.exists(), f"Refusing existing result: {path}")
        with path.with_suffix(".log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        require(audit.checkpoint(checkpoint)["sha256"] == provenance["checkpoint_sha256"], "Checkpoint changed during evaluation")
        require(all(sha256(ROOT / name) == value for name, value in source_hashes.items()), "Evaluation source changed during execution")
        payload = load_json(path)
        validator(payload)
        payload["repair_provenance"] = provenance
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        return payload

    for entry in entries:
        env = entry["env_id"]
        eval_path = out_root / f"eval_{env}.json"
        command = [sys.executable, "-m", "code.eval.closed_loop", "--ckpt", str(checkpoint),
                   "--env", entry.get("clo_env", CLO_ENVS.get(env, env)), "--data", entry["path"],
                   "--out", str(eval_path), "--n-episodes", str(episodes), "--n-seeds", "1",
                   "--horizon", "5", "--eval-budget", "50", "--history-size", str(entry["history_size"]),
                   "--goal-offset", str(entry["goal_offset"]), "--pad-obs-eval", "128", "--action-dim-eval", "56",
                   "--device", device]
        ev = produce(command, eval_path, lambda value: validate_closed_loop(value, episodes=episodes))
        require(ev["protocol"].get("checkpoint") == str(checkpoint), "Evaluation used a different checkpoint")
        require(ev["env_id"] == native_env_id(entry.get("clo_env", env)),
                "Evaluation used a different native environment")
        require(ev["protocol"].get("data_loader_protocol") == audit.payload["evaluation_data_protocol_version"],
                "Wrong evaluation data generation")
        row = {"env": env, "env-SR": metric(ev, "success_rate_env"),
               "eval_source": str(eval_path), "eval_sha256": sha256(eval_path)}
        if env in diagnostic_envs:
            diagnostic_path = out_root / f"align_{env}.json"
            diagnostic = produce([
                sys.executable, "-m", "code.scripts.event_align", "--ckpt", str(checkpoint),
                "--model", model, "--env", env, "--out", str(diagnostic_path),
                "--n-steps", str(steps), "--n-resets", "2", "--seed", str(seed),
                "--pad-obs-to", "128", "--action-dim-eval", "56", "--device", device,
            ], diagnostic_path, lambda value: validate_diagnostic(value, env=env, model=model, seed=seed))
            require(diagnostic["n_steps"] == steps, "Incomplete diagnostic budget")
            row.update(div=metric(diagnostic, "divergence"),
                       resp=metric(diagnostic, "responsiveness", nullable=True),
                       event_rho=metric(diagnostic, "event_rho", nullable=True),
                       diagnostic_source=str(diagnostic_path), diagnostic_sha256=sha256(diagnostic_path))
        rows.append(row)
    return rows, provenance


def aggregate(per_env, expected_envs=DMC_ENVS):
    require(len(per_env) == len(expected_envs) and {row["env"] for row in per_env} == set(expected_envs),
            "Missing/duplicate budget evaluation cells")
    return {key: summarize(metric(row, key, nullable=key in ("resp", "event_rho")) for row in per_env)
            for key in ("env-SR", "div", "resp", "event_rho")}


def evaluate_budget(model, frac, checkpoint, out_root, audit, *, seed=0, episodes=3, steps=200, device="cpu"):
    checkpoint = Path(checkpoint).resolve()
    require(frac in (0.5, 1.0, 2.0), "Unsupported data-budget fraction")
    if frac == 1.0:
        require(checkpoint.parent.parent.name == model and checkpoint.parent.parent.parent.name == "generalist_G16",
                "The 1x cell must use its audited full-G16 checkpoint")
    else:
        require(checkpoint.parent.parent.name == str(frac) and checkpoint.parent.parent.parent.name == model,
                "Requested data-budget fraction does not match checkpoint identity")
    entries = [entry for entry in eval_entries(ROOT / "configs/generalist_G16_eval.json") if entry["env_id"] in DMC_ENVS]
    rows, provenance = evaluate_checkpoint(checkpoint, model, out_root, audit, entries, DMC_ENVS,
                                          seed=seed, episodes=episodes, steps=steps, device=device)
    summary = {"status": "completed", "protocol_version": 2, "measurement_object": "forward.emb",
               "model": model, "frac": frac, "seed": seed, "n_envs": len(rows), "per_env": rows,
               "metrics": aggregate(rows), "repair_provenance": provenance}
    write_new_json(Path(out_root) / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frac", type=float, required=True, choices=(0.5, 1.0, 2.0))
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate_budget(args.model, args.frac, args.ckpt, args.out_root, TrainingAudit(args.training_manifest),
                    seed=args.seed, episodes=args.n_episodes, steps=args.n_steps, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

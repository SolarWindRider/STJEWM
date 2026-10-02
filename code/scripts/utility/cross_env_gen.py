"""Evaluate full-G16/held-out-G16 audited checkpoints without historical reuse."""
from __future__ import annotations

import argparse
from pathlib import Path

from code.scripts.audited_results import (
    ROOT, TrainingAudit, fmt, load_json, metric, read_diagnostic, require,
    sha256, summarize, validate_closed_loop, write_new_json,
)
from code.scripts.utility.budget_scaling import eval_entries, evaluate_checkpoint, native_env_id

TARGET_MODELS = (
    "stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only", "stjewm_no_trace",
    "stjewm_hidden_leak", "stjewm_membrane_readout", "alif_timecell_baseline",
    "gru_baseline", "lewm_baseline_v2", "stacked_lif_trace", "stacked_lif_free", "mlp_baseline",
)
HELDOUT_ENVS = ("walker", "humanoid")
IN_DOMAIN_DIAGNOSTIC_ENVS = ("cartpole_2d", "pendulum_2d", "finger", "ball_in_cup", "cheetah")
METRIC_ENVS = IN_DOMAIN_DIAGNOSTIC_ENVS + HELDOUT_ENVS
REGIMES = {"full_G16": "generalist_G16", "minus_walker_humanoid": "generalist_G16_minus_walker_humanoid"}


def read_cell(root, audit, checkpoint, env, model, seed):
    eval_path = root / f"eval_{env}.json"
    ev = load_json(eval_path)
    validate_closed_loop(ev)
    audit.validate_provenance(ev, checkpoint)
    require(ev["protocol"].get("checkpoint") == str(checkpoint)
            and ev["env_id"] == native_env_id(env), "Evaluation identity mismatch")
    require(ev["protocol"].get("data_loader_protocol") == audit.payload["evaluation_data_protocol_version"],
            "Wrong evaluation data generation")
    cell = {"env": env, "env_sr": metric(ev, "success_rate_env"),
            "eval_source": str(eval_path), "eval_sha256": sha256(eval_path)}
    if env in METRIC_ENVS:
        path = root / f"align_{env}.json"
        diagnostic = read_diagnostic(path, audit, checkpoint, env=env, model=model, seed=seed)
        cell.update(divergence=metric(diagnostic, "divergence"),
                    responsiveness=metric(diagnostic, "responsiveness", nullable=True),
                    event_rho=metric(diagnostic, "event_rho", nullable=True),
                    diagnostic_source=str(path), diagnostic_sha256=sha256(path))
    return cell


def run_model(args, audit, model):
    entries = eval_entries(ROOT / "configs/generalist_G16_eval.json")
    require(len(entries) == 16, "Expected all 16 non-stress generalist environments")
    for regime, family in REGIMES.items():
        checkpoint = audit.results_root / family / model / f"seed_{args.seed}" / "final.pt"
        root = args.out_root / regime / model
        rows, provenance = evaluate_checkpoint(checkpoint, model, root, audit, entries, METRIC_ENVS,
                                              seed=args.seed, episodes=args.n_episodes, steps=args.n_steps, device=args.device)
        write_new_json(root / "completion.json", {
            "status": "completed", "model": model, "regime": regime, "seed": args.seed,
            "expected_envs": [entry["env_id"] for entry in entries], "per_env": rows,
            "repair_provenance": provenance,
        })


def aggregate(out_root, table_path, audit, seed=0):
    entries = eval_entries(ROOT / "configs/generalist_G16_eval.json")
    expected_envs = [entry["env_id"] for entry in entries]
    require(len(expected_envs) == 16, "Expected 16 non-stress environments")
    train_envs = [entry["env_id"] for entry in load_json(ROOT / "configs/generalist_G16_minus_walker_humanoid.json")]
    require(set(train_envs) == set(expected_envs) - set(HELDOUT_ENVS), "Held-out training scope mismatch")
    rows = []
    for regime, family in REGIMES.items():
        for model in TARGET_MODELS:
            checkpoint = audit.results_root / family / model / f"seed_{seed}" / "final.pt"
            root = out_root / regime / model
            completed = load_json(root / "completion.json")
            require(completed.get("status") == "completed" and (completed["model"], completed["regime"], completed["seed"]) == (model, regime, seed),
                    f"Incomplete/mismatched cross-env run: {root}")
            require(completed["expected_envs"] == expected_envs, "Incomplete cross-env planned coverage")
            audit.validate_provenance(completed, checkpoint)
            require(len(completed["per_env"]) == len(expected_envs)
                    and {cell["env"] for cell in completed["per_env"]} == set(expected_envs),
                    "Incomplete/duplicate completion manifest")
            for cell in completed["per_env"]:
                require(sha256(cell["eval_source"]) == cell["eval_sha256"], "Changed evaluation result")
                if cell["env"] in METRIC_ENVS:
                    require(sha256(cell["diagnostic_source"]) == cell["diagnostic_sha256"], "Changed diagnostic result")
            cells = {env: read_cell(root, audit, checkpoint, env, model, seed) for env in expected_envs}
            in_domain = {"env_sr": summarize(cells[env]["env_sr"] for env in train_envs)}
            in_domain.update({key: summarize(cells[env][key] for env in IN_DOMAIN_DIAGNOSTIC_ENVS)
                              for key in ("divergence", "responsiveness", "event_rho")})
            heldout = {}
            for env in HELDOUT_ENVS:
                cell = cells[env]
                gaps = {key: None if stats["mean"] is None or cell[key] is None else stats["mean"] - cell[key]
                        for key, stats in in_domain.items()}
                heldout[env] = {"metrics": cell, "gap_in_minus_holdout": gaps}
            rows.append({"model": model, "regime": regime, "seed": seed, "in_domain": in_domain,
                         "heldout": heldout, "coverage": {"expected": 16, "completed": len(cells)},
                         "repair_provenance": completed["repair_provenance"]})
    lines = ["# Audited cross-environment generalisation", "",
             "The full-G16 and minus-walker/humanoid regimes use identical planned environments and one training seed.",
             "No training-seed confidence interval is available. Undefined correlations are reported, not replaced with zero.", "",
             "| regime | model | held-out env | env-SR | divergence | responsiveness | event_rho | in-domain rho defined/expected |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        rho = row["in_domain"]["event_rho"]
        for env, heldout in row["heldout"].items():
            values = " | ".join(fmt(heldout["metrics"][key]) for key in ("env_sr", "divergence", "responsiveness", "event_rho"))
            lines.append(f"| {row['regime']} | {row['model']} | {env} | {values} | {rho['n_defined']}/{rho['n_expected']} |")
    audit.protect_output(table_path)
    audit.protect_output(table_path.with_suffix(".json"))
    write_new_json(table_path.with_suffix(".json"), {
        "status": "completed", "training_manifest": str(audit.path), "training_manifest_sha256": audit.digest,
        "planned_rows": 24, "seed_std": None, "rows": rows,
    })
    with table_path.open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    return rows


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--training-manifest", type=Path, required=True)
    result.add_argument("--out-root", type=Path, required=True)
    result.add_argument("--table-path", type=Path, required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--n-episodes", type=int, default=3)
    result.add_argument("--n-steps", type=int, default=200)
    result.add_argument("--device", default="cpu")
    result.add_argument("--aggregate-only", action="store_true")
    return result


def main():
    cli = parser()
    cli.add_argument("--model", choices=TARGET_MODELS)
    args = cli.parse_args()
    audit = TrainingAudit(args.training_manifest)
    if args.aggregate_only:
        aggregate(args.out_root, args.table_path, audit, args.seed)
    else:
        require(args.model is not None, "--model is required unless --aggregate-only")
        run_model(args, audit, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

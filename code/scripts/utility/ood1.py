"""Historical OOD1 cross-benchmark-family diagnostics, not publication inputs.

The declared design trains on one family and diagnoses the other three. The
checked-in DMC training spec is usable; the PushT/Reacher/TwoRoom training specs
and the DMC evaluation-family entries are prerequisites, not invented defaults.
The existing model list has ten entries, not twelve. This source preserves that
list and does not add scientific cells.

Measurements use the canonical protocol-2 single-frame readout and separate
observation-embedding diagnostics, with reset transitions excluded. Historical
raw results cannot substitute for a completed audited publication manifest.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from code.eval.closed_loop import make_env
from code.scripts.audited_results import fmt, load_json, require, validate_diagnostic, write_new_json
from code.scripts.event_align import ENV_KIND_MAP, build_model
from code.scripts.latent_rollout import collect_state_rollout


SPLITS = [
    ("ood1_dmc_train", "dmc", ["pusht", "reacher_4d", "tworoom"]),
    ("ood1_pusht_train", "pusht", ["dmc", "reacher_4d", "tworoom"]),
    ("ood1_reacher_train", "reacher_4d", ["dmc", "pusht", "tworoom"]),
    ("ood1_tworoom_train", "tworoom", ["dmc", "pusht", "reacher_4d"]),
]
DEFAULT_CKPT_BUDGET = [
    "stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only",
    "stjewm_no_trace", "stjewm_hidden_leak", "stjewm_membrane_readout",
    "mlp_baseline", "gru_baseline", "alif_timecell_baseline", "stacked_lif_trace",
]


def measure_diagnostic_cross_family(
    ckpt_path: str, env_kind: str, env_path: str, env_id: str,
    n_steps: int = 200, seed: int = 0, device: str = "cpu",
    action_dim: int | None = None,
) -> dict:
    """Strict checkpoint-backed, episode-safe diagnostics across native envs."""
    native_kind = (
        ENV_KIND_MAP.get(env_id, env_id) if env_kind == "dmc"
        else "reacher" if env_kind == "reacher_4d" else env_kind
    )
    env = make_env(native_kind, data_path=env_path)
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved = checkpoint["args"]
        state_dict = {key.replace("_orig_mod.", ""): value for key, value in checkpoint["model"].items()}
        state_dim = saved.get("pad_obs_to") or saved.get("state_dim") or env.spec.obs_dim
        trained_action_dim = saved.get("action_dim") or env.spec.action_dim
        require(action_dim is None or action_dim == trained_action_dim,
                "Requested action dimension differs from the checkpoint")
        model_name = Path(ckpt_path).parent.parent.name
        model = build_model(model_name, state_dim, trained_action_dim, saved, state_dict=state_dict)
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
        result, _ = collect_state_rollout(
            model, env, state_dim, trained_action_dim,
            n_steps=n_steps, n_resets=1, seed=seed, device=device,
        )
        result.update(
            ckpt=str(Path(ckpt_path).resolve()), model=model_name, env=env_id,
            env_kind=env_kind, weights_loaded_strict=True,
        )
        return result
    finally:
        env.close()


def event_align_cross_family(
    ckpt_path: str, env_kind: str, env_path: str, env_id: str,
    n_steps: int = 100, seed: int = 0, device: str = "cpu",
    action_dim: int | None = None,
) -> dict:
    """Use canonical event_rho, including None when correlation is undefined."""
    return measure_diagnostic_cross_family(
        ckpt_path, env_kind, env_path, env_id,
        n_steps=n_steps, seed=seed, device=device, action_dim=action_dim,
    )


def train_one_ckpt(
    split: str, model: str, base_seed: int = 0, out_dir: str | Path = "results/ood1",
) -> Path:
    """Train only from an existing family spec, in the requested result root."""
    target = Path("configs") / f"{split}.json"
    if not target.is_file():
        raise FileNotFoundError(f"OOD1 prerequisite missing: training family spec {target}")
    load_json(target)
    cell_dir = Path(out_dir) / split / model / f"seed_{base_seed}"
    ckpt = cell_dir / "final.pt"
    if ckpt.is_file():
        return ckpt
    subprocess.run([
        "/bin/bash", "code/scripts/generalist_v0_7_5/train_one.sh",
        model, str(target), str(cell_dir), str(base_seed),
    ], check=True)
    require(ckpt.is_file(), f"Trainer returned without checkpoint: {ckpt}")
    return ckpt


def run_one_cell(
    ckpt: Path, env_id: str, env_kind: str, env_path: str,
    seed: int = 0, device: str = "cpu",
) -> dict:
    paths = (ckpt.parent / f"div_{env_id}.json", ckpt.parent / f"rho_{env_id}.json")
    require(not any(path.exists() for path in paths),
            f"Archive existing historical OOD1 outputs before rerunning: {paths}")
    diag = measure_diagnostic_cross_family(
        str(ckpt), env_kind, env_path, env_id, n_steps=200, seed=seed, device=device,
    )
    align = event_align_cross_family(
        str(ckpt), env_kind, env_path, env_id, n_steps=100, seed=seed, device=device,
    )
    write_new_json(paths[0], diag)
    write_new_json(paths[1], align)
    return {"div": diag["divergence"], "resp": diag["responsiveness"], "rho": align["event_rho"]}


def planned_cells(out_dir: Path, splits: list[str], models: list[str], seed: int, eval_spec: Path):
    """Require declared family specs and coverage before any training starts."""
    entries = load_json(eval_spec)["specs"]
    definitions = {split: unseen for split, _, unseen in SPLITS}
    require(splits and models, "At least one split and model must be requested")
    require(len(splits) == len(set(splits)) and len(models) == len(set(models)), "Duplicate requested split/model")
    require(not set(splits) - definitions.keys(), "Unknown OOD1 split")
    cells, missing = [], []
    for split in splits:
        train_spec = Path("configs") / f"{split}.json"
        if not train_spec.is_file():
            missing.append(f"training family spec: {train_spec}")
        else:
            load_json(train_spec)
        unseen = definitions[split]
        selected = [entry for entry in entries if entry["env_kind"] in unseen]
        absent_families = set(unseen) - {entry["env_kind"] for entry in selected}
        missing.extend(f"{split}: evaluation entries for unseen family {family}" for family in sorted(absent_families))
        env_ids = [entry["env_id"] for entry in selected]
        require(len(env_ids) == len(set(env_ids)), f"Duplicate evaluation environment in {split}")
        for entry in selected:
            if not Path(entry["path"]).is_file():
                missing.append(f"{split}/{entry['env_id']}: offline data {entry['path']}")
            for model in models:
                ckpt = out_dir / split / model / f"seed_{seed}" / "final.pt"
                cells.append((split, model, ckpt, entry))
    if missing:
        raise FileNotFoundError("OOD1 prerequisites unavailable:\n- " + "\n- ".join(missing))
    require(cells, "No requested OOD1 evaluation cells")
    return cells


def aggregate(
    out_dir: str | Path = "results/ood1", *, splits: list[str] | None = None,
    models: list[str] | None = None, seed: int = 0,
    eval_spec: str | Path = "configs/ood1_eval.json",
) -> dict:
    """Inspect the requested lattice; absent runs are never completed cells."""
    cells = planned_cells(
        Path(out_dir), splits if splits is not None else [split for split, _, _ in SPLITS],
        models if models is not None else DEFAULT_CKPT_BUDGET, seed, Path(eval_spec),
    )
    rows, missing = [], []
    for split, model, ckpt, entry in cells:
        env_id = entry["env_id"]
        identity = {"split": split, "model": model, "seed": seed, "env": env_id}
        paths = (ckpt, ckpt.parent / f"div_{env_id}.json", ckpt.parent / f"rho_{env_id}.json")
        absent = [str(path) for path in paths if not path.is_file()]
        if absent:
            missing.append({**identity, "missing": absent})
            continue
        diag, align = load_json(paths[1]), load_json(paths[2])
        for payload, steps in ((diag, 200), (align, 100)):
            validate_diagnostic(payload, env=env_id, model=model, seed=seed)
            require(payload["n_steps"] == steps, f"Wrong OOD1 diagnostic length: {identity}")
            require(Path(payload["ckpt"]).resolve() == ckpt.resolve(), f"Wrong OOD1 checkpoint: {identity}")
        rows.append({**identity, "divergence": diag["divergence"],
                     "responsiveness": diag["responsiveness"], "event_rho": align["event_rho"]})
    result = {
        "status": "incomplete" if missing else "complete", "publication_ready": False,
        "protocol_version": 2, "n_required": len(cells), "n_complete": len(rows),
        "missing": missing, "cells": rows,
    }
    write_new_json(Path(out_dir) / "coverage.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=[split for split, _, _ in SPLITS],
                        default=[split for split, _, _ in SPLITS])
    parser.add_argument("--models", nargs="+", default=DEFAULT_CKPT_BUDGET)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results/ood1"))
    parser.add_argument("--eval-spec", type=Path, default=Path("configs/ood1_eval.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cells = planned_cells(args.out_dir, args.splits, args.models, args.seed, args.eval_spec)
    if not args.aggregate_only:
        if args.skip_train:
            missing = sorted({str(ckpt) for _, _, ckpt, _ in cells if not ckpt.is_file()})
            require(not missing, "--skip-train requires every requested checkpoint: " + ", ".join(missing))
        for split, model, ckpt, entry in cells:
            if not args.skip_train and not ckpt.is_file():
                train_one_ckpt(split, model, args.seed, args.out_dir)
            result = run_one_cell(ckpt, entry["env_id"], entry["env_kind"], entry["path"], args.seed, args.device)
            print(f"[{split}/{model}/{entry['env_id']}] div={fmt(result['div'])} "
                  f"resp={fmt(result['resp'])} rho={fmt(result['rho'])}")
    result = aggregate(args.out_dir, splits=args.splits, models=args.models, seed=args.seed, eval_spec=args.eval_spec)
    print(f"[ood1] {result['status']}: {result['n_complete']}/{result['n_required']} cells")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

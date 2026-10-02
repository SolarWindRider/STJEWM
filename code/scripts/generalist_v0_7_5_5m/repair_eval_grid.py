#!/usr/bin/env python3
"""Run the complete E1/E2/E4/E7 state grids against audited canonical checkpoints.

The default is a read-only plan. --execute requires a new --out-root; --resume
continues that exact plan without replacing any result or retrying failed cells.
Checkpoint selection limits this invocation, never the recorded complete grid.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
GRID_PROTOCOL = "complete_state_grid_consolidated_20260916"
# Maze evaluations run under the dedicated second-phase episode-safe loader;
# every other cell must match the manifest evaluation protocol.
MAZE_EVAL_PROTOCOL = "episode_safe_maze_20260916"
SPLITS = (
    "oodc_F1", "oodc_F2", "oodc_F3", "oodc_F1F2", "oodc_F1F3", "oodc_F2F3",
    "cross_benchmark_F1", "cross_benchmark_F2", "cross_benchmark_F3", "generalist_16env",
)
SEED_SPLITS = ("cross_benchmark_F1", "oodc_F2", "generalist_16env")
SIGREG_SPLITS = ("cross_benchmark_F1", "oodc_F2")
STJEWM = tuple("stjewm_" + name for name in (
    "trace_only", "spike_only", "rate_only", "no_trace", "hidden_leak", "membrane_readout",
))
BASELINES = (
    "alif_timecell_baseline", "gru_baseline", "lewm_baseline_v2", "stacked_lif_trace",
    "stacked_lif_free", "mlp_baseline", "lif_transformer_baseline",
)
MODELS = STJEWM + BASELINES
LAMBDAS = ("0.09", "0.01", "0.001", "0.0")
ALIASES = {"cartpole_2d": "cartpole", "pendulum_2d": "pendulum", "humanoid_CMU": "humanoid_cmu"}
BUDGET = {
    "n_episodes": 5, "n_seeds": 1, "horizon": 25, "eval_budget": 50,
    "history_size": 1, "cem_samples": 300, "cem_elites": 30,
}
SOURCE_FILES = (
    "code/eval/closed_loop.py", "code/core/cem.py", "code/core/encode.py",
    "code/core/envs/dmc_env.py", "code/core/envs/delayed_t_maze.py",
    "code/core/envs/reacher_env.py", "code/data/loaders.py", "code/data/base.py",
    "code/core/envs/swm_envs.py",
    "code/scripts/event_align.py", "code/scripts/generalist_v0_7_5_5m/repair_eval_grid.py",
)
RUNNER_SOURCE = "code/scripts/generalist_v0_7_5_5m/repair_eval_grid.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant {value} in {path}")
    with path.open() as handle:
        return json.load(handle, parse_constant=invalid)


def write_json(path: Path, value, *, replace: bool = False) -> None:
    if not replace:
        with path.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def evaluation_entries(path: Path) -> list[dict]:
    raw = load_json(path)
    entries = raw["specs"] if isinstance(raw, dict) else raw
    declared = raw.get("_train_envs") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Empty or malformed specification: {path}")
    if declared is not None and len(declared) != len(entries):
        raise ValueError(f"Declared task list does not align with source rows: {path}")
    normalized = []
    for index, entry in enumerate(entries):
        source_env = entry["env_id"]
        task = declared[index] if declared is not None else source_env
        source_declared_env = task
        condition = "native_state"
        correction = None
        if task != source_env:
            # These original specs used the same qpos dataset for native and
            # masked cheetah, causing a filename collision in oodc_F1F3. The
            # declared task selects the actual runtime observation wrapper;
            # this does NOT claim that training applied an observation mask.
            if (source_env, task) != ("cheetah", "cheetah_velhidden"):
                raise ValueError(f"Unresolved declared/source task mismatch: {path}:{index}: {entry}")
            correction = "Use aligned _train_envs task; preserve original unmasked training row"
        env = entry.get("clo_env") or ALIASES.get(task, task)
        if task != source_env and env != task:
            raise ValueError(f"Explicit clo_env contradicts declared task: {path}:{index}")
        if env in ("cheetah_velhidden", "cheetah_qpos_masked"):
            task = env = "cheetah_qpos_masked"
            condition = "qpos_indices_3_4_5_masked_not_velocity"
        data = (ROOT / entry["path"]).resolve()
        if not data.is_file():
            raise FileNotFoundError(f"Required data for {path}:{index}: {data}")
        # These four experiments have no stress flag bundles. Fail closed if
        # their source protocol changes rather than allowing an override of
        # the fixed budgets or output/checkpoint arguments.
        if entry.get("extra_flags"):
            raise ValueError(f"Unreviewed extra_flags in {path}:{index}: {entry['extra_flags']}")
        normalized.append({
            "source_row": index, "source_entry": entry, "env_id": task, "clo_env": env,
            "source_declared_env": source_declared_env,
            "data": str(data), "data_size": data.stat().st_size,
            "goal_offset": int(entry.get("goal_offset", 25)),
            "observation_condition": condition, "specification_correction": correction,
        })
    names = [entry["env_id"] for entry in normalized]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate task/output identities remain in {path}: {names}")
    return normalized


def build_plan(manifest_path: Path, manifest: dict, experiments: tuple[str, ...], out_root: Path) -> dict:
    required_metadata = manifest.get("required_checkpoint_metadata")
    if not isinstance(required_metadata, dict) or not required_metadata:
        raise ValueError("A consolidated model/loss/data repair manifest with required_checkpoint_metadata is mandatory; data-only audits are obsolete")
    for field in ("training_source_sha256", "loader_source_sha256"):
        if not isinstance(manifest.get(field), dict) or not manifest[field]:
            raise ValueError(f"Consolidated manifest requires nonempty {field}")
    for job in manifest["affected_checkpoints"]:
        data_protocol = job.get("required_data_protocol_version")
        if not isinstance(data_protocol, str) or not data_protocol:
            raise ValueError(f"Missing per-checkpoint data protocol: {job['checkpoint']}")
        if type(job.get("expected_step")) is not int or job["expected_step"] < 1:
            raise ValueError(f"Missing positive approved step budget: {job['checkpoint']}")
        if required_metadata.get("data_protocol_version", data_protocol) != data_protocol:
            raise ValueError("Common required metadata contradicts a per-checkpoint data protocol")
    results_root = Path(manifest["results_root"]).resolve()
    affected = {str(Path(row["checkpoint"]).resolve()): row for row in manifest["affected_checkpoints"]}
    safe = {str(Path(path).resolve()) for path in manifest["unaffected_checkpoints"]}
    if set(affected) & safe:
        raise ValueError("Affected and unaffected checkpoint sets overlap")
    loader_module = ast.parse((ROOT / "code/data/loaders.py").read_text())
    loader_protocols = [
        node.value.value for node in loader_module.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        and any(isinstance(target, ast.Name) and target.id == "DATA_LOADER_PROTOCOL_VERSION"
                for target in node.targets)
    ]
    expected_eval_protocol = manifest["evaluation_data_protocol_version"]
    if len(loader_protocols) != 1 or loader_protocols[0] != expected_eval_protocol:
        raise ValueError("Active evaluator loader protocol differs from the consolidated repair manifest")
    eval_data_protocol = loader_protocols[0]
    # A fresh sibling directory is allowed, but no output can land inside a
    # canonical checkpoint family, archive or training staging directory.
    protected = [results_root / family for family in manifest["family_counts"]]
    protected += [Path(manifest["archive_root"]), Path(manifest["staging_root"])]
    if out_root == results_root or any(out_root == path or path in out_root.parents for path in protected):
        raise ValueError(f"--out-root must be separate from checkpoint/archive/staging roots: {out_root}")
    cells, source_hashes = [], {relative: sha256(ROOT / relative) for relative in SOURCE_FILES}
    for registry in ("training_source_sha256", "loader_source_sha256"):
        for relative, expected_hash in manifest[registry].items():
            actual_hash = source_hashes.get(relative)
            if actual_hash is None:
                actual_hash = sha256(ROOT / relative)
            if actual_hash != expected_hash:
                raise ValueError(f"Evaluation source differs from consolidated {registry}: {relative}")
            source_hashes[relative] = actual_hash
    specs = {}

    def add(experiment, family, split, model, seed, spec_kind):
        spec_path = ROOT / "configs" / spec_kind / f"{split}.json"
        relative_spec = str(spec_path.relative_to(ROOT))
        if relative_spec not in specs:
            specs[relative_spec] = evaluation_entries(spec_path)
            source_hashes[relative_spec] = sha256(spec_path)
        checkpoint = str(results_root / family / split / model / "seed_0" / "final.pt")
        if checkpoint not in affected and checkpoint not in safe:
            raise ValueError(f"Planned checkpoint is absent from the training impact audit: {checkpoint}")
        output_split = "heldout_" + split if experiment == "E4" else split
        for entry in specs[relative_spec]:
            cell_id = f"{experiment}/{output_split}/{model}/seed_{seed}/{entry['env_id']}"
            output = out_root / experiment / output_split / model / f"seed_{seed}" / f"eval_{entry['env_id']}.json"
            cell = dict(entry, id=cell_id, experiment=experiment, family=family, split=split,
                        model=model, training_seed=seed, checkpoint=checkpoint,
                        checkpoint_class="affected" if checkpoint in affected else "safe",
                        expected_data_protocol=eval_data_protocol,
                        source_spec=relative_spec, output=str(output),
                        log=str(out_root / "logs" / (cell_id + ".log")),
                        cem_iters=30 if entry["clo_env"].startswith("pusht") else 10)
            command = [sys.executable, "-u", "-m", "code.eval.closed_loop", "--env", cell["clo_env"],
                       "--ckpt", checkpoint, "--data", cell["data"], "--out", str(output),
                       "--device", "cuda:0", "--split", "in_dist"]
            for key, value in dict(BUDGET, goal_offset=cell["goal_offset"], cem_iters=cell["cem_iters"],
                                   pad_obs_eval=128, action_dim_eval=56).items():
                command.extend(("--" + key.replace("_", "-"), str(value)))
            cell["command"] = command
            cells.append(cell)

    for experiment in experiments:
        if experiment in ("E1", "E4"):
            for split in SPLITS if experiment == "E1" else SPLITS[:-1]:
                for model in MODELS:
                    add(experiment, "5m_5mpar" if model in STJEWM else "5m", split, model, 0,
                        "oodc_5m" if experiment == "E1" else "heldout_eval")
        elif experiment == "E2":
            # Both original B2 and G5 suites: all 13 models, seeds 1 and 2.
            # Their checkpoint directory is seed_0, not seed_<training seed>.
            for seed in (1, 2):
                for split in SEED_SPLITS:
                    for model in MODELS:
                        add(experiment, f"5m_seed{seed}", split, model, seed, "oodc_5m")
        elif experiment == "E7":
            for value in LAMBDAS:
                for split in SIGREG_SPLITS:
                    add(experiment, "5m_sigreg_sweep", split, "stjewm_trace_only_sig" + value, 0, "oodc_5m")
    if len({cell["id"] for cell in cells}) != len(cells):
        raise ValueError("Grid contains duplicate cell IDs")
    summary = {}
    for experiment in experiments:
        group = [cell for cell in cells if cell["experiment"] == experiment]
        summary[experiment] = {
            "planned_cells": len(group), "checkpoints": len({cell["checkpoint"] for cell in group}),
            "by_split": dict(Counter(cell["split"] for cell in group)),
            "by_checkpoint_class": dict(Counter(cell["checkpoint_class"] for cell in group)),
        }
    plan = {
        "grid_protocol": GRID_PROTOCOL, "data_protocol": eval_data_protocol,
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "required_checkpoint_metadata": required_metadata,
        "execution_ready": manifest.get("execution_ready") is True,
        "source_sha256": source_hashes, "out_root": str(out_root), "experiments": list(experiments),
        "budget": dict(BUDGET, pad_obs_eval=128, action_dim_eval=56, cem_iters="PushT:30; otherwise:10"),
        "summary": summary, "planned_cells": len(cells), "cells": cells,
        "interpretation_notes": [
            "E2 reruns only existing seed1/seed2 checkpoints; use E1 for the seed0 reference.",
            "E1 original configs have100 rows/model; old1248 omitted39 maze cells and collided13 cheetah filenames.",
            "E1 _train_envs declares cheetah_velhidden but source training rows are unmasked cheetah; record this discrepancy.",
            "Legacy cheetah_velhidden provenance maps to cheetah_qpos_masked: qpos indices3:6, not velocities.",
            "E4 source and existing raw grid both contain42 cells/model (39 OOD+3cross):546 total, unchanged.",
            "E7 full four-lambda grid is84; old sweep directory had63, missing both lambda0.09 evaluations.",
            "Checkpoint eligibility comes exclusively from the consolidated model/loss/data audit, never task or model names.",
        ],
    }
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def resume_view(plan: dict) -> dict:
    """Plan identity for --resume, tolerating only this runner's own fixes.

    The runner hashes itself into source_sha256, so a validator fix would
    otherwise block resuming an identical grid. Every other tracked source,
    the manifest, specifications and all cell content stay strictly compared.
    """
    view = json.loads(json.dumps(plan))
    view.pop("plan_sha256", None)
    runner_hash = view.get("source_sha256", {}).get(RUNNER_SOURCE)
    if runner_hash is not None:
        view["source_sha256"][RUNNER_SOURCE] = "runner-self"
    return view


def metadata_mismatches(actual, expected, prefix="checkpoint") -> list[str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [prefix]
        mismatches = []
        for key, value in expected.items():
            name = prefix + "." + key
            if key not in actual:
                mismatches.append(name)
            else:
                mismatches.extend(metadata_mismatches(actual[key], value, name))
        return mismatches
    return [] if actual == expected else [prefix]


def checkpoint_state(path: str, audit: dict | None) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return {"status": "pending_checkpoint" if audit is not None else "failed",
                "reason": "Canonical checkpoint has not been promoted" if audit is not None else "Missing audited safe checkpoint"}
    import torch
    try:
        before = checkpoint_path.stat()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
        version = checkpoint.get("data_protocol_version")
        if audit is not None:
            mismatches = metadata_mismatches(checkpoint, audit["required_metadata"])
            if mismatches:
                return {"status": "pending_checkpoint",
                        "reason": "Canonical checkpoint awaits consolidated repair provenance: " + ", ".join(mismatches)}
            receipt = load_json(checkpoint_path.parent / "training_repair.json")
            if receipt.get("status") != "completed" or receipt.get("checkpoint") != path:
                raise ValueError("Affected checkpoint has no matching successful atomic-promotion receipt")
            if checkpoint.get("training_provenance", {}).get("source_sha256") != audit["training_source_sha256"]:
                raise ValueError("Repaired checkpoint training-source fingerprints differ from the consolidated audit")
            provenance = checkpoint["data_provenance"]
            if provenance["loader_protocol"] != audit["data_protocol_version"]:
                raise ValueError("Repaired checkpoint loader provenance mismatch")
            if provenance["loader_source_sha256"] != audit["loader_source_sha256"]:
                raise ValueError("Repaired checkpoint loader fingerprints differ from its required repair audit")
            job = audit["job"]
            if checkpoint.get("step") != job["expected_step"]:
                raise ValueError("Repaired checkpoint did not complete the approved legal training-step budget")
            if provenance["spec"]["sha256"] != job["spec_sha256"]:
                raise ValueError("Repaired checkpoint training specification fingerprint mismatch")
            expected = {key: value for key, value in job["args"].items() if key != "out"}
            actual = {key: value for key, value in checkpoint["args"].items() if key != "out"}
            if actual != expected:
                raise ValueError("Repaired checkpoint arguments differ from the audited training budget")
        digest = sha256(checkpoint_path)
        after = checkpoint_path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError("Checkpoint changed while checking provenance")
        return {"status": "ready", "sha256": digest, "size": after.st_size,
                "mtime_ns": after.st_mtime_ns, "inode": after.st_ino,
                "data_protocol_version": version, "step": checkpoint["step"],
                "training_source_sha256": audit["training_source_sha256"] if audit is not None else None,
                "verified_required_metadata": audit["required_metadata"] if audit is not None else None,
                "repair_manifest": audit["manifest"] if audit is not None else None,
                "repair_manifest_sha256": audit["manifest_sha256"] if audit is not None else None}
    except Exception:
        return {"status": "failed", "reason": traceback.format_exc()}


def validate_result(cell: dict, dependency: dict) -> None:
    payload = load_json(Path(cell["output"]))
    if not isinstance(payload, dict):
        raise ValueError("Evaluator result must be a JSON object")

    def inspect(value):
        if isinstance(value, dict):
            if any(key in value for key in ("error", "skipped", "fallback")):
                raise ValueError("Error/skipped/fallback records are not evaluation results")
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Evaluation contains non-finite values")
    inspect(payload)
    numeric_fields = (
        "success_rate_lewm", "success_rate_lewm_std", "success_rate_lewm_005", "success_rate_lewm_001",
        "success_rate_env", "success_rate_env_std", "mean_cos_dist", "mean_cos_dist_std",
        "mean_phys_dist", "mean_phys_dist_std", "mean_reward", "mean_reward_std", "wall_time_sec",
    )
    for key in numeric_fields:
        if type(payload.get(key)) not in (int, float):
            raise ValueError(f"Missing/malformed aggregate numeric metric: {key}")
        if key.startswith("success_rate") and not 0 <= payload[key] <= 1:
            raise ValueError(f"Invalid success-rate metric: {key}")
        if (key.endswith("_std") or key == "wall_time_sec") and payload[key] < 0:
            raise ValueError(f"Invalid nonnegative metric: {key}")
    expected_env = {
        "pusht": "swm/PushT-v1", "tworoom": "swm/TwoRoom-v1",
        "cheetah_qpos_masked": "mujoco/cheetah",
        "delayed_t_maze": "delayed_t_maze/delay50_cue3",
    }.get(cell["clo_env"], "mujoco/" + cell["clo_env"])
    if payload.get("env_id") != expected_env:
        raise ValueError(f"Wrong/missing native result environment: {payload.get('env_id')!r}")
    for key, value in dict(BUDGET, goal_offset=cell["goal_offset"], cem_iters=cell["cem_iters"]).items():
        if type(payload.get(key)) is not int or payload[key] != value:
            raise ValueError(f"Incomplete/wrong evaluation budget: {key}={payload.get(key)!r}; expected{value}")
    protocol = payload["protocol"]
    expected_loader_protocol = (MAZE_EVAL_PROTOCOL if cell["clo_env"] == "delayed_t_maze"
                                else cell["expected_data_protocol"])
    if (protocol.get("version") != 2 or protocol.get("data_loader_protocol") != expected_loader_protocol
            or protocol.get("cem_cost") != "squared_l2"
            or protocol.get("latent_representation") != "forward.emb, independent observed frames, zero action"
            or protocol.get("replanning") != "observed history after executing up to H actions"):
        raise ValueError("Result is not the repaired observed-state planning protocol")
    if protocol.get("checkpoint") != cell["checkpoint"]:
        raise ValueError("Result came from a different checkpoint")
    if protocol.get("checkpoint_data_protocol_version") != dependency["data_protocol_version"]:
        raise ValueError("Result checkpoint provenance differs from the checked checkpoint")
    if protocol.get("model_action_dim") != 56 or protocol.get("model_state_dim") != 128:
        raise ValueError("Result does not use the original padded model interface")
    native_dim = protocol.get("cem_action_dim")
    if type(native_dim) is not int or not 0 < native_dim <= 56:
        raise ValueError("Missing native CEM action dimensionality")
    if cell["clo_env"] == "cheetah_qpos_masked":
        mask = protocol.get("observation_mask", {})
        if mask != {"indices": [3, 4, 5], "quantity": "qpos", "goal_masked": True}:
            raise ValueError("Masked-qpos task lacks explicit matching observed/goal mask evidence")
    if not isinstance(protocol.get("physical_success"), dict):
        raise ValueError("Missing native physical-goal metric protocol")
    episodes = payload["per_episode"]
    if len(episodes) != 5 or len(payload["per_seed"]) != 1 or payload["per_seed"][0]["n"] != 5:
        raise ValueError("Result does not contain all five planned episodes")
    if len({(episode["seed"], episode["episode_idx"]) for episode in episodes}) != 5:
        raise ValueError("Evaluation episode identities are duplicated")
    for episode in episodes:
        if episode["seed"] != 0 or not 1 <= episode["actions_taken"] <= 50 or not 1 <= episode["planning_calls"] <= 50:
            raise ValueError("Invalid closed-loop episode seed/action/planning counts")
        for key in ("init_state", "goal_state", "final_state"):
            if (not isinstance(episode[key], list) or not episode[key]
                    or any(type(value) not in (int, float) for value in episode[key])):
                raise ValueError(f"Missing recorded native {key}")
        for key in ("env_success", "lewm_success", "lewm_success_at_005", "lewm_success_at_001"):
            if type(episode[key]) is not bool:
                raise ValueError(f"Malformed episode success metric: {key}")
        for key in ("cos_dist", "phys_dist", "episode_reward", "plan_time_sec"):
            if type(episode[key]) not in (int, float):
                raise ValueError(f"Malformed episode numeric metric: {key}")
    for top, per_episode in (("success_rate_env", "env_success"), ("success_rate_lewm", "lewm_success"),
                             ("mean_cos_dist", "cos_dist"), ("mean_phys_dist", "phys_dist"),
                             ("mean_reward", "episode_reward")):
        value = payload[top]
        if type(value) not in (int, float) or not math.isclose(value, sum(ep[per_episode] for ep in episodes) / 5,
                                                           rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"Aggregate metric does not describe the five completed episodes: {top}")


def run_cell(cell: dict, gpu: str, dependency: dict, threads: int, timeout: float) -> dict:
    outcome = {"status": "failed", "gpu": gpu, "started_at": utc_now(), "checkpoint": dependency}
    try:
        path = Path(cell["checkpoint"])
        info = path.stat()
        if (info.st_ino, info.st_size, info.st_mtime_ns) != (dependency["inode"], dependency["size"], dependency["mtime_ns"]):
            raise RuntimeError("Checkpoint changed after dependency validation")
        output, log = Path(cell["output"]), Path(cell["log"])
        if output.exists():
            raise FileExistsError(f"Refusing to replace an existing evaluation: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu,
                         MUJOCO_GL="egl", PYTHONPATH=str(ROOT), OMP_NUM_THREADS=str(threads),
                         MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                         NUMEXPR_NUM_THREADS=str(threads))
        with log.open("x") as handle:
            process = subprocess.run(cell["command"], cwd=ROOT, env=child_env, stdout=handle,
                                     stderr=subprocess.STDOUT, timeout=timeout)
        outcome["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError(f"Evaluator exited{process.returncode}; see {log}")
        validate_result(cell, dependency)
        outcome.update(status="completed", output_sha256=sha256(output))
    except Exception:
        outcome["failure"] = traceback.format_exc()
    outcome["finished_at"] = utc_now()
    return outcome


def execute(plan: dict, manifest: dict, args) -> int:
    if manifest.get("execution_ready") is not True:
        raise ValueError("Consolidated manifest has not been released for execution")
    out_root = Path(plan["out_root"])
    if args.resume:
        previous_plan = load_json(out_root / "plan.json")
        if resume_view(previous_plan) != resume_view(plan):
            raise ValueError("Cannot resume: planned cells, configuration, manifest or evaluator source changed")
        # Status must always advertise the plan contract on disk, not the
        # re-computed fingerprint of the current runner generation.
        contract_plan_sha256 = previous_plan["plan_sha256"]
    else:
        out_root.mkdir(parents=True, exist_ok=False)
        write_json(out_root / "plan.json", plan)
        contract_plan_sha256 = plan["plan_sha256"]
    with (out_root / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = load_json(out_root / "status.json") if args.resume and (out_root / "status.json").exists() else {}
        records = previous.get("cells", {})
        fingerprint = sha256(Path(plan["manifest"]))
        audits = {
            str(Path(job["checkpoint"]).resolve()): {
                "job": job, "manifest": plan["manifest"], "manifest_sha256": fingerprint,
                "data_protocol_version": job["required_data_protocol_version"],
                "loader_source_sha256": manifest["loader_source_sha256"],
                "training_source_sha256": manifest["training_source_sha256"],
                "required_metadata": dict(manifest["required_checkpoint_metadata"],
                                          data_protocol_version=job["required_data_protocol_version"]),
            }
            for job in manifest["affected_checkpoints"]
        }

        def dependency_state(path):
            return checkpoint_state(path, audits.get(path))

        dependencies = {}
        cells = {cell["id"]: cell for cell in plan["cells"]}
        started = time.monotonic()
        invocation = {"started_at": utc_now(), "checkpoint_selection": args.checkpoint_selection,
                      "physical_gpus": args.gpus, "workers_per_gpu": args.workers_per_gpu,
                      "threads": args.threads, "checkpoint_wait_seconds": args.checkpoint_wait_seconds,
                      "cell_timeout_seconds": args.cell_timeout_seconds}
        selected = set()
        for cell in cells.values():
            classification = cell["checkpoint_class"]
            selection = args.checkpoint_selection
            choose = selection == "all" or (selection == "safe" and classification == "safe") or (selection == "repaired" and classification == "affected")
            if selection == "ready":
                if cell["checkpoint"] not in dependencies:
                    dependencies[cell["checkpoint"]] = dependency_state(cell["checkpoint"])
                choose = dependencies[cell["checkpoint"]]["status"] != "pending_checkpoint"
            if choose:
                selected.add(cell["id"])
            records.setdefault(cell["id"], {"status": "planned"})
        invocation["selected_cells"] = len(selected)
        event_path = out_root / "events.jsonl"

        def record(cell_id, value):
            records[cell_id] = value
            with event_path.open("a") as handle:
                handle.write(json.dumps({"at": utc_now(), "cell": cell_id, **value}, allow_nan=False) + "\n")

        def save(finished=False):
            counts = Counter(value["status"] for value in records.values())
            selected_complete = bool(selected) and all(records[key]["status"] == "completed" for key in selected)
            status = {
                "grid_protocol": GRID_PROTOCOL, "plan_sha256": contract_plan_sha256,
                "updated_at": utc_now(), "invocation": invocation, "invocation_finished": finished,
                "planned_cells": len(cells), "selected_cells": len(selected),
                "selected_scope_complete": selected_complete,
                "full_grid_complete": counts.get("completed", 0) == len(cells),
                "counts": dict(counts), "selected_counts": dict(Counter(records[key]["status"] for key in selected)),
                "by_experiment": {experiment: dict(Counter(records[cell["id"]]["status"] for cell in cells.values()
                                                            if cell["experiment"] == experiment))
                                  for experiment in plan["experiments"]},
                "cells": records,
            }
            write_json(out_root / "status.json", status, replace=True)
            return status

        # A file's existence is never enough to resume successfully: a prior
        # zero subprocess exit, the exact output hash and full schema are needed.
        for key, cell in cells.items():
            prior = records[key]
            validation_failed = (prior["status"] == "failed" and prior.get("returncode") == 0
                                 and "in validate_result" in prior.get("failure", ""))
            interrupted = prior["status"] == "running"
            if ((validation_failed or interrupted)
                    and Path(cell["output"]).is_file() and Path(cell["log"]).is_file()
                    and prior["checkpoint"].get("status") == "ready"):
                # The evaluator process finished its work (complete schema-valid
                # output plus log) or succeeded and only the acceptance check
                # itself was wrong. Re-validate the untouched output, still
                # bound to the original checkpoint fingerprint.
                note = ("revalidated_output: accepted after validator fix; evaluator exit and log unchanged"
                        if validation_failed else
                        "accepted after graceful stop: complete schema-valid output and log with unchanged checkpoint")
                try:
                    if sha256(Path(cell["checkpoint"])) != prior["checkpoint"]["sha256"]:
                        raise ValueError("Canonical checkpoint changed since the failed validation")
                    validate_result(cell, prior["checkpoint"])
                    record(key, {"status": "completed", "gpu": prior.get("gpu"),
                                 "started_at": prior.get("started_at"),
                                 "finished_at": prior.get("finished_at", utc_now()),
                                 "checkpoint": prior["checkpoint"],
                                 "returncode": prior.get("returncode", 0),
                                 "output_sha256": sha256(Path(cell["output"])),
                                 "revalidated_output": note})
                except Exception:
                    record(key, {"status": "failed", "gpu": prior.get("gpu"),
                                 "checkpoint": prior["checkpoint"], "returncode": prior.get("returncode", 0),
                                 "failure": "Re-validation after validator fix rejected the output:\n" + traceback.format_exc()})
                continue
            if interrupted and not Path(cell["output"]).exists():
                # Graceful stop before the evaluator wrote anything; after the
                # orphan drain there is no writer left, so the cell is clean.
                # The partial log is rotated, never deleted, so the fresh run
                # can create its own log without colliding.
                log_path = Path(cell["log"])
                if log_path.is_file():
                    log_path.rename(log_path.with_name(log_path.name + ".interrupted"))
                record(key, {"status": "planned", "gpu": prior.get("gpu"),
                             "rescheduled_after": "graceful stop; no output artifact existed"})
                continue
            if (prior["status"] == "failed" and prior.get("returncode") is None
                    and "log.open" in prior.get("failure", "")
                    and "FileExistsError" in prior.get("failure", "")
                    and not Path(cell["output"]).exists()):
                # The cell was rescheduled onto its own stale partial log; no
                # evaluator ever ran. Rotate the stale log and rerun cleanly.
                log_path = Path(cell["log"])
                if log_path.is_file():
                    log_path.rename(log_path.with_name(log_path.name + ".interrupted"))
                record(key, {"status": "planned", "gpu": prior.get("gpu"),
                             "rescheduled_after": "log collision from an earlier graceful stop; evaluator never ran"})
                continue
            if prior["status"] == "completed":
                try:
                    if prior.get("returncode") != 0:
                        raise ValueError("Completed record lacks successful evaluator process exit evidence")
                    if cell["checkpoint"] not in dependencies:
                        dependencies[cell["checkpoint"]] = dependency_state(cell["checkpoint"])
                    current = dependencies[cell["checkpoint"]]
                    if current["status"] != "ready" or current["sha256"] != prior["checkpoint"]["sha256"]:
                        raise ValueError("Canonical checkpoint is no longer the one used by this completed evaluation")
                    if sha256(Path(cell["output"])) != prior["output_sha256"]:
                        raise ValueError("Completed output changed since the prior invocation")
                    validate_result(cell, prior["checkpoint"])
                except Exception:
                    record(key, {"status": "failed", "failure": traceback.format_exc()})
            elif key in selected and prior["status"] not in ("failed",):
                if prior["status"] == "running" or Path(cell["output"]).exists() or Path(cell["log"]).exists():
                    record(key, {"status": "failed", "failure": "Interrupted/untracked artifacts require a new out-root; never overwrite or accept without exit evidence"})
                else:
                    record(key, {"status": "planned"})
            elif key not in selected and prior["status"] not in ("failed",):
                record(key, {"status": "not_selected", "reason": f"Invocation checkpoint selection: {args.checkpoint_selection}"})
        save()
        pending = {key for key in selected if records[key]["status"] == "planned"}
        slots = args.gpus * args.workers_per_gpu
        active = {}
        next_poll = 0.0
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            while pending or active:
                now = time.monotonic()
                if now >= next_poll:
                    for path in {cells[key]["checkpoint"] for key in pending}:
                        if path not in dependencies or dependencies[path]["status"] == "pending_checkpoint":
                            dependencies[path] = dependency_state(path)
                    next_poll = now + args.poll_seconds
                for key in list(pending):
                    dependency = dependencies[cells[key]["checkpoint"]]
                    if dependency["status"] == "failed":
                        record(key, {"status": "failed", "failure": dependency["reason"]})
                        pending.remove(key)
                    elif dependency["status"] == "pending_checkpoint":
                        if now - started >= args.checkpoint_wait_seconds:
                            record(key, {"status": "failed", "failure": "Timed out waiting for canonical episode-safe checkpoint promotion"})
                            pending.remove(key)
                        elif records[key]["status"] != "pending_checkpoint":
                            record(key, {"status": "pending_checkpoint", "reason": dependency["reason"]})
                # Scan the complete pending queue for ready dependencies. An
                # early missing checkpoint cannot stall safe work on any GPU.
                for key in cells:
                    if not slots:
                        break
                    if key not in pending or dependencies[cells[key]["checkpoint"]]["status"] != "ready":
                        continue
                    gpu = slots.pop(0)
                    dependency = dependencies[cells[key]["checkpoint"]]
                    record(key, {"status": "running", "gpu": gpu, "started_at": utc_now(), "checkpoint": dependency})
                    active[pool.submit(run_cell, cells[key], gpu, dependency, args.threads, args.cell_timeout_seconds)] = (key, gpu)
                    pending.remove(key)
                save()
                if active:
                    done, _ = wait(active, timeout=args.poll_seconds, return_when=FIRST_COMPLETED)
                    for future in done:
                        key, gpu = active.pop(future)
                        record(key, future.result())
                        slots.append(gpu)
                        print(f"[{records[key]['status']}] {key} GPU{gpu}", flush=True)
                elif pending:
                    time.sleep(args.poll_seconds)
        final = save(finished=True)
        with (out_root / "invocations.jsonl").open("a") as handle:
            handle.write(json.dumps({**invocation, "finished_at": utc_now(), "counts": final["counts"],
                                     "full_grid_complete": final["full_grid_complete"],
                                     "selected_scope_complete": final["selected_scope_complete"]}) + "\n")
        print(json.dumps({key: value for key, value in final.items() if key != "cells"}, indent=2), flush=True)
        if final["counts"].get("failed", 0):
            return 1
        return 0 if final["selected_scope_complete"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Consolidated model/loss/data repair manifest")
    parser.add_argument("--out-root", type=Path, required=True, help="Fresh separate evaluation directory")
    parser.add_argument("--experiments", default="E1,E2,E4,E7")
    parser.add_argument("--checkpoint-selection", choices=("all", "safe", "repaired", "ready"), default="all",
                        help="ready snapshots available dependencies; all/repaired wait for missing promotions")
    parser.add_argument("--gpus", default="0,1,3", help="Comma-separated physical GPU indices")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint-wait-seconds", type=float, default=86400)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--cell-timeout-seconds", type=float, default=7200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-plan", action="store_true", help="Print every planned cell, not just summary")
    args = parser.parse_args()
    args.gpus = args.gpus.split(",")
    experiments = tuple(args.experiments.split(","))
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(not gpu.isdigit() for gpu in args.gpus):
        parser.error("--gpus requires unique physical GPU indices")
    if len(set(experiments)) != len(experiments) or any(value not in ("E1", "E2", "E4", "E7") for value in experiments):
        parser.error("--experiments must select distinct E1,E2,E4,E7 names")
    if min(args.workers_per_gpu, args.threads) < 1:
        parser.error("Worker and thread bounds must be positive")
    if not all(math.isfinite(value) and value > 0 for value in (args.poll_seconds, args.cell_timeout_seconds)):
        parser.error("Polling interval and cell timeout must be finite and positive")
    if not math.isfinite(args.checkpoint_wait_seconds) or args.checkpoint_wait_seconds < 0:
        parser.error("Checkpoint wait must be finite and nonnegative")
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    plan = build_plan(manifest_path, manifest, experiments, args.out_root.resolve())
    if not args.execute:
        print(json.dumps(plan if args.print_plan else {key: value for key, value in plan.items() if key != "cells"}, indent=2))
        return 0
    return execute(plan, manifest, args)


if __name__ == "__main__":
    raise SystemExit(main())

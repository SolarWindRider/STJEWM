#!/usr/bin/env python3
"""Run the audited auxiliary grids without evaluating archived training weights.

Without --execute, print the complete plan. Training manifests are ordered: a
later affected-checkpoint entry supersedes an earlier audit of the same path.
Each execution needs a new --run-dir; validated canonical results are reusable,
but incomplete, incompatible, or failed results are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from code.scripts.generalist_v0_7_5_5m.measure_latent_stats_5m import DMC_ENVS, MODELS
from code.scripts.probe import ENV_PROBE, ENV_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path("/data/lx/tmp/results")
GROUPS = ("probes", "state", "scale", "rollouts")
PROBE_ENVS = ("cartpole_2d", "pendulum_2d", "finger", "ball_in_cup", "cheetah", "walker", "hopper")
STATS_ENVS = tuple(name for name, _ in DMC_ENVS[:7])
ROLLOUT_ENVS = ("cartpole_2d", "cheetah")
SCALE_FAMILIES = ("generalist_G4", "generalist_G8", "generalist_G16")
SIGREG_MODELS = tuple(f"stjewm_trace_only_sig{weight}" for weight in ("0.09", "0.01", "0.001", "0.0"))
SOURCE_FILES = (
    "code/scripts/repair_auxiliary_grid.py", "code/scripts/probe.py",
    "code/scripts/event_align.py", "code/scripts/latent_rollout.py",
    "code/scripts/generalist_v0_7_5_5m/measure_latent_stats_5m.py",
    "code/data/__init__.py", "code/data/base.py", "code/data/loaders.py", "code/data/multi_env.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def finite_tree(value, location="result"):
    if isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite number at {location}")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{location}[{index}]")


def load_json(path):
    value = json.loads(Path(path).read_text())
    finite_tree(value, str(path))
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def atomic_status(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def publish(source, destination):
    """Publish complete bytes atomically, using a no-clobber hard link."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".auxiliary-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with Path(source).open("rb") as original:
                shutil.copyfileobj(original, handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def core_family(model):
    return "5m_5mpar" if model.startswith("stjewm") else "5m"


def read_audits(paths, results):
    dependencies, manifests = {}, []
    for path in paths:
        path = path.resolve()
        audit = load_json(path)
        require(Path(audit["results_root"]).resolve() == results, f"Results root differs in {path}")
        required_metadata = audit.get("required_checkpoint_metadata", {})
        summary = {"path": str(path), "sha256": fingerprint(path),
                   "required_checkpoint_metadata": required_metadata,
                   "execution_ready": audit.get("execution_ready") is True}
        manifests.append(summary)
        for checkpoint in audit.get("unaffected_checkpoints", []):
            dependencies.setdefault(checkpoint, {"affected": False, "manifest": summary})
        for entry in audit["affected_checkpoints"]:
            checkpoint = entry["checkpoint"]
            relative = Path(checkpoint).relative_to(results)
            require(relative.parts[0] not in ("_repair_archive", "_repair_staging"), f"Noncanonical checkpoint: {checkpoint}")
            dependencies[checkpoint] = {
                "affected": True, "manifest": summary, "audit": entry,
                "loader_source_sha256": audit.get("loader_source_sha256"),
                "training_source_sha256": audit.get("training_source_sha256"),
                "required_data_protocol_version": entry.get("required_data_protocol_version", audit.get("data_protocol_version")),
            }
    return dependencies, manifests


def budget_for(group):
    if group == "probes":
        return {"seed": 0, "probe_seed": 12345, "epochs": 1, "max_windows": 5000,
                "batch": 128, "lr": 0.001, "val_frac": 0.2, "history_size": 1,
                "goal_offset": 25, "context_steps": 2, "pad_obs_to": 128, "action_dim": 56,
                "representation": "readout", "target_metric": "raw_unclipped_r2"}
    return {"seed": 0, "n_steps": 200, "requested_segments": 2,
            "context_steps": 1, "pad_obs_to": 128, "action_dim": 56}


def command_for(job, stage, cells):
    command = [sys.executable, "-u", "-m"]
    common = ["--device", "cuda:0"]
    if job["group"] == "probes":
        cell = cells[0]
        return command + ["code.scripts.probe", "--env", cell["env"], "--model", job["model"],
                          "--ckpt", job["checkpoint"], "--probe-target", "position",
                          "--representation", "readout", "--out", str(stage / cell["stage_json"]),
                          "--epochs", "1", "--max-windows", "5000", "--batch", "128", "--lr", "0.001",
                          "--val-frac", "0.2", "--pad-obs-to", "128", "--action-dim-eval", "56"] + common
    if job["group"] in ("state", "scale"):
        return command + ["code.scripts.generalist_v0_7_5_5m.measure_latent_stats_5m",
                          "--results", job["producer_results"], "--out", str(stage),
                          "--splits", job["split"], "--models", job["model"],
                          "--envs", *[cell["env"] for cell in cells], "--n-steps", "200", "--seed", "0"] + common
    cell = cells[0]
    # This is the same collect_state_rollout producer used by
    # rerun_representation_grid.py, including the four sigreg checkpoints.
    return command + ["code.scripts.latent_rollout", "--model", job["model"],
                      "--ckpt", job["checkpoint"], "--env", cell["env"],
                      "--out", str(stage / cell["stage_json"]), "--n-steps", "200",
                      "--n-resets", "2", "--seed", "0", "--pad-obs-to", "128",
                      "--action-dim-eval", "56"] + common


def build_plan(args, dependencies, manifests):
    jobs = []
    results, archive = args.results, Path(load_json(args.training_manifest[0])["archive_root"])
    core = tuple(MODELS)
    require(len(core) == 13 and len(set(core)) == 13, "The canonical core model set is not the original 13 models")

    def add(group, family, split, model, envs):
        checkpoint = results / family
        if group != "scale":
            checkpoint /= split
        checkpoint = checkpoint / model / "seed_0/final.pt"
        dependency = dependencies.get(str(checkpoint))
        require(dependency is not None, f"Checkpoint has no supplied audit: {checkpoint}")
        job_id = f"{group}__{family}__{split}__{model}"
        if group in ("probes", "rollouts"):
            job_id += f"__{envs[0]}"
        cells = []
        for env in envs:
            if group == "probes":
                output = results / "g2_probe" / family / model / f"{env}.json"
                stage_json = "result.json"
                require(ENV_REGISTRY[env][2:] == (1, 25), f"Changed original probe window budget for {env}")
            elif group == "state":
                stats_root = "5m_stats_fair" if family == "5m_5mpar" else "5m_stats"
                stage_json = f"{split}/{model}/latent_stats_{env}.json"
                output = results / stats_root / stage_json
            elif group == "scale":
                stage_json = f"{split}/{model}/latent_stats_{env}.json"
                output = results / f"{family}_stats" / stage_json
            else:
                stage_json = "result.json"
                output = results / "latent_rollout" / split / model / f"{env}.json"
            original = archive / output.relative_to(results)
            require(output.resolve() == output, f"Output resolves outside its canonical location: {output}")
            cell = {"env": env, "output": str(output), "stage_json": stage_json,
                    "original_output": str(original), "original_output_present": original.is_file(), "copies": []}
            if group == "probes":
                cell["data_path"] = ENV_REGISTRY[env][1]
                cell["probe_dim"] = ENV_PROBE[env]["pos"][1] - ENV_PROBE[env]["pos"][0]
            if group == "rollouts":
                cell["npz"] = str(output.with_suffix(".npz"))
                if model in core:
                    cell["copies"] = [str(results / "g1" / family / model / f"{env}_{split}.json")]
            cells.append(cell)
        job = {"id": job_id, "group": group, "family": family, "split": split, "model": model,
               "checkpoint": str(checkpoint), "training_dependency": dependency["manifest"],
               "requires_retraining": dependency["affected"], "budget": budget_for(group), "cells": cells}
        if group in ("state", "scale"):
            job["producer_results"] = str(results if group == "scale" else results / family)
        job["log"] = str(args.run_dir / "logs" / f"{job_id}.log")
        job["stage"] = str(args.run_dir / "staging" / job_id)
        job["command"] = command_for(job, Path(job["stage"]), cells)
        jobs.append(job)

    if "probes" in args.groups:
        for model in core:
            for env in PROBE_ENVS:
                add("probes", core_family(model), "oodc_F1", model, [env])
    state_splits = sorted({Path(path).relative_to(results).parts[1]
                           for path in dependencies
                           if Path(path).relative_to(results).parts[0] in ("5m", "5m_5mpar")})
    if "state" in args.groups:
        require(len(state_splits) == 10, f"Training audit omits original state splits: {state_splits}")
        for split in state_splits:
            for model in core:
                add("state", core_family(model), split, model, STATS_ENVS)
    scale_models = {}
    if "scale" in args.groups:
        expected = set(core) - {"lif_transformer_baseline"}
        for family in SCALE_FAMILIES:
            audited = {Path(path).parent.parent.name for path, dependency in dependencies.items()
                       if dependency["affected"] and Path(path).relative_to(results).parts[0] == family}
            require(audited == expected and len(audited) == 12, f"Scale checkpoint inventory mismatch: {family}: {sorted(audited)}")
            scale_models[family] = sorted(audited)
            for model in core:
                if model in audited:
                    add("scale", family, family, model, STATS_ENVS)
    if "rollouts" in args.groups:
        for split in ("cross_benchmark_F1", "oodc_F2"):
            for model in (*core, *SIGREG_MODELS):
                family = "5m_sigreg_sweep" if model in SIGREG_MODELS else core_family(model)
                for env in ROLLOUT_ENVS:
                    add("rollouts", family, split, model, [env])

    cells = [cell for job in jobs for cell in job["cells"]]
    outputs = [path for cell in cells for path in (cell["output"], *cell["copies"], *([cell["npz"]] if "npz" in cell else []))]
    require(len(outputs) == len(set(outputs)), "The planned jobs overlap canonical outputs")
    audited_safe_outputs = [
        path for job in jobs if not job["requires_retraining"] for cell in job["cells"]
        for path in (cell["output"], *cell["copies"], *([cell["npz"]] if "npz" in cell else []))
    ]
    source_names = set(SOURCE_FILES)
    for job in jobs:
        dependency = dependencies[job["checkpoint"]]
        source_names.update(dependency.get("loader_source_sha256") or {})
        source_names.update(dependency.get("training_source_sha256") or {})
    return {"protocol_version": 2, "results_root": str(results), "groups": args.groups,
            "training_manifests": manifests, "state_splits": state_splits, "scale_models": scale_models,
            "totals": {"jobs": len(jobs), "cells": len(cells), "json": len(cells) + sum(len(c["copies"]) for c in cells),
                       "npz": sum("npz" in cell for cell in cells),
                       "cells_by_group": dict(Counter(job["group"] for job in jobs for _ in job["cells"]))},
            "physical_gpus": args.gpus, "workers_per_gpu": args.workers_per_gpu, "threads_per_worker": args.threads,
            "source_files": {name: {"sha256": fingerprint(ROOT / name), "signature": signature(ROOT / name)} for name in sorted(source_names)},
            "audited_safe_outputs": audited_safe_outputs, "jobs": jobs}


def checkpoint_provenance(job, dependency):
    import torch

    path = Path(job["checkpoint"])
    if not path.is_file():
        return None
    require(path.resolve() == path, f"Refusing a checkpoint symlink or noncanonical source: {path}")
    before = signature(path)
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    saved = checkpoint["args"]
    require(saved.get("seed") == 0 and saved.get("pad_obs_to") == 128 and saved.get("action_dim") == 56,
            f"Checkpoint is not the original seed0/pad128/action56 contract: {path}")
    if dependency["affected"]:
        audit = dependency["audit"]
        expected_protocol = dependency["required_data_protocol_version"]
        required_metadata = dependency["manifest"]["required_checkpoint_metadata"]
        require(required_metadata and isinstance(expected_protocol, str), f"Final audit lacks required checkpoint protocols: {path}")
        if (checkpoint.get("data_protocol_version") != expected_protocol
                or any(checkpoint.get(key) != value for key, value in required_metadata.items())):
            return None
        provenance = checkpoint["data_provenance"]
        require(provenance["loader_protocol"] == expected_protocol, f"Loader protocol mismatch: {path}")
        require(provenance["loader_source_sha256"] == dependency["loader_source_sha256"], f"Training source fingerprint mismatch: {path}")
        require(checkpoint["training_provenance"]["source_sha256"] == dependency["training_source_sha256"],
                f"Training/model source fingerprint mismatch: {path}")
        require(provenance["spec"]["sha256"] == audit["spec_sha256"], f"Training spec fingerprint mismatch: {path}")
        expected_args = audit["args"]
        require({k: v for k, v in saved.items() if k != "out"} == {k: v for k, v in expected_args.items() if k != "out"},
                f"Approved training arguments changed: {path}")
        require(checkpoint["step"] == audit["expected_step"], f"Approved legal-window training step budget changed: {path}")
        promotion = load_json(path.parent / "training_repair.json")
        require(promotion["status"] == "completed" and promotion["checkpoint"] == str(path)
                and promotion["step"] == checkpoint["step"] and promotion["data_provenance"] == provenance,
                f"Canonical checkpoint has no matching completed promotion: {path}")
    digest = fingerprint(path)
    require(signature(path) == before, f"Checkpoint changed while checking provenance: {path}")
    return {"checkpoint": str(path), "checkpoint_sha256": digest, "checkpoint_signature": before,
            "checkpoint_step": checkpoint["step"], "checkpoint_readout_mode": saved.get("readout_mode"),
            "training_data_protocol_version": checkpoint.get("data_protocol_version"),
            "training_protocol_version": checkpoint.get("training_protocol_version"),
            "training_audit": dependency["manifest"], "training_affected": dependency["affected"]}


def validate_payload(payload, job, cell):
    finite_tree(payload)
    require(payload.get("skipped") is False and not payload.get("error") and not payload.get("errors"), "Producer returned a skipped/error record")
    for key, expected in {"protocol_version": 2, "measurement_object": "forward.emb", "weights_loaded_strict": True,
                          "model": job["model"], "env": cell["env"], "split": job["split"]}.items():
        require(payload.get(key) == expected, f"Unexpected {key}: {payload.get(key)!r}; expected {expected!r}")
    require(payload.get("checkpoint", payload.get("ckpt")) == job["checkpoint"], "Producer used a different checkpoint")
    if job["group"] == "probes":
        for key, expected in {"representation": "readout", "probe_seed": 12345, "target_metric": "raw_unclipped_r2",
                              "metric": "r2", "binary": False, "probe_target": "position", "n_train": 4000,
                              "n_val": 1000, "probe_dim": cell["probe_dim"]}.items():
            require(payload.get(key) == expected, f"Probe contract mismatch: {key}")
        require(isinstance(payload.get("r2"), (int, float)), "Missing raw R2")
        require(len(payload["per_dim_r2"]) == cell["probe_dim"] and len(payload["near_const_dims"]) == cell["probe_dim"], "Incomplete per-dimension probe result")
        require(all(isinstance(value, (int, float)) for value in payload["per_dim_r2"]), "Missing per-dimension R2")
        require(math.isclose(payload["r2"], sum(payload["per_dim_r2"]) / cell["probe_dim"], rel_tol=1e-10, abs_tol=1e-12), "Mean R2 disagrees with raw per-dimension R2")
        return
    for key, expected in {"seed": 0, "n_steps": 200, "requested_segments": 2, "context_steps": 1,
                          "observation_embedding_object": "forward.emb_pre_cell"}.items():
        require(payload.get(key) == expected, f"Diagnostic budget/interface mismatch: {key}")
    require(isinstance(payload.get("n_episodes"), int) and 2 <= payload["n_episodes"] <= 198, "Invalid episode count")
    require(payload["n_resets"] == payload["n_episodes"] - 1, "Reset count disagrees with episode count")
    for name in ("readout", "observation_embedding"):
        metrics = payload["representations"][name]
        for key in ("divergence", "per_dim_std_max", "per_dim_std_min", "mean_norm_latent", "mean_d_obs", "mean_d_lat"):
            require(isinstance(metrics.get(key), (int, float)) and metrics[key] >= 0, f"Missing/invalid {name}.{key}")
        require(metrics["n_transitions"] == 200 - payload["n_episodes"], "Reset boundaries were included in diagnostic transitions")
        # The canonical protocol uses null for mathematically undefined
        # correlations, not NaN, an invented zero, or an omitted cell.
        require(metrics["event_rho"] is None or isinstance(metrics["event_rho"], (int, float)) and abs(metrics["event_rho"]) <= 1 + 1e-12,
                f"Invalid {name}.event_rho")
        response = metrics["responsiveness"]
        require((response is None and metrics["mean_d_obs"] == 0) or isinstance(response, (int, float)) and response >= 0,
                f"Invalid {name}.responsiveness")
    for key, value in payload["representations"]["readout"].items():
        require(payload.get(key) == value, f"Primary metric does not describe forward.emb: {key}")
    correlation = payload["corr_obs_rate"]
    require(correlation is None or isinstance(correlation, (int, float)) and abs(correlation) <= 1 + 1e-12,
            "Invalid observation/spike-rate correlation")


def validate_arrays(path, payload):
    import numpy as np

    with np.load(path, allow_pickle=False) as arrays:
        for name in ("obs_arr", "action_arr", "episode_id", "lat_arr", "embedding_arr"):
            require(name in arrays, f"Missing rollout array: {name}")
        for name in arrays.files:
            require(np.issubdtype(arrays[name].dtype, np.number) and np.isfinite(arrays[name]).all(), f"Nonfinite/nonnumeric rollout array: {name}")
        require(arrays["obs_arr"].shape == (200, 128) and arrays["action_arr"].shape == (200, 56), "Rollout padding/budget mismatch")
        for name in ("lat_arr", "embedding_arr"):
            require(arrays[name].ndim == 2 and arrays[name].shape[0] == 200 and arrays[name].shape[1] > 0, f"Invalid representation trajectory: {name}")
        episodes = arrays["episode_id"]
        require(episodes.shape == (200,) and np.issubdtype(episodes.dtype, np.integer), "Invalid episode trajectory")
        require(episodes[0] == 0 and np.all((np.diff(episodes) == 0) | (np.diff(episodes) == 1)), "Noncontiguous episode trajectory")
        require(int(episodes[-1]) + 1 == payload["n_episodes"], "JSON/NPZ episode mismatch")


def run_job(job, gpu, checkpoint, plan, args):
    record = {"id": job["id"], "gpu": gpu, "log": job["log"], "started_at": time.time(), "cells": len(job["cells"])}
    try:
        for name, source in plan["source_files"].items():
            require(fingerprint(ROOT / name) == source["sha256"], f"Evaluation source changed after planning: {name}")
        for manifest in plan["training_manifests"]:
            require(fingerprint(manifest["path"]) == manifest["sha256"], f"Training audit changed after planning: {manifest['path']}")
        require(signature(job["checkpoint"]) == checkpoint["checkpoint_signature"], "Checkpoint changed before evaluation")
        provenance = {**checkpoint, "evaluation_source_sha256": {name: source["sha256"] for name, source in plan["source_files"].items()},
                      "budget": job["budget"]}
        pending = []
        for cell in job["cells"]:
            output = Path(cell["output"])
            if output.exists():
                payload = load_json(output)
                validate_payload(payload, job, cell)
                recorded = payload.get("repair_provenance")
                require(recorded == provenance or recorded is None and not checkpoint["training_affected"],
                        f"Existing output lacks matching repair provenance: {output}")
                if "npz" in cell:
                    validate_arrays(cell["npz"], payload)
                    if "trajectory_sha256" in payload:
                        require(payload["trajectory_sha256"] == fingerprint(cell["npz"]), f"Paired trajectory changed: {cell['npz']}")
                    else:
                        require(not checkpoint["training_affected"], f"Missing paired trajectory provenance: {cell['npz']}")
                for copy in cell["copies"]:
                    if Path(copy).exists():
                        require(fingerprint(copy) == fingerprint(output), f"G1 copy differs: {copy}")
                    else:
                        publish(output, copy)
            else:
                require(not any(Path(path).exists() for path in (*cell["copies"], *([cell["npz"]] if "npz" in cell else []))),
                        f"Incomplete existing result bundle, archive it explicitly before rerunning: {output}")
                pending.append(cell)
        record["reused_cells"] = len(job["cells"]) - len(pending)
        log_path = Path(job["log"])
        with log_path.open("x") as log:
            if pending:
                stage = Path(job["stage"])
                stage.mkdir(parents=True, exist_ok=False)
                command = command_for(job, stage, pending)
                record["command"] = command
                child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu,
                                 MUJOCO_GL="egl", PYTHONPATH=str(ROOT), PYTHONHASHSEED="0",
                                 OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                                 OPENBLAS_NUM_THREADS=str(args.threads), NUMEXPR_NUM_THREADS=str(args.threads))
                log.write(json.dumps({"command": command, "gpu": gpu, "checkpoint": checkpoint}, allow_nan=False) + "\n")
                log.flush()
                process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returncode = process.wait(timeout=args.job_timeout)
                except BaseException:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
                require(returncode == 0, f"Producer exited {returncode}; see {log_path}")
                require(signature(job["checkpoint"]) == checkpoint["checkpoint_signature"], "Checkpoint changed during evaluation")
                # Validate the entire producer job before publishing any cell.
                for cell in pending:
                    path = stage / cell["stage_json"]
                    payload = load_json(path)
                    payload.setdefault("split", job["split"])
                    validate_payload(payload, job, cell)
                    if "npz" in cell:
                        validate_arrays(path.with_suffix(".npz"), payload)
                        payload["trajectory_sha256"] = fingerprint(path.with_suffix(".npz"))
                    payload["repair_provenance"] = provenance
                    write_json(path, payload)
                for cell in pending:
                    path = stage / cell["stage_json"]
                    if "npz" in cell:
                        publish(path.with_suffix(".npz"), cell["npz"])
                    publish(path, cell["output"])
                    for copy in cell["copies"]:
                        publish(path, copy)
            else:
                log.write("All canonical cells passed finite-result and checkpoint-provenance checks; no producer launched.\n")
        record.update(status="completed", produced_cells=len(pending))
    except Exception:
        record.update(status="failed", error=traceback.format_exc())
    record["finished_at"] = time.time()
    return record


def execute(plan, dependencies, args):
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "logs").mkdir()
    write_json(args.run_dir / "plan.json", plan)
    states = {job["id"]: {"id": job["id"], "status": "pending", "cells": len(job["cells"])} for job in plan["jobs"]}
    pending, active, checkpoints = list(plan["jobs"]), {}, {}
    unready = {}
    slots = args.gpus * args.workers_per_gpu
    deadline = time.monotonic() + args.checkpoint_timeout
    status_path = args.run_dir / "status.json"

    def save_status(final=False):
        completed = sum(item["cells"] for item in states.values() if item["status"] == "completed")
        status = "completed" if final and completed == plan["totals"]["cells"] else "failed" if final else "running"
        atomic_status(status_path, {"status": status, "updated_at": time.time(), "planned": plan["totals"],
                                   "completed_cells": completed, "jobs": list(states.values())})

    print(f"[auxiliary] plan={args.run_dir / 'plan.json'} totals={plan['totals']}", flush=True)
    save_status()
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        while pending or active:
            for job in pending[:]:
                if not slots:
                    break
                checkpoint_path = job["checkpoint"]
                try:
                    if checkpoint_path not in checkpoints:
                        current = signature(checkpoint_path) if Path(checkpoint_path).is_file() else None
                        if checkpoint_path in unready and unready[checkpoint_path] == current:
                            states[job["id"]]["status"] = "waiting_checkpoint"
                            continue
                        ready = checkpoint_provenance(job, dependencies[checkpoint_path])
                        if ready is None:
                            unready[checkpoint_path] = current
                            states[job["id"]]["status"] = "waiting_checkpoint"
                            continue
                        checkpoints[checkpoint_path] = ready
                    gpu = slots.pop(0)
                    states[job["id"]].update(status="running", gpu=gpu, log=job["log"])
                    active[pool.submit(run_job, job, gpu, checkpoints[checkpoint_path], plan, args)] = (job, gpu)
                    pending.remove(job)
                except Exception:
                    states[job["id"]].update(status="failed", error=traceback.format_exc())
                    pending.remove(job)
            save_status()
            if active:
                finished, _ = wait(active, timeout=min(args.poll_seconds, 5), return_when=FIRST_COMPLETED)
                for future in finished:
                    job, gpu = active.pop(future)
                    states[job["id"]] = future.result()
                    slots.append(gpu)
                    print(f"[auxiliary] {states[job['id']]['status']} {job['id']}", flush=True)
            elif pending:
                if not args.wait_checkpoints or time.monotonic() >= deadline:
                    for job in pending:
                        states[job["id"]].update(status="failed", error=f"Canonical checkpoint was not promoted before the dependency deadline: {job['checkpoint']}")
                    pending.clear()
                else:
                    time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    save_status(final=True)
    failures = [item for item in states.values() if item["status"] != "completed"]
    print(f"[auxiliary] completed_jobs={len(states) - len(failures)} failed_jobs={len(failures)} status={status_path}", flush=True)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--training-manifest", type=Path, action="append",
                        help="Repeat in training-repair order; affected entries take precedence over safe entries")
    parser.add_argument("--groups", choices=GROUPS, nargs="+", default=list(GROUPS))
    parser.add_argument("--run-dir", type=Path, required=True, help="New log/status/staging directory; canonical result paths do not change")
    parser.add_argument("--gpus", default="0,1,3", help="Comma-separated physical GPU indices, excluding reserved GPUs")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--job-timeout", type=float, default=3600)
    parser.add_argument("--wait-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-timeout", type=float, default=172800)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.gpus = args.gpus.split(",")
    if any(not gpu.isascii() or not gpu.isdigit() for gpu in args.gpus):
        parser.error("--gpus must contain physical GPU indices")
    args.gpus = [str(int(gpu)) for gpu in args.gpus]
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain unique physical GPU indices")
    if args.workers_per_gpu < 1 or args.threads < 1 or len(set(args.groups)) != len(args.groups):
        parser.error("Worker/thread counts must be positive and groups must not repeat")
    if any(not math.isfinite(value) or value <= 0 for value in (args.job_timeout, args.checkpoint_timeout, args.poll_seconds)):
        parser.error("All timeouts and polling intervals must be positive and finite")
    args.results, args.run_dir = args.results.resolve(), args.run_dir.resolve()
    if not args.training_manifest:
        parser.error("--training-manifest is required; obsolete phase-specific audits are never selected implicitly")
    dependencies, manifests = read_audits(args.training_manifest, args.results)
    plan = build_plan(args, dependencies, manifests)
    if not args.execute:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    require(all(manifest["execution_ready"] for manifest in manifests), "Training audit is provisional; execution_ready must be true")
    for job in plan["jobs"]:
        dependency = dependencies[job["checkpoint"]]
        if dependency["affected"]:
            require(dependency["manifest"]["required_checkpoint_metadata"]
                    and dependency["loader_source_sha256"] and dependency["training_source_sha256"],
                    f"Final audit is missing frozen training/data metadata: {job['checkpoint']}")
            for field in ("loader_source_sha256", "training_source_sha256"):
                for name, expected_hash in dependency[field].items():
                    require(plan["source_files"][name]["sha256"] == expected_hash,
                            f"Current source differs from the final frozen training audit: {name}")
    return execute(plan, dependencies, args)


if __name__ == "__main__":
    raise SystemExit(main())

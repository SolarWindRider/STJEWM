#!/usr/bin/env python3
"""Final consolidated repair: retrain all 371 affected checkpoints in one generation.

The manifest lists the 371 audited checkpoints (144 state-data, 39 maze-data,
130 pixel-data) with their original arguments, approved argument deltas and the
required data protocol per entry. Without --execute, validate the whole plan
against the live repository and archives and print the finite training command
per checkpoint. With --execute, archive whole original seed directories that
still exist, train fresh staged runs, verify protocol provenance, expected
steps, tensor shapes and loss logs, then atomically promote each success.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# Import the repository package before torch can cache the stdlib code module.
import code as _code_pkg  # noqa: F401

import torch

TRAINING_PROTOCOL_VERSION = "causal_grad_sigreg_B_20260916"
GENERIC_DATA_PROTOCOL = "episode_safe_20260916"
MAZE_DATA_PROTOCOL = "episode_safe_maze_20260916"
PIXEL_DATA_PROTOCOL = "pixel_domains_episode_safe_20260916"
KNOWN_DATA_PROTOCOLS = {
    GENERIC_DATA_PROTOCOL, MAZE_DATA_PROTOCOL, PIXEL_DATA_PROTOCOL,
}


def training_command(approved_args: dict, staging_dir: Path) -> list[str]:
    command = [sys.executable, "-u", "-m", "code.train.train"]
    for key, value in approved_args.items():
        if key == "out":
            value = str(staging_dir)
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif isinstance(value, (str, int, float)):
            command.extend((flag, str(value)))
        else:
            raise ValueError(f"Unsupported approved argument {key}={value!r}")
    return command


def expected_tensor_reference(job: dict) -> dict:
    """Return the state dict the new checkpoint's tensors must match.

    Normal jobs replay the original architecture byte-for-byte. Jobs with an
    approved argument delta (the LeWM width-288 restoration) intentionally
    change tensor shapes, so the reference is a freshly built model from the
    approved arguments rather than the archived original.
    """
    from code.train.train import build_model

    if not job.get("approved_arg_deltas"):
        original = torch.load(Path(job["original_checkpoint"]), map_location="cpu",
                              mmap=True, weights_only=False)
        try:
            return original["model"]
        finally:
            del original
    args = job["args"]
    model = build_model(
        args["model"], args["pad_obs_to"], args["action_dim"], args["n_layers"],
        args.get("readout_mode", "hidden_leak"),
        embed_dim=args.get("embed_dim"),
        hidden_dim=args.get("hidden_dim"),
        mlp_hidden=args.get("mlp_hidden"), mlp_layers=args.get("mlp_layers"),
        stacked_lif_layers=args.get("stacked_lif_layers"),
        stacked_lif_din=args.get("stacked_lif_din"),
        image_size=args.get("image_size", 0),
    )
    reference = model.state_dict()
    del model
    return reference


def validate_completion(job: dict, staging_dir: Path, manifest: dict) -> dict:
    checkpoint = torch.load(staging_dir / "final.pt", map_location="cpu", mmap=True,
                            weights_only=False)
    expected_args = dict(job["args"], out=str(staging_dir))
    if checkpoint["args"] != expected_args:
        raise ValueError("Completed training arguments differ from the approved arguments")
    if checkpoint.get("training_protocol_version") != TRAINING_PROTOCOL_VERSION:
        raise ValueError("Completed checkpoint lacks the final training protocol version")
    provenance = checkpoint.get("training_provenance", {})
    if provenance.get("protocol_version") != TRAINING_PROTOCOL_VERSION:
        raise ValueError("Completed checkpoint training provenance has the wrong protocol")
    if provenance.get("source_sha256") != manifest["training_source_sha256"]:
        raise ValueError("Completed checkpoint training source registry differs from the manifest")
    data_provenance = checkpoint["data_provenance"]
    required = job["required_data_protocol_version"]
    if checkpoint.get("data_protocol_version") != required:
        raise ValueError(f"Completed checkpoint data protocol is not {required}")
    if data_provenance["loader_protocol"] != required:
        raise ValueError(f"Completed checkpoint loader protocol is not {required}")
    if data_provenance["loader_source_sha256"] != manifest["loader_source_sha256"]:
        raise ValueError("Completed checkpoint data source registry differs from the manifest")
    if data_provenance["spec"]["sha256"] != job["spec_sha256"]:
        raise ValueError("Training specification changed after the impact audit")
    if checkpoint["step"] != job["expected_step"]:
        raise ValueError(
            f"Completed step {checkpoint['step']} differs from the audited expectation "
            f"{job['expected_step']}"
        )
    new_state = checkpoint["model"]
    reference_state = expected_tensor_reference(job)
    try:
        if new_state.keys() != reference_state.keys():
            raise ValueError("Completed checkpoint tensor keys differ from the expected architecture")
        for key, value in new_state.items():
            previous = reference_state[key]
            if value.shape != previous.shape or value.dtype != previous.dtype:
                raise ValueError(f"Completed checkpoint shape/dtype mismatch: {key}")
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                raise ValueError(f"Completed checkpoint contains nonfinite weights: {key}")
    finally:
        del new_state, reference_state
        checkpoint.pop("model", None)
    losses = json.loads((staging_dir / "loss_log.json").read_text())
    if checkpoint["step"] <= 0 or losses["step"] != checkpoint["step"]:
        raise ValueError("Checkpoint and loss log do not describe completed training")
    for entry in losses["losses"]:
        if any(not math.isfinite(value) for value in entry.values() if isinstance(value, (int, float))):
            raise ValueError("Training loss log contains nonfinite values")
    return {"step": checkpoint["step"], "data_provenance": data_provenance}


def validate_manifest_sources(manifest: dict) -> None:
    if manifest.get("execution_ready") is not True:
        raise ValueError("Refusing to run a manifest that is not marked execution_ready")
    if manifest.get("training_protocol_version") != TRAINING_PROTOCOL_VERSION:
        raise ValueError("Unexpected training protocol version")
    registries = {
        "training_source_sha256": manifest.get("training_source_sha256"),
        "loader_source_sha256": manifest.get("loader_source_sha256"),
    }
    for registry_name, registry in registries.items():
        if not isinstance(registry, dict) or not registry:
            raise ValueError(f"Manifest registry {registry_name} is missing")
        for relative, expected_hash in registry.items():
            path = ROOT / relative
            if not path.is_file():
                raise ValueError(f"Registered source is absent: {relative}")
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Source fingerprint mismatch in {registry_name}: {relative}")


def plan_job(job: dict, manifest: dict, results_root: Path, archive_root: Path,
             staging_root: Path):
    source_dir = Path(job["checkpoint"]).parent.resolve()
    relative = source_dir.relative_to(results_root)
    quarantine_dir = archive_root / "checkpoints_final" / relative
    if Path(job["archived_checkpoint"]) != quarantine_dir / "final.pt":
        raise ValueError("Quarantine destination differs from the approved repair manifest")
    staging_dir = staging_root / relative
    if staging_dir.exists():
        raise FileExistsError(f"Refusing to replace a staged run: {staging_dir}")
    if quarantine_dir.exists():
        raise FileExistsError(f"Quarantine copy already exists: {relative}")
    original_path = Path(job["original_checkpoint"])
    canonical_path = source_dir / "final.pt"
    if original_path.exists():
        reference_path = original_path
    elif canonical_path.exists() and job["original_checkpoint"] == job["archived_checkpoint"]:
        # Pre-quarantine state for lineages whose canonical copy is the original.
        reference_path = canonical_path
    else:
        raise FileNotFoundError(
            f"Neither the original reference nor the canonical checkpoint exists: {relative}")
    reference = torch.load(reference_path, map_location="cpu", mmap=True, weights_only=False)
    try:
        if reference["args"] != job["original_args"]:
            raise ValueError(f"Original checkpoint changed after impact audit: {reference_path}")
        if reference["step"] != job["original_step"]:
            raise ValueError(f"Original checkpoint step changed after impact audit: {reference_path}")
    finally:
        del reference
    spec_path = (ROOT / job["args"]["multi_env_spec"]).resolve()
    if hashlib.sha256(spec_path.read_bytes()).hexdigest() != job["spec_sha256"]:
        raise ValueError(f"Training spec changed after impact audit: {spec_path}")
    if job["required_data_protocol_version"] not in KNOWN_DATA_PROTOCOLS:
        raise ValueError(f"Unknown required data protocol: {job['required_data_protocol_version']}")
    deltas = job.get("approved_arg_deltas") or {}
    for key, delta in deltas.items():
        if job["args"].get(key) != delta["approved"] or job["original_args"].get(key) != delta["original"]:
            raise ValueError(f"Approved argument delta mismatch for {key}: {reference_path}")
    command = training_command(job["args"], staging_dir)
    return job, source_dir, quarantine_dir, staging_dir, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,3", help="Physical GPUs available for training")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Bounded concurrent training processes per GPU (default: 1)")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if not gpus or len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        parser.error("--gpus must be a comma-separated list of unique physical GPU indices")
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be positive")
    worker_gpus = gpus * args.workers_per_gpu
    manifest = json.loads(args.manifest.read_text())
    validate_manifest_sources(manifest)
    results_root = Path(manifest["results_root"]).resolve()
    archive_root = args.archive_root.resolve()
    staging_root = args.staging_root.resolve()
    jobs = []
    for job in manifest["affected_checkpoints"]:
        planned = plan_job(job, manifest, results_root, archive_root, staging_root)
        jobs.append(planned)
        if not args.execute:
            print(json.dumps({"checkpoint": str(planned[1] / "final.pt"), "command": planned[4]}))
    if len(jobs) != manifest["affected_checkpoint_count"]:
        raise ValueError("Manifest job count differs from its declared affected_checkpoint_count")
    if manifest["unaffected_checkpoint_count"] != len(manifest["unaffected_checkpoints"]):
        raise ValueError("Manifest unaffected inventory count mismatch")
    unaffected_paths = set(manifest["unaffected_checkpoints"])
    affected_paths = {job[0]["checkpoint"] for job in jobs}
    if unaffected_paths & affected_paths:
        raise ValueError("Checkpoints listed as both affected and unaffected")
    audited_paths = set(manifest.get("audited_checkpoint_paths", []))
    if audited_paths != affected_paths | unaffected_paths:
        raise ValueError("Audited inventory does not cover exactly the affected and unaffected sets")
    print(f"[repair_training] {len(jobs)} checkpoints; GPUs {', '.join(gpus)}; "
          f"workers_per_gpu={args.workers_per_gpu}; execute={args.execute}", flush=True)
    if not args.execute:
        return

    # Quarantine whatever currently sits at each canonical seed directory
    # (originals and any partial first-wave promotions) before launching any
    # optimization. Whole-directory moves also remove stale eval artifacts
    # from the canonical namespace. A checksum record proves what was replaced.
    quarantine_records = {}
    for _, source_dir, quarantine_dir, _, _ in jobs:
        if source_dir.exists():
            quarantine_dir.parent.mkdir(parents=True, exist_ok=True)
            source_dir.rename(quarantine_dir)
            replaced = quarantine_dir / "final.pt"
            if replaced.exists():
                quarantine_records[str(replaced)] = hashlib.sha256(
                    replaced.read_bytes()).hexdigest()
    if quarantine_records:
        with (archive_root / "checkpoints_final_quarantine_manifest.json").open("x") as handle:
            json.dump(quarantine_records, handle, indent=2, allow_nan=False)

    def worker(gpu: str, assigned_jobs: list) -> list[dict]:
        outcomes = []
        child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu,
                         MUJOCO_GL="egl", PYTHONPATH=str(ROOT))
        for job, source_dir, quarantine_dir, staging_dir, command in assigned_jobs:
            record = {"checkpoint": str(source_dir / "final.pt"),
                      "original_checkpoint": job["original_checkpoint"],
                      "quarantined_checkpoint": str(quarantine_dir / "final.pt"),
                      "staging_dir": str(staging_dir), "gpu": gpu, "command": command}
            try:
                staging_dir.mkdir(parents=True, exist_ok=False)
                print(f"[repair_training] GPU{gpu} START {source_dir}", flush=True)
                with (staging_dir / "train.log").open("x") as log:
                    result = subprocess.run(command, cwd=ROOT, env=child_env,
                                            stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Training exited {result.returncode}; see {staging_dir / 'train.log'}")
                record.update(validate_completion(job, staging_dir, manifest))
                record["status"] = "completed"
                with (staging_dir / "training_repair.json").open("x") as handle:
                    json.dump(record, handle, indent=2, allow_nan=False)
                if source_dir.exists():
                    raise FileExistsError(f"Canonical path reappeared during training: {source_dir}")
                staging_dir.rename(source_dir)
                print(f"[repair_training] GPU{gpu} PROMOTED {source_dir}", flush=True)
            except Exception:
                record["status"] = "failed"
                record["error"] = traceback.format_exc()
                print(record["error"], file=sys.stderr, flush=True)
            outcomes.append(record)
        return outcomes

    outcomes = []
    with ThreadPoolExecutor(max_workers=len(worker_gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs[index::len(worker_gpus)])
                   for index, gpu in enumerate(worker_gpus)]
        for future in as_completed(futures):
            outcomes.extend(future.result())
    status_path = Path(manifest["status_path"])
    with status_path.open("x") as handle:
        json.dump({"training_protocol_version": TRAINING_PROTOCOL_VERSION,
                   "outcomes": outcomes}, handle, indent=2, allow_nan=False)
    failures = [item for item in outcomes if item["status"] != "completed"]
    print(f"[repair_training] completed={len(outcomes) - len(failures)} failed={len(failures)} "
          f"status={status_path}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

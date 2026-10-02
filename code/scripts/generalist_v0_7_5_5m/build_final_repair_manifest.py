#!/usr/bin/env python3
"""Assemble the final consolidated 371-checkpoint repair manifest.

Merges the three audited lineages (first-wave state data repair, maze repair,
pixel collection repair) into one generation under the final training protocol.
Every LeWM checkpoint receives the explicitly approved embed_dim 288 delta that
restores the intended 5M width; every entry carries the data protocol its
datasets actually require and the audited expected step count. The 11 truly
unaffected checkpoints are inventoried for StateGrid gating without retraining.

The driver (repair_training.py) performs the authoritative per-checkpoint
verification against the live files at plan time; this generator only merges
audited facts and refuses ambiguous inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from code.train.train import (  # noqa: E402
    DATA_SOURCE_FILES,
    TRAINING_PROTOCOL_VERSION,
    TRAINING_SOURCE_FILES,
    source_sha256,
)

RESULTS_ROOT = Path("/data/lx/tmp/results")
ARCHIVE_ROOT = RESULTS_ROOT / "_repair_archive/20260916T102154Z"
STAGING_ROOT = RESULTS_ROOT / "_repair_staging_final"
STATE_MANIFEST = ARCHIVE_ROOT / "training_repair_manifest.json"
MAZE_MANIFEST = ARCHIVE_ROOT / "training_repair_maze_manifest.json"
PIXEL_AUDIT = ARCHIVE_ROOT / "pixel_training_data_audit.json"
OUTPUT_PATH = ARCHIVE_ROOT / "training_final_repair_manifest.json"
STATUS_PATH = ARCHIVE_ROOT / "training_final_repair_status.json"
GENERIC_DATA_PROTOCOL = "episode_safe_20260916"
MAZE_DATA_PROTOCOL = "episode_safe_maze_20260916"
PIXEL_DATA_PROTOCOL = "pixel_domains_episode_safe_20260916"
LEWM_WIDTH_DELTA = {"embed_dim": {"original": 192, "approved": 288}}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


QUARANTINE_DIR = "checkpoints_final"


def quarantine_checkpoint_path(checkpoint: str) -> str:
    relative = Path(checkpoint).parent.relative_to(RESULTS_ROOT)
    return str(ARCHIVE_ROOT / QUARANTINE_DIR / relative / "final.pt")


def archive_checkpoint_path(checkpoint: str) -> str:
    relative = Path(checkpoint).parent.relative_to(RESULTS_ROOT)
    return str(ARCHIVE_ROOT / "checkpoints" / relative / "final.pt")


def apply_approved_delta(entry: dict) -> dict:
    args = entry["original_args"]
    if args["model"] == "lewm_baseline":
        if args["embed_dim"] != 192:
            raise ValueError(f"Unexpected original LeWM width for {entry['checkpoint']}")
        entry["args"] = dict(args, embed_dim=288)
        entry["approved_arg_deltas"] = dict(LEWM_WIDTH_DELTA)
        entry["expected_tensor_rule"] = "rebuild_from_approved_args"
    return entry


def state_entry(entry: dict) -> dict:
    checkpoint = entry["checkpoint"]
    if entry["archived_checkpoint"] != archive_checkpoint_path(checkpoint):
        raise ValueError(f"First-wave archive destination drifted for {checkpoint}")
    return apply_approved_delta({
        "checkpoint": checkpoint,
        "original_checkpoint": entry["archived_checkpoint"],
        "archived_checkpoint": quarantine_checkpoint_path(checkpoint),
        "family": entry["family"],
        "original_args": entry["args"],
        "original_step": entry["step"],
        "args": entry["args"],
        "expected_step": entry["step"],
        "spec_sha256": entry["spec_sha256"],
        "required_data_protocol_version": GENERIC_DATA_PROTOCOL,
        "approved_arg_deltas": {},
        "expected_tensor_rule": "match_archived_original",
        "lineage": "state_data_repair_144",
    })


def maze_entry(entry: dict) -> dict:
    checkpoint = entry["checkpoint"]
    if entry["archived_checkpoint"] != archive_checkpoint_path(checkpoint):
        raise ValueError(f"Maze archive destination drifted for {checkpoint}")
    if entry["expected_step"] != entry["step"]:
        raise ValueError(f"Maze expected step drifted for {checkpoint}")
    return apply_approved_delta({
        "checkpoint": checkpoint,
        "original_checkpoint": quarantine_checkpoint_path(checkpoint),
        "archived_checkpoint": quarantine_checkpoint_path(checkpoint),
        "family": entry["family"],
        "original_args": entry["args"],
        "original_step": entry["step"],
        "args": entry["args"],
        "expected_step": entry["expected_step"],
        "spec_sha256": entry["spec_sha256"],
        "required_data_protocol_version": MAZE_DATA_PROTOCOL,
        "approved_arg_deltas": {},
        "expected_tensor_rule": "match_archived_original",
        "lineage": "maze_episode_repair_39",
    })


def pixel_entry(model: dict, split: dict) -> dict:
    checkpoint = model["checkpoint_path"]
    if model["original_saved_step"] != model["legacy_reconstructed_steps"]:
        raise ValueError(f"Pixel legacy step reconstruction drifted for {checkpoint}")
    return apply_approved_delta({
        "checkpoint": checkpoint,
        "original_checkpoint": quarantine_checkpoint_path(checkpoint),
        "archived_checkpoint": quarantine_checkpoint_path(checkpoint),
        "family": "5m_pixel",
        "original_args": model["original_args"],
        "original_step": model["original_saved_step"],
        "args": model["original_args"],
        "expected_step": model["expected_corrected_steps"],
        "spec_sha256": split["spec_sha256"],
        "required_data_protocol_version": PIXEL_DATA_PROTOCOL,
        "approved_arg_deltas": {},
        "expected_tensor_rule": "match_archived_original",
        "lineage": "pixel_collection_repair_130",
        "collection_budget": {
            "legacy_dataset_windows_per_epoch": split["legacy_dataset_windows"],
            "expected_dataset_windows_per_epoch": split["expected_corrected_dataset_windows"],
        },
    })


def model_repair_entry(checkpoint: str, payload_args: dict, step: int) -> dict:
    spec_relative = payload_args["multi_env_spec"]
    spec_path = ROOT / spec_relative
    return apply_approved_delta({
        "checkpoint": checkpoint,
        "original_checkpoint": quarantine_checkpoint_path(checkpoint),
        "archived_checkpoint": quarantine_checkpoint_path(checkpoint),
        "family": str(Path(checkpoint).parent.relative_to(RESULTS_ROOT).parts[0]),
        "original_args": payload_args,
        "original_step": step,
        "args": payload_args,
        "expected_step": step,
        "spec_sha256": sha256_path(spec_path),
        "required_data_protocol_version": GENERIC_DATA_PROTOCOL,
        "approved_arg_deltas": {},
        "expected_tensor_rule": "match_archived_original",
        "lineage": "model_repair_58",
    })


def is_model_affected(payload_args: dict) -> bool:
    """ModelAudit rule: the 3-term JEPA loss with lambda_sigreg > 0 (STJEWM,
    LeWM, and the GRU/MLP baselines that dispatch to the same loss), ALIF,
    or Stacked-LIF trace."""
    model = payload_args["model"]
    if model in ("stjewm", "lewm_baseline", "gru_baseline", "mlp_baseline"):
        return float(payload_args["lambda_sigreg"]) > 0.0
    return model in ("alif_timecell_baseline", "stacked_lif_trace")


def unaffected_entry(checkpoint: str, family: str) -> dict:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Unaffected checkpoint missing: {checkpoint}")
    import torch

    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    entry = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_path(path),
        "family": family,
        "args": payload["args"],
        "step": payload["step"],
        "metadata_keys": sorted(payload.keys()),
        "retrain_required": False,
        "required_data_protocol_version": GENERIC_DATA_PROTOCOL,
        "required_training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "lineage": "modelaudit_truly_unaffected_11",
    }
    del payload
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Write the manifest (refuses to overwrite an existing file)")
    options = parser.parse_args()

    state_manifest = json.loads(STATE_MANIFEST.read_text())
    maze_manifest = json.loads(MAZE_MANIFEST.read_text())
    pixel_audit = json.loads(PIXEL_AUDIT.read_text())

    entries = []
    for entry in state_manifest["affected_checkpoints"]:
        entries.append(state_entry(entry))
    for entry in maze_manifest["affected_checkpoints"]:
        entries.append(maze_entry(entry))
    for split in pixel_audit["splits"]:
        for model in split["models"]:
            entries.append(pixel_entry(model, split))

    checkpoints = [entry["checkpoint"] for entry in entries]
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError("Duplicate checkpoint across repair lineages")
    affected_paths = set(checkpoints)

    # The remaining first-wave-unaffected state checkpoints split by the
    # ModelAudit model rule: 58 model-only repairs plus the 11 truly safe.
    import torch

    maze_paths = {e["checkpoint"] for e in maze_manifest["affected_checkpoints"]}
    residual = [path for path in state_manifest["unaffected_checkpoints"]
                if not path.startswith(f"{RESULTS_ROOT}/5m_pixel/")
                and path not in maze_paths]
    model_affected = []
    model_safe = []
    for checkpoint in residual:
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        try:
            args = payload["args"]
            step = payload["step"]
        finally:
            del payload
        if is_model_affected(args):
            model_affected.append((checkpoint, args, step))
        else:
            model_safe.append(checkpoint)
    if len(model_affected) != 58 or len(model_safe) != 11:
        raise ValueError(f"ModelAudit classification drifted: {len(model_affected)} "
                         f"model-only affected, {len(model_safe)} safe")
    for checkpoint, args, step in model_affected:
        entries.append(model_repair_entry(checkpoint, args, step))
        affected_paths.add(checkpoint)

    unaffected_paths = [
        f"{RESULTS_ROOT}/5m/{split}/{model}/seed_0/final.pt"
        for split in ("oodc_F1", "oodc_F2", "oodc_F1F2")
        for model in ("stacked_lif_free", "lif_transformer_baseline")
    ] + [
        f"{RESULTS_ROOT}/5m_seed{seed}/oodc_F2/{model}/seed_0/final.pt"
        for seed in (1, 2)
        for model in ("stacked_lif_free", "lif_transformer_baseline")
    ] + [
        f"{RESULTS_ROOT}/5m_sigreg_sweep/oodc_F2/stjewm_trace_only_sig0.0/seed_0/final.pt",
    ]
    if affected_paths & set(unaffected_paths):
        raise ValueError("Unaffected checkpoint also listed as affected")
    if len(unaffected_paths) != 11:
        raise ValueError("Unexpected unaffected inventory size")
    if set(unaffected_paths) != set(model_safe):
        raise ValueError("Derived safe set differs from the ModelAudit inventory:\n"
                         f"derived-only: {sorted(set(model_safe) - set(unaffected_paths))}\n"
                         f"audit-only: {sorted(set(unaffected_paths) - set(model_safe))}")

    audited_paths = sorted(affected_paths | set(unaffected_paths))
    first_wave_total = (state_manifest["audit_checkpoint_count"]
                        if "audit_checkpoint_count" in state_manifest else 382)
    if len(audited_paths) != first_wave_total:
        raise ValueError(
            f"Final coverage {len(audited_paths)} differs from the audited {first_wave_total}")

    lewm = [entry for entry in entries if entry["args"]["model"] == "lewm_baseline"]
    if len(lewm) != 29:
        raise ValueError(f"Expected 29 LeWM checkpoints, assembled {len(lewm)}")
    for entry in lewm:
        if entry["approved_arg_deltas"] != LEWM_WIDTH_DELTA:
            raise ValueError(f"LeWM checkpoint lacks the width delta: {entry['checkpoint']}")

    unaffected = [unaffected_entry(path, str(Path(path).relative_to(RESULTS_ROOT).parts[0]))
                  for path in unaffected_paths]

    family_counts = {
        family: {
            "total": sum(Path(path).relative_to(RESULTS_ROOT).parts[0] == family
                         for path in audited_paths),
            "affected": sum(entry["family"] == family for entry in entries),
        }
        for family in sorted(state_manifest["family_counts"])
    }

    manifest = {
        "phase": "final_consolidated_repair_20260916",
        "execution_ready": None,  # set to true only after the source freeze announcement
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "required_checkpoint_metadata": {
            "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        },
        "evaluation_data_protocol_version": GENERIC_DATA_PROTOCOL,
        "results_root": str(RESULTS_ROOT),
        "archive_root": str(ARCHIVE_ROOT),
        "staging_root": str(STAGING_ROOT),
        "status_path": str(STATUS_PATH),
        "training_source_sha256": source_sha256(TRAINING_SOURCE_FILES),
        "loader_source_sha256": source_sha256(DATA_SOURCE_FILES),
        "audit_checkpoint_count": len(audited_paths),
        "affected_checkpoint_count": len(entries),
        "unaffected_checkpoint_count": len(unaffected),
        "family_counts": family_counts,
        "lineage_counts": {
            "state_data_repair_144": sum(
                e["lineage"] == "state_data_repair_144" for e in entries),
            "maze_episode_repair_39": sum(
                e["lineage"] == "maze_episode_repair_39" for e in entries),
            "pixel_collection_repair_130": sum(
                e["lineage"] == "pixel_collection_repair_130" for e in entries),
            "model_repair_58": sum(e["lineage"] == "model_repair_58" for e in entries),
        },
        "data_protocol_counts": {
            protocol: sum(e["required_data_protocol_version"] == protocol for e in entries)
            for protocol in (GENERIC_DATA_PROTOCOL, MAZE_DATA_PROTOCOL, PIXEL_DATA_PROTOCOL)
        },
        "approved_arg_delta_policy": {
            "lewm_baseline": {
                "field": "embed_dim",
                "original": 192,
                "approved": 288,
                "checkpoint_count": len(lewm),
                "reason": "Restore the intended 5M width-288 LeWM; explicit per-entry "
                          "exception to original-argument replay, verified by tensor-shape "
                          "rebuild at completion.",
            },
        },
        "expected_step_policy": {
            "state_data_repair_144": "original saved steps (episode-safe loaders preserve "
                                     "the audited window budgets for these specs)",
            "maze_episode_repair_39": "maze impact audit expected_step (capped 2000-window "
                                      "budgets preserved by the episode-safe parser)",
            "pixel_collection_repair_130": "pixel budget audit expected_corrected_steps "
                                           "(legal in-episode windows with honored caps)",
            "model_repair_58": "original saved steps (data budgets were already correct; "
                               "only model/loss sources changed)",
        },
        "unaffected_policy": "ModelAudit classification; StateGrid gates membership by "
                             "canonical path. These checkpoints predate protocol metadata "
                             "and are exempt from retraining.",
        "lineage_sources": {
            "state": str(STATE_MANIFEST),
            "maze": str(MAZE_MANIFEST),
            "pixel": str(PIXEL_AUDIT),
        },
        "affected_checkpoints": sorted(entries, key=lambda e: e["checkpoint"]),
        "unaffected_checkpoints": sorted(unaffected_paths),
        "unaffected_checkpoint_metadata": sorted(unaffected, key=lambda e: e["checkpoint"]),
        "audited_checkpoint_paths": audited_paths,
    }

    print(f"[build_final_repair_manifest] affected={len(entries)} "
          f"unaffected={len(unaffected)} lewm288={len(lewm)} "
          f"protocols={manifest['data_protocol_counts']}")
    if not options.write:
        return
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to replace {OUTPUT_PATH}")
    manifest["execution_ready"] = True
    with OUTPUT_PATH.open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    print(f"[build_final_repair_manifest] wrote {OUTPUT_PATH} "
          f"(sha256 {sha256_path(OUTPUT_PATH)})")


if __name__ == "__main__":
    main()

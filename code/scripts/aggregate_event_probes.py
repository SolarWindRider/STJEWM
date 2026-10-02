"""Inspect an explicit environment/model/event-target lattice.

Unsupported environment/target combinations are not required cells. Missing,
undefined, skipped, and obsolete supported cells prevent a complete summary.
This historical raw-output inspector does not authorize publication.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from code.scripts.audited_results import load_json, metric, require, summarize, write_new_json
from code.scripts.probe import ENV_REGISTRY, EVENT_BINARY_TARGETS, EVENT_TARGET_PROTOCOL, event_target_supported


def aggregate(probe_dir: Path, envs: list[str], models: list[str], targets: list[str]) -> dict:
    requested = {(env, model, target) for env in envs for model in models for target in targets}
    expected = {cell for cell in requested if event_target_supported(cell[0], cell[2])}
    require(expected, "The requested lattice has no supported event targets")
    records = {}
    for path in sorted(probe_dir.glob("*.json")):
        payload = load_json(path)
        cell = (payload.get("env"), payload.get("model"), payload.get("probe_target"))
        if cell not in expected:
            continue
        require(cell not in records, f"Duplicate event-probe cell: {cell}")
        records[cell] = (path, payload)

    cells, missing, unavailable = [], [], []
    for env, model, target in sorted(expected):
        identity = {"env": env, "model": model, "probe_target": target}
        record = records.get((env, model, target))
        if record is None:
            missing.append(identity)
            continue
        path, payload = record
        try:
            require(payload.get("skipped") is False and payload.get("status") == "complete",
                    payload.get("reason") or "Probe did not complete")
            require(not payload.get("error"), "Probe reported an error")
            require(payload.get("protocol_version") == 2 and
                    payload.get("event_target_protocol") == EVENT_TARGET_PROTOCOL,
                    "Obsolete event labels: rerun with the supported-target protocol")
            require(payload.get("weights_loaded_strict") is True, "Checkpoint weights were not loaded strictly")
            require(payload.get("binary") is True and payload.get("metric") == "auroc", "Expected binary AUROC")
            require(payload.get("representation") in ("readout", "observation_embedding"), "Unknown representation")
            require(payload.get("measurement_object") == (
                "forward.emb" if payload["representation"] == "readout" else "forward.emb_pre_cell"
            ), "Mismatched representation interface")
            require(payload.get("n_train", 0) > 0 and payload.get("n_val", 0) > 0, "Empty train/validation set")
            require(0 < metric(payload, "base_rate") < 1, "AUROC undefined for a single observed class")
            auroc, auprc = metric(payload, "r2"), metric(payload, "auprc")
            require(0 <= auroc <= 1 and 0 <= auprc <= 1, "Classification metric outside [0, 1]")
        except ValueError as exc:
            unavailable.append({**identity, "path": str(path), "status": payload.get("status", "invalid"),
                                "reason": str(exc)})
            continue
        cells.append({**identity, "path": str(path), "auroc": auroc, "auprc": auprc,
                      "representation": payload["representation"]})

    representations = {cell["representation"] for cell in cells}
    require(len(representations) <= 1, "Cannot pool different probe representations")
    complete = not missing and not unavailable
    return {
        "status": "complete" if complete else "incomplete",
        "publication_ready": False,
        "event_target_protocol": EVENT_TARGET_PROTOCOL,
        "representation": next(iter(representations), None),
        "n_required": len(expected), "n_complete": len(cells),
        "unsupported": [{"env": env, "model": model, "probe_target": target}
                        for env, model, target in sorted(requested - expected)],
        "missing": missing, "unavailable": unavailable, "cells": cells,
        "per_model": {
            model: {"auroc": summarize(cell["auroc"] for cell in cells if cell["model"] == model),
                    "auprc": summarize(cell["auprc"] for cell in cells if cell["model"] == model)}
            for model in sorted(set(models))
        } if complete else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--envs", nargs="+", required=True, choices=sorted(ENV_REGISTRY))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", required=True, choices=sorted(EVENT_BINARY_TARGETS))
    parser.add_argument("--out", type=Path, required=True, help="New coverage JSON; existing output is never overwritten")
    args = parser.parse_args()
    require(args.probe_dir.is_dir(), f"Missing probe result directory: {args.probe_dir}")
    result = aggregate(args.probe_dir, args.envs, args.models, args.targets)
    write_new_json(args.out, result)
    print(f"[aggregate_event_probes] {result['status']}: {result['n_complete']}/{result['n_required']} supported cells")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

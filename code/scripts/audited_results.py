"""Fail-closed evidence contracts shared by diagnostic/report consumers."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

TRAINING_PROTOCOL = "causal_grad_sigreg_B_20260916"
ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def metric(payload, key, *, nullable=False):
    require(key in payload, f"Missing metric: {key}")
    value = payload[key]
    if value is None and nullable:
        return None
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid metric {key}: {value!r}")
    return float(value)


def summarize(values):
    values = list(values)
    require(values, "Cannot summarize an empty planned metric set")
    defined = [value for value in values if value is not None]
    require(all(type(value) in (int, float) and math.isfinite(value) for value in defined), "Non-finite metric")
    return {"mean": statistics.mean(defined) if defined else None,
            "std": statistics.stdev(defined) if len(defined) > 1 else None,
            "n_expected": len(values), "n_defined": len(defined), "n_undefined": len(values) - len(defined)}


def fmt(value, digits=3):
    return "undefined" if value is None else f"{value:.{digits}f}"


class TrainingAudit:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.payload = load_json(self.path)
        self.digest = sha256(self.path)
        required = self.payload.get("required_checkpoint_metadata", {})
        require(required.get("training_protocol_version") == TRAINING_PROTOCOL,
                "A final causal/gradient/SIGReg-B training audit is required; old data-only generations are invalid")
        require(self.payload.get("execution_ready") is True, "Training audit is not execution-ready")
        for key in ("training_source_sha256", "loader_source_sha256"):
            require(isinstance(self.payload.get(key), dict) and self.payload[key], f"Missing {key}")
        self.results_root = Path(self.payload["results_root"]).resolve()
        self.affected = {str(Path(row["checkpoint"]).resolve()): row for row in self.payload["affected_checkpoints"]}
        self.safe = {str(Path(path).resolve()) for path in self.payload["unaffected_checkpoints"]}
        require(len(self.affected) == len(self.payload["affected_checkpoints"]), "Duplicate affected checkpoints")
        require(not set(self.affected) & self.safe, "Conflicting checkpoint classification")
        self._cache = {}

    def checkpoint(self, path):
        from code.scripts.generalist_v0_7_5_5m.repair_eval_grid import checkpoint_state
        path = str(Path(path).resolve())
        require(path in self.affected or path in self.safe, f"Checkpoint absent from final audit: {path}")
        stat = Path(path).stat()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        cached = self._cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        job = self.affected.get(path)
        audit = None if job is None else {
            "job": job, "manifest": str(self.path), "manifest_sha256": self.digest,
            "data_protocol_version": job["required_data_protocol_version"],
            "loader_source_sha256": self.payload["loader_source_sha256"],
            "training_source_sha256": self.payload["training_source_sha256"],
            "required_metadata": dict(self.payload["required_checkpoint_metadata"],
                                      data_protocol_version=job["required_data_protocol_version"]),
        }
        state = checkpoint_state(path, audit)
        require(state["status"] == "ready", f"Checkpoint is not eligible: {path}: {state.get('reason')}")
        self._cache[path] = (signature, state)
        return state

    def provenance(self, checkpoint):
        checkpoint = str(Path(checkpoint).resolve())
        state = self.checkpoint(checkpoint)
        return {"checkpoint": checkpoint, "checkpoint_sha256": state["sha256"],
                "training_audit": {"path": str(self.path), "sha256": self.digest},
                "training_protocol_version": TRAINING_PROTOCOL if checkpoint in self.affected else None,
                "training_affected": checkpoint in self.affected}

    def validate_provenance(self, payload, checkpoint):
        checkpoint = str(Path(checkpoint).resolve())
        expected = self.provenance(checkpoint)
        actual = payload.get("repair_provenance")
        require(isinstance(actual, dict), f"Missing generation provenance: {checkpoint}")
        for key in ("checkpoint", "checkpoint_sha256"):
            require(actual.get(key) == expected[key], f"Mismatched {key}: {checkpoint}")
        audit = actual.get("training_audit", {})
        require(audit.get("path") == str(self.path) and audit.get("sha256") == self.digest,
                f"Wrong training generation: {checkpoint}")
        if checkpoint in self.affected:
            require(actual.get("training_protocol_version") == TRAINING_PROTOCOL, "Obsolete training protocol")

    def protect_output(self, path):
        path = Path(path).resolve()
        protected = [self.results_root / family for family in self.payload["family_counts"]]
        protected += [Path(self.payload[key]).resolve() for key in ("archive_root", "staging_root")]
        require(path != self.results_root and not any(path == root or root in path.parents for root in protected),
                f"Output must be separate from checkpoint/archive/staging roots: {path}")
        require(not path.exists(), f"Refusing existing output: {path}")


def validate_diagnostic(payload, *, env=None, model=None, seed=None):
    require(not payload.get("error") and not payload.get("errors") and payload.get("skipped") is False,
            "Missing/failed/skipped diagnostics cannot be aggregated")
    for key, expected in (("protocol_version", 2), ("measurement_object", "forward.emb"),
                          ("observation_embedding_object", "forward.emb_pre_cell"), ("weights_loaded_strict", True)):
        require(payload.get(key) == expected, f"Wrong diagnostic contract: {key}")
    for key, expected in (("env", env), ("model", model), ("seed", seed)):
        if expected is not None:
            require(payload.get(key) == expected, f"Wrong diagnostic identity: {key}")
    for key in ("divergence", "responsiveness", "event_rho"):
        value = metric(payload, key, nullable=key != "divergence")
        require(payload["representations"]["readout"].get(key) == value, f"Metric is not the readout: {key}")
        if value is not None:
            require(-1 <= value <= 1 if key == "event_rho" else value >= 0, f"Invalid {key}")
    require(type(payload.get("n_steps")) is int and payload["n_steps"] >= 3, "Incomplete diagnostic trajectory")


def validate_closed_loop(payload, *, episodes=None):
    require(not payload.get("error") and not payload.get("errors") and not payload.get("skipped"), "Failed evaluation")
    protocol = payload.get("protocol", {})
    require(protocol.get("version") == 2 and protocol.get("cem_cost") == "squared_l2"
            and protocol.get("latent_representation") == "forward.emb, independent observed frames, zero action",
            "Obsolete closed-loop protocol")
    require(type(payload.get("n_episodes")) is int and payload["n_episodes"] > 0, "Missing episode count")
    require(len(payload.get("per_episode", [])) == payload["n_episodes"], "Incomplete evaluation episodes")
    if episodes is not None:
        require(payload["n_episodes"] == episodes, "Wrong evaluation episode budget")
    for key in ("success_rate_env", "success_rate_lewm", "success_rate_lewm_005", "mean_cos_dist", "mean_phys_dist"):
        value = metric(payload, key)
        if key.startswith("success_rate"):
            require(0 <= value <= 1, f"Invalid {key}")
    episode_rows = payload["per_episode"]
    require(len({(row["seed"], row["episode_idx"]) for row in episode_rows}) == len(episode_rows),
            "Duplicate closed-loop episodes")
    cosines = [metric(row, "cos_dist") for row in episode_rows]
    distances = [metric(row, "phys_dist") for row in episode_rows]
    require(all(-1e-6 <= value <= 1 + 1e-6 for value in cosines) and all(value >= 0 for value in distances),
            "Invalid terminal distances")
    require(all(type(row["env_success"]) is bool for row in episode_rows), "Invalid native success")
    for key, expected in (
        ("success_rate_env", statistics.mean(row["env_success"] for row in episode_rows)),
        ("success_rate_lewm", sum(value < 0.1 for value in cosines) / len(cosines)),
        ("success_rate_lewm_005", sum(value < 0.05 for value in cosines) / len(cosines)),
        ("mean_cos_dist", statistics.mean(cosines)), ("mean_phys_dist", statistics.mean(distances)),
    ):
        require(math.isclose(metric(payload, key), expected, rel_tol=1e-9, abs_tol=1e-12),
                f"Closed-loop metric/episode mismatch: {key}")


def read_diagnostic(path, audit, checkpoint, *, audited_safe=False, **identity):
    audit.checkpoint(checkpoint)
    payload = load_json(path)
    validate_diagnostic(payload, **identity)
    if not (audited_safe and str(Path(checkpoint).resolve()) in audit.safe and payload.get("repair_provenance") is None):
        audit.validate_provenance(payload, checkpoint)
    require(str(Path(payload.get("checkpoint", payload.get("ckpt", ""))).resolve()) == str(Path(checkpoint).resolve()),
            f"Diagnostic checkpoint identity mismatch: {path}")
    return payload


def validate_pixel_grid(status_path, audit):
    """Resolve the exact 130-checkpoint/1690-cell primary grid from final receipts."""
    from code.scripts.generalist_v0_7_5_5m.repair_eval_grid import MODELS, SPLITS
    from code.scripts.generalist_v0_7_5_5m_pixel.eval_pixel_ckpt import DMC_ENVS
    model_map = {"stjewm_trace_only": "stjewm", "lewm_baseline_v2": "lewm_baseline"}
    models = {model_map.get(model, model) for model in MODELS}
    envs = tuple(DMC_ENVS)
    require(len(models) == 13 and len(SPLITS) == 10 and len(envs) == 13, "Primary pixel inventory changed")
    status_path = Path(status_path).resolve()
    status = load_json(status_path)
    require(status.get("protocol_version") == 2 and status.get("training_protocol_version") == TRAINING_PROTOCOL,
            "Obsolete pixel grid generation")
    require(status["training_manifest"] == str(audit.path) and status["training_manifest_sha256"] == audit.digest,
            "Pixel grid belongs to a different training generation")
    require(status["planned_envs"] == list(envs) and status["planned_checkpoints"] == 130
            and status["planned_cells"] == 1690 and len(status["outcomes"]) == 130, "Incomplete pixel grid")
    expected = {(split, model) for split in SPLITS for model in models}
    seen, rows, outputs = set(), [], set()
    for outcome in status["outcomes"]:
        require(outcome.get("status") == "completed" and outcome.get("returncode") == 0
                and not outcome.get("error"), "Failed/incomplete pixel outcome")
        checkpoint = Path(outcome["checkpoint"]).resolve()
        relative = checkpoint.relative_to(audit.results_root)
        require(len(relative.parts) == 5 and relative.parts[0] == "5m_pixel"
                and relative.parts[-2:] == ("seed_0", "final.pt"), "Wrong pixel checkpoint storage identity")
        key = (relative.parts[1], relative.parts[2])
        require(key in expected and key not in seen, "Missing/duplicate/unexpected pixel model/split")
        seen.add(key)
        output = Path(outcome["output"]).resolve()
        require(status_path.parent in output.parents and output not in outputs, "Pixel output is outside its fresh grid root")
        outputs.add(output)
        require(audit.checkpoint(checkpoint)["sha256"] == outcome["checkpoint_sha256"],
                "Pixel checkpoint changed since evaluation")
        summary_path = output / "eval_summary.json"
        require(sha256(summary_path) == outcome["summary_sha256"], "Pixel summary fingerprint mismatch")
        summary = load_json(summary_path)
        audit.validate_provenance(summary, checkpoint)
        require(summary["training_protocol_version"] == TRAINING_PROTOCOL, "Wrong pixel training generation")
        validate_pixel_summary(summary, checkpoint, output, envs)
        for env in envs:
            rows.append({"split": key[0], "model": key[1], "seed": 0, "env": env,
                         "checkpoint": str(checkpoint), "checkpoint_sha256": outcome["checkpoint_sha256"],
                         "source": str(summary_path), "source_sha256": outcome["summary_sha256"],
                         "metrics": summary["results_per_env"][env]})
    require(seen == expected, "Pixel grid omits a planned checkpoint")
    return rows


def validate_pixel_summary_row(payload, env, episodes):
    require(not payload.get("error") and not payload.get("skipped"), f"Failed pixel cell: {env}")
    require(payload.get("protocol_version") == 2 and payload.get("measurement_object") == "forward.emb",
            "Obsolete pixel measurement")
    require(payload.get("display_env_kind") == env, "Wrong pixel display environment key")
    require(payload.get("env_id") == f"mujoco/{'humanoid_cmu' if env == 'humanoid_CMU' else env}_pixel", "Wrong native pixel environment")
    for key, expected in (("n_episodes", episodes), ("n_seeds", 1), ("horizon", 5), ("eval_budget", 50),
                          ("cem_samples", 300), ("cem_elites", 30), ("cem_iters", 10)):
        require(payload.get(key) == expected, f"Pixel budget mismatch: {key}")
    require(payload["episode_seeds"] == list(range(episodes)) and payload["planner_seeds"] == list(range(episodes)),
            "Unseeded/incomplete pixel evaluation")
    episode_rows = payload["per_episode"]
    require(len(episode_rows) == episodes
            and [entry["episode_idx"] for entry in episode_rows] == list(range(episodes)), "Incomplete pixel episodes")
    cosines = [metric(entry, "cos_dist") for entry in episode_rows]
    distances = [metric(entry, "phys_dist") for entry in episode_rows]
    require(all(-1e-6 <= value <= 1 + 1e-6 for value in cosines), "Invalid normalized cosine distance")
    require(all(value >= 0 for value in distances), "Invalid physical distance")
    require(all(type(entry["env_success"]) is bool and 1 <= entry["actions_taken"] <= 50 for entry in episode_rows),
            "Incomplete native pixel trajectory")
    expected = {
        "success_rate_env": statistics.mean(entry["env_success"] for entry in episode_rows),
        "mean_cos_dist": statistics.mean(cosines), "mean_phys_dist": statistics.mean(distances),
        "success_rate_lewm": sum(value < 0.1 for value in cosines) / episodes,
        "success_rate_lewm_005": sum(value < 0.05 for value in cosines) / episodes,
        "success_rate_lewm_001": sum(value < 0.01 for value in cosines) / episodes,
    }
    for key, value in expected.items():
        require(math.isclose(metric(payload, key), value, rel_tol=1e-9, abs_tol=1e-12), f"Pixel metric/episode mismatch: {key}")


def load_auxiliary(run_dir, audit, groups):
    """Read only a completed auxiliary plan, not a directory-discovered subset."""
    from code.scripts.repair_auxiliary_grid import validate_payload, validate_arrays
    run_dir = Path(run_dir)
    plan, status = load_json(run_dir / "plan.json"), load_json(run_dir / "status.json")
    require(plan.get("protocol_version") == 2 and status.get("status") == "completed", "Auxiliary grid is incomplete")
    require(status["planned"] == plan["totals"] and status["completed_cells"] == plan["totals"]["cells"],
            "Auxiliary coverage mismatch")
    manifests = plan["training_manifests"]
    require(len(manifests) == 1 and manifests[0]["path"] == str(audit.path)
            and manifests[0]["sha256"] == audit.digest, "Auxiliary grid belongs to a different training generation")
    require(all(sha256(ROOT / name) == source["sha256"] for name, source in plan["source_files"].items()),
            "Auxiliary evaluation source changed since planning")
    states = {row["id"]: row for row in status["jobs"]}
    require(len(states) == len(plan["jobs"]) and len(states) == len(status["jobs"]), "Duplicate/missing auxiliary jobs")
    rows = []
    for job in plan["jobs"]:
        require(states[job["id"]]["status"] == "completed" and states[job["id"]]["cells"] == len(job["cells"]),
                f"Incomplete auxiliary job: {job['id']}")
        if job["group"] not in groups:
            continue
        for cell in job["cells"]:
            payload = read_diagnostic(
                cell["output"], audit, job["checkpoint"], env=cell["env"], model=job["model"], seed=0,
                audited_safe=cell["output"] in plan["audited_safe_outputs"],
            )
            validate_payload(payload, job, cell)
            if "npz" in cell:
                validate_arrays(cell["npz"], payload)
            require(payload["n_steps"] == job["budget"]["n_steps"], "Diagnostic budget mismatch")
            rows.append({"group": job["group"], "family": job["family"], "split": job["split"],
                         "model": job["model"], "env": cell["env"], "seed": 0, "checkpoint": job["checkpoint"],
                         "source": cell["output"], "source_sha256": sha256(cell["output"]), **payload})
    require(rows and set(groups) <= {row["group"] for row in rows}, "Requested diagnostic group is absent")
    return rows


def load_state_grid(run_dir, audit):
    """Validate every planned state cell and its process/checkpoint/output receipt."""
    from code.scripts.generalist_v0_7_5_5m.repair_eval_grid import GRID_PROTOCOL, validate_result
    run_dir = Path(run_dir)
    plan, status = load_json(run_dir / "plan.json"), load_json(run_dir / "status.json")
    require(plan.get("grid_protocol") == GRID_PROTOCOL and status.get("grid_protocol") == GRID_PROTOCOL,
            "Obsolete state grid generation")
    require(plan["manifest"] == str(audit.path) and plan["manifest_sha256"] == audit.digest, "Wrong training generation")
    require(status.get("full_grid_complete") is True and status["plan_sha256"] == plan["plan_sha256"],
            "State grid is incomplete or belongs to a different plan")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() == plan["plan_sha256"],
            "State plan fingerprint mismatch")
    require(plan["planned_cells"] == len(plan["cells"]) == len(status["cells"]), "State coverage mismatch")
    rows = []
    for cell in plan["cells"]:
        receipt = status["cells"][cell["id"]]
        require(receipt["status"] == "completed" and receipt["returncode"] == 0, f"Incomplete state cell: {cell['id']}")
        current = audit.checkpoint(cell["checkpoint"])
        require(current["sha256"] == receipt["checkpoint"]["sha256"], "Checkpoint changed since evaluation")
        require(sha256(cell["output"]) == receipt["output_sha256"], "State result fingerprint mismatch")
        validate_result(cell, receipt["checkpoint"])
        payload = load_json(cell["output"])
        rows.append({**cell, "env": cell["env_id"], "seed": cell["training_seed"],
                     "source": cell["output"], "source_sha256": receipt["output_sha256"], "metrics": payload})
    require(len({row["id"] for row in rows}) == len(rows), "Duplicate state cells")
    return rows


def validate_pixel_summary(summary, checkpoint, output, envs, *, episodes=5):
    require(summary.get("protocol_version") == 2, "Obsolete pixel protocol")
    require(Path(summary["ckpt"]).resolve() == Path(checkpoint).resolve(), "Wrong pixel checkpoint")
    require(set(summary["evaluated_envs"]) == set(envs)
            and len(summary["evaluated_envs"]) == len(envs)
            and set(summary["results_per_env"]) == set(envs), "Incomplete pixel environment coverage")
    require(summary["protocol"]["name"] == "pixel_static_qpos_native_action_cem"
            and summary["protocol"]["latent_distance"] == "(1 - cosine_similarity(final_z, goal_z)) / 2",
            "Pixel distance is not the canonical latent cosine metric")
    for env in envs:
        row = summary["results_per_env"][env]
        validate_pixel_summary_row(row, env, episodes)
        require(load_json(Path(output) / f"eval_{env}.json") == row, f"Pixel summary/per-env mismatch: {env}")


def heldout_native_success(env_kind, env, final_state, goal_state):
    """Reproduce closed_loop's env-native success, including the reacher override."""
    import numpy as np
    from code.core.envs.dmc_env import DMC_ENVS
    if env_kind == "reacher":
        distance = float(np.linalg.norm(np.asarray(final_state)[:2] - np.asarray(goal_state)[:2]) / np.sqrt(2))
        return bool(distance < DMC_ENVS["reacher"][4]), distance
    success, distance = env.check_success(final_state, goal_state)
    return bool(success), float(distance)

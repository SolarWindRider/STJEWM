"""P12 reconstruction: synthetic ground-truth encoders through the canonical diagnostics.

The original P12 producer, raw sweep and seeds were not found in any searched
local archive or OBS prefix (recovery ledger 2026-09-16). This reconstruction
follows the documented narrative spec (results_draft.md §2.3): six named
ground-truth encoders plus gain/noise sweeps evaluated on a 200-step random
cartpole trajectory with the divergence (mean per-dim std), responsiveness
(mean ||dz|| / mean ||dobs||) and event-rho (within-episode Pearson of
transition norms) definitions used by code/scripts/latent_rollout.py.

Documented thresholds: divergence 0.05 separates calibrated from over-reactive;
event rho 0.3 separates inliers from noise. Encoders are applied to the raw
recorded observations (no rescaling) and every rule/scale/seed below is a
reconstruction decision, not a recovered original.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import code  # noqa: F401

import numpy as np

from code.scripts.latent_rollout import trajectory_metrics

PROTOCOL = "p12_synthetic_reconstruction_20260917"
NAMED_ENCODERS = (
    "constant", "identity_k0.2", "identity_k1.0", "gain_k10",
    "noisy_sigma0.1", "uncorrelated",
)
GAIN_SWEEP = (0.02, 0.04, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 10.0)
NOISE_SWEEP = (0.01, 0.02, 0.05, 0.1, 0.2)
DIV_THRESHOLD = 0.05
RHO_THRESHOLD = 0.3
CONSTANT_EPS = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_window(n_steps: int, seed: int):
    """One seeded, episode-safe observation window from the cartpole benchmark data."""
    from code.data import load_dataset
    from code.scripts.event_align import ENV_DATA

    path = ENV_DATA["cartpole_2d"]
    # history(1) + goal_offset + 1 = n_steps, so one legal window is the full
    # trajectory; episode-safe starts still exclude any boundary-crossing row.
    ds = load_dataset(env_kind="dmc", path=path, history_size=1,
                      goal_offset=n_steps - 2, max_windows=None, pad_obs_to=None)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(ds)))
    item = ds[start]
    window = item["state"].numpy()  # (history+goal_offset+1, obs_dim)
    if window.shape[0] < n_steps:
        raise ValueError(f"Dataset window shorter than requested: {window.shape}")
    observations = window[:n_steps].astype(np.float64)
    episode_ids = np.zeros(n_steps, dtype=np.int64)
    return observations, episode_ids, Path(path).resolve(), sha256(Path(path))


def encoder_series(kind: str, obs: np.ndarray, rng: np.ndarray) -> np.ndarray:
    if kind == "constant":
        return np.full_like(obs, obs.mean())
    if kind.startswith(("gain_k", "identity_k")):
        return obs * float(kind.rsplit("_k", 1)[1])
    if kind.startswith("noisy_sigma"):
        return obs + float(kind[12:]) * rng.standard_normal(obs.shape)
    if kind == "uncorrelated":
        return rng.standard_normal(obs.shape) * obs.std(axis=0, keepdims=True)
    raise ValueError(f"Unknown encoder kind: {kind}")


def classify(metrics: dict) -> str:
    """Explicit reconstruction rules; each boundary is disclosed in the output."""
    if metrics["divergence"] < CONSTANT_EPS:
        return "collapsed"
    rho = metrics["event_rho"]
    if rho is None or rho < RHO_THRESHOLD:
        return "noise"
    if metrics["divergence"] > DIV_THRESHOLD:
        return "over_reactive"
    return "calibrated"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=199,
                        help="cartpole_250k episodes are 200 rows with a terminal "
                             "marker, so 199 is the longest episode-safe window; "
                             "the narrative's 200-step trajectory is not "
                             "reproducible episode-safely and this deviation is "
                             "disclosed in the output.")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--out", type=Path,
                        default=Path("/data/lx/tmp/results/p12_synthetic_reconstruction"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Archive the previous reconstruction first: {args.out}")
    args.out.mkdir(parents=True)

    observations, episode_ids, data_path, data_hash = load_window(args.n_steps, args.seed)

    def evaluate(kind: str) -> dict:
        # zlib.crc32 is process-stable, unlike str.hash (randomized per run).
        rng = np.random.default_rng(args.seed + zlib.crc32(kind.encode()) % 100000)
        latents = encoder_series(kind, observations, rng)
        metrics = trajectory_metrics(observations, latents, episode_ids)
        return {"encoder": kind, "regime": classify(metrics), **metrics}

    named = [evaluate(kind) for kind in NAMED_ENCODERS]
    gain_rows = [evaluate(f"gain_k{k}") for k in GAIN_SWEEP]
    noise_rows = [evaluate(f"noisy_sigma{s}") for s in NOISE_SWEEP]
    uncorrelated = next(row for row in named if row["encoder"] == "uncorrelated")

    gains = np.array([row["divergence"] for row in gain_rows])
    ks = np.array(GAIN_SWEEP)
    crossing = None
    for index in range(1, len(ks)):
        if gains[index - 1] <= DIV_THRESHOLD < gains[index]:
            lower, upper = gains[index - 1], gains[index]
            crossing = float(ks[index - 1] + (DIV_THRESHOLD - lower)
                             * (ks[index] - ks[index - 1]) / (upper - lower))
            break
    rho_sigma_crossing = None
    sigmas = np.array(NOISE_SWEEP)
    rhos = np.array([row["event_rho"] for row in noise_rows])
    for index in range(1, len(sigmas)):
        if rhos[index - 1] >= RHO_THRESHOLD > rhos[index]:
            lower, upper = rhos[index - 1], rhos[index]
            rho_sigma_crossing = float(sigmas[index - 1] + (lower - RHO_THRESHOLD)
                                       * (sigmas[index] - sigmas[index - 1])
                                       / (lower - upper))
            break

    summary = {
        "protocol": PROTOCOL,
        "provenance": {
            "spec_source": "results_draft.md §2.3 (archived documents/, verbatim passages in P12Evidence ledger)",
            "original_producer": "not found in searched local archives or OBS (see /data/lx/tmp/results/_repair_archive/20260916T102154Z raw-scan ledger)",
            "trajectory": {"data_path": str(data_path), "data_sha256": data_hash,
                           "window_seed": args.seed, "n_steps": args.n_steps,
                           "episode_safe": True,
                           "obs_per_dim_std": observations.std(axis=0).tolist(),
                           "divergence_of_identity": float(trajectory_metrics(
                               observations, observations, episode_ids)["divergence"])},
            "source_sha256": {str(path): sha256(path) for path in (
                ROOT / "code/scripts/p12_synthetic_validation.py",
                ROOT / "code/scripts/latent_rollout.py",
                ROOT / "code/data/loaders.py",
            )},
        },
        "rules": {
            "collapsed": f"divergence < {CONSTANT_EPS}",
            "noise": f"event_rho < {RHO_THRESHOLD} (undefined counts as noise)",
            "over_reactive": f"divergence > {DIV_THRESHOLD}",
            "calibrated": "remaining encoders",
            "encoder_scale": "raw recorded observations, no rescaling (reconstruction decision)",
            "uncorrelated_scale": "per-dim iid normal scaled to the window's per-dim std (reconstruction decision)",
            "noise_rng": "per-encoder Generator seeded from --seed plus stable zlib.crc32(name) (PYTHONHASHSEED-independent)",
            "rho_noise_floor": "rho monotonicity is only meaningful above the noise floor; with n-3=196 transitions the standard error is about 0.07, so near-zero rho values can invert order by sampling fluctuation (observed between sigma=0.1 and 0.2)",
        },
        "named_encoders": named,
        "gain_sweep": {"k": list(GAIN_SWEEP), "rows": gain_rows,
                       "div_threshold_crossing_k": crossing},
        "noise_sweep": {"sigma": list(NOISE_SWEEP), "rows": noise_rows,
                        "rho_threshold_crossing_sigma": rho_sigma_crossing},
        "documented_claims_check": {
            "constant_is_collapsed": next(r["regime"] == "collapsed" for r in named if r["encoder"] == "constant"),
            "uncorrelated_is_noise": uncorrelated["regime"] == "noise",
            "div_increases_monotonically_in_k": bool(np.all(np.diff(gains) > 0)),
            "rho_decreases_with_sigma": bool(np.all(np.diff(rhos) < 0)),
            "documented_div_boundary_between_k03_and_k05": crossing is not None and 0.3 <= crossing <= 0.5,
            "documented_rho_boundary_between_sigma002_and_sigma005": rho_sigma_crossing is not None and 0.02 <= rho_sigma_crossing <= 0.05,
        },
        "scale_disclosure": {
            "note": "The divergence criterion is observation-scale-dependent: "
                    "div(k*obs) = k * div(obs). The narrative's k-crossing in "
                    "(0.3, 0.5) implies the original trajectory's mean per-dim "
                    "observation std was about 0.10-0.17, which does not match "
                    "this repo's raw cartpole scale. The crossing below is "
                    "reported at the measured scale; k_per_unit_scale is the "
                    "scale-free crossing (threshold / div(identity)).",
            "div_threshold_crossing_k_per_unit_scale": (
                DIV_THRESHOLD / trajectory_metrics(observations, observations, episode_ids)["divergence"]
            ),
        },
    }
    with (args.out / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps({"named": {r["encoder"]: r["regime"] for r in named},
                      "crossings": {"div_k": crossing, "rho_sigma": rho_sigma_crossing},
                      "claims": summary["documented_claims_check"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

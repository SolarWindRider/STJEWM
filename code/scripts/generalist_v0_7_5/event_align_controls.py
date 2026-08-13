"""Event-alignment negative controls (Workstream C of v0.7.6-fixes-plan).

The headline ρ ≈ 0.99 for STJEWM-trace looks too perfect. This script applies
five (technically six + the untrained case = seven) perturbations to the latent
stream ONLY and recomputes Pearson(‖Δobs‖, ‖Δz_perturbed‖) using the same
``pearson()`` helper that ``event_align.py`` uses.

Per the v0.7.6 plan:
- The original "none" control reproduces ``event_align.py``'s reported ρ.
- time-shift-k shifts ``lat_list`` by +k steps before ‖Δz‖.
- latent-shuffle randomly permutes ``lat_list`` indices.
- obs-copy treats the first 16 obs dims as the surrogate latent.
- action-only uses the action-encoder output as the surrogate latent.
- untrained_trace instantiates STJEWM(trace_only) with random weights.

For each (model, env, control) we emit one JSON to:
    results/generalist_G16/event_align_controls/<model>_<env>_<control>.json

After all runs, the script aggregates into:
    results/aggregate/generalist_align_controls_table.md

Budget: ~5s/run × 12 ckpts × ~7 controls × 2 envs ≈ 10 min on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root on sys.path (event_align.py expects this).
sys.path.insert(0, "/home/lx/snn")

from code.scripts.event_align import (
    ENV_DATA,
    ENV_KIND_MAP,
    build_model,
    pearson,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# 12 G16 generalist ckpts (with seed_0/final.pt present).
G16_MODELS: List[str] = [
    "stjewm_trace_only",
    "stjewm_spike_only",
    "stjewm_rate_only",
    "stjewm_no_trace",
    "stjewm_hidden_leak",
    "stjewm_membrane_readout",
    "alif_timecell_baseline",
    "gru_baseline",
    "lewm_baseline_v2",
    "stacked_lif_trace",
    "stacked_lif_free",
    "mlp_baseline",
]

# Two envs are sufficient and match Figure 5.
CONTROL_ENVS: List[str] = ["cheetah", "finger"]

# Table rows in the order the paper references.
TABLE_ROWS: List[str] = [
    "stjewm_trace_only",
    "stjewm_spike_only",
    "stjewm_rate_only",
    "stjewm_hidden_leak",
    "stjewm_membrane_readout",
    "alif_timecell_baseline",
    "gru_baseline",
    "mlp_baseline",
    "untrained_trace",
]

# Control columns (the first one reproduces event_align.py's "none").
CONTROLS: List[str] = [
    "none",
    "time_shift_1",
    "time_shift_5",
    "time_shift_10",
    "latent_shuffle",
    "obs_copy",
    "action_only",
]

# A row of "untrained_trace" only runs the "none" control (i.e. ρ on
# untrained STJEWM-trace). Other controls on untrained weights are
# meaningless (shuffling a random init has nothing to test against) and
# the plan only specifies the "none" case for the untrained model.
UNTRAINED_CONTROLS: List[str] = ["none"]


# --------------------------------------------------------------------------
# Trajectory collection — shared between trained ckpts and untrained STJEWM.
# Returns:
#   obs_arr  (T, state_dim) padded obs
#   lat_arr  (T, D)         pre-cell state embedding (matches event_align.py)
#   act_arr  (T, action_dim)
#   meta     dict with state_dim, action_dim, n_steps, n_resets
# --------------------------------------------------------------------------
def _collect_trajectory(
    model: torch.nn.Module,
    env,
    state_dim: int,
    action_dim: int,
    n_steps: int = 100,
    n_resets: int = 1,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Run a random policy and record obs / latent / action trajectories.

    Mirrors ``event_align.py``'s main loop exactly so that the "none" control
    reproduces the existing ρ value to within ±0.001.
    """
    obs_list: List[np.ndarray] = []
    lat_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []

    a_low = env.spec.action_low
    a_high = env.spec.action_high
    steps_per_reset = max(1, n_steps // n_resets)

    env.reset(seed=seed)
    obs = env.get_state()
    if len(obs) < state_dim:
        obs = np.concatenate([obs, np.zeros(state_dim - len(obs), dtype=np.float32)])

    n_done = 0
    t = 0
    while t < n_steps:
        a = np.random.uniform(a_low, a_high).astype(np.float32)
        out, _, done, _ = env.step(a)
        obs = out.get("state", list(out.values())[0])
        obs = np.asarray(obs, dtype=np.float32)
        if len(obs) < state_dim:
            obs = np.concatenate([obs, np.zeros(state_dim - len(obs), dtype=np.float32)])
        s_t = torch.from_numpy(obs).reshape(1, 1, -1).to(device)
        a_padded = np.zeros(action_dim, dtype=np.float32)
        a_padded[: len(a)] = a
        a_t = torch.from_numpy(a_padded).reshape(1, 1, -1).to(device)
        with torch.no_grad():
            enc = model.encode(s_t, a_t)
        lat_list.append(enc["emb"][0, 0].cpu().numpy())
        obs_list.append(obs)
        act_list.append(a_padded)
        t += 1
        if done and t < n_steps:
            n_done += 1
            env.reset(seed=n_done)

    return {
        "obs_arr": np.stack(obs_list, axis=0),
        "lat_arr": np.stack(lat_list, axis=0),
        "act_arr": np.stack(act_list, axis=0),
        "n_steps": int(len(obs_list)),
        "n_resets": int(n_done),
    }


# --------------------------------------------------------------------------
# Controls — each takes the trajectory and returns a (T-1,) surrogate Δz.
# --------------------------------------------------------------------------
def _delta_l2(arr: np.ndarray) -> np.ndarray:
    """First-difference L2 norm along the time axis. Shape: (T-1,)."""
    if arr.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(np.diff(arr, axis=0), axis=1).astype(np.float32)


def _ctrl_none(traj: Dict[str, np.ndarray], **_unused) -> np.ndarray:
    return _delta_l2(traj["lat_arr"])


def _ctrl_time_shift(traj: Dict[str, np.ndarray], *, k: int = 1, **_unused) -> np.ndarray:
    """Shift ``lat_arr`` by +k steps (drop the last k, prepend k copies of
    ``lat_arr[0]``). The diff then reflects the misalignment between the
    obs at time t and the latent at time t-k.
    """
    lat = traj["lat_arr"]
    if k >= lat.shape[0]:
        return np.zeros((0,), dtype=np.float32)
    shifted = np.concatenate([np.broadcast_to(lat[0:1], (k,) + lat.shape[1:]), lat[:-k]], axis=0)
    return _delta_l2(shifted)


def _ctrl_latent_shuffle(traj: Dict[str, np.ndarray], *, seed: int = 0, **_unused) -> np.ndarray:
    lat = traj["lat_arr"].copy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(lat.shape[0])
    return _delta_l2(lat[perm])


def _ctrl_obs_copy(traj: Dict[str, np.ndarray], **_unused) -> np.ndarray:
    """Use the first 16 obs dims as the surrogate latent."""
    obs = traj["obs_arr"][:, :16]
    return _delta_l2(obs)


def _ctrl_action_only(traj: Dict[str, np.ndarray], **_unused) -> np.ndarray:
    """Use the raw padded action as the surrogate latent (the action
    vector that event_align.py actually feeds the model)."""
    return _delta_l2(traj["act_arr"])


CONTROL_FNS = {
    "none": _ctrl_none,
    "time_shift_1": lambda t, **kw: _ctrl_time_shift(t, k=1, **kw),
    "time_shift_5": lambda t, **kw: _ctrl_time_shift(t, k=5, **kw),
    "time_shift_10": lambda t, **kw: _ctrl_time_shift(t, k=10, **kw),
    "latent_shuffle": _ctrl_latent_shuffle,
    "obs_copy": _ctrl_obs_copy,
    "action_only": _ctrl_action_only,
}


# --------------------------------------------------------------------------
# Compute ρ_obs_perturbed from trajectory + control.
# --------------------------------------------------------------------------
def _compute_rho(traj: Dict[str, np.ndarray], control: str, seed: int = 0) -> Tuple[float, int]:
    """Return (rho, n_pairs). rho is NaN-flagged as None on degenerate inputs."""
    d_obs = _delta_l2(traj["obs_arr"])
    fn = CONTROL_FNS[control]
    if control == "latent_shuffle":
        d_lat = fn(traj, seed=seed)
    else:
        d_lat = fn(traj)
    L = min(d_obs.shape[0], d_lat.shape[0])
    if L < 2:
        return 0.0, 0
    d_obs = d_obs[:L]
    d_lat = d_lat[:L]
    return pearson(d_obs, d_lat), L


# --------------------------------------------------------------------------
# Per-(model, env) run — collect trajectory once, apply all controls.
# --------------------------------------------------------------------------
def _load_trained_model(ckpt_path: str, env_name: str, model_name: str, device: str):
    """Mirror ``event_align.py``'s ckpt loading + build_model routing."""
    from code.eval.closed_loop import make_env
    env_kind = ENV_KIND_MAP[env_name]
    env = make_env(env_kind, data_path=None)
    state_dim = env.spec.obs_dim
    action_dim = env.spec.action_dim
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {}) or {}
    pad_obs_to = ck_args.get("pad_obs_to")
    if pad_obs_to is not None:
        state_dim = pad_obs_to
    ck_action_dim = ck_args.get("action_dim")
    if ck_action_dim is not None:
        action_dim = ck_action_dim
    model = build_model(model_name, state_dim, action_dim, ck_args)
    try:
        model.load_state_dict(ck["model"])
    except Exception as e:  # noqa: BLE001 — surface as a skipped entry
        return None, None, None, None, f"ckpt load failed: {e}"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, env, state_dim, action_dim, None


def _build_untrained_trace(env_name: str, device: str):
    """Instantiate STJEWM with readout_mode='trace_only' but DO NOT load any
    ckpt — random init."""
    from code.eval.closed_loop import make_env
    from code.stjewm import STJEWM
    env_kind = ENV_KIND_MAP[env_name]
    env = make_env(env_kind, data_path=None)
    state_dim = env.spec.obs_dim
    action_dim = env.spec.action_dim
    # Match the args used by stjewm_trace_only G16 ckpt (see train_one.sh / aggregate_master.py).
    model = STJEWM(
        d_hid=192, embed_dim=192, action_dim=action_dim, action_emb_dim=192,
        state_dim=state_dim, cell_n_layers=4, n_d=3,
        trace_beta=0.9, freeze_encoder=True,
        readout_mode="trace_only",
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, env, state_dim, action_dim


def _run_model_env(
    model_name: str,
    env_name: str,
    controls: List[str],
    ckpt_dir: Path,
    out_dir: Path,
    n_steps: int = 100,
    n_resets: int = 1,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Run one model × env × {all controls}. Writes one JSON per (model, env, control)."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if model_name == "untrained_trace":
        model, env, state_dim, action_dim = _build_untrained_trace(env_name, device)
        skip_reason: Optional[str] = None
    else:
        ckpt_path = ckpt_dir / model_name / "seed_0" / "final.pt"
        if not ckpt_path.exists():
            skip_reason = f"no ckpt at {ckpt_path}"
            model = env = state_dim = action_dim = None
        else:
            model, env, state_dim, action_dim, skip_reason = _load_trained_model(
                str(ckpt_path), env_name, model_name, device,
            )

    for ctrl in controls:
        out_path = out_dir / f"{model_name}_{env_name}_{ctrl}.json"
        out[(model_name, ctrl)] = None  # placeholder
        if skip_reason is not None or model is None:
            out[(model_name, ctrl)] = {"skipped": True, "reason": skip_reason}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out[(model_name, ctrl)], indent=2))
            print(f"[controls] {model_name:28s} {env_name:7s} {ctrl:15s} SKIPPED ({skip_reason or chr(34)+chr(34)})")
            continue

        try:
            traj = _collect_trajectory(
                model, env, state_dim, action_dim,
                n_steps=n_steps, n_resets=n_resets, seed=seed, device=device,
            )
        except Exception as e:  # noqa: BLE001
            out[(model_name, ctrl)] = {"skipped": True, "reason": f"trajectory failed: {e}"}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out[(model_name, ctrl)], indent=2))
            print(f"[controls] {model_name:28s} {env_name:7s} {ctrl:15s} SKIPPED ({skip_reason or chr(34)+chr(34)})")
            continue

        rho, n_pairs = _compute_rho(traj, ctrl, seed=seed)
        rec = {
            "skipped": False,
            "reason": None,
            "env": env_name,
            "model": model_name,
            "control": ctrl,
            "rho": float(rho),
            "n_steps": int(n_pairs),
            "n_resets": int(traj["n_resets"]),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
        }
        out[(model_name, ctrl)] = rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"[controls] {model_name:28s} {env_name:7s} {ctrl:15s} "
              f"rho={rho:+.4f}  n={n_pairs}")

    return out


# --------------------------------------------------------------------------
# Aggregate → generalist_align_controls_table.md
# --------------------------------------------------------------------------
def _load_results(out_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    rows: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for fp in sorted(out_dir.glob("*.json")):
        stem = fp.stem  # "<model>_<env>_<control>"
        # env names are known and contain underscores only in cartpole_2d etc.
        # control names are stable; iterate from the right.
        if "_" not in stem:
            continue
        for ctrl in CONTROLS:
            if stem.endswith("_" + ctrl):
                control = ctrl
                rest = stem[: -(len(ctrl) + 1)]
                for env in CONTROL_ENVS:
                    if rest.endswith("_" + env):
                        env_name = env
                        model_name = rest[: -(len(env) + 1)]
                        try:
                            d = json.loads(fp.read_text())
                        except Exception:
                            continue
                        rows[(model_name, env_name, control)] = d
                        break
                break
    return rows


def _format_rho_cell(model_name: str, env: str, ctrl: str,
                     rho_map: Dict[Tuple[str, str, str], Optional[float]]) -> str:
    rho = rho_map.get((model_name, env, ctrl))
    if rho is None:
        return "—"
    txt = f"{rho:.3f}"
    if rho < 0.3:
        return f"**{txt}**"  # killed by control → alignment was real
    if rho > 0.7:
        return f"*{txt}*"   # control didn't affect → possible artifact
    return txt


def _write_table(rows: Dict[Tuple[str, str, str], Dict[str, Any]],
                 table_path: Path) -> None:
    """For each (model, control) row, produce one row in the table with
    columns: control | ρ(cheetah) | ρ(finger).

    The plan's table has 9 models × controls. We emit one row per model with
    sub-rows per control, OR — simpler — pivot the table to (model, control)
    cells. The plan's headline layout is "rows = model × controls" so we
    follow that.
    """
    rho_map: Dict[Tuple[str, str, str], Optional[float]] = {}
    for k, v in rows.items():
        rho_map[k] = None if v.get("skipped") else v.get("rho")

    table_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Event-alignment negative controls (G16 generalist, cheetah + finger)\n")
    lines.append(
        "Pearson ρ between obs-event (‖Δobs‖) and a *perturbed* latent-event "
        "(‖Δz_perturbed‖). The metric is `event_align.pearson` exactly.\n"
    )
    lines.append(
        "**Interpretation**:\n"
        "- **bold** = ρ < 0.3 → control killed the alignment (the metric was real)\n"
        "- *italic* = ρ > 0.7 → control did not break alignment (possible metric artifact)\n"
        "- plain = ρ in [0.3, 0.7] → intermediate / inconclusive\n"
    )
    lines.append("")

    # Layout: one section per model. For each model, one sub-table with
    # control rows × env columns. This makes the matrix readable when each
    # model has a different control set.
    header = (
        "| control | ρ (cheetah) | ρ (finger) |\n"
        "|---|---|---|"
    )
    lines.append(header)
    for m in TABLE_ROWS:
        lines.append(f"\n### {m}\n")
        controls = UNTRAINED_CONTROLS if m == "untrained_trace" else CONTROLS
        lines.append(header)
        for ctrl in controls:
            cheetah_cell = _format_rho_cell(m, "cheetah", ctrl, rho_map)
            finger_cell = _format_rho_cell(m, "finger", ctrl, rho_map)
            lines.append(f"| {ctrl} | {cheetah_cell} | {finger_cell} |")

    # Acceptance block.
    lines.append("\n## Acceptance check\n")
    def _fmt(v): return "—" if v is None else f"{v:.3f}"
    ok_lines = []
    for m in ["stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only", "stjewm_hidden_leak"]:
        rho_c = rho_map.get((m, "cheetah", "none"))
        rho_f = rho_map.get((m, "finger", "none"))
        rho_sc = rho_map.get((m, "cheetah", "latent_shuffle"))
        rho_sf = rho_map.get((m, "finger", "latent_shuffle"))
        ok = (rho_c is not None and rho_c >= 0.95 and rho_f is not None and rho_f >= 0.95
              and rho_sc is not None and rho_sc <= 0.3 and rho_sf is not None and rho_sf <= 0.3)
        flag = "PASS" if ok else "FAIL"
        ok_lines.append(
            f"- {flag}  {m}: none=(cheetah {_fmt(rho_c)}, finger {_fmt(rho_f)}), "
            f"latent_shuffle=(cheetah {_fmt(rho_sc)}, finger {_fmt(rho_sf)})"
        )
    rho_uc = rho_map.get(("untrained_trace", "cheetah", "none"))
    rho_uf = rho_map.get(("untrained_trace", "finger", "none"))
    ok_untrained = (rho_uc is not None and rho_uc <= 0.3 and rho_uf is not None and rho_uf <= 0.3)
    flag = "PASS" if ok_untrained else "FAIL"
    ok_lines.append(
        f"- {flag}  untrained_trace: none=(cheetah {_fmt(rho_uc)}, finger {_fmt(rho_uf)}) — must be ≤ 0.3"
    )
    lines.extend(ok_lines)
    lines.append("")

    table_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Event-alignment negative controls.")
    ap.add_argument("--suite", default="G16", choices=["G16"])
    ap.add_argument("--out-dir", default="/home/lx/snn/results/generalist_G16/event_align_controls")
    ap.add_argument("--ckpt-dir", default="/home/lx/snn/results/generalist_G16")
    ap.add_argument("--aggregate-out",
                    default="/home/lx/snn/results/aggregate/generalist_align_controls_table.md")
    ap.add_argument("--n-steps", type=int, default=100,
                    help="Match run_align.sh: 100 steps -> 99 valid diff pairs.")
    ap.add_argument("--n-resets", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Override the model list (defaults to all 12 G16 ckpts + untrained_trace).")
    ap.add_argument("--envs", nargs="*", default=None,
                    help="Override env list (defaults to cheetah + finger).")
    ap.add_argument("--controls", nargs="*", default=None,
                    help="Override control list (defaults to all 7 controls).")
    ap.add_argument("--skip-aggregate", action="store_true",
                    help="Skip the markdown aggregation step.")
    args = ap.parse_args()

    models = args.models or (G16_MODELS + ["untrained_trace"])
    envs = args.envs or CONTROL_ENVS
    controls = args.controls or CONTROLS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)

    print(f"[controls] running {len(models)} models × {len(envs)} envs × "
          f"{len(controls)} controls → {out_dir}")

    for m in models:
        for env in envs:
            ctrls = UNTRAINED_CONTROLS if m == "untrained_trace" else controls
            _run_model_env(
                model_name=m, env_name=env, controls=ctrls,
                ckpt_dir=ckpt_dir, out_dir=out_dir,
                n_steps=args.n_steps, n_resets=args.n_resets,
                seed=args.seed, device=args.device,
            )

    if args.skip_aggregate:
        return 0

    print(f"[controls] aggregating → {args.aggregate_out}")
    rows = _load_results(out_dir)
    _write_table(rows, Path(args.aggregate_out))
    print(f"[controls] wrote {args.aggregate_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
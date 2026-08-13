"""OOD path-C: 3-family DMC cross-sub-family transfer (v0.7.10b).

Per docs/OOD_PATH_C_PLAN.md:
  - 3 DMC sub-families: F1 classic control, F2 locomotion, F3 sparse-POMDP.
  - 6 splits total: 3 OOD1 (1 train -> 2 unseen) + 3 OOD2 (2 train -> 1 unseen).
  - 12 ckpts per split: 6 STJEWM readouts + 6 baselines.
  - 1 seed, 2K windows/env, 25 min/ckpt (sequential on 1-CPU ~30 hr; parallel 4-way ~8 hr).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, "/home/lx/snn")

import numpy as np
import torch


SPLIT_NAMES = [
    "oodc_F1",
    "oodc_F2",
    "oodc_F3",
    "oodc_F1F2",
    "oodc_F1F3",
    "oodc_F2F3",
]

DEFAULT_CKPT_BUDGET = [
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


def resolve_env_kind(env_id: str) -> str:
    mapping = {
        "cartpole_2d": "cartpole",
        "pendulum_2d": "pendulum",
        "humanoid_CMU": "humanoid_cmu",
    }
    return mapping.get(env_id, env_id)


def _flat_obs(o):
    if isinstance(o, dict):
        for k in ("state", "obs", "observation"):
            if k in o:
                o = o[k]
                break
        else:
            for v in o.values():
                if hasattr(v, "astype"):
                    o = v
                    break
    arr = np.asarray(o).flatten().astype(np.float32)
    return arr


def train_one_ckpt(split: str, model_name: str, seed: int, out_dir: str) -> Path:
    ckpt_dir = Path(out_dir) / split / model_name / f"seed_{seed}"
    final_pt = ckpt_dir / "final.pt"
    if final_pt.exists():
        print(f"  [skip-train] {final_pt} already exists")
        return final_pt

    spec_path = Path(f"configs/oodc/{split}.json")
    spec = json.loads(spec_path.read_text())
    train_specs = spec["train_specs"]

    spec_dir = Path("configs/oodc/merged")
    spec_dir.mkdir(parents=True, exist_ok=True)
    merged_path = spec_dir / f"{split}_{model_name}_seed{seed}.json"
    merged_path.write_text(json.dumps(train_specs, indent=2))

    cmd = [
        "bash", "code/scripts/generalist_v0_7_5/train_one.sh",
        model_name,
        str(merged_path),
        str(ckpt_dir),
        str(seed),
    ]
    print(f"  [train] {model_name} split={split} seed={seed} -> {ckpt_dir}")
    log_path = ckpt_dir / "train.log"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not final_pt.exists():
        raise RuntimeError(f"Training failed for {model_name} on {split}; see {log_path}")
    return final_pt


def measure_diagnostic_dmc(
    ckpt_path: str, env_id: str, env_path: str, n_steps: int = 200,
    seed: int = 0, device: str = "cpu",
) -> Dict[str, float]:
    from code.eval.closed_loop import make_env, _PadObsWrapper
    from code.scripts.utility.latent_goal_mpc import build_model_from_ckpt
    from code.core.encode import encode_obs as _encode_obs

    env_kind = resolve_env_kind(env_id)
    env = make_env(env_kind=env_kind, data_path=env_path)
    if hasattr(env, "seed"):
        env.seed(seed)

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    pad_obs_to = ck_args.get("pad_obs_to") or ck_args.get("state_dim") or env.spec.obs_dim
    action_dim = ck_args.get("action_dim") or env.spec.action_dim
    if pad_obs_to and pad_obs_to > env.spec.obs_dim:
        env = _PadObsWrapper(env, pad_obs_to)
    state_dim = pad_obs_to

    mdl = build_model_from_ckpt(ck_args, state_dim, action_dim, device)
    mdl.load_state_dict(ck["model"], strict=False)
    mdl.eval()

    cur_obs = _flat_obs(env.reset(seed=seed))
    obs_list, lat_list = [], []
    for t in range(n_steps):
        if hasattr(env, "action_space") and hasattr(env.action_space, "sample"):
            action = env.action_space.sample()
        else:
            action = np.random.uniform(
                env.spec.action_low, env.spec.action_high
            ).astype(np.float32)
        z = _encode_obs(mdl, torch.as_tensor(cur_obs, dtype=torch.float32),
                       action_dim, device)
        lat_list.append(z.detach().cpu().numpy())
        obs_list.append(cur_obs)
        step_out, _, done, _ = env.step(action)
        cur_obs = _flat_obs(step_out)
        if done:
            cur_obs = _flat_obs(env.reset(seed=seed))

    obs_arr = np.stack(obs_list)
    lat_arr = np.stack(lat_list)
    d_obs = np.diff(obs_arr, axis=0)
    d_lat = np.diff(lat_arr, axis=0)
    per_dim_std = lat_arr.std(axis=0)
    divergence = float(per_dim_std.mean())
    ratio = (
        np.linalg.norm(d_lat, axis=1)
        / (np.linalg.norm(d_obs, axis=1) + 1e-9)
    )
    responsiveness = float(ratio.mean())
    ndo = np.linalg.norm(d_obs, axis=1)
    ndl = np.linalg.norm(d_lat, axis=1)
    if ndo.std() > 1e-9 and ndl.std() > 1e-9:
        rho = float(np.corrcoef(ndo, ndl)[0, 1])
    else:
        rho = float("nan")
    return {
        "ckpt": str(ckpt_path),
        "env_id": env_id,
        "n_steps": int(n_steps),
        "divergence": round(divergence, 4),
        "responsiveness": round(responsiveness, 4),
        "rho": round(rho, 3) if not np.isnan(rho) else None,
        "per_dim_std_max": round(float(per_dim_std.max()), 4),
        "per_dim_std_min": round(float(per_dim_std.min()), 4),
    }


def measure_env_sr(
    ckpt_path: str, env_id: str, env_path: str, n_episodes: int = 3,
    seed: int = 0,
) -> Dict[str, float]:
    """Call closed_loop via subprocess; parse JSON from --out.

    Critical: env_kind must be lowercase (closed_loop dispatch table is
    lowercase, so "humanoid_CMU" fails but "humanoid_cmu" works).
    Stress envs need their specific CLI flag (--vel-hidden-mask-obs-ratio
    for cheetah_velhidden; --flicker-mask-ratio for flicker variants).
    For cartpole_2d/pendulum_2d we omit --goal-offset so closed_loop uses
    its per-env default (25) — passing --goal-offset 25 collides with the
    dataset's own goal_offset metadata and causes eval to fail.
    """
    out_json = Path(ckpt_path).parent / f"eval_{env_id}.json"
    env_kind_lower = env_id.lower() if env_id != "delayed_t_maze" else env_id
    cmd = [
        "/home/lx/miniconda3/envs/snn/bin/python", "-m", "code.eval.closed_loop",
        "--env", env_kind_lower,
        "--ckpt", str(ckpt_path),
        "--data", str(env_path),
        "--out", str(out_json),
        "--n-episodes", str(n_episodes),
        "--n-seeds", "1",
        "--cem-samples", "50",
        "--cem-elites", "5",
        "--cem-iters", "5",
        "--horizon", "5",
        "--eval-budget", "30",
        "--history-size", "1",
    ]
    # Per-env goal offset for envs that need it; skip for cartpole/pendulum
    # (closed_loop uses dataset's own goal_offset metadata there)
    if env_id not in ("cartpole_2d", "pendulum_2d"):
        cmd += ["--goal-offset", "25"]
    # Stress envs need their specific mask flag
    if env_id == "cheetah_velhidden":
        cmd += ["--vel-hidden-mask-obs-ratio", "0.0"]
    if env_id == "cartpole_flicker":
        cmd += ["--flicker-mask-ratio", "0.5"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return {"env_id": env_id, "env_sr": None, "env_sr_error": result.stderr[-500:]}
    except Exception as e:
        return {"env_id": env_id, "env_sr": None, "env_sr_error": f"{type(e).__name__}: {e}"}
    try:
        out = json.loads(out_json.read_text())
    except Exception as e:
        return {"env_id": env_id, "env_sr": None, "env_sr_error": f"read failed: {e}"}
    return {
        "env_id": env_id,
        "n_episodes": n_episodes,
        "env_sr": float(out.get("success_rate_env", out.get("success_rate", 0.0))),
        "env_sr_lewm": float(out.get("success_rate_lewm", 0.0)),
    }


def run_one_cell(
    ckpt: Path, env_id: str, env_path: str,
    seed: int, n_steps: int = 200, n_episodes: int = 3,
) -> Dict[str, Any]:
    out = {"ckpt": str(ckpt), "env_id": env_id, "env_path": env_path}
    try:
        out.update(measure_diagnostic_dmc(str(ckpt), env_id, env_path, n_steps=n_steps, seed=seed))
    except Exception as e:
        out["diagnostic_error"] = f"{type(e).__name__}: {e}"
    try:
        out.update(measure_env_sr(str(ckpt), env_id, env_path, n_episodes=n_episodes, seed=seed))
    except Exception as e:
        out["env_sr_error"] = f"{type(e).__name__}: {e}"
    return out


def aggregate(out_dir: str):
    out_root = Path(out_dir)
    cells = []
    for split_dir in sorted(out_root.iterdir()):
        if not split_dir.is_dir():
            continue
        for model_dir in sorted(split_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for seed_dir in sorted(model_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                for env_json in sorted(seed_dir.glob("*.json")):
                    if env_json.name.startswith("_") or env_json.name.startswith("eval_"):
                        continue
                    try:
                        d = json.loads(env_json.read_text())
                    except Exception:
                        continue
                    cells.append({
                        "split": split_dir.name,
                        "model": model_dir.name,
                        "seed": seed_dir.name,
                        "env_id": env_json.stem,
                        "div": d.get("divergence", float("nan")),
                        "resp": d.get("responsiveness", float("nan")),
                        "rho": d.get("rho", float("nan")),
                        "env_sr": d.get("env_sr", float("nan")),
                    })
    if not cells:
        print(f"No OODC cells found in {out_root}")
        return

    out_md = Path("results/utility/ood1_table.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w") as f:
        f.write("# OOD Path-C: 3-family DMC cross-sub-family transfer (v0.7.10b)\n\n")
        f.write("3 DMC sub-families: F1 classic control, F2 locomotion, F3 sparse-POMDP.\n")
        f.write("6 splits (3 OOD1: F1, F2, F3 trained; 3 OOD2: F1F2, F1F3, F2F3 trained).\n")
        f.write("12 ckpts per split, 1 seed, 2K windows/env, 3 episodes per held-out env.\n\n")
        f.write("Per-cell metric: `div` (latent per-dim std), `resp` (mean |delta-lat|/|delta-obs|), ")
        f.write("`rho` (corr ||delta-obs|| vs ||delta-lat||), `env_sr` (closed-loop success rate).\n\n")
        f.write(f"Total: {len(cells)} ckpt x env cells.\n\n")
        f.write("## Per-cell\n\n")
        f.write("| split | model | env | div | resp | rho | env_sr |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for c in cells:
            def f1(x):
                if isinstance(x, float):
                    if x != x:
                        return "nan"
                    return f"{x:.4f}"
                return str(x) if x is not None else "nan"
            def f2(x):
                if isinstance(x, float):
                    if x != x:
                        return "nan"
                    return f"{x:.3f}"
                return str(x) if x is not None else "nan"
            f.write(f"| {c['split']} | {c['model']} | {c['env_id']} | "
                    f"{f1(c['div'])} | {f2(c['resp'])} | {f2(c['rho'])} | {f2(c['env_sr'])} |\n")

        from collections import defaultdict
        agg = defaultdict(list)
        for c in cells:
            agg[(c["split"], c["model"])].append((c["div"], c["resp"], c["rho"], c["env_sr"]))
        f.write("\n## Mean per (split, model) over held-out envs\n\n")
        f.write("| split | model | n_envs | mean_div | mean_resp | mean_rho | mean_env_sr |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for k, vs in sorted(agg.items()):
            ds = [v[0] for v in vs if isinstance(v[0], float) and v[0] == v[0]]
            rs = [v[1] for v in vs if isinstance(v[1], float) and v[1] == v[1]]
            hos = [v[2] for v in vs if isinstance(v[2], float) and v[2] == v[2]]
            srs = [v[3] for v in vs if isinstance(v[3], float) and v[3] == v[3]]
            def avg(lst):
                return f"{sum(lst)/len(lst):.4f}" if lst else "nan"
            f.write(f"| {k[0]} | {k[1]} | {len(vs)} | {avg(ds)} | {avg(rs)} | {avg(hos)} | {avg(srs)} |\n")
        f.write("\n## Per-split, per-family mean\n\n")
        f.write("STJEWM = trace, spike, rate, no_trace, hidden_leak, membrane_readout.\n")
        f.write("SNN-baselines = alif_timecell, stacked_lif_trace, stacked_lif_free.\n")
        f.write("non-SNN baselines = mlp, gru, lewm.\n\n")
        f.write("| split | family | n_cells | mean_div | mean_resp | mean_rho | mean_env_sr |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        family_assign = {
            "stjewm_trace_only": "STJEWM",
            "stjewm_spike_only": "STJEWM",
            "stjewm_rate_only": "STJEWM",
            "stjewm_no_trace": "STJEWM",
            "stjewm_hidden_leak": "STJEWM",
            "stjewm_membrane_readout": "STJEWM",
            "alif_timecell_baseline": "SNN-baselines",
            "stacked_lif_trace": "SNN-baselines",
            "stacked_lif_free": "SNN-baselines",
            "mlp_baseline": "non-SNN",
            "gru_baseline": "non-SNN",
            "lewm_baseline_v2": "non-SNN",
        }
        fam_agg = defaultdict(list)
        for c in cells:
            fam = family_assign.get(c["model"], "other")
            fam_agg[(c["split"], fam)].append((c["div"], c["resp"], c["rho"], c["env_sr"]))
        for k, vs in sorted(fam_agg.items()):
            ds = [v[0] for v in vs if isinstance(v[0], float) and v[0] == v[0]]
            rs = [v[1] for v in vs if isinstance(v[1], float) and v[1] == v[1]]
            hos = [v[2] for v in vs if isinstance(v[2], float) and v[2] == v[2]]
            srs = [v[3] for v in vs if isinstance(v[3], float) and v[3] == v[3]]
            def avg(lst):
                return f"{sum(lst)/len(lst):.4f}" if lst else "nan"
            f.write(f"| {k[0]} | {k[1]} | {len(vs)} | {avg(ds)} | {avg(rs)} | {avg(hos)} | {avg(srs)} |\n")

    print(f"Wrote {out_md}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=SPLIT_NAMES)
    ap.add_argument("--models", nargs="*", default=DEFAULT_CKPT_BUDGET)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--out-dir", default="results/ood1")
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--n-episodes", type=int, default=3)
    args = ap.parse_args()

    if not args.aggregate_only:
        for split in args.splits:
            spec_path = Path(f"configs/oodc/{split}.json")
            if not spec_path.exists():
                print(f"[skip] {split}: spec not found")
                continue
            spec = json.loads(spec_path.read_text())
            eval_specs = spec["eval_specs"]
            print(f"=== split {split}: train={spec['train_envs']} held-out={spec['eval_envs']} ===")
            for model in args.models:
                if not args.skip_train:
                    ckpt = train_one_ckpt(split, model, args.seed, args.out_dir)
                else:
                    ckpt = Path(args.out_dir) / split / model / f"seed_{args.seed}" / "final.pt"
                for env_entry in eval_specs:
                    env_id = env_entry["env_id"]
                    env_path = env_entry["path"]
                    cell = run_one_cell(ckpt, env_id, env_path, args.seed,
                                          n_steps=args.n_steps, n_episodes=args.n_episodes)
                    out_json = Path(args.out_dir) / split / model / f"seed_{args.seed}" / f"{env_id}.json"
                    out_json.parent.mkdir(parents=True, exist_ok=True)
                    out_json.write_text(json.dumps(cell, indent=2))
                    print(f"  [{model}/{env_id}] div={cell.get('divergence', 'NA')} "
                          f"resp={cell.get('responsiveness', 'NA')} rho={cell.get('rho', 'NA')} "
                          f"env_sr={cell.get('env_sr', 'NA')}")

    aggregate(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

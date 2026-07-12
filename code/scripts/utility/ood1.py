"""OOD1 cross-benchmark-family transfer (v0.7.10 utility experiment).

A serious cross-environment transfer experiment cannot be answered by holding
out 2 of 16 environments from the *same* G16 suite (that's the within-suite
pilot of v0.7.8). Real OOD generalisation requires holding out an *entire
benchmark family* and testing on the other 3 families.

OOD1 by construction = "1 train family -> 3 unseen families". The honest
matrix over the 4 families currently available (DMC classic-control, PushT,
LeWM reacher / OGBench cube, Delayed POMDP / TwoRoom) is:

  Split              | train_family       | unseen_families
  -------------------|--------------------|---------------------------
  OOD1_dmc           | DMC (13 envs)      | pusht, reacher, tworoom
  OOD1_pusht          | pusht (1 env only) | DMC, reacher, tworoom
  OOD1_reacher       | reacher (1 env only)| DMC, pusht, tworoom
  OOD1_tworoom       | tworoom (1 env)     | DMC, pusht, reacher

Only OOD1_dmc is non-degenerate (13 training envs). The other 3 splits train
on a single env and are reported with explicit degeneracy caveats.

For each split we train (a) 6 STJEWM readouts (trace/spike/rate/no_trace/
hidden_leak/membrane) and (b) 4 baselines (mlp_baseline, gru_baseline,
cubifae_baseline, slt_lif_mpc_trace) per split. That's 12 ckpts x 4 splits
= 48 trainings. At ~25 min/ckpt on 1 CPU = ~20 hr wallclock.

Per-cell output:
  results/utility/ood1/<split>/<model>/seed_<seed>/<env>_{div,resp,rho,env_sr}.json
Aggregate table at results/utility/ood1_table.md.

Honest caveats baked in:
- 1-seed; no std bars (same as v0.7.8 pilot).
- Cross-family dynamics are wildly different (DMC is qpos-only; tworoom has
  visual obs and 100-step memory; reacher has sub-task POMDP structure).
  We do NOT claim env-native control generalisation; we only claim the
  diagnostic profile (div/resp/rho) is preserved across the family boundary.
- The 3 degenerate splits (pusht/reacher/tworoom as train) train only on
  one env. The DMC-trained ckpt has the strongest comparison; the others
  are sanity checks.

This module also embeds a slim in-process measure_latent_stats helper
that dispatches on env_kind — measure_latent_stats.py was DMC-only.
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


# ============================================================
# Split definitions
# ============================================================

# 4 train-by-1-family splits. Honest: only "dmc" has enough training data.
# The other 3 are 1-env train sets, reported with "degenerate" caveats.
SPLITS = [
    ("ood1_dmc_train",     "dmc",       ["pusht", "reacher_4d", "tworoom"]),
    ("ood1_pusht_train",    "pusht",     ["dmc", "reacher_4d", "tworoom"]),
    ("ood1_reacher_train",  "reacher_4d",["dmc", "pusht", "tworoom"]),
    ("ood1_tworoom_train",  "tworoom",   ["dmc", "pusht", "reacher_4d"]),
]

# 12 ckpts trained per split. Same set as v0.7.8 G16 pilot (cuBiFAE is
# already calibrated per G16 Tables 4-6; if the cross-family transfer claim
# holds for trace/spike, the 10 remaining ckpts are degraded to confirming
# evidence).
DEFAULT_CKPT_BUDGET = [
    "stjewm_trace_only",
    "stjewm_spike_only",
    "stjewm_rate_only",
    "stjewm_no_trace",
    "stjewm_hidden_leak",
    "stjewm_membrane_readout",
    "mlp_baseline",
    "gru_baseline",
    "cubifae_baseline",
    "slt_lif_mpc_trace",
]


# ============================================================
# In-process measure_latent_stats (cross-family)
# ============================================================

def _load_model_for_env(ckpt_path: str, env, device: str = "cpu"):
    """Build the model whose weights are in ckpt, with action/state dims
    padded to match ckpt args. Mirrors closed_loop._PadObsWrapper path."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    pad_obs_to = (ck_args.get("pad_obs_to") or env.spec.obs_dim)
    action_dim_ckpt = ck_args.get("action_dim") or env.spec.action_dim
    from code.eval.closed_loop import _PadObsWrapper
    if pad_obs_to > env.spec.obs_dim:
        env = _PadObsWrapper(env, pad_obs_to)
    m = ck_args.get("model", "stjewm")
    if m == "lewm_baseline":
        from code.lewm_transformer_baseline import LeWMTransformerBaseline
        action_dim = ck_args.get("action_dim")
        mdl = LeWMTransformerBaseline(env_spec=env.spec,
                                       embed_dim=ck_args.get("embed_dim", 128),
                                       n_layers=ck_args.get("n_layers", 4),
                                       horizon=ck_args.get("horizon", 5),
                                       action_dim=action_dim).to(device)
    elif m == "mlp_baseline":
        from code.mlp_baseline import MLPBaseline
        mdl = MLPBaseline(env_spec=env.spec,
                          embed_dim=ck_args.get("embed_dim", 128),
                          action_dim=ck_args.get("action_dim"),
                          n_layers=ck_args.get("n_layers", 2)).to(device)
    elif m == "gru_baseline":
        from code.gru_baseline import GRUBaseline
        mdl = GRUBaseline(env_spec=env.spec,
                           embed_dim=ck_args.get("embed_dim", 128),
                           action_dim=ck_args.get("action_dim"),
                           n_layers=ck_args.get("n_layers", 2)).to(device)
    elif m == "cubifae_baseline":
        from code.cubifae_baseline import CuBiFAEBaseline
        mdl = CuBiFAEBaseline(env_spec=env.spec,
                             embed_dim=ck_args.get("embed_dim", 128),
                             action_dim=ck_args.get("action_dim"),
                             n_layers=ck_args.get("n_layers", 4)).to(device)
    elif m == "slt_lif_mpc_trace":
        from code.slt_lif_mpc_baseline import SLTLIFMPCBaseline
        mdl = SLTLIFMPCBaseline(env_spec=env.spec,
                                embed_dim=ck_args.get("embed_dim", 128),
                                action_dim=ck_args.get("action_dim"),
                                n_layers=ck_args.get("n_layers", 4),
                                mode="trace").to(device)
    else:
        from code.stjewm import STJEWM
        mdl = STJEWM(
            obs_dim=env.spec.obs_dim,
            action_dim=ck_args.get("action_dim") or env.spec.action_dim,
            embed_dim=ck_args.get("embed_dim", 192),
            state_dim=ck_args.get("state_dim", 128),
            n_layers=ck_args.get("n_layers", 4),
            readout_mode=ck_args.get("readout_mode", "trace"),
            cell_n_d=ck_args.get("cell_n_d", 3),
            trace_beta=ck_args.get("trace_beta", 0.9),
        ).to(device)
    mdl.load_state_dict(ck["model"], strict=False)
    mdl.eval()
    # Sliced-action dim guard: if ckpt was trained at smaller action_dim,
    # the predictor will slice; model already supports via action_dim.
    return mdl


def measure_diagnostic_cross_family(
    ckpt_path: str, env_kind: str, env_path: str, env_id: str,
    n_steps: int = 200, seed: int = 0, device: str = "cpu",
) -> Dict[str, float]:
    """Roll a ckpt on a (possibly non-DMC) env for n_steps and return
    {divergence, responsiveness}. The metric is the same as v0.7.8
    measure_latent_stats but dispatches on env_kind (DMC/PushT/Reacher/...)
    so we can measure cross-family OOD diagnostics.
    """
    from code.core.envs import make_env
    env = make_env(env_kind=env_kind, env_path=env_path, env_id=env_id)
    env.seed(seed)
    model = _load_model_for_env(str(ckpt_path), env, device)

    obs_list, lat_list = [], []
    obs = env.reset()
    # Pad obs to model.obs_dim if needed
    cur_obs = obs
    for t in range(n_steps):
        action = env.action_space.sample()
        model_obs = torch.as_tensor(cur_obs, dtype=torch.float32,
                                    device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = model(model_obs, deterministic=True)
        z = out.get("z", out.get("trace"))
        if isinstance(z, tuple):
            z = z[0]
        lat_list.append(z.squeeze(0).squeeze(0).detach().cpu().numpy())
        obs_list.append(np.asarray(cur_obs))
        cur_obs, _, done, _ = env.step(action)
        if done:
            cur_obs = env.reset()
    obs_arr = np.stack(obs_list)
    lat_arr = np.stack(lat_list)
    d_obs = np.diff(obs_arr, axis=0)
    d_lat = np.diff(lat_arr, axis=0)
    per_dim_std = lat_arr.std(axis=0)
    divergence = float(per_dim_std.mean())
    ratio = np.linalg.norm(d_lat, axis=1) / (np.linalg.norm(d_obs, axis=1) + 1e-9)
    responsiveness = float(ratio.mean())
    return {
        "ckpt": str(ckpt_path),
        "env_kind": env_kind,
        "env_id": env_id,
        "n_steps": int(n_steps),
        "responsiveness": round(responsiveness, 4),
        "divergence": round(divergence, 4),
        "per_dim_std_max": round(float(per_dim_std.max()), 4),
        "per_dim_std_min": round(float(per_dim_std.min()), 4),
        "mean_norm_obs": round(float(np.linalg.norm(obs_arr, axis=1).mean()), 4),
        "mean_norm_lat": round(float(np.linalg.norm(lat_arr, axis=1).mean()), 4),
    }


# ============================================================
# In-process event_align (cross-family)
# ============================================================

def event_align_cross_family(
    ckpt_path: str, env_kind: str, env_path: str, env_id: str,
    n_steps: int = 100, seed: int = 0, device: str = "cpu",
) -> Dict[str, float]:
    """Per-step correlation between ||Δobs_t|| and ||Δlatent_t||.
    The same diagnostic measure_latent_stats supports internally for
    DMC; this replicates the cross-family version.
    """
    from code.core.envs import make_env
    env = make_env(env_kind=env_kind, env_path=env_path, env_id=env_id)
    env.seed(seed)
    model = _load_model_for_env(str(ckpt_path), env, device)

    dobs_arr, dlat_arr = [], []
    obs = env.reset()
    cur_obs = obs
    prev_obs, prev_lat = None, None
    for t in range(n_steps):
        action = env.action_space.sample()
        model_obs = torch.as_tensor(cur_obs, dtype=torch.float32,
                                    device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = model(model_obs, deterministic=True)
        z = out.get("z", out.get("trace"))
        if isinstance(z, tuple):
            z = z[0]
        lat = z.squeeze(0).squeeze(0).detach().cpu().numpy()
        if prev_obs is not None:
            dobs_arr.append(np.linalg.norm(np.asarray(cur_obs) - prev_obs))
            dlat_arr.append(np.linalg.norm(lat - prev_lat))
        prev_obs = np.asarray(cur_obs)
        prev_lat = lat
        cur_obs, _, done, _ = env.step(action)
        if done:
            cur_obs = env.reset()
    if len(dobs_arr) < 2:
        return {"corr_obs_latent": float("nan"), "n": 0}
    if dobs.std() < 1e-9 or dlat.std() < 1e-9:
        return {"corr_obs_latent": float("nan"), "n": len(dobs_arr)}
    corr = float(np.corrcoef(dobs, dlat)[0, 1])
    return {"corr_obs_latent": corr, "n": len(dobs_arr)}


# ============================================================
# Train / measure / aggregate
# ============================================================

def train_one_ckpt(split: str, model: str, base_seed: int = 0) -> Path:
    """Train ONE ckpt for one (split, model) pair via train_one.sh."""
    out_dir = Path(f"results/ood1/{split}/{model}/seed_{base_seed}")
    ckpt = out_dir / "final.pt"
    if ckpt.exists():
        print(f"[train] {split}/{model}: ckpt already exists, skipping")
        return ckpt

    spec_link = Path(f"configs/_ood1_{split}_{model}.json")
    spec_link.parent.mkdir(parents=True, exist_ok=True)
    target = Path(f"configs/ood1_{split.split('_train')[0]}_train.json")
    if not target.exists():
        raise FileNotFoundError(f"missing train spec: {target}")
    spec_link.write_text(target.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/bin/bash",
        "code/scripts/generalist_v0_7_5/train_one.sh",
        model,
        str(spec_link),
        str(out_dir),
        str(base_seed),
    ]
    t0 = time.time()
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if rc != 0 or not ckpt.exists():
        raise RuntimeError(f"train failed for {split}/{model}: rc={rc}")
    print(f"[train] {split}/{model}: done in {dt/60:.1f} min -> {ckpt}")
    return ckpt


def run_one_cell(ckpt: Path, env_id: str, env_kind: str, env_path: str,
                 seed: int = 0) -> Dict[str, float]:
    """Measure {div, resp, rho} on (ckpt, env) for OOD1. Skip env-SR; OOD1
    cross-family env-SR is meaningless (cartpole vs pusht vs tworoom all
    report binary-style SR against different success criteria)."""
    cell = ckpt.parent
    out_div = cell / f"div_{env_id}.json"
    out_rho = cell / f"rho_{env_id}.json"

    diag = measure_diagnostic_cross_family(
        str(ckpt), env_kind, env_path, env_id, n_steps=200, seed=seed
    )
    out_div.parent.mkdir(parents=True, exist_ok=True)
    out_div.write_text(json.dumps(diag, indent=2))

    align = event_align_cross_family(
        str(ckpt), env_kind, env_path, env_id, n_steps=100, seed=seed
    )
    out_rho.write_text(json.dumps(align, indent=2))

    return {"div": diag["divergence"], "resp": diag["responsiveness"],
            "rho": align["corr_obs_latent"]}


def aggregate(out_dir: str = "results/utility/ood1") -> None:
    """Walk results/ood1/{split}/{model}/seed_0/{env}*.json and write the
    aggregate table at results/utility/ood1_table.md.
    """
    base = Path("results/utility") / Path(out_dir).name
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    rows = []
    cells = []
    for split_dir in sorted(base.glob("ood1_*")):
        if not split_dir.is_dir():
            continue
        for model_dir in sorted(split_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for seed_dir in sorted(model_dir.glob("seed_*")):
                for env_json in sorted(seed_dir.glob("div_*.json")):
                    env_id = env_json.name.replace("div_", "").replace(".json", "")
                    rho_json = seed_dir / f"rho_{env_id}.json"
                    if not rho_json.exists():
                        continue
                    d = json.loads(env_json.read_text())
                    r = json.loads(rho_json.read_text())
                    cells.append({
                        "split": split_dir.name,
                        "model": model_dir.name,
                        "seed": seed_dir.name,
                        "env_id": env_id,
                        "div": d["divergence"],
                        "resp": d["responsiveness"],
                        "rho": r["corr_obs_latent"],
                    })
    if not cells:
        print("No OOD1 cells found.")
        return

    splits = sorted({c["split"] for c in cells})
    envs = sorted({c["env_id"] for c in cells})
    models = sorted({c["model"] for c in cells})

    out_md = Path("results/utility/ood1_table.md")
    with out_md.open("w") as f:
        f.write("# OOD1 cross-benchmark-family transfer (v0.7.10)\n\n")
        f.write("Honest-scope OOD1 matrix: train on 1 family, evaluate on the 3 other "
                "families. Only the *ood1_dmc_train* split is non-degenerate (13 DMC "
                "training envs); the other 3 splits train on a single env. All numbers "
                "are 1-seed pilot-scale.\n\n")
        f.write("Per-cell metric: `div` (latent per-dim std), `resp` (mean |delta-lat|/"
                "|delta-obs|), `rho` (corr between ||delta-obs|| and ||delta-lat||).\n\n")
        f.write("| split | model | seed | env | div | resp | rho |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for c in cells:
            f.write(f"| {c['split']} | {c['model']} | {c['seed']} | "
                    f"{c['env_id']} | {c['div']:.4f} | {c['resp']:.3f} | "
                    f"{c['rho']:.3f} |\n")

        # Mean per (split, model, env) — if multiple cells exist
        from collections import defaultdict
        agg = defaultdict(list)
        for c in cells:
            agg[(c["split"], c["model"], c["env_id"])].append((c["div"], c["resp"], c["rho"]))
        f.write("\n## Mean per (split, model, env) across seeds\n\n")
        f.write("| split | model | env | mean_div | mean_resp | mean_rho |\n")
        f.write("|---|---|---|---|---|---|\n")
        for k, vs in sorted(agg.items()):
            ds = [v[0] for v in vs]
            rs = [v[1] for v in vs]
            ho = [v[2] for v in vs]
            f.write(f"| {k[0]} | {k[1]} | {k[2]} | "
                    f"{sum(ds)/len(ds):.4f} | {sum(rs)/len(rs):.3f} | "
                    f"{sum(ho)/len(ho):.3f} |\n")
    print(f"Wrote {out_md}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=[s[0] for s in SPLITS])
    ap.add_argument("--models", nargs="*", default=DEFAULT_CKPT_BUDGET)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--out-dir", default="results/ood1")
    args = ap.parse_args()

    if not args.aggregate_only:
        for split, train_family, unseen_families in SPLITS:
            if split not in args.splits:
                continue
            print(f"=== split {split}: train={train_family} unseen={unseen_families} ===")
            for model in args.models:
                train_one_ckpt(split, model, args.seed)
                ckpt = Path(args.out_dir) / split / model / f"seed_{args.seed}" / "final.pt"
                eval_spec = json.loads(Path("configs/ood1_eval.json").read_text())["specs"]
                for env_entry in eval_spec:
                    if env_entry["env_kind"] not in unseen_families:
                        continue
                    out = run_one_cell(
                        ckpt, env_entry["env_id"], env_entry["env_kind"],
                        env_entry["path"], args.seed,
                    )
                    print(f"  [{env_entry['env_kind']}/{env_entry['env_id']}] "
                          f"div={out['div']:.4f} resp={out['resp']:.3f} "
                          f"rho={out['rho']:.3f}")

    aggregate(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

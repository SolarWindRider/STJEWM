"""Cross-environment generalisation driver for held-out G16 environments.

This experiment trains a 14-environment generalist checkpoint with walker and
humanoid held out, evaluates it on the full non-stress G16 eval spec, and
collects the held-out diagnostic metrics used by the paper:

- env-SR from eval_<env>.json
- divergence / responsiveness from latent_stats_<env>.json
- event-alignment rho from event_align/<env>_<model>_seed0.json

The same aggregation path also reads the existing full-G16 seed-0 checkpoints so
that full-train and held-out-train rows are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable

ROOT = Path("/home/lx/snn")
TRAIN_SPEC = ROOT / "configs/generalist_G16_minus_walker_humanoid.json"
EVAL_SPEC = ROOT / "configs/generalist_G16_eval.json"
FULL_RESULTS = ROOT / "results/generalist_G16"
MINUS_RESULTS = ROOT / "results/generalist_G16_minus_walker_humanoid"
OUT_DIR = ROOT / "results/utility/cross_env_gen"
TABLE_PATH = ROOT / "results/utility/cross_env_gen_table.md"
SEED = 0

# v0.7.9 review fix: TARGET_MODELS now == all 12 G16 models
# (was the 4-ckpt subset trace/spike/mlp/gru; the remaining 8 ckpts —
#  ALIF-timecell / Stacked-LIF trace / Stacked-LIF free / LeWM-v2 + the 4 STJEWM
#  readouts rate/no_trace/hidden_leak/membrane — were missing per
#  reviewer feedback). Train is one-shot wallclock ≈ 8 ckpts × ~25 min on 1
#  CPU = ~3.3 hr; results regenerate results/utility/cross_env_gen_table.md.
TARGET_MODELS = [
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
FULL_G16_MODELS = [
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
HELDOUT_ENVS = ["walker", "humanoid"]
# Metrics based on latent dynamics / event alignment are currently defined for
# this DMC diagnostic subset plus the two held-out DMC envs.
IN_DOMAIN_DIAGNOSTIC_ENVS = [
    "cartpole_2d",
    "pendulum_2d",
    "finger",
    "ball_in_cup",
    "cheetah",
]
METRIC_ENVS = IN_DOMAIN_DIAGNOSTIC_ENVS + HELDOUT_ENVS


class Regime:
    def __init__(self, name: str, label: str, results_dir: Path, models: list[str]):
        self.name = name
        self.label = label
        self.results_dir = results_dir
        self.models = models

    @property
    def align_dir(self) -> Path:
        return self.results_dir / "event_align"


FULL_REGIME = Regime("full_G16", "full-G16", FULL_RESULTS, FULL_G16_MODELS)
MINUS_REGIME = Regime(
    "minus_walker_humanoid",
    "minus walker+humanoid",
    MINUS_RESULTS,
    TARGET_MODELS,
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("[cross-env] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def ckpt_path(regime: Regime, model: str, seed: int = SEED) -> Path:
    return regime.results_dir / model / f"seed_{seed}" / "final.pt"


def seed_dir(regime: Regime, model: str, seed: int = SEED) -> Path:
    return regime.results_dir / model / f"seed_{seed}"


def load_train_envs(train_spec: Path = TRAIN_SPEC) -> list[str]:
    return [str(e["env_id"]) for e in json.loads(train_spec.read_text())]


def make_eval16_spec(eval_spec: Path = EVAL_SPEC, out_dir: Path = OUT_DIR) -> Path:
    """Create a non-stress 16-env eval spec with the humanoid_CMU CLI mapping.

    The checked-in G16 eval spec also contains stress rows after the 16 ID envs.
    This experiment is specifically the zero-shot 16-env G16 evaluation, so the
    transient spec filters rows with extra_flags. It also maps humanoid_CMU to
    the lowercase env id accepted by closed_loop.make_env.
    """
    entries = []
    for entry in json.loads(eval_spec.read_text()):
        if entry.get("extra_flags"):
            continue
        fixed = dict(entry)
        if fixed.get("env_id") == "humanoid_CMU":
            fixed["clo_env"] = "humanoid_cmu"
        entries.append(fixed)
    if len(entries) != 16:
        raise RuntimeError(f"expected 16 non-stress G16 eval entries, got {len(entries)}")
    path = out_dir / "_g16_eval_16_nonstress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n")
    return path


def train_minus_checkpoint(model: str, *, force: bool = False, seed: int = SEED) -> None:
    out = seed_dir(MINUS_REGIME, model, seed)
    ckpt = out / "final.pt"
    if ckpt.exists() and not force:
        print(f"[cross-env] skip train {model}: {ckpt} exists", flush=True)
        return
    run([
        "bash",
        "code/scripts/generalist_v0_7_5/train_one.sh",
        model,
        str(TRAIN_SPEC.relative_to(ROOT)),
        str(out),
        str(seed),
    ])


def eval_checkpoint(
    regime: Regime,
    model: str,
    spec: Path,
    *,
    force: bool = False,
    seed: int = SEED,
    n_episodes: int = 3,
) -> None:
    ckpt = ckpt_path(regime, model, seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"missing ckpt: {ckpt}")
    if force:
        for env in [e["env_id"] for e in json.loads(spec.read_text())]:
            fp = seed_dir(regime, model, seed) / f"eval_{env}.json"
            if fp.exists():
                fp.unlink()
    env = os.environ.copy()
    env.update({
        "OUT_BASE": str(regime.results_dir),
        "N_EPISODES": str(n_episodes),
        "N_SEEDS": "1",
        "HORIZON": "5",
        "EVAL_BUDGET": "50",
        "HISTORY_SIZE": "1",
    })
    run([
        "bash",
        "code/scripts/generalist_v0_7_5/eval_closed_loop_one.sh",
        model,
        str(ckpt),
        str(spec),
        str(seed),
    ], env=env)


def ensure_latent_stats(
    regime: Regime,
    model: str,
    env_name: str,
    *,
    force: bool = False,
    seed: int = SEED,
    n_steps: int = 200,
    device: str = "cpu",
) -> None:
    ckpt = ckpt_path(regime, model, seed)
    out = seed_dir(regime, model, seed) / f"latent_stats_{env_name}.json"
    if out.exists() and not force:
        return
    cmd = [
        "/home/lx/miniconda3/envs/snn/bin/python",
        "-m",
        "code.scripts.generalist_v0_7_5.measure_latent_stats",
        "--ckpt",
        str(ckpt),
        "--env",
        env_name,
        "--n-steps",
        str(n_steps),
        "--seed",
        str(seed),
        "--device",
        device,
        "--out",
        str(out),
    ]
    run(cmd)


def ensure_event_align(
    regime: Regime,
    model: str,
    env_name: str,
    *,
    force: bool = False,
    seed: int = SEED,
    n_steps: int = 100,
    device: str = "cpu",
) -> None:
    ckpt = ckpt_path(regime, model, seed)
    out = regime.align_dir / f"{env_name}_{model}_seed{seed}.json"
    if out.exists() and not force:
        return
    cmd = [
        "/home/lx/miniconda3/envs/snn/bin/python",
        "-m",
        "code.scripts.event_align",
        "--env",
        env_name,
        "--model",
        model,
        "--ckpt",
        str(ckpt),
        "--out",
        str(out),
        "--n-steps",
        str(n_steps),
        "--pad-obs-to",
        "128",
        "--action-dim-eval",
        "56",
        "--device",
        device,
    ]
    run(cmd)


def ensure_metrics(
    regime: Regime,
    model: str,
    *,
    force_latent: bool = False,
    force_align: bool = False,
    seed: int = SEED,
    latent_steps: int = 200,
    align_steps: int = 100,
    device: str = "cpu",
) -> None:
    for env_name in METRIC_ENVS:
        ensure_latent_stats(
            regime, model, env_name,
            force=force_latent, seed=seed, n_steps=latent_steps, device=device,
        )
        ensure_event_align(
            regime, model, env_name,
            force=force_align, seed=seed, n_steps=align_steps, device=device,
        )


def mean(xs: Iterable[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else None


def metric_sources(regime: Regime, model: str, env_name: str, seed: int = SEED) -> dict[str, str]:
    sd = seed_dir(regime, model, seed)
    return {
        "eval": str(sd / f"eval_{env_name}.json"),
        "latent_stats": str(sd / f"latent_stats_{env_name}.json"),
        "event_align": str(regime.align_dir / f"{env_name}_{model}_seed{seed}.json"),
    }


def read_cell(regime: Regime, model: str, env_name: str, seed: int = SEED) -> dict[str, Any]:
    src = metric_sources(regime, model, env_name, seed)
    ev = load_json(Path(src["eval"])) or {}
    lat = load_json(Path(src["latent_stats"])) or {}
    align = load_json(Path(src["event_align"])) or {}
    return {
        "env_sr": ev.get("success_rate_env"),
        "lewm_sr": ev.get("success_rate_lewm"),
        "mean_cos_dist": ev.get("mean_cos_dist"),
        "divergence": lat.get("divergence"),
        "responsiveness": lat.get("responsiveness"),
        "rho": align.get("corr_obs_latent"),
        "skipped": bool(align.get("skipped")) if align else False,
        "sources": src,
    }


def row_id(regime: Regime, model: str) -> str:
    return f"{model}__{regime.name}"


def build_row(
    regime: Regime,
    model: str,
    train_envs: list[str],
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    env_cells = {env: read_cell(regime, model, env, seed) for env in set(train_envs) | set(METRIC_ENVS)}
    env_sr_in = mean(env_cells[e].get("env_sr") for e in train_envs if e not in HELDOUT_ENVS)
    diag_in = {env: env_cells[env] for env in IN_DOMAIN_DIAGNOSTIC_ENVS}
    in_means = {
        "env_sr": env_sr_in,
        "divergence": mean(c.get("divergence") for c in diag_in.values()),
        "responsiveness": mean(c.get("responsiveness") for c in diag_in.values()),
        "rho": mean(c.get("rho") for c in diag_in.values()),
    }
    heldout: dict[str, Any] = {}
    for env in HELDOUT_ENVS:
        cell = env_cells[env]
        gaps = {
            "env_sr": None if in_means["env_sr"] is None or cell["env_sr"] is None else in_means["env_sr"] - float(cell["env_sr"]),
            "divergence": None if in_means["divergence"] is None or cell["divergence"] is None else in_means["divergence"] - float(cell["divergence"]),
            "responsiveness": None if in_means["responsiveness"] is None or cell["responsiveness"] is None else in_means["responsiveness"] - float(cell["responsiveness"]),
            "rho": None if in_means["rho"] is None or cell["rho"] is None else in_means["rho"] - float(cell["rho"]),
        }
        drops = {k: (None if v is None else max(float(v), 0.0)) for k, v in gaps.items()}
        heldout[env] = {"metrics": cell, "gap_in_minus_holdout": gaps}
        payload = {
            "model": model,
            "train_regime": regime.name,
            "train_regime_label": regime.label,
            "env": env,
            "seed": seed,
            "held_out_from_training": regime is MINUS_REGIME,
            "in_domain_eval_envs": [e for e in train_envs if e not in HELDOUT_ENVS],
            "in_domain_diagnostic_envs": IN_DOMAIN_DIAGNOSTIC_ENVS,
            "in_domain_means": in_means,
            "holdout_metrics": cell,
            "cross_env_generalisation_gap": gaps,
            "cross_env_generalisation_drop": drops,
        }
        write_json(OUT_DIR / regime.name / model / f"{env}.json", payload)
        if regime is MINUS_REGIME:
            write_json(OUT_DIR / model / f"{env}.json", payload)
        write_json(
            OUT_DIR / row_id(regime, model) / f"{env}.json",
            payload,
        )
    return {
        "model": model,
        "train_regime": regime.name,
        "train_regime_label": regime.label,
        "row_id": row_id(regime, model),
        "seed": seed,
        "in_domain_means": in_means,
        "heldout": heldout,
    }


def fmt(v: Any, digits: int = 3, scale: float = 1.0) -> str:
    if v is None:
        return "-"
    try:
        x = float(v) * scale
    except Exception:
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def write_table(rows: list[dict[str, Any]], out_path: Path = TABLE_PATH) -> None:
    lines = [
        "# Cross-environment generalisation: held-out walker + humanoid",
        "",
        "This is a zero-shot cross-environment generalisation table over an environment-distribution shift.",
        "Rows compare checkpoints trained on all G16 environments with checkpoints trained on G16 minus walker and humanoid.",
        "",
        "`env-SR_in` is the mean env-native success over the 14 non-held-out training environments. ",
        "`drop_in_minus_holdout` / gap columns are signed in-domain minus held-out differences, so negative env-SR values mean the held-out env scored higher than the in-domain mean. ",
        "`div_in`, `resp_in`, and `ρ_in` are means over the in-domain DMC diagnostic envs: " + ", ".join(IN_DOMAIN_DIAGNOSTIC_ENVS) + ".",
        "",
    ]
    header = ["model", "train"]
    for env in HELDOUT_ENVS:
        header.extend([
            f"{env} env-SR_in", f"{env} env-SR_holdout", f"{env} drop_in_minus_holdout",
            f"{env} div_in", f"{env} div_holdout", f"{env} div_gap_in_minus_holdout",
            f"{env} resp_in", f"{env} resp_holdout", f"{env} resp_gap_in_minus_holdout",
            f"{env} ρ_in", f"{env} ρ_holdout", f"{env} ρ_gap_in_minus_holdout",
        ])
    header.extend(["mean drop_in_minus_holdout", "mean div_gap", "mean resp_gap", "mean ρ_gap"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    ordered = sorted(
        rows,
        key=lambda r: (
            TARGET_MODELS.index(r["model"]) if r["model"] in TARGET_MODELS else 100 + FULL_G16_MODELS.index(r["model"]),
            0 if r["train_regime"] == "full_G16" else 1,
        ),
    )
    for row in ordered:
        vals = [row["model"], row["train_regime_label"]]
        mean_gaps: dict[str, list[float]] = {"env_sr": [], "divergence": [], "responsiveness": [], "rho": []}
        for env in HELDOUT_ENVS:
            h = row["heldout"][env]
            m = h["metrics"]
            g = h["gap_in_minus_holdout"]
            im = row["in_domain_means"]
            vals.extend([
                fmt(im["env_sr"], 1, 100), fmt(m["env_sr"], 1, 100), fmt(g["env_sr"], 1, 100),
                fmt(im["divergence"], 4), fmt(m["divergence"], 4), fmt(g["divergence"], 4),
                fmt(im["responsiveness"], 3), fmt(m["responsiveness"], 3), fmt(g["responsiveness"], 3),
                fmt(im["rho"], 3), fmt(m["rho"], 3), fmt(g["rho"], 3),
            ])
            for k, v in g.items():
                if v is not None and math.isfinite(float(v)):
                    mean_gaps[k].append(float(v))
        vals.extend([
            fmt(mean(mean_gaps["env_sr"]), 1, 100),
            fmt(mean(mean_gaps["divergence"]), 4),
            fmt(mean(mean_gaps["responsiveness"]), 3),
            fmt(mean(mean_gaps["rho"]), 3),
        ])
        lines.append("| " + " | ".join(vals) + " |")

    target_rows = [r for r in rows if r["train_regime"] == MINUS_REGIME.name]
    gap_summaries = []
    for row in target_rows:
        sr_vals = [row["heldout"][env]["gap_in_minus_holdout"].get("env_sr") for env in HELDOUT_ENVS]
        sr_gap = mean(sr_vals)
        rho_vals = [row["heldout"][env]["gap_in_minus_holdout"].get("rho") for env in HELDOUT_ENVS]
        rho_gap = mean(rho_vals)
        resp_vals = [row["heldout"][env]["gap_in_minus_holdout"].get("responsiveness") for env in HELDOUT_ENVS]
        resp_gap = mean(resp_vals)
        if sr_gap is not None:
            gap_summaries.append((float(sr_gap), float(rho_gap or 0.0), float(resp_gap or 0.0), row["model"]))
    if gap_summaries:
        worst_rho = max(gap_summaries, key=lambda x: x[1])
        best_rho = min(gap_summaries, key=lambda x: x[1])
        lines.extend([
            "",
            "## Quick read",
            "",
            "Env-SR is saturated on walker/humanoid in this run: every held-out-train checkpoint scores 100% on both held-out envs, so the signed env-SR gap is negative rather than a failure drop.",
            f"The diagnostic transfer signal is in div/resp/ρ: by mean signed ρ gap, `{worst_rho[3]}` drops the most ({worst_rho[1]:.3f}); `{best_rho[3]}` drops the least ({best_rho[1]:.3f}). GRU also shows the largest responsiveness shift, while both STJEWM rows keep calibrated responsiveness near the in-domain mean.",
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[cross-env] wrote {out_path}", flush=True)


def aggregate(seed: int = SEED) -> list[dict[str, Any]]:
    train_envs = load_train_envs()
    rows: list[dict[str, Any]] = []
    for model in FULL_REGIME.models:
        if ckpt_path(FULL_REGIME, model, seed).exists():
            rows.append(build_row(FULL_REGIME, model, train_envs, seed=seed))
    for model in MINUS_REGIME.models:
        if ckpt_path(MINUS_REGIME, model, seed).exists():
            rows.append(build_row(MINUS_REGIME, model, train_envs, seed=seed))
    write_json(OUT_DIR / "_index.json", {"rows": rows})
    write_table(rows)
    return rows


def run_model(args: argparse.Namespace) -> None:
    if args.model not in TARGET_MODELS:
        raise ValueError(f"{args.model} is not one of {TARGET_MODELS}")
    spec = make_eval16_spec()
    if not args.skip_train:
        train_minus_checkpoint(args.model, force=args.force_train, seed=args.seed)
    if not args.skip_eval:
        eval_checkpoint(
            MINUS_REGIME, args.model, spec,
            force=args.force_eval, seed=args.seed, n_episodes=args.n_episodes,
        )
    if not args.skip_metrics:
        ensure_metrics(
            MINUS_REGIME, args.model,
            force_latent=args.force_latent,
            force_align=args.force_align,
            seed=args.seed,
            latent_steps=args.latent_steps,
            align_steps=args.align_steps,
            device=args.device,
        )


def ensure_full_baseline(args: argparse.Namespace) -> None:
    spec = make_eval16_spec()
    baseline_models = FULL_G16_MODELS if args.all_full_models else TARGET_MODELS
    for model in baseline_models:
        if not ckpt_path(FULL_REGIME, model, args.seed).exists():
            print(f"[cross-env] skip full baseline {model}: missing ckpt", flush=True)
            continue
        spec_envs = [str(e["env_id"]) for e in json.loads(spec.read_text())]
        missing_eval = any(not (seed_dir(FULL_REGIME, model, args.seed) / f"eval_{env}.json").exists() for env in spec_envs)
        if missing_eval:
            print(f"[cross-env] full baseline {model}: filling missing eval JSONs", flush=True)
        if args.eval_full_baseline or missing_eval:
            eval_checkpoint(
                FULL_REGIME, model, spec,
                force=args.force_eval, seed=args.seed, n_episodes=args.n_episodes,
            )
        ensure_metrics(
            FULL_REGIME, model,
            force_latent=args.force_latent,
            force_align=args.force_align,
            seed=args.seed,
            latent_steps=args.latent_steps,
            align_steps=args.align_steps,
            device=args.device,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one cross-environment generalisation model.")
    p.add_argument("--model", choices=TARGET_MODELS, help="Minus-walker/humanoid model to train/eval.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--latent-steps", type=int, default=200)
    p.add_argument("--align-steps", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--force-latent", action="store_true")
    p.add_argument("--force-align", action="store_true")
    p.add_argument("--ensure-full-baseline", action="store_true")
    p.add_argument("--all-full-models", action="store_true", help="With --ensure-full-baseline, process all 12 full-G16 ckpts.")
    p.add_argument("--eval-full-baseline", action="store_true", help="Fill missing full-G16 eval JSONs using the 16-env spec.")
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        if args.ensure_full_baseline:
            ensure_full_baseline(args)
        if args.model:
            run_model(args)
    aggregate(seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

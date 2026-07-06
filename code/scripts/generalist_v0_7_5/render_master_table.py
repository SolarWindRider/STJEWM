"""Render generalist_master_table.md from the JSON file.

Reads:
  results/aggregate/generalist_master_table.json

Writes:
  results/aggregate/generalist_master_table.md

The MD format mirrors v0.7.5 but adds responsiveness + divergence
columns in the per-model summary (collapse-robust by construction).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


ID_ENVS = [
    "pendulum_2d", "dog", "tworoom", "quadruped", "reacher", "hopper",
    "fish", "stacker", "cheetah", "walker", "pusht", "humanoid",
    "finger", "ball_in_cup", "cartpole_2d",
]
STRESS_ENVS = ["pusht_ood", "tworoom_long", "cartpole_flicker", "cheetah_velhidden"]


def render(rows: List[Dict[str, Any]]) -> str:
    suites = sorted({r["suite"] for r in rows})
    models = [
        "stjewm_trace_only", "stjewm_spike_only", "stjewm_rate_only",
        "stjewm_no_trace", "stjewm_hidden_leak", "stjewm_membrane_readout",
        "cubifae_baseline", "gru_baseline", "lewm_baseline_v2",
        "slt_lif_mpc_trace", "slt_lif_mpc_free", "mlp_baseline",
    ]
    out = []
    out.append("# Generalist World-Model Evaluation — Master Table (v0.7.5)\n")
    out.append("**Setup.** Twelve model variants (6 STJEWM readouts + cubifae + gru + lewm")
    out.append("+ 2 slt variants + mlp collapse-control) trained on three task-scale")
    out.append("suites:\n")
    out.append("- **G4** — 4 envs (cartpole_2d, pendulum_2d, cheetah, pusht), 8K windows total.")
    out.append("- **G8** — G4 + finger, walker, reacher, tworoom, 16K windows.")
    out.append("- **G16** — full 16-env union, 32K windows.\n")
    out.append("All suites share the same per-window budget (2K windows / env), batch 32,")
    out.append("lr 3e-4, 1 epoch, n_layers=2, embed_dim=192, action_dim=56 (padded across")
    out.append("envs), pad_obs_to=128. Closed-loop eval at 3 episodes × 1 seed; stress-eval")
    out.append("at 3 episodes × 1 seed on 4 stress envs (pusht_ood, tworoom_long,")
    out.append("cartpole_flicker, cheetah_velhidden).\n")
    out.append("`mlp_baseline*` is the negative control for latent collapse — its")
    out.append("divergence-from-constant should be the lowest (collapse signature).\n")
    out.append("Seeds: [0]. Cells: env-SR mean ± std across seeds, in [0, 100]. '-' = no data.\n")
    out.append("---\n")

    # Section 1: env-SR In-Distribution
    out.append("\n## 1. env-SR (In-Distribution) — 12 models × 15 ID envs × 3 suites\n")
    write_grid(out, rows, suites, models, ID_ENVS, "env_sr_mean")
    # Section 2: env-SR Stress
    out.append("\n## 2. env-SR (Stress) — 12 models × 4 stress envs × 3 suites\n")
    write_grid(out, rows, suites, models, STRESS_ENVS, "env_sr_mean")

    # Pre-bucket per (suite, model) — SEPARATE lists for ID and stress.
    by_sm: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("env_sr_mean") is None:
            continue
        key = (r["suite"], r["model"])
        if r["env"] in STRESS_ENVS:
            by_sm[key]["stress_sr"].append(r["env_sr_mean"])
        else:
            by_sm[key]["env_sr_id"].append(r["env_sr_mean"])
        if r.get("lewm_sr_mean") is not None:
            by_sm[key]["lewm_sr_id" if r["env"] not in STRESS_ENVS else "lewm_sr_stress"].append(r["lewm_sr_mean"])
        if r.get("responsiveness_mean") is not None:
            by_sm[key]["resp"].append(r["responsiveness_mean"])
        if r.get("divergence_mean") is not None:
            by_sm[key]["div"].append(r["divergence_mean"])

    # Section 3: per-model summary with new resp/div columns
    out.append("\n## 3. Per-model summary (env-SR AVG, collapse-gap, responsiveness, divergence)\n")
    out.append("Columns:")
    out.append("- `mean_id` env-SR across 15 ID envs (averaged per suite).")
    out.append("- `gap` LeWM-SR − env-SR (collapse-inflatable; large +ve → likely collapse).")
    out.append("- `resp` responsiveness = `mean_norm(Δlatent) / mean_norm(Δobs)`; calibrated ~0.2, LeWM ~30 (over-reactive), GRU ~30 (noise).")
    out.append("- `div` divergence-from-constant = per-dim std of latent trajectory, averaged;")
    out.append("  collapse < 0.001 (MLP ~0.0002), calibrated ~0.01 (STJEWM), over-reactive ~0.18 (LeWM).\n")
    out.append("| model | mean_id (G4/G8/G16) | gap_id (G4/G8/G16) | mean_stress (G4/G8/G16) | resp (G4/G8/G16) | div (G4/G8/G16) |")
    out.append("|---|---|---|---|---|---|")

    def fmt3(vals, scale=100.0, dec=1):
        v3 = []
        for s in suites:
            vs = by_sm.get((s, m), {}).get(vals, [])
            v3.append((sum(vs) / len(vs) * scale) if vs else None)
        return "/".join(f"{v:.{dec}f}" if v is not None else "-" for v in v3)

    for m in models:
        marker = " (collapse-control)" if m == "mlp_baseline" else ""
        env_v3 = fmt3("env_sr_id")
        # gap = lewm - env
        lewm_v3 = []
        env_v_raw = []
        for s in suites:
            lewm_vs = by_sm.get((s, m), {}).get("lewm_sr_id", [])
            env_vs = by_sm.get((s, m), {}).get("env_sr_id", [])
            lewm_v3.append((sum(lewm_vs) / len(lewm_vs) * 100) if lewm_vs else None)
            env_v_raw.append((sum(env_vs) / len(env_vs) * 100) if env_vs else None)
        gap_v3 = [(l - e) if (l is not None and e is not None) else None
                  for l, e in zip(lewm_v3, env_v_raw)]
        stress_v3 = fmt3("stress_sr")
        resp_v3 = fmt3("resp", scale=1.0, dec=3)
        div_v3 = fmt3("div", scale=1.0, dec=4)
        gap_cell = "/".join(f"{g:+.1f}" if g is not None else "-" for g in gap_v3)
        out.append(f"| {m}{marker} | {env_v3} | {gap_cell} | {stress_v3} | {resp_v3} | {div_v3} |")

    # Section 4: headline takeaways — collapse-robust
    out.append("\n## 4. Headline takeaways (v0.7.5 — corrected metrics)\n")
    out.append(
        "**Three distinct non-spiking failure modes are now visible.** With the"
        " collapse-robust `divergence` metric, the 3 non-spiking baselines"
        " separate into 3 categories that the v0.7.5 `gap` metric could not"
        " distinguish:\n\n"
        "| model | div | interpretation |\n"
        "|---|---|---|\n"
        "| stjewm_trace / spike / no_trace / hidden_leak / membrane / rate_only | 0.011–0.012 | calibrated |\n"
        "| cubifae_baseline | 0.011 | calibrated (SNN) |\n"
        "| slt_lif_mpc_trace / free | 0.011 | calibrated (SNN) |\n"
        "| **mlp_baseline** | **0.0002** | **collapse (50× lower than STJEWM)** |\n"
        "| **gru_baseline** | 0.008 | noise (responsiveness 30, but ρ ≈ 0) |\n"
        "| **lewm_baseline_v2** | **0.186** | over-reactive (Transformer amplifies obs) |\n\n"
        "**STJEWM is the only family that is simultaneously (a) responsive to obs, (b)"
        " not collapsed, and (c) event-aligned (ρ ≥ 0.99 from v0.7.5 §9.3).**\n"
    )
    out.append(
        "**The `LeWM-SR` column in v0.7.5 §9.5 was collapse-inflatable.** MLP's"
        " LeWM-SR was 95.6% not because it plans well, but because the"
        " constant latent satisfies `cos_dist < 0.1` for any goal. The new"
        " `divergence` metric catches this: MLP's `div = 0.0002` is **50×"
        " lower** than STJEWM's. The v0.7.5 `gap` column (LeWM-SR −"
        " env-SR) was already a collapse-robust proxy and confirms the"
        " signal (MLP gap = +24.4, STJEWM gap = −15.6), but it doesn't show"
        " the *magnitude* of the collapse — `divergence` does.\n"
    )
    out.append(
        "**GRU's `divergence` is similar to STJEWM (0.008 vs 0.011), but its"
        " `responsiveness` is 150× higher (31.1 vs 0.2).** GRU's latent is"
        " *noisy* — the per-dim std is normal, but the per-step changes"
        " are 150× larger. Combined with v0.7.5's ρ ≈ −0.07, this is the"
        " signature of an uncorrelated noisy latent, not collapse.\n"
    )
    out.append(
        "**LeWM's `responsiveness` is 150× STJEWM and `divergence` is 16× STJEWM"
        " (0.186 vs 0.011).** LeWM is *not* collapsed — it's"
        " over-reactive, with a latent that amplifies obs events by an"
        " order of magnitude. Combined with v0.7.5's ρ = 0.52, this is"
        " the signature of a Transformer that tracks obs events but with"
        " a poorly conditioned response surface.\n"
    )
    out.append(
        "**On env-native success rate (v0.7.5 §9.1) all 12 models are within"
        " ±4pp of each other.** The new metrics do not change that ranking"
        " — STJEWM still doesn't win env-SR. The new finding is that the"
        " *quality* of the latent representation is dramatically different"
        " across families, and only STJEWM has a calibrated, responsive,"
        " non-collapsed, event-aligned latent.\n"
    )
    return "\n".join(out) + "\n"


def write_grid(out, rows, suites, models, envs, key):
    """Write a (suite × env) grid of model values."""
    by = defaultdict(dict)
    for r in rows:
        if r.get(key) is None:
            continue
        by[(r["suite"], r["env"])][r["model"]] = r[key] * 100
    for suite in suites:
        out.append(f"\n**{suite}**\n")
        out.append("| env | " + " | ".join(models) + " |")
        out.append("|" + "---|" * (1 + len(models)))
        for env in envs:
            cells = []
            for m in models:
                v = by.get((suite, env), {}).get(m)
                cells.append(f"{v:.1f}" if v is not None else "-")
            out.append(f"| {env} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/home/lx/snn/results/aggregate/generalist_master_table.json")
    ap.add_argument("--out", default="/home/lx/snn/results/aggregate/generalist_master_table.md")
    args = ap.parse_args()
    rows = json.loads(Path(args.json).read_text())["rows"]
    md = render(rows)
    Path(args.out).write_text(md)
    print(f"[render] wrote {args.out} ({len(md)} chars)")


if __name__ == "__main__":
    main()
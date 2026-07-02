#!/usr/bin/env python
"""Aggregate stress-sweep results into a markdown table + short curve prose.

Reads all eval_<difficulty>.json files under
    /home/lx/snn/results/<env>/<model>_seed<s>/eval_*.json

Writes:
    /home/lx/snn/results/aggregate/stress_sweep_table.md   (table)
    /home/lx/snn/results/aggregate/stress_sweep_curve.md   (prose)
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from collections import defaultdict

RESULTS = Path("/home/lx/snn/results")
AGG = RESULTS / "aggregate"
AGG.mkdir(parents=True, exist_ok=True)
ENV_BLOCKS = [
    ("cartpole_flicker", "cartpole_flicker", [
        ("f025", "flicker=0.25"),
        ("f050", "flicker=0.50"),
        ("f075", "flicker=0.75"),
    ]),
    ("pusht_ood", "pusht_ood (unseen_goal)", [
        ("g50",  "goal_off=50"),
        ("g100", "goal_off=100"),
        ("g200", "goal_off=200"),
    ]),
    ("tworoom_long", "tworoom_long (in_dist)", [
        ("g50",  "goal_off=50"),
        ("g100", "goal_off=100"),
        ("g200", "goal_off=200"),
    ]),
    ("cheetah_velhidden", "cheetah_velhidden (in_dist)", [
        ("vh00", "baseline"),
    ]),
]

MODELS = [
    "stjewm_trace_only",
    "stjewm_hidden_leak",
    "stjewm_spike_only",
    "stjewm_no_trace",
    "stjewm_membrane_readout",
]

SEEDS = [0, 1, 2]


def fmt_pct(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{100.0 * v:5.1f}"


def load_cell(env_outdir: str, model: str, seed: int, tag: str) -> dict | None:
    p = RESULTS / env_outdir / f"{model}_seed{seed}" / f"eval_{tag}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"error": f"parse_fail: {e}"}


def aggregate_over_seeds(cells: list[dict | None]) -> dict | None:
    """Average env-SR over seeds, ignoring missing."""
    valid = [c for c in cells if c is not None and "error" not in c]
    if not valid:
        return None
    srs = [c["success_rate_env"] for c in valid]
    lewm = [c["success_rate_lewm"] for c in valid]
    return {
        "env_sr": sum(srs) / len(srs),
        "lewm_sr": sum(lewm) / len(lewm),
        "n_seeds": len(valid),
        "n_total_seeds": len(cells),
    }


def build_table() -> tuple[str, dict]:
    """Return (markdown_table, raw_data).

    Produces TWO tables:
      1. Env-native success rate (%)
      2. LeWM latent goal-matching success rate (%) — cos_dist < 0.1
    """
    # header: model | <env.difficulty> ...
    headers = ["model"]
    for _, env_lbl, diffs in ENV_BLOCKS:
        for tag, lbl in diffs:
            headers.append(f"{env_lbl}.{lbl}")

    # Rows
    env_rows = []
    lewm_rows = []
    raw = defaultdict(dict)
    for model in MODELS:
        env_row = [model]
        lewm_row = [model]
        for env_outdir, env_lbl, diffs in ENV_BLOCKS:
            for tag, lbl in diffs:
                cells = [load_cell(env_outdir, model, s, tag) for s in SEEDS]
                agg = aggregate_over_seeds(cells)
                raw[model][f"{env_lbl}.{lbl}"] = agg
                if agg is None:
                    env_row.append("n/a")
                    lewm_row.append("n/a")
                else:
                    env_row.append(fmt_pct(agg["env_sr"]))
                    lewm_row.append(fmt_pct(agg["lewm_sr"]))
        env_rows.append(env_row)
        lewm_rows.append(lewm_row)

    lines = []
    lines.append("# Stress-difficulty sweep — success rate (%)")
    lines.append("")
    lines.append("Cells: rows = STJEWM mode, columns = (env, difficulty). "
                 "Values are averaged over available seeds (n_seeds=1 by default for this sweep). "
 "`n/a` = ckpt missing or eval errored.")
    lines.append("")
    lines.append("Two metrics are reported:")
    lines.append("")
    lines.append("- **Env-native SR**: env's own success criterion (often very strict; "
                 "will be ~0% for envs whose task is hard to complete from latent plans).")
    lines.append("- **LeWM-SR (cos_dist < 0.1)**: latent goal-matching success — the "
                 "metric LeWM itself reports. This is the primary comparison signal.")
    lines.append("")

    # Env-native SR table
    lines.append("## Env-native success rate (%)")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in env_rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")

    # LeWM-SR table
    lines.append("## LeWM-SR (cos_dist < 0.1) — latent goal-matching success (%)")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in lewm_rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")

    lines.append("Source: /home/lx/snn/results/<env>/<model>_seed<s>/eval_<tag>.json")
    return "\n".join(lines), raw


def cell_pct(model: str, col_key: str, raw: dict, metric: str = "lewm_sr") -> float | None:
    agg = raw.get(model, {}).get(col_key)
    if agg is None:
        return None
    return agg.get(metric)


def curve_prose(raw: dict) -> str:
    """Short prose summary highlighting monotonicity and trace-vs-membrane gap.

    Uses LeWM-SR (latent goal-matching) as the primary metric since env-native
    SR is mostly 0% on the harder stress envs and is not informative.
    """
    out = []
    out.append("# Stress-difficulty sweep — curve trends")
    out.append("")
    out.append("Notation: SR = LeWM-SR (cos_dist < 0.1, the latent goal-matching "
               "metric LeWM reports). Averaged over available seeds. "
               "`ΔSR` is the drop from the easiest to hardest difficulty. "
               "Positive `trace−membrane` means the trace model beats the "
               "membrane model on that env/difficulty (positive = membrane is weaker).")
    out.append("")

    # Per-env monotonicity
    for env_outdir, env_lbl, diffs in ENV_BLOCKS:
        out.append(f"## {env_lbl}")
        out.append("")
        diff_keys = [f"{env_lbl}.{lbl}" for _, lbl in diffs]
        cells_per_model = {}
        for m in MODELS:
            cells_per_model[m] = [cell_pct(m, k, raw, metric="lewm_sr") for k in diff_keys]
        easier_first = True

        out.append("")
        out.append("LeWM-SR (%)")
        out.append("")
        out.append("| model | " + " | ".join(diffs[i][1] for i in range(len(diffs))) + " | ΔSR (easy→hard) |")
        out.append("|" + "|".join(["---"] * (2 + len(diffs))) + "|")
        for m in MODELS:
            vals = cells_per_model[m]
            row = [m]
            for v in vals:
                row.append(fmt_pct(v))
            valid = [v for v in vals if v is not None]
            if len(valid) >= 2:
                dsr = (valid[0] - valid[-1]) if easier_first else (valid[-1] - valid[0])
                row.append(f"{100.0 * dsr:+.1f}")
            else:
                row.append("n/a")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

        # Trace vs membrane gap per difficulty
        if any(k.startswith(env_lbl) for k in raw.get(MODELS[0], {}).keys()):
            out.append("Trace − membrane gap (LeWM-SR, percentage points):")
            out.append("")
            out.append("| difficulty | trace | membrane | trace − membrane |")
            out.append("|---|---|---|---|")
            for k in diff_keys:
                tr = cell_pct("stjewm_trace_only", k, raw, metric="lewm_sr")
                mb = cell_pct("stjewm_membrane_readout", k, raw, metric="lewm_sr")
                gap = (None if tr is None or mb is None else tr - mb)
                out.append(f"| {k.split('.', 1)[1]} | {fmt_pct(tr)} | {fmt_pct(mb)} | "
                           f"{'n/a' if gap is None else f'{100.0*gap:+.1f}'} |")
            out.append("")

        tr_vals = cells_per_model.get("stjewm_trace_only", [])
        mb_vals = cells_per_model.get("stjewm_membrane_readout", [])
        tr_valid = [v for v in tr_vals if v is not None]
        mb_valid = [v for v in mb_vals if v is not None]
        if tr_valid and mb_valid:
            # Sign convention: dsr = first - last, so POSITIVE = SR decreases
            # with difficulty. Some envs (e.g. tworoom_long) show SR increasing
            # at higher goal_offset — that yields a negative dsr and is reported
            # as "SR goes UP" rather than "drops".
            tr_diff = (tr_valid[0] - tr_valid[-1])
            mb_diff = (mb_valid[0] - mb_valid[-1])
            tr_dir = "drops" if tr_diff >= 0 else "rises"
            mb_dir = "drops" if mb_diff >= 0 else "rises"
            avg_gap_easy = (tr_valid[0] - mb_valid[0]) if len(tr_valid) >= 1 and len(mb_valid) >= 1 else 0.0
            avg_gap_hard = (tr_valid[-1] - mb_valid[-1]) if len(tr_valid) >= 1 and len(mb_valid) >= 1 else 0.0
            out.append(f"**{env_lbl}**: trace {tr_dir} "
                       f"{100.0 * abs(tr_diff):.1f}pp from easy to hard; "
                       f"membrane {mb_dir} {100.0 * abs(mb_diff):.1f}pp. "
                       f"trace−membrane gap = "
                       f"{100.0 * avg_gap_easy:+.1f}pp (easy) → "
                       f"{100.0 * avg_gap_hard:+.1f}pp (hard).")
            out.append("")
        else:
            out.append(f"**{env_lbl}**: insufficient data for trend summary.")
            out.append("")

    out.append("## Headline")
    out.append("")
    rows = []
    for env_outdir, env_lbl, diffs in ENV_BLOCKS:
        if len(diffs) < 2:
            continue
        diff_keys = [f"{env_lbl}.{lbl}" for _, lbl in diffs]
        tr_easy = cell_pct("stjewm_trace_only", diff_keys[0], raw, metric="lewm_sr")
        tr_hard = cell_pct("stjewm_trace_only", diff_keys[-1], raw, metric="lewm_sr")
        mb_easy = cell_pct("stjewm_membrane_readout", diff_keys[0], raw, metric="lewm_sr")
        mb_hard = cell_pct("stjewm_membrane_readout", diff_keys[-1], raw, metric="lewm_sr")
        if None in (tr_easy, tr_hard, mb_easy, mb_hard):
            continue
        rows.append((env_lbl, tr_easy, tr_hard, mb_easy, mb_hard))
    if rows:
        for env_lbl, t_e, t_h, m_e, m_h in rows:
            out.append(f"- **{env_lbl}**: trace "
                       f"{100.0*t_e:.1f} → {100.0*t_h:.1f}% "
                       f"(Δ {100.0*(t_e-t_h):.1f}pp); membrane "
                       f"{100.0*m_e:.1f} → {100.0*m_h:.1f}% "
                       f"(Δ {100.0*(m_e-m_h):.1f}pp).")
    else:
        out.append("(no envs with ≥2 difficulty levels and complete trace/membrane data)")
    out.append("")
    return "\n".join(out)


def main():
    table_md, raw = build_table()
    curve_md = curve_prose(raw)
    (AGG / "stress_sweep_table.md").write_text(table_md)
    (AGG / "stress_sweep_curve.md").write_text(curve_md)
    print(f"[aggregate] wrote {AGG / 'stress_sweep_table.md'}")
    print(f"[aggregate] wrote {AGG / 'stress_sweep_curve.md'}")


if __name__ == "__main__":
    main()
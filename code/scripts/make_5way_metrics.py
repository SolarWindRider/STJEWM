"""5-condition comparison: STJEWM-trace, STJEWM-leak, STJEWM-spike,
STJEWM-no-trace, and LeWM.

Reads:
  results/<env>/<model>/eval.json
  results/aggregate/eval_v1_readout/<env>_<mode>.json   (new readout-mode evals)

Writes:
  results/aggregate/summary_5way.md
"""
import json
from pathlib import Path

base = Path("/home/lx/snn/results")
agg = base / "aggregate"

envs = [
    "ball_in_cup", "cartpole_2d", "cheetah", "dog", "finger", "fish", "hopper",
    "humanoid", "humanoid_CMU", "pendulum_2d", "pusht", "quadruped", "reacher",
    "stacker", "tworoom", "walker",
]

# (display name, subdir or eval_v1_readout prefix, env_id)
models = [
    ("STJEWM-trace",   "stjewm_trace_only",  "eval_v1_readout"),
    ("STJEWM-rate",    "stjewm_rate_only",   "eval_v2_5way"),
    ("STJEWM-spike",   "stjewm_spike_only",  "eval_v1_readout"),
    ("STJEWM-leak",    "stjewm_hidden_leak", "eval_v1_readout"),
    ("STJEWM-no-trace","stjewm_no_trace",    "eval_v2_5way"),
    ("STJEWM-membrane","stjewm_membrane_readout","eval_v1_readout"),
    ("LeWM",           "lewm_baseline_v2",   "regular"),
    ("GRU",            "gru_baseline",       "eval_v2_5way"),
    ("MLP",            "mlp_baseline",       "eval_v2_5way"),
]

# (display name, json key, format, scale_to_pct)
# scale=100 means multiply by 100 to get percent; scale=1 means raw value
metrics = [
    ("LeWM-SR (%)",      "success_rate_lewm", "{:6.1f}%", 100),
    ("Env-SR (%)",       "success_rate_env",  "{:6.1f}%", 100),
    ("cos_dist",         "mean_cos_dist",     "{:6.3f}",  1),
    ("phys_dist",        "mean_phys_dist",    "{:7.2f}",  1),
]


def fmt_avg(vals, fmt, scale, jkey):
    """Compute AVG row. For phys_dist, use median because pusht dominates."""
    valid = [v for v in vals if v is not None and v == v]  # filter NaN
    if not valid:
        return "   —"
    if jkey == "mean_phys_dist" and len(valid) >= 3:
        s = sorted(valid)
        return fmt.format(s[len(s) // 2] * scale) + " (med)"
    return fmt.format(sum(valid) / len(valid) * scale)


# Map model_dir_name to (filename, subdir)
# eval_v1_readout uses short names (no stjewm_ prefix); eval_v2_5way uses full names
model_to_filename = {
    "stjewm_trace_only":          ("trace_only",         "eval_v1_readout"),
    "stjewm_hidden_leak":         ("hidden_leak",        "eval_v1_readout"),
    "stjewm_spike_only":          ("spike_only",         "eval_v1_readout"),
    "stjewm_no_trace":            ("stjewm_no_trace",    "eval_v2_5way"),
    "stjewm_membrane_readout":    ("membrane_readout",   "eval_v1_readout"),
    "stjewm_rate_only":           ("stjewm_rate_only",   "eval_v2_5way"),
    "lewm_baseline_v2":           ("lewm_baseline",      "regular"),
    "gru_baseline":               ("gru_baseline",       "eval_v2_5way"),
    "mlp_baseline":               ("mlp_baseline",       "eval_v2_5way"),
}


def collect_data():
    """data[metric_key][model][env] = value"""
    data = {}
    for _mname, jkey, _fmt, _scale in metrics:
        data[jkey] = {m[0]: {} for m in models}
    for env in envs:
        for mname, dir_name, src in models:
            for _mn, jkey, _fmt, _scale in metrics:
                fname, subdir = model_to_filename.get(dir_name, (dir_name, src))
                if subdir == "regular":
                    p = base / env / dir_name / "eval.json"
                else:
                    p = agg / subdir / f"{env}_{fname}.json"
                if p and p.exists():
                    try:
                        d = json.loads(p.read_text())
                        data[jkey][mname][env] = d.get(jkey)
                    except Exception:
                        pass
    return data


def render_table(data, mname, jkey, fmt, scale):
    cols = [m[0] for m in models]
    lines = []
    lines.append(f"## {mname}\n")
    lines.append("| Env | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    vals_per_col = {c: [] for c in cols}
    for env in envs:
        cells = []
        for c in cols:
            v = data[jkey][c].get(env)
            if v is None:
                cells.append("   —")
            else:
                cells.append(fmt.format(v * scale))
                vals_per_col[c].append(v)
        lines.append(f"| {env} | " + " | ".join(cells) + " |")
    avg_row = [fmt_avg(vals_per_col[c], fmt, scale, jkey) for c in cols]
    lines.append("| **AVG** | " + " | ".join(f"**{v}**" for v in avg_row) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    data = collect_data()
    lines = [
        "# 5-Condition Comparison (post-bugfix)\n",
        "STJEWM under 4 readout modes + LeWM baseline.\n",
        "Note: STJEWM-{trace,leak,spike,no_trace} are retrained on the same 16-env suite",
        "with the same hyper-params (3 epochs, batch 64, max_windows=10000).",
        "Their avg LeWM-SR is the membrane-forbidden protocol headline (Sec. 4.1, Table 1).\n",
    ]
    for mname, jkey, fmt, scale in metrics:
        lines.append(render_table(data, mname, jkey, fmt, scale))
    lines.append("## Metric definitions")
    lines.append("- **LeWM-SR (%)**: Fraction of CEM plans whose final latent is within cos_dist < 0.1 of goal latent (LeWM paper primary metric).")
    lines.append("- **Env-SR (%)**: Fraction of plans that achieve env-native goal.")
    lines.append("- **cos_dist**: Mean (1-cos_sim)/2. Lower is better. 0 = identical, 1 = orthogonal.")
    lines.append("- **phys_dist**: Mean physical distance. Lower is better. AVG uses median to avoid pusht/tworoom dominating the scale.\n")
    lines.append("## Reading the table")
    lines.append("- **STJEWM-trace** is the only model that respects the membrane-forbidden protocol (Sec. 2.1).")
    lines.append("- **STJEWM-leak** is the legacy default (hidden + trace).")
    lines.append("- **STJEWM-spike** masks the hidden state by the post-spike activation.")
    lines.append("- **STJEWM-no-trace** drops the trace branch entirely (ablation).")
    lines.append("- **LeWM** is a 4-layer Transformer + AdaLN-zero with no trace.\n")
    out = agg / "summary_5way.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

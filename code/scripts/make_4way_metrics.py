"""4-condition comparison with all 4 metrics:
1. success_rate_lewm  (LeWM paper: cos_dist < 0.1)
2. success_rate_env   (env-native success)
3. mean_cos_dist     (LeWM paper fig 1 x-axis)
4. mean_phys_dist    (physical distance to goal)
"""
import json
from pathlib import Path

base = Path("/home/lx/snn/results")
envs = ["ball_in_cup", "cartpole_2d", "cheetah", "dog", "finger", "fish", "hopper", "humanoid", "humanoid_CMU", "pendulum_2d", "pusht", "quadruped", "reacher", "stacker", "tworoom", "walker"]

models = [
    ("STJEWM (with goal)",   "stjewm_v2"),
    ("STJEWM (no goal)",     "stjewm_nogoal"),
    ("LeWM (with goal)",     "lewm_baseline_v2"),
    ("LeWM (no goal)",       "lewm_baseline_no_goal"),
]

# Metrics: (display name, json key, format, scale_to_pct)
# scale=100 means multiply by 100 to get percent; scale=1 means raw value
metrics = [
    ("LeWM-SR (%)",     "success_rate_lewm",   "{:5.0f}%",  100),
    ("Env-SR (%)",      "success_rate_env",    "{:5.0f}%",  100),
    ("cos_dist",        "mean_cos_dist",       "{:.3f}",     1),
    ("phys_dist",       "mean_phys_dist",      "{:6.2f}",    1),
]


def fmt_avg(vals, fmt, scale, jkey):
    """Compute AVG row. For phys_dist, use median because pusht dominates."""
    valid = [v for v in vals if v is not None and not (v != v)]
    if not valid:
        return "-"
    if jkey == "mean_phys_dist":
        # Use median to avoid pusht (~1000) and tworoom (~100) dominating
        sorted_v = sorted(valid)
        mid = sorted_v[len(sorted_v) // 2]
        return f"{fmt.format(mid * scale)} (median)"
    return fmt.format(sum(valid) / len(valid) * scale)

# Collect data
data = {}
for _, jkey, _, _ in metrics:
    data[jkey] = {m[1]: {} for m in models}

for env in envs:
    for _, dir_name in models:
        p = base / env / dir_name / "eval.json"
        if p.exists():
            d = json.load(open(p))
            for _, jkey, _, _ in metrics:
                data[jkey][dir_name][env] = d.get(jkey)

# Print to stdout
for mname, jkey, fmt, scale in metrics:
    print(f"\n=== {mname} ===")
    hdr = f"{'env':<14s}"
    for name, _ in models:
        hdr += f" | {name:>20s}"
    print(hdr)
    print("-" * len(hdr))
    for env in envs:
        line = f"{env:<14s}"
        for _, dir_name in models:
            v = data[jkey][dir_name].get(env)
            if v is None:
                line += f" | {'-':>20s}"
            else:
                line += f" | {fmt.format(v * scale)}"
        print(line)
    print("-" * len(hdr))
    line = f"{'AVG':<14s}"
    for _, dir_name in models:
        vals = [v for v in data[jkey][dir_name].values()]
        line += f" | {fmt_avg(vals, fmt, scale, jkey):>20s}"
    print(line)
lines = ["# 4-Condition Comparison (post-fix only)\n"]
lines.append("Only valid (post-fix) variants. v1 (broken goal) deleted per user request.\n")
lines.append("Four metrics per model × env. Lower is better for cos_dist/phys_dist; higher is better for SR.\n")
lines.append("Note: phys_dist AVG uses **median** (not mean) because pusht (~1000) and tworoom (~100) would dominate the scale and hide other-env differences.\n")

for mname, jkey, fmt, scale in metrics:
    lines.append(f"## {mname}\n")
    lines.append("| Env |" + " | ".join(name for name, _ in models) + " |")
    lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
    for env in envs:
        row = f"| {env}"
        for _, dir_name in models:
            v = data[jkey][dir_name].get(env)
            row += f" | {fmt.format(v * scale)}" if v is not None else " | -"
        row += " |"
        lines.append(row)
    avg_row = "| **AVG**"
    for _, dir_name in models:
        vals = [v for v in data[jkey][dir_name].values()]
        avg_row += f" | **{fmt_avg(vals, fmt, scale, jkey)}**"
    avg_row += " |"
    lines.append(avg_row)
    lines.append("")

lines.append("## Metric definitions")
lines.append("- **LeWM-SR (%)**: Fraction of CEM plans whose final latent is within cos_dist < 0.1 of goal latent (LeWM paper primary metric).")
lines.append("- **Env-SR (%)**: Fraction of plans that achieve env-native goal (env-specific success criterion).")
lines.append("- **cos_dist**: Mean (1-cos_sim)/2 between final latent and goal latent. Lower is better. 0 = identical, 1 = orthogonal.")
lines.append("- **phys_dist**: Mean physical distance between plan trajectory and goal. Lower is better. Scale varies by env (DMC ~0-2, pusht ~0-1000, tworoom ~100).")
lines.append("")
lines.append("## Analysis")
lines.append("Compare with-goal vs no-goal for both STJEWM and LeWM:")
lines.append("- If no-goal ≈ with-goal: goal loss term is negligible on these evals.")
lines.append("- If no-goal < with-goal: goal loss DOES help (improves SR / reduces distance).")
lines.append("- If no-goal > with-goal: goal loss HURTS on this env.")
lines.append("")
lines.append("v1 (broken goal) was deleted per user request — see `docs/GOAL_LOSS_FIX.md` for the bug analysis.")
lines.append("Tworoom was previously NaN due to a missing env.reset() call (see `docs/TWOROOM_BUGFIX.md`); now fixed.")

out = base / "aggregate" / "summary_4way.md"
with open(out, "w") as f:
    f.write("\n".join(lines))

print(f"\nWrote {out}")

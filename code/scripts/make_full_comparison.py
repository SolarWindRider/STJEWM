"""Full 6-condition comparison:
- stjewm (v1, broken goal): orig
- stjewm_v2 (v2, fixed goal)
- stjewm_nogoal (v2, no goal)
- lewm_baseline (v1, broken goal)
- lewm_baseline_v2 (v2, fixed goal)
- lewm_baseline_no_goal (v2, no goal)
"""
import json
from pathlib import Path

base = Path("/home/lx/snn/results")
envs = ["ball_in_cup", "cartpole_2d", "cheetah", "dog", "finger", "fish", "hopper", "humanoid", "humanoid_CMU", "pendulum_2d", "pusht", "quadruped", "reacher", "stacker", "tworoom", "walker"]

# Models = (display_name, dir_name). Only valid (post-fix) variants.
models = [
    ("STJEWM (with goal)",   "stjewm_v2"),
    ("STJEWM (no goal)",     "stjewm_nogoal"),
    ("LeWM (with goal)",     "lewm_baseline_v2"),
    ("LeWM (no goal)",       "lewm_baseline_no_goal"),
]

# Print table
hdr = f"{'env':<14s}"
for name, _ in models:
    hdr += f" | {name:>22s}"
print(hdr)
print("-" * len(hdr))

data = {m[1]: {} for m in models}
for env in envs:
    line = f"{env:<14s}"
    for name, dir_name in models:
        p = base / env / dir_name / "eval.json"
        if p.exists():
            d = json.load(open(p))
            sr = d["success_rate_lewm"] * 100
            data[dir_name][env] = sr
            line += f" | {sr:6.0f}%"
        else:
            data[dir_name][env] = None
            line += f" | {'-':>22s}"
    print(line)

# Average row
print("-" * len(hdr))
avg_line = f"{'AVG':<14s}"
for name, dir_name in models:
    vals = [v for v in data[dir_name].values() if v is not None]
    if vals:
        avg = sum(vals) / len(vals)
        avg_line += f" | {avg:6.1f}%"
    else:
        avg_line += f" | {'-':>22s}"
print(avg_line)

# Save to md
out = base / "aggregate" / "summary_4way.md"
lines = ["# 4-Condition Comparison (post-fix only)\n"]
lines.append("Only valid (post-fix) variants. v1 (broken goal) deleted per user request.\n")
lines.append("| Env |" + " | ".join(name for name, _ in models) + " |")
lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
for env in envs:
    row = f"| {env}"
    for name, dir_name in models:
        v = data[dir_name].get(env)
        row += f" | {v:.0f}%" if v is not None else " | -"
    row += " |"
    lines.append(row)
# Average row
avg_row = "| **AVG**"
for name, dir_name in models:
    vals = [v for v in data[dir_name].values() if v is not None]
    if vals:
        avg_row += f" | **{sum(vals)/len(vals):.1f}%**"
    else:
        avg_row += " | -"
avg_row += " |"
lines.append(avg_row)
lines.append("")
lines.append("## Analysis")
lines.append("Compare with-goal vs no-goal for both STJEWM and LeWM:")
lines.append("- If no-goal ≈ with-goal: goal loss term is negligible on these evals.")
lines.append("- If no-goal < with-goal: goal loss DOES help (improves SR).")
lines.append("- If no-goal > with-goal: goal loss HURTS on this env.")
lines.append("")
lines.append("v1 (broken goal) was deleted per user request — see `docs/GOAL_LOSS_FIX.md` for the bug analysis.")
lines.append("v2 (fixed) is identical to v1 on all 16 STJEWM envs (model saturated, see `docs/SATURATION_ANALYSIS.md`).")

with open(out, "w") as f:
    f.write("\n".join(lines))

print(f"\nWrote {out}")

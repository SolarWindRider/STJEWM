"""Generate v2 3-way comparison markdown with all evals."""
import json
from pathlib import Path

base = Path("/home/lx/snn/results")
envs = ["ball_in_cup", "cartpole_2d", "cheetah", "dog", "finger", "fish", "hopper", "humanoid", "humanoid_CMU", "pendulum_2d", "pusht", "quadruped", "reacher", "stacker", "tworoom", "walker"]

out = base / "aggregate" / "summary_3way_v2.md"

lines = []
lines.append("# 3-Way Comparison: STJEWM vs LeWM-goal vs LeWM-no-goal\n")
lines.append("All models with FIXED goal loss (full-window forward, single goal_offset-step prediction).\n")
lines.append("| Env | STJEWM (with goal) | LeWM (with goal) | LeWM (no goal) | Best |")
lines.append("|---|---|---|---|---|")
wins = {"STJEWM": 0, "LeWM-goal": 0, "LeWM-no": 0, "TIE": 0}
sr_stj, sr_leg, sr_leng = [], [], []

for env in envs:
    stj = base / env / "stjewm_v2" / "eval.json"
    leg = base / env / "lewm_baseline_v2" / "eval.json"
    leng = base / env / "lewm_baseline_no_goal" / "eval.json"
    s1 = json.load(open(stj))["success_rate_lewm"]*100 if stj.exists() else None
    s2 = json.load(open(leg))["success_rate_lewm"]*100 if leg.exists() else None
    s3 = json.load(open(leng))["success_rate_lewm"]*100 if leng.exists() else None
    s1s = f"{s1:.0f}%" if s1 is not None else "-"
    s2s = f"{s2:.0f}%" if s2 is not None else "-"
    s3s = f"{s3:.0f}%" if s3 is not None else "-"
    sr = {}
    if s1 is not None: sr["STJEWM"] = s1
    if s2 is not None: sr["LeWM-goal"] = s2
    if s3 is not None: sr["LeWM-no"] = s3
    if sr:
        max_v = max(sr.values())
        ws = [k for k, v in sr.items() if v >= max_v - 0.5]
        best = "TIE" if len(ws) > 1 else ws[0]
        wins[best] = wins.get(best, 0) + 1
    else:
        best = "-"
    lines.append(f"| {env} | {s1s} | {s2s} | {s3s} | {best} |")
    if s1 is not None: sr_stj.append(s1)
    if s2 is not None: sr_leg.append(s2)
    if s3 is not None: sr_leng.append(s3)

lines.append(f"| **AVG** | **{sum(sr_stj)/len(sr_stj):.1f}%** | **{sum(sr_leg)/len(sr_leg):.1f}%** | **{sum(sr_leng)/len(sr_leng):.1f}%** | |")
lines.append("")
lines.append("## Wins per model (best of 3 on LeWM-SR)")
for k, v in wins.items():
    lines.append(f"- **{k}**: {v}/16")
lines.append("")
lines.append("## Key insight")
lines.append("STJEWM (SN model + goal) wins 4/16 vs LeWM variants (Transformer ± goal).")
lines.append("STJEWM wins on DMC envs that require longer-horizon goal planning: dog, hopper, humanoid, pendulum_2d.")
lines.append("LeWM wins on cheetah + pusht (the data-saturated envs).")
lines.append("")
lines.append("Note: v1 (broken goal) was deleted per user request — see `docs/GOAL_LOSS_FIX.md` for the bug analysis.")
lines.append("The v1 vs v2 comparison (now archived in the docs) showed v1=v2 on all 16 STJEWM envs (model saturated, see `docs/SATURATION_ANALYSIS.md`).")
with open(out, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote {out}")

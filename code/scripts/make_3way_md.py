"""Generate a clean 3-way comparison markdown table."""
import json
from pathlib import Path

base = Path("/home/lx/snn/results")
envs = sorted([d.name for d in base.iterdir() if d.is_dir() and d.name != "aggregate"])

print("# 3-Way Comparison: STJEWM vs LeWM-with-goal vs LeWM-no-goal\n")
print("STJEWM (5.03M params, SNN with goal loss) vs:")
print("- LeWM-with-goal (5.07M params, Transformer with goal loss)")
print("- LeWM-no-goal (5.07M params, Transformer without goal loss — proper LeWM paper baseline)")
print()
print("| Env | STJEWM LeWM-SR | LeWM-goal SR | LeWM-no SR | Best (LeWM-SR) |")
print("|---|---|---|---|---|")

wins = {"STJEWM": 0, "LeWM-goal": 0, "LeWM-no": 0, "TIE": 0}
sr_stj = []
sr_leg = []
sr_leng = []
for env in envs:
    if env == "lewm_baseline_no_goal":
        continue
    row_data = {"env": env}
    for model, key in [("stjewm", "STJEWM"), ("lewm_baseline", "LeWM-goal"), ("lewm_baseline_no_goal", "LeWM-no")]:
        p = base / env / model / "eval.json"
        if p.exists():
            d = json.load(open(p))
            row_data[key] = d
    line = f"| {env}"
    for key in ["STJEWM", "LeWM-goal", "LeWM-no"]:
        if key in row_data:
            line += f" | {row_data[key]['success_rate_lewm']*100:.0f}%"
            if key == "STJEWM": sr_stj.append(row_data[key]['success_rate_lewm']*100)
            elif key == "LeWM-goal": sr_leg.append(row_data[key]['success_rate_lewm']*100)
            else: sr_leng.append(row_data[key]['success_rate_lewm']*100)
        else:
            line += " | -"
    # best
    sr = {k: row_data[k]['success_rate_lewm']*100 for k in ["STJEWM", "LeWM-goal", "LeWM-no"] if k in row_data}
    if sr:
        max_v = max(sr.values())
        ws = [k for k, v in sr.items() if v >= max_v - 0.5]
        if len(ws) > 1:
            best = "TIE"
        else:
            best = ws[0]
        wins[best] = wins.get(best, 0) + 1
    else:
        best = "-"
    line += f" | {best} |"
    print(line)

# Aggregate
print()
print("## Summary (LeWM-SR)")
print(f"STJEWM mean LeWM-SR:   {sum(sr_stj)/len(sr_stj):.1f}%")
print(f"LeWM-goal mean LeWM-SR: {sum(sr_leg)/len(sr_leg):.1f}%")
print(f"LeWM-no mean LeWM-SR:   {sum(sr_leng)/len(sr_leng):.1f}%")
print()
print("## Wins per model (best of 3 on LeWM-SR)")
for k, v in wins.items():
    print(f"  {k}: {v}/16")

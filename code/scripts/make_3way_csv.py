"""3-way comparison CSV: STJEWM vs LeWM-with-goal vs LeWM-no-goal."""
import csv
import json
from pathlib import Path

base = Path("/home/lx/snn/results")
out = base / "aggregate" / "summary_3way.csv"

envs = sorted([d.name for d in base.iterdir() if d.is_dir() and d.name != "aggregate"])

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "env",
        "stjewm_lewm_sr", "lewm_goal_lewm_sr", "lewm_nogoal_lewm_sr",
        "stjewm_env_sr", "lewm_goal_env_sr", "lewm_nogoal_env_sr",
        "stjewm_cos", "lewm_goal_cos", "lewm_nogoal_cos",
        "stjewm_final_loss", "lewm_goal_final_loss", "lewm_nogoal_final_loss",
        "best_model",
    ])
    for env in envs:
        row = [env]
        for model in ("stjewm", "lewm_baseline", "lewm_baseline_no_goal"):
            p = base / env / model / "eval.json"
            lp = base / env / model / "loss_log.json"
            if p.exists():
                d = json.load(open(p))
                row.extend([
                    f"{d.get('success_rate_lewm', 0)*100:.1f}",
                    f"{d.get('success_rate_env', 0)*100:.1f}",
                    f"{d.get('mean_cos_dist', 0):.4f}",
                ])
            else:
                row.extend(["-", "-", "-"])
            if lp.exists():
                lld = json.load(open(lp))
                if lld.get("losses"):
                    row.append(f"{lld['losses'][-1].get('total', 0):.3f}")
                else:
                    row.append("-")
            else:
                row.append("-")
        # Best of 3
        sr = {}
        for m, col in [("STJEWM", 1), ("LeWM-goal", 2), ("LeWM-no", 3)]:
            v = row[col] if col < len(row) else "-"
            try:
                sr[m] = float(v.rstrip("%"))
            except:
                sr[m] = None
        valid_sr = {k: v for k, v in sr.items() if v is not None}
        if valid_sr:
            best_val = max(valid_sr.values())
            winners = [k for k, v in valid_sr.items() if v >= best_val - 0.5]
            if len(winners) > 1:
                best = "TIE"
            else:
                best = winners[0]
        else:
            best = "-"
        row.append(best)
        w.writerow(row)

print(f"Wrote {out}")

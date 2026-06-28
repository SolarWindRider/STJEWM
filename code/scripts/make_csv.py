"""Make a single CSV summarizing all 32 evals (16 envs × 2 models).

Output: /home/lx/snn/results/aggregate/summary.csv
"""
import csv
import json
from pathlib import Path

base = Path("/home/lx/snn/results")
out = base / "aggregate" / "summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)

envs = sorted([d.name for d in base.iterdir() if d.is_dir() and d.name != "aggregate"])

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "env",
        "stjewm_lewm_sr_pct", "lewm_lewm_sr_pct", "delta_lewm_sr_pp",
        "stjewm_env_sr_pct", "lewm_env_sr_pct", "delta_env_sr_pp",
        "stjewm_cos_dist", "lewm_cos_dist",
        "stjewm_final_loss", "lewm_final_loss",
        "winner_lewm_sr", "winner_env_sr",
    ])
    for env in envs:
        s_p = base / env / "stjewm" / "eval.json"
        l_p = base / env / "lewm_baseline" / "eval.json"
        s_loss_p = base / env / "stjewm" / "loss_log.json"
        l_loss_p = base / env / "lewm_baseline" / "loss_log.json"
        if not s_p.exists() or not l_p.exists():
            continue
        s = json.load(open(s_p))
        l = json.load(open(l_p))
        s_lsr = s["success_rate_lewm"] * 100
        l_lsr = l["success_rate_lewm"] * 100
        s_esr = s["success_rate_env"] * 100
        l_esr = l["success_rate_env"] * 100
        d_lsr = round(s_lsr - l_lsr, 1)
        d_esr = round(s_esr - l_esr, 1)
        s_loss = "-"
        l_loss = "-"
        if s_loss_p.exists():
            d_loss = json.load(open(s_loss_p))
            if d_loss.get("losses"):
                s_loss = round(d_loss["losses"][-1].get("total", 0), 3)
        if l_loss_p.exists():
            d_loss = json.load(open(l_loss_p))
            if d_loss.get("losses"):
                l_loss = round(d_loss["losses"][-1].get("total", 0), 3)
        w_lsr = "STJEWM" if d_lsr > 1 else ("LeWM" if d_lsr < -1 else "TIE")
        w_esr = "STJEWM" if d_esr > 1 else ("LeWM" if d_esr < -1 else "TIE")
        w.writerow([
            env,
            f"{s_lsr:.1f}", f"{l_lsr:.1f}", d_lsr,
            f"{s_esr:.1f}", f"{l_esr:.1f}", d_esr,
            f"{s['mean_cos_dist']:.4f}", f"{l['mean_cos_dist']:.4f}",
            s_loss, l_loss,
            w_lsr, w_esr,
        ])

print(f"Wrote {out}")

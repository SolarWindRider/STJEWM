"""Digest the lambda-arm closed-loop evaluations against the E1 trace baseline.

lambda=0 = E1 stjewm_trace_only (identical checkpoint as access lam0_ref,
same eval seed 0 -> instance-aligned pairing by episode_idx).
Outputs lambda_closedloop_summary.csv and lambda_paired.csv.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("/data/lx/tmp/results/accessibility_closedloop_20261120")
AGG = Path("/data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json")
OUT = Path("/home/lx/audit_20261120")
ACCESS = json.load(open("/home/lx/snn/results/mech_0917_b/accessibility/access_summary.json"))
ARMS = {"lam0": {"label": "lam0_ref(=E1 trace)", "c20": 1.361e-4, "decoder_score": None},
        "lam00_control": {"label": "lam0 continued-training control", "c20": 2.42e-7, "decoder_score": -0.0986},
        "lam01": {"label": "lam0.1", "c20": 34.705, "decoder_score": 0.1981},
        "lam10": {"label": "lam1.0", "c20": 46.335, "decoder_score": 0.4441},
        "lam100": {"label": "lam10", "c20": 20.820, "decoder_score": 0.5309}}


def main() -> int:
    agg = json.load(open(AGG))
    base_cells = {}
    for e in agg["evals"]:
        if e.get("experiment") == "E1" and e["model"] == "stjewm_trace_only":
            base_cells[(e["split"], e["env_id"])] = e
    print("E1 trace baseline cells:", len(base_cells))

    summary_rows = []
    paired_rows = []
    for arm, meta in ARMS.items():
        if arm == "lam0":
            cells = {(k[0], k[1]): v for k, v in base_cells.items()}
        else:
            cells = {}
            for p in (BASE / arm).rglob("eval_*.json"):
                d = json.load(open(p))
                split = p.parts[-4]
                env = p.name[len("eval_"):-len(".json")]
                cells[(split, env)] = {"metrics": d, "output": str(p),
                                       "id": f"P1/{arm}/{split}/{env}",
                                       "checkpoint": str(p).replace("/eval_" + env + ".json", "/final.pt")}
        srs = []
        eps_all = []
        base_keys = set(base_cells.keys())
        for (sp, env), e in sorted(cells.items()):
            if arm != "lam0" and (sp, env) not in base_keys:
                continue  # grid-drift cells (new-config envs absent from E1) excluded
            m = e["metrics"]
            srs.append((sp, env, m["success_rate_env"], m.get("mean_cos_dist")))
            d = json.load(open(e["output"]))
            for pe in d["per_episode"]:
                eps_all.append((sp, env, pe))
        sr = sum(v for _, _, v, _ in srs) / len(srs)
        summary_rows.append({"arm": arm, "label": meta["label"],
                             "checkpoint": arm_ckpt(arm),
                             "c20_fig2c_protocol": meta["c20"],
                             "decoder_score": meta["decoder_score"],
                             "env_success_rate": f"{sr:.4f}",
                             "n_cells": len(srs), "n_episodes": len(eps_all),
                             "planning_calls_per_episode": 2,
                             "note": ("lambda=0 = original checkpoint (E1 reuse)" if arm == "lam0"
                                      else "matched continued-training control (lambda=0, same schedule)" if arm == "lam00_control"
                                      else "1-epoch aux-decoder fine-tune of the same checkpoint")})
        for (sp, env, sr_arm, cos_arm) in srs:
            if arm == "lam0":
                continue
            b = base_cells.get((sp, env))
            if b is None:
                continue
            sr0 = b["metrics"]["success_rate_env"]
            paired_rows.append({"arm": arm, "split": sp, "env": env,
                                "sr_lambda": sr_arm, "sr_lambda0": sr0,
                                "paired_diff_pp": f"{(sr_arm - sr0) * 100:+.1f}",
                                "n_episodes": 5,
                                "checkpoint": cells[(sp, env)]["checkpoint"]})
    with open(OUT / "lambda_closedloop_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    with open(OUT / "lambda_paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        w.writeheader(); w.writerows(paired_rows)
    for r in summary_rows:
        print(r)
    diffs = defaultdict(list)
    for r in paired_rows:
        diffs[r["arm"]].append(float(r["paired_diff_pp"]))
    for arm, d in diffs.items():
        print(f"{arm}: mean paired diff = {sum(d)/len(d):+.2f} pp (n={len(d)})")
    return 0


def arm_ckpt(arm):
    if arm == "lam0":
        return "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt"
    if arm == "lam00_control":
        return "/data/lx/tmp/results/accessibility_20260917/oodc_F1/stjewm_trace_only_lam00/seed_0/final.pt"
    return f"/data/lx/tmp/results/accessibility_20260917/oodc_F1/stjewm_trace_only_{arm}/seed_0/final.pt"


if __name__ == "__main__":
    sys.exit(main())

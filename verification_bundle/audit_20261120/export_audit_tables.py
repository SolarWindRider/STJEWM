"""Re-export all audit tables with a strict 14-column schema.

Run: cd /home/lx/snn && /home/lx/miniconda3/envs/snn/bin/python \
      /home/lx/audit_20261120/export_audit_tables.py
Outputs into /home/lx/audit_20261120/: results.csv (14 cols, aligned),
fig2c_arms.csv, fig4b_heatmap.csv, fig5b_two_aggregations.csv,
fig1c_values.csv, maskobs_table.csv, e2_paired_summary.csv.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/lx/snn")
OUT = Path("/home/lx/audit_20261120")
FD = OUT / "fd_perenv.csv"

FIELDS = ["experiment", "figure_panel", "model", "checkpoint", "train_seed",
          "eval_seed", "split", "env", "modality", "metric", "value", "n",
          "source_path", "note"]


def wcsv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {path.name}: {len(rows)} rows")


def main() -> int:
    scaling = json.load(open(ROOT / "results/mech_0917_b/scaling/scaling_summary.json"))
    table = scaling["table"]
    access = json.load(open(ROOT / "results/mech_0917_b/accessibility/access_summary.json"))
    mask = json.load(open(ROOT / "results/mech_0917_b/maskobs/maskobs_summary.json"))
    f1 = json.load(open("/home/lx/figs_nmi/20260918/fig1_source.json"))
    models = list(table)

    rows: list[dict] = []

    # ---- Fig.2c arms (exact values; fixes the v1 question-marked guesses)
    arm_ckpt = {
        "lam0_ref": "/data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt",
        "lam0.1": "/data/lx/tmp/results/accessibility_20260917/oodc_F1/stjewm_trace_only_lam01/seed_0/final.pt",
        "lam1.0": "/data/lx/tmp/results/accessibility_20260917/oodc_F1/stjewm_trace_only_lam10/seed_0/final.pt",
        "lam10.0": "/data/lx/tmp/results/accessibility_20260917/oodc_F1/stjewm_trace_only_lam100/seed_0/final.pt",
    }
    arm_rows = []
    for arm, a in access["arms"].items():
        per = {e: a[e]["c20_contraction"] for e in ("cartpole_2d", "cheetah")}
        mean = sum(per.values()) / len(per)
        dec = [a[e].get("action_sens_rel") for e in per]
        arm_rows.append({"experiment": "P0.2", "figure_panel": "Fig2c",
                         "model": "stjewm_trace_only", "checkpoint": arm_ckpt[arm],
                         "train_seed": 0, "eval_seed": 0, "split": "oodc_F1",
                         "env": "cartpole+cheetah", "modality": "state",
                         "metric": "c20_dose_mean",
                         "value": f"{mean:.6g}", "n": "32x2",
                         "source_path": "results/mech_0917_b/accessibility/access_summary.json",
                         "note": f"per-env {per['cartpole_2d']:.4g}/{per['cheetah']:.4g}; "
                                 f"action_sens_rel {dec[0]:.3g}/{dec[1]:.3g}; "
                                 + ("original checkpoint, NOT fine-tuned"
                                    if arm == "lam0_ref" else
                                    "1-epoch fine-tune, all params + Linear(192,128) aux decoder")})
    wcsv(OUT / "fig2c_arms.csv", arm_rows)
    for r in arm_rows:
        rows.append({**r, "figure_panel": "Fig2c"})

    # ---- Fig.4b heatmap cells (definition: h1 = pooled FD gain at h=1;
    #      h5 cell = h1 gain x mean-dz ratio (approximates gain(h5) up to
    #      baseline-norm difference); SLIF obs h5 = underflow -> n/a in v2 panel)
    heat_rows = []
    for m in models:
        t = table[m]
        cells = {"obs_h1": t["gain_o_rel"], "obs_h5": t["gain_o_rel"] * t["dzo_decay5_over_1"],
                 "act_h1": t["gain_a_rel"], "act_h5": t["gain_a_rel"] * t["dza_decay5_over_1"]}
        for k, v in cells.items():
            if m == "slif_trace" and k == "obs_h5":
                note = "underflow: denominator below numerical resolution; v2 panel = grey n/a cell (v1 placeholder 17.9 retired)"
                val = "n/a"
            else:
                note = "h5 cell = h1 gain x mean-dz(5)/mean-dz(1); approximates gain(h5) up to ||z_base(h)|| difference"
                val = f"{v:.4g}"
            heat_rows.append({"experiment": "P0.1", "figure_panel": "Fig4b",
                              "model": m, "checkpoint": "", "train_seed": 0,
                              "eval_seed": 0, "split": "cross_benchmark_F1",
                              "env": "cartpole+cheetah", "modality": "state",
                              "metric": k, "value": val, "n": 128,
                              "source_path": "results/mech_0917_b/scaling/scaling_summary.json table",
                              "note": note})
    wcsv(OUT / "fig4b_heatmap.csv", heat_rows)
    rows.extend(heat_rows)

    # ---- Fig.5b two CSR aggregations (+ Spearman variants)
    agg_rows = []
    for m in models:
        t = table[m]
        agg_rows.append({"experiment": "P0.1", "figure_panel": "Fig5b",
                         "model": m, "checkpoint": "", "train_seed": 0,
                         "eval_seed": 0, "split": "cross_benchmark_F1",
                         "env": "cartpole+cheetah", "modality": "state",
                         "metric": "csr_as_published_mean_of_per_env_ratios",
                         "value": f"{t['csr']:.6g}", "n": 2,
                         "source_path": "scaling_summary.json table; formula gain_a(h1)/(gain_o(h1)+1e-8) per env, then mean",
                         "note": "ratio-then-aggregate"})
        agg_rows.append({"experiment": "P0.1", "figure_panel": "Fig5b",
                         "model": m, "checkpoint": "", "train_seed": 0,
                         "eval_seed": 0, "split": "cross_benchmark_F1",
                         "env": "cartpole+cheetah", "modality": "state",
                         "metric": "ratio_of_pooled_mean_gains",
                         "value": f"{t['gain_a_rel'] / t['gain_o_rel']:.6g}", "n": 2,
                         "source_path": "scaling_summary.json table",
                         "note": "aggregate-then-ratio (Fig.4a coordinate ratio)"})
    for name, key, n in [("as_published", "spearman_n8", 8), ("n6", "spearman_n6_no_mlp_slif", 6)]:
        sp_block = scaling[key]
        for met in ("csr", "c20"):
            d = sp_block[met]
            agg_rows.append({"experiment": "P0.1", "figure_panel": "Fig5",
                             "model": "ALL", "checkpoint": "", "train_seed": 0,
                             "eval_seed": 0, "split": "", "env": "", "modality": "state",
                             "metric": f"spearman_{met}_vs_env_sr_{name}",
                             "value": f"rho={d['rho']:+.3f} p={d['p']:.3g}", "n": n,
                             "source_path": "scaling_summary.json",
                             "note": "average-rank ties; env_sr has no ties"})
    wcsv(OUT / "fig5b_two_aggregations.csv", agg_rows)
    rows.extend(agg_rows)

    # ---- Fig.1c values (as plotted) + recomputed correlations
    d = {}
    for r in f1["data"]:
        if r["panel"] == "c":
            d.setdefault(r["model"], {})["sr"] = r["value"]
        elif r["panel"] == "c-left":
            d.setdefault(r["model"], {})["e1"] = r["value"]
        elif r["panel"] == "c-right":
            d.setdefault(r["model"], {})["r2"] = r["value"]
    fig1c_rows = []
    for m, v in d.items():
        for met, key in [("env_success_rate", "sr"),
                         ("teacher_forced_rollout_rel_L2_h1", "e1"),
                         ("linear_probe_R2_s_t", "r2")]:
            fig1c_rows.append({"experiment": "P0.3", "figure_panel": "Fig1c",
                               "model": m, "checkpoint": "", "train_seed": 0,
                               "eval_seed": 0, "split": "E1 10 splits",
                               "env": "100 cells" if key == "sr" else "cartpole+cheetah",
                               "modality": "state", "metric": met,
                               "value": f"{v[key]:.4g}", "n": 100 if key == "sr" else 256,
                               "source_path": "fig1_source.json; rollout_summary.json; nonlinear_probe_summary.json",
                               "note": "e1 = teacher-forced latent rollout, E_lat(1)=mean||zhat-z*||/(||z*||+1e-8)"})
    def spearman(x, y):
        def rank(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v); i = 0
            while i < len(o):
                j = i
                while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                    j += 1
                for k in range(i, j + 1):
                    r[o[k]] = (i + j) / 2.0
                i = j + 1
            return r
        rx, ry = rank(x), rank(y)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return num / den if den else float("nan")
    ms = list(d)
    sr = [d[m]["sr"] for m in ms]
    e1 = [d[m]["e1"] for m in ms]
    r2 = [d[m]["r2"] for m in ms]
    keep = [i for i, m in enumerate(ms) if m not in ("MLP", "Stacked-LIF trace")]
    fig1c_rows.append({"experiment": "P0.3", "figure_panel": "Fig1c", "model": "ALL",
                       "checkpoint": "", "train_seed": 0, "eval_seed": 0, "split": "",
                       "env": "", "modality": "state",
                       "metric": "spearman_e1_vs_sr_all8/n6",
                       "value": f"{spearman(e1, sr):+.3f}/{spearman([e1[i] for i in keep], [sr[i] for i in keep]):+.3f}",
                       "n": 8, "source_path": "recomputed", "note": ""})
    fig1c_rows.append({"experiment": "P0.3", "figure_panel": "Fig1c", "model": "ALL",
                       "checkpoint": "", "train_seed": 0, "eval_seed": 0, "split": "",
                       "env": "", "modality": "state",
                       "metric": "spearman_linearR2_vs_sr_all8/n6",
                       "value": f"{spearman(r2, sr):+.3f}/{spearman([r2[i] for i in keep], [sr[i] for i in keep]):+.3f}",
                       "n": 8, "source_path": "recomputed", "note": "negative = dissociation"})
    wcsv(OUT / "fig1c_values.csv", fig1c_rows)
    rows.extend(fig1c_rows)

    # ---- Fig.4c maskobs table
    mask_rows = []
    quartet = ["trace", "lewm", "gru", "membrane"]
    hints = {"trace": "stjewm_trace_only", "lewm": "lewm_baseline_v2",
             "gru": "gru_baseline", "membrane": "stjewm_membrane_readout"}
    for m in quartet:
        rel = mask["pooled"][hints[m]]["rel_l2"]
        for cond in ("last25", "win25"):
            mask_rows.append({"experiment": "P0.1", "figure_panel": "Fig4c",
                              "model": m, "checkpoint": "", "train_seed": 0,
                              "eval_seed": 0, "split": "cross_benchmark_F1",
                              "env": "cartpole+cheetah", "modality": "state",
                              "metric": f"rel_l2_{cond}",
                              "value": f"{rel[cond]:.4f}", "n": 768,
                              "source_path": "results/mech_0917_b/maskobs/maskobs_summary.json",
                              "note": "zeroing same 25% dims on last frame only vs every frame; 3 shared mask seeds; rel=||dz||/(||z_clean||+1e-8)"})
    wcsv(OUT / "maskobs_table.csv", mask_rows)
    rows.extend(mask_rows)

    # ---- E2 paired summary (trace vs gru/lewm, per training seed)
    e2 = list(csv.DictReader(open("/home/lx/snn/verification_bundle/e2_seed_paired.csv")))
    by = defaultdict(dict)
    for r in e2:
        if r["experiment"] == "E2":
            by[(r["split"], r["env_id"], r["seed"])][r["model"]] = float(r["success_rate_env"])
    e2_rows = []
    for base in ("gru_baseline", "lewm_baseline_v2"):
        per_seed = defaultdict(list)
        for (sp, e, sd), mm in by.items():
            if "stjewm_trace_only" in mm and base in mm:
                per_seed[sd].append(mm["stjewm_trace_only"] - mm[base])
        for sd, diffs in sorted(per_seed.items()):
            wins = sum(1 for x in diffs if x > 0)
            ties = sum(1 for x in diffs if x == 0)
            e2_rows.append({"experiment": "P1", "figure_panel": "Results-multiseed",
                            "model": f"trace_vs_{base}", "checkpoint": "",
                            "train_seed": sd, "eval_seed": 0,
                            "split": "3 splits pooled", "env": "37 cells",
                            "modality": "state", "metric": "mean_paired_diff",
                            "value": f"{sum(diffs) / len(diffs):+.3f}", "n": len(diffs),
                            "source_path": "verification_bundle/e2_seed_paired.csv",
                            "note": f"win/tie/lose={wins}/{ties}/{len(diffs) - wins - ties}; shared (init,goal) instances"})
    wcsv(OUT / "e2_paired_summary.csv", e2_rows)
    rows.extend(e2_rows)

    # ---- Fig.5b per-env raw FD gains (from deterministic re-export)
    if FD.exists():
        with open(FD) as f:
            for r in csv.DictReader(f):
                rows.append({"experiment": "P0.1", "figure_panel": "Fig5b-appendix",
                             "model": r["model"],
                             "checkpoint": r.get("checkpoint", ""),
                             "train_seed": 0, "eval_seed": 0,
                             "split": "cross_benchmark_F1", "env": r["env"],
                             "modality": "state",
                             "metric": f"gains_{r.get('rel', '')}",
                             "value": f"g_a(h1)={r['gain_a_h1']}; g_o(h1)={r['gain_o_h1']}; csr_per_env={r['csr_per_env_r1']}",
                             "n": r.get("n_windows") or "",
                             "source_path": "fd_perenv.csv (SEED_WIN=0 deterministic re-export)",
                             "note": "per-env raw values; original per-env FD was not persisted"})
    else:
        print("[warn] fd_perenv.csv not present yet; rerun export_audit_tables.py after bg job")

    wcsv(OUT / "results.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

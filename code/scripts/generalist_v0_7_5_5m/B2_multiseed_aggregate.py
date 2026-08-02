#!/usr/bin/env python3
"""B2 multi-seed aggregator.

Reads:
  results/5m/<split>/<MODEL>/seed_0/eval_<env>.json
  results/5m_seed1/<split>/<MODEL>/seed_0/eval_<env>.json
  results/5m_seed2/<split>/<MODEL>/seed_0/eval_<env>.json

Computes per-model cos_dist mean+/-std across 3 seeds x 3 splits, plus
LeWM@0.05/env-SR. Writes summary.json + summary.md to
results/journal_prep/B2_multiseed/.
"""
import os, sys, json, math, statistics
from pathlib import Path

ROOT = Path("/home/lx/snn")
SPLITS = ["cross_benchmark_F1", "oodc_F2", "generalist_16env"]
MODELS = ["stjewm_trace_only", "stjewm_spike_only", "slt_lif_mpc_trace",
          "lewm_baseline_v2", "mlp_baseline"]

# Path resolution: same naming convention as training/eval
def eval_path(seed_dir, split, model):
    """seed_dir: 'results/5m' or 'results/5m_seedN'"""
    return ROOT / seed_dir / split / model / "seed_0"

SEED_DIRS = {0: "results/5m", 1: "results/5m_seed1", 2: "results/5m_seed2"}

def load_eval_json(p):
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None

def gather_seed_per_env(seed_dir, splits):
    """For a given seed_dir, return:
       {(split, model, env): {mean_cos_dist, success_rate_lewm_005, ...}}
    """
    out = {}
    for split in splits:
        for model in MODELS:
            d = eval_path(seed_dir, split, model)
            if not d.exists():
                continue
            for f in sorted(d.glob("eval_*.json")):
                env = f.stem.replace("eval_", "")
                j = load_eval_json(f)
                if j is None:
                    continue
                out[(split, model, env)] = {
                    "mean_cos_dist": j.get("mean_cos_dist"),
                    "mean_phys_dist": j.get("mean_phys_dist"),
                    "success_rate_lewm": j.get("success_rate_lewm"),
                    "success_rate_lewm_005": j.get("success_rate_lewm_005"),
                    "success_rate_lewm_001": j.get("success_rate_lewm_001"),
                    "success_rate_env": j.get("success_rate_env"),
                }
    return out

def main():
    # Per-seed per-env tables
    per_seed = {}
    for seed, sdir in SEED_DIRS.items():
        per_seed[seed] = gather_seed_per_env(sdir, SPLITS)

    # Find the common envs (those present in all 3 seeds for a given split+model)
    common = {}  # (split, model, env)
    split_model_envs_2 = {}  # (split, model) -> set(envs)
    for k in per_seed[2].keys():
        common[k] = True
    for k in per_seed[1].keys():
        if k in common:
            pass
        else:
            common.pop(k, None)
    for k in per_seed[0].keys():
        if k in common:
            pass
        else:
            common.pop(k, None)

    # Per-(split, model) aggregation across seeds
    # Per-seed cos_dist: average over envs (or use cum mean_cos_dist * n / 5)
    # json has mean over n_episodes (5); can do weighted avg over envs.
    def per_seed_means(per_seed_data, split, model):
        envs = [k for k in per_seed_data.keys() if k[0] == split and k[1] == model]
        if not envs:
            return None, None
        cos_vals = [per_seed_data[k]["mean_cos_dist"] for k in envs
                    if per_seed_data[k]["mean_cos_dist"] is not None]
        sr5_vals = [per_seed_data[k]["success_rate_lewm_005"] for k in envs
                    if per_seed_data[k]["success_rate_lewm_005"] is not None]
        if not cos_vals:
            return None, None
        cos = sum(cos_vals) / len(cos_vals)
        sr5 = sum(sr5_vals) / len(sr5_vals) if sr5_vals else None
        return cos, sr5

    rows = []
    for split in SPLITS:
        for model in MODELS:
            seed_cos = []
            seed_sr5 = []
            seed_summary = {}
            for seed in [0, 1, 2]:
                cos, sr5 = per_seed_means(per_seed[seed], split, model)
                if cos is not None:
                    seed_cos.append(cos)
                    seed_sr5.append(sr5 if sr5 is not None else float("nan"))
                    seed_summary[f"seed{seed}_cos_dist"] = cos
                    seed_summary[f"seed{seed}_sr_at_005"] = sr5
            row = {
                "split": split,
                "model": model,
                "n_envs": len([k for k in per_seed[0] if k[0] == split and k[1] == model]),
                "n_seeds_with_data": len(seed_cos),
            }
            if seed_cos:
                row["cos_dist_mean_across_seeds"] = sum(seed_cos) / len(seed_cos)
                if len(seed_cos) > 1:
                    row["cos_dist_std_across_seeds"] = statistics.stdev(seed_cos)
                else:
                    row["cos_dist_std_across_seeds"] = 0.0
                # 95% CI of the mean via t-distribution (approx with normal for n=3)
                # For n=3, t_0.025(2) = 4.303
                tcrit = 4.303 if len(seed_cos) == 3 else 1.96
                row["cos_dist_95ci_low"] = row["cos_dist_mean_across_seeds"] - tcrit * row["cos_dist_std_across_seeds"] / math.sqrt(len(seed_cos))
                row["cos_dist_95ci_high"] = row["cos_dist_mean_across_seeds"] + tcrit * row["cos_dist_std_across_seeds"] / math.sqrt(len(seed_cos))
                # Also aggregate sr_at_005
                sr5_clean = [v for v in seed_sr5 if not (isinstance(v, float) and math.isnan(v))]
                if sr5_clean:
                    row["sr005_mean_across_seeds"] = sum(sr5_clean) / len(sr5_clean)
                    if len(sr5_clean) > 1:
                        row["sr005_std_across_seeds"] = statistics.stdev(sr5_clean)
                    else:
                        row["sr005_std_across_seeds"] = 0.0
            row.update(seed_summary)
            rows.append(row)

    # Per-model aggregated across splits (the headline "family partition" claim)
    model_rows = []
    for model in MODELS:
        cos_per_seed = {0: [], 1: [], 2: []}
        for split in SPLITS:
            seed_cos_for_row = []
            for seed in [0, 1, 2]:
                cos, _ = per_seed_means(per_seed[seed], split, model)
                if cos is not None:
                    cos_per_seed[seed].append(cos)
        # Average across splits for each seed
        seed_means_per_split = []
        for seed in [0, 1, 2]:
            if cos_per_seed[seed]:
                seed_means_per_split.append(sum(cos_per_seed[seed]) / len(cos_per_seed[seed]))
        if not seed_means_per_split:
            continue
        mean = sum(seed_means_per_split) / len(seed_means_per_split)
        std = statistics.stdev(seed_means_per_split) if len(seed_means_per_split) > 1 else 0.0
        tcrit = 4.303 if len(seed_means_per_split) == 3 else 1.96
        model_rows.append({
            "model": model,
            "n_seeds": len(seed_means_per_split),
            "seed_means_per_split": seed_means_per_split,
            "mean_cos_dist": mean,
            "std_cos_dist": std,
            "ci95_low": mean - tcrit * std / math.sqrt(len(seed_means_per_split)),
            "ci95_high": mean + tcrit * std / math.sqrt(len(seed_means_per_split)),
        })

    out_dir = ROOT / "results/journal_prep/B2_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
        json.dump({"per_split_model": rows, "per_model_aggregate": model_rows,
                   "splits": SPLITS, "models": MODELS}, f, indent=2)
    print(f"Wrote {out_dir / 'summary.json'}")

    # Quick console dump
    print()
    print(f"{'Model':<25s} {'mean±std':<18s} {'95% CI':<20s} n")
    for r in model_rows:
        print(f"{r['model']:<25s} {r['mean_cos_dist']:.4f}±{r['std_cos_dist']:.4f}    "
              f"[{r['ci95_low']:.4f},{r['ci95_high']:.4f}]  {r['n_seeds']}")
    print()
    print("Per-split breakdown:")
    for split in SPLITS:
        print(f"  === {split} ===")
        for r in rows:
            if r["split"] == split:
                cs = r.get("cos_dist_mean_across_seeds")
                sd = r.get("cos_dist_std_across_seeds")
                if cs is not None:
                    print(f"    {r['model']:<25s} {cs:.4f}±{sd:.4f}  n_seeds={r['n_seeds_with_data']}")

if __name__ == "__main__":
    main()

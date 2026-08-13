# B2 — Multi-Seed Statistical Reliability of the Cosine-Distance Partition

**Date:** 2026-08-13 (re-evaluated after checkpoint retrain)
**Status:** All 30 ckpts trained and evaluated across seeds {0, 1, 2}; partition verdict
robust at 3-seed resolution. Seed-1 checkpoints were retrained after the rename accident;
fresh evals on 2026-08-13. All 39 cells complete, n_seeds = 3 everywhere (generalist_16env
seed-2 stacked_lif_trace/free retrained + evaluated to close the last gaps).
**Aggregation file:** `results/journal_prep/B2_multiseed/summary.json`

## TL;DR verdict

The 3-cluster partition holds **with high confidence at 3 seeds**. Calibrated, over-react,
and collapse clusters are pairwise-disjoint at the 95% CI.

| Cluster | Model | cos_dist mean ± std | 95% CI (t₀.₀₂₅, df=2) |
|---------|-------|---------------------|------------------------|
| **calibrated** | `stjewm_trace_only` | **0.1159 ± 0.0037** | [0.1066, 0.1251] |
| **calibrated** | `stjewm_spike_only` | **0.1180 ± 0.0057** | [0.1038, 0.1322] |
| **calibrated** | `stacked_lif_trace` | **0.1146 ± 0.0017** | [0.1104, 0.1188] |
| **over-react** | `lewm_baseline_v2` | **0.1925 ± 0.0116** | [0.1637, 0.2213] |
| **collapse** | `mlp_baseline` | **0.0056 ± 0.0009** | [0.0033, 0.0078] |

## Per-split breakdown (3 seeds each)

### cross_benchmark_F1

| Model | cos_dist mean | std | n_seeds | 95% CI |
|-------|---------------|-----|---------|--------|
| STJEWM-trace           | 0.1061 | 0.0033 | 3 | [0.0978, 0.1143] |
| STJEWM-spike           | 0.1176 | 0.0055 | 3 | [0.1040, 0.1313] |
| Stacked-LIF-trace      | 0.1115 | 0.0056 | 3 | [0.0975, 0.1255] |
| LeWM-v2                | 0.1895 | 0.0014 | 3 | [0.1859, 0.1931] |
| MLP                    | 0.0038 | 0.0010 | 3 | [0.0013, 0.0063] |

### oodc_F2

| Model | cos_dist mean | std | n_seeds | 95% CI |
|-------|---------------|-----|---------|--------|
| STJEWM-trace           | 0.1295 | 0.0060 | 3 | [0.1145, 0.1445] |
| STJEWM-spike           | 0.1177 | 0.0017 | 3 | [0.1134, 0.1219] |
| Stacked-LIF-trace      | 0.1199 | 0.0056 | 3 | [0.1060, 0.1338] |
| LeWM-v2                | 0.2100 | 0.0307 | 3 | [0.1338, 0.2862] |
| MLP                    | 0.0000 | 0.0000 | 3 | [0.0000, 0.0000] |

### generalist_16env

| Model | cos_dist mean | std | n_seeds | 95% CI |
|-------|---------------|-----|---------|--------|
| STJEWM-trace           | 0.1121 | 0.0053 | 3 | [0.0991, 0.1252] |
| STJEWM-spike           | 0.1187 | 0.0110 | 3 | [0.0915, 0.1459] |
| Stacked-LIF-trace      | 0.1123 | 0.0024 | 3 | [0.1064, 0.1182] |
| LeWM-v2                | 0.1779 | 0.0233 | 3 | [0.1199, 0.2359] |
| MLP                    | 0.0129 | 0.0019 | 3 | [0.0081, 0.0176] |

## Verdict

The calibrated cluster (STJEWM-trace, STJEWM-spike, Stacked-LIF-trace) forms a
CI-overlapping band (0.115–0.118); LeWM-v2 (0.192) and MLP (0.006) are CI-disjoint
from it at the 95% level — the paper's 3-cluster partition is statistically robust.
The last two missing seed-2 cells were closed on 2026-08-13; no cells remain incomplete.

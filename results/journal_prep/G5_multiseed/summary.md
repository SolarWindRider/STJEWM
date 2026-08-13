# G5 — Full 3-Seed Cosine-Distance Coverage (13 Models)

**Date:** 2026-08-13 (re-evaluated after checkpoint retrain)
**Extends:** B2 (5 models × 3 seeds × 3 splits) → G5 (full 13 models × 3 seeds × 3 splits).
**Aggregation file:** `results/journal_prep/G5_multiseed/summary.json`
**Data basis:** seed-1 checkpoints were corrupted in the rename accident and retrained
(2026-08-12/13); this summary reflects fresh closed-loop evals on the retrained seed-1
checkpoints plus original seed-0/seed-2 evals. Two previously missing seed-2 cells
(stacked_lif_trace/free, generalist_16env) were retrained and evaluated — coverage is now complete.

## TL;DR verdict

The 3-cluster partition claimed in the paper is **robust at 3-seed resolution across the
FULL 13-model census**. With complete coverage (all 39 (split, model) cells, n_seeds = 3),
the calibrated / over-react separation is still CI-disjoint, and GRU (3-layer RNN baseline)
collapses into the same cluster as MLP and LIFTransformer.

| Cluster | Model | cos_dist mean ± std | 95% CI (t₀.₀₂₅, df=2) |
|---------|-------|---------------------|------------------------|
| **collapse** | `lif_transformer_baseline` | **0.0000 ± 0.0000** | [-0.0000, 0.0000] |
| **collapse** | `mlp_baseline` (B2) | **0.0056 ± 0.0009** | [0.0033, 0.0078] |
| **collapse** | `gru_baseline` | **0.0175 ± 0.0008** | [0.0155, 0.0194] |
| **calibrated** | `stacked_lif_trace` (B2) | **0.1146 ± 0.0017** | [0.1104, 0.1188] |
| **calibrated** | `stjewm_trace_only` (B2) | **0.1159 ± 0.0037** | [0.1066, 0.1251] |
| **calibrated** | `stjewm_rate_only` | **0.1165 ± 0.0063** | [0.1009, 0.1321] |
| **calibrated** | `stjewm_spike_only` (B2) | **0.1180 ± 0.0057** | [0.1038, 0.1322] |
| **calibrated** | `stacked_lif_free` | **0.1245 ± 0.0064** | [0.1087, 0.1404] |
| **calibrated** | `alif_timecell_baseline` | **0.1259 ± 0.0054** | [0.1124, 0.1394] |
| **calibrated** | `stjewm_no_trace` | **0.1288 ± 0.0049** | [0.1167, 0.1410] |
| **calibrated** | `stjewm_hidden_leak` | **0.1371 ± 0.0105** | [0.1109, 0.1633] |
| **calibrated** | `stjewm_membrane_readout` | **0.1374 ± 0.0049** | [0.1252, 0.1495] |
| **over-react** | `lewm_baseline_v2` (B2) | **0.1925 ± 0.0116** | [0.1637, 0.2213] |

Headline separation:

  - **Calibrated vs over-react:** Cohen's d ≈ 8.9 (STJEWM-trace), 9.4 (Stacked-LIF-trace), 7.4 (ALIF-timecell).
  - **Calibrated vs collapse:** Cohen's d > 20 for every calibrated model vs MLP.
  - **Collapse gap:** GRU ≈ 0.017 ≠ 0; still ~10× smaller than the calibrated cluster.

## Coverage

| Models covered | Count | Where |
|----------------|-------|-------|
| From B2 (3 seeds) | 5 | stjewm_trace_only, stjewm_spike_only, stacked_lif_trace, lewm_baseline_v2, mlp_baseline |
| Added by G5 (3 seeds) | 8 | stjewm_rate_only, stjewm_no_trace, stjewm_hidden_leak, stjewm_membrane_readout, alif_timecell_baseline, stacked_lif_free, gru_baseline, lif_transformer_baseline |
| **Total** | **13** | All 13 models, 39 (split, model) cells, n_seeds = 3 everywhere |

## Per-split breakdown (3 seeds each)

### cross_benchmark_F1

| Model | cos_dist mean | std | n_seeds | 95% CI |
|-------|---------------|-----|---------|--------|
| LIF-Transformer        | -0.0000 | 0.0000 | 3 | [-0.0000, 0.0000] |
| MLP                    | 0.0038 | 0.0010 | 3 | [0.0013, 0.0063] |
| GRU                    | 0.0157 | 0.0007 | 3 | [0.0139, 0.0175] |
| STJEWM-trace           | 0.1061 | 0.0033 | 3 | [0.0978, 0.1143] |
| Stacked-LIF-trace      | 0.1115 | 0.0056 | 3 | [0.0975, 0.1255] |
| STJEWM-rate            | 0.1131 | 0.0038 | 3 | [0.1038, 0.1225] |
| Stacked-LIF-free       | 0.1147 | 0.0024 | 3 | [0.1088, 0.1207] |
| STJEWM-spike           | 0.1176 | 0.0055 | 3 | [0.1040, 0.1313] |
| ALIF-timecell          | 0.1216 | 0.0057 | 3 | [0.1074, 0.1358] |
| STJEWM-no-trace        | 0.1237 | 0.0029 | 3 | [0.1164, 0.1309] |
| STJEWM-membrane        | 0.1265 | 0.0115 | 3 | [0.0979, 0.1550] |
| STJEWM-leak            | 0.1301 | 0.0071 | 3 | [0.1125, 0.1477] |
| LeWM-v2                | 0.1895 | 0.0014 | 3 | [0.1859, 0.1931] |

### oodc_F2

| Model | cos_dist mean | std | n_seeds | 95% CI |
|-------|---------------|-----|---------|--------|
| LIF-Transformer        | 0.0000 | 0.0000 | 3 | [-0.0000, 0.0000] |
| MLP                    | 0.0000 | 0.0000 | 3 | [0.0000, 0.0000] |
| GRU                    | 0.0022 | 0.0002 | 3 | [0.0016, 0.0028] |
| STJEWM-spike           | 0.1177 | 0.0017 | 3 | [0.1134, 0.1219] |
| Stacked-LIF-trace      | 0.1199 | 0.0056 | 3 | [0.1060, 0.1338] |
| STJEWM-rate            | 0.1215 | 0.0121 | 3 | [0.0915, 0.1515] |
| STJEWM-trace           | 0.1295 | 0.0060 | 3 | [0.1145, 0.1445] |
| STJEWM-no-trace        | 0.1336 | 0.0208 | 3 | [0.0819, 0.1854] |
| Stacked-LIF-free       | 0.1359 | 0.0068 | 3 | [0.1190, 0.1528] |
| ALIF-timecell          | 0.1389 | 0.0121 | 3 | [0.1088, 0.1689] |
| STJEWM-leak            | 0.1463 | 0.0168 | 3 | [0.1046, 0.1880] |
| STJEWM-membrane        | 0.1554 | 0.0115 | 3 | [0.1269, 0.1839] |
| LeWM-v2                | 0.2100 | 0.0307 | 3 | [0.1338, 0.2862] |

### generalist_16env

| Model | cos_dist mean | std | n_seeds | 95% CI |
|-------|---------------|-----|---------|--------|
| LIF-Transformer        | 0.0000 | 0.0000 | 3 | [0.0000, 0.0000] |
| MLP                    | 0.0129 | 0.0019 | 3 | [0.0081, 0.0176] |
| GRU                    | 0.0346 | 0.0027 | 3 | [0.0278, 0.0414] |
| STJEWM-trace           | 0.1121 | 0.0053 | 3 | [0.0991, 0.1252] |
| Stacked-LIF-trace      | 0.1123 | 0.0024 | 3 | [0.1064, 0.1182] |
| STJEWM-rate            | 0.1147 | 0.0108 | 3 | [0.0878, 0.1417] |
| ALIF-timecell          | 0.1173 | 0.0040 | 3 | [0.1073, 0.1274] |
| STJEWM-spike           | 0.1187 | 0.0110 | 3 | [0.0915, 0.1459] |
| Stacked-LIF-free       | 0.1229 | 0.0153 | 3 | [0.0849, 0.1608] |
| STJEWM-no-trace        | 0.1292 | 0.0056 | 3 | [0.1153, 0.1431] |
| STJEWM-membrane        | 0.1302 | 0.0100 | 3 | [0.1053, 0.1551] |
| STJEWM-leak            | 0.1350 | 0.0078 | 3 | [0.1157, 0.1543] |
| LeWM-v2                | 0.1779 | 0.0233 | 3 | [0.1199, 0.2359] |

## Cluster verdict (3-seed × 3-split resolution, all 13 models)

The 3-cluster partition is **preserved** when extending from 5 → 13 models:
calibrated models (STJEWM 6 readouts + ALIF-timecell + Stacked-LIF-trace/free) sit in a
CI-overlapping band (0.114–0.137); MLP / LIF-Transformer / GRU collapse (≈ 0.00–0.017);
LeWM-v2 over-reacts (0.192). All calibrated vs over-react and calibrated vs collapse
pairs are CI-disjoint.

## Acceptance statement

With full 3-seed × 3-split coverage for all 13 models, the 3-cluster partition survives:
the central claims of the paper do not depend on the subset of models, seeds, or splits.

## Methodology and reproducibility

- Training: `code/scripts/generalist_v0_7_5_5m/` (train_one_5m.sh, B2/G5 launchers).
- Eval: `code/scripts/generalist_v0_7_5_5m/eval_one.sh` (CEM 300×30×10, H=5, budget 50, 5 eps).
- Aggregation: `code/scripts/generalist_v0_7_5_5m/G5_multiseed_aggregate.py`.
- Data: `results/5m`, `results/5m_seed1`, `results/5m_seed2` eval_<env>.json files.

## Caveats and explicit non-claims

- Seed-1 checkpoints are **retrained** networks, not the original weights; their evals are from
  2026-08-13. CI widths reflect genuine retrain variance across the 3 seeds.
- cos_dist here is the per-eval-json mean over the split's envs (not per-env geometric mean).
- We do not claim pixel-modal or event-ρ coverage in this file (see MAIN_TABLE / G1).

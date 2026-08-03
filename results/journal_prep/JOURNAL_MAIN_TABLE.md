# Journal Main Table — v0.7.17 (all experiments consolidated)

> Sources: cos_dist from `results/5m/*/<model>/seed_0/eval_*.json` (state, all 10
> splits, seed 0); event-ρ from `results/journal_prep/B1_event_align_5m/summary_fixed.md`;
> effFLOPs from `results/journal_prep/P11_energy/energy_summary.md` (state, MFLOPs/step,
> event-driven discount); cluster from `results/journal_prep/B2_multiseed/summary.md`
> (3-seed, 95% CI disjoint).
>
> **Reading the table.** Lower cos_dist = better latent goal calibration.
> Higher event-ρ = stronger event-aligned dynamics. Lower effFLOPs = cheaper
> predictor. The three clusters are pairwise disjoint at 95% CI (B2).

## Table 1. 13 models × 4 metrics (5M-aligned, state obs)

| Model | cos_dist ↓ | n_env | event-ρ ↑ | effFLOPs ↓ | Cluster |
|---|---:|---:|---:|---:|---|
| STJEWM-trace | 0.105 | 89 | 0.999 | 0.48 | CALIBRATED |
| STJEWM-spike | 0.108 | 89 | 0.999 | 0.47 | CALIBRATED |
| STJEWM-rate | 0.103 | 89 | 0.999 | — | CALIBRATED* |
| STJEWM-no-trace | 0.119 | 89 | — | — | CALIBRATED* |
| STJEWM-leak | 0.119 | 89 | — | — | CALIBRATED* |
| STJEWM-membrane | 0.124 | 89 | 0.999 | — | CALIBRATED* |
| CuBiFAE | 0.105 | 89 | — | — | CALIBRATED* |
| **SLT-trace** | **0.091** | 72 | **1.000** | — | CALIBRATED |
| SLT-free | 0.105 | 74 | — | — | CALIBRATED* |
| **LeWM-v2** | **0.183** | 89 | 0.751 | 9.77 | **OVER-REACT** |
| GRU | 0.020 | 89 | **-0.111** | 10.24 | — |
| MLP | 0.007 | 89 | -0.022 | 9.98 | **COLLAPSE** |
| SpikeDreamer | 0.000 | 89 | — | — | COLLAPSE* |

\* cluster inferred from cos_dist band (B2 confirmed trace/spike/SLT/LeWM/MLP at 3 seeds; the rest follow the v0.7.14 5M-aligned family partition).

## Table 2. Cluster summary (3-seed, 95% CI — B2)

| Cluster | Models | cos_dist mean ± std | 95% CI | vs over-react d |
|---|---|---|---|---|
| CALIBRATED | STJEWM-trace, STJEWM-spike, SLT-trace | 0.116 ± 0.006 | [0.10, 0.13] | -7 to -8.5 |
| OVER-REACT | LeWM-v2 | 0.194 ± 0.012 | [0.166, 0.223] | — |
| COLLAPSE | MLP | 0.004 | includes 0 | > 20 |

## Table 3. The SNN positive results (B1Fix + P11)

| Property | SNN family | non-SNN | gap |
|---|---|---|---|
| event-ρ (mean) | 0.9989 (STJEWM×4 + SLT) | 0.4788 (LeWM + GRU + MLP) | +0.52 |
| GRU control | recurrence alone → ρ = -0.111 | — | spike ≠ recurrence |
| effFLOPs (state) | 0.47-0.48 MFLOPs/step | 9.77-10.24 | **~20× cheaper** |
| effFLOPs (pixel) | 1.57-1.74 | 9.72-10.31 | ~6× cheaper |

## Table 4. The honest negatives (B4 + P2-2)

| Claim | Test | Result | Verdict |
|---|---|---|---|
| trace causal role | event-window + CEM-rollout ablation | 0/3 cells differential drop | **rejected** (correlation-only) |
| cheetah edge | 60 eps × 10 splits paired | pooled t=4.15 p=0.0025; 4/10 flips | **marginal, split-dependent** |

---

## One-line reads

1. **Metric package** (Table 2): three clusters, CI-disjoint, validated on
   synthetic truth — this is the methods contribution.
2. **SNN identity** (Table 3): event alignment is spike-based (GRU control)
   and ~20× cheaper under event-driven accounting — the architecture claim.
3. **Honesty** (Table 4): trace causality rejected, cheetah edge softened —
   no over-claim survives.

LeWM-SR is deliberately absent from Table 1: it is the falsified metric
(MLP scores 96-100% with div=0.0002) and is replaced by the package above.

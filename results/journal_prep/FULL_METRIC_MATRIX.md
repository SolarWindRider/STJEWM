# Full Metric Matrix — 13 models × all experiments (v0.7.17)

> Each column = one experiment's metric. `—` = not measured for that model.
> **Columns and sources:**
> - `cos_dist`, `LeWM@0.05`, `env-SR`: `results/5m/*/<m>/seed_0/eval_*.json` (state, 10 splits, seed 0, CEM 300×30×10 H=5 budget 50)
> - `event-ρ`: B1+B1Fix (`event_align.py`, 200-step random policy, 4 envs × 2 splits, Pearson corr obs↔latent first-diff)
> - `event-AUROC`: 5M-aligned event probes (6 targets: contact/entered/k10/k5/motion/target), 481 OK cells
> - `effFLOPs/dense/sparsity/trainable`: P11 (`measure_energy.py`, state, per-step, event-driven discount, predictor-only)
> - `3-seed cos`: B2 (5 models × 3 splits × 3 seeds, mean±std, 95% CI)
> - `pos R²/future_k R²/goal_dir R²`: B3-fixed linear probes (cross_benchmark_F1)

## Table 1. Full matrix

| Model | n_env | cos_dist ↓ | LeWM@.05 | env-SR | event-ρ ↑ | AUROC-cont | AUROC-k5 | AUROC-motion | effFLOPs ↓ | dense | spar% | train. | 3-seed cos ± | pos R² | fut R² | goal R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STJEWM-trace | 89 | 0.105 | 0.373 | 0.000 | 0.9987 | 0.517 | 0.540 | 0.477 | 0.483 | 5.23 | 93.3 | 2.70 | 0.119±0.002 | -0.017 | -0.024 | -0.086 |
| STJEWM-spike | 89 | 0.108 | 0.373 | 0.000 | 0.9988 | 0.517 | 0.542 | 0.491 | 0.465 | 5.16 | 93.6 | 2.70 | 0.114±0.010 | -0.066 | -0.035 | -0.053 |
| STJEWM-rate | 89 | 0.103 | 0.416 | 0.000 | 0.9988 | 0.525 | 0.532 | 0.496 | — | — | — | — | — | — | — | — |
| STJEWM-no-trace | 89 | 0.119 | 0.364 | 0.000 | — | 0.516 | 0.526 | 0.485 | — | — | — | — | — | — | — | — |
| STJEWM-leak | 89 | 0.119 | 0.398 | 0.000 | — | 0.524 | 0.565 | 0.477 | — | — | — | — | — | — | — | — |
| STJEWM-membrane | 89 | 0.124 | 0.375 | 0.000 | 0.9987 | 0.526 | 0.504 | 0.490 | — | — | — | — | — | — | — | — |
| CuBiFAE | 89 | 0.105 | 0.422 | 0.000 | — | 0.507 | 0.518 | 0.497 | — | — | — | — | — | — | — | — |
| SLT-trace | 72 | 0.091 | 0.422 | 0.000 | 0.9996 | — | — | — | — | — | — | — | 0.115±0.004 | — | — | — |
| SLT-free | 74 | 0.105 | 0.386 | 0.000 | — | — | — | — | — | — | — | — | — | — | — | — |
| LeWM-v2 | 89 | 0.183 | 0.225 | 0.000 | 0.7515 | — | — | — | 9.770 | 9.77 | 0.0 | 4.97 | 0.194±0.011 | 0.605 | 0.396 | 0.168 |
| GRU | 89 | 0.020 | 0.894 | 0.000 | -0.1111 | 0.595 | 0.560 | 0.545 | 10.241 | 10.24 | 0.0 | 5.13 | — | — | — | — |
| MLP | 89 | 0.007 | 0.948 | 0.000 | -0.0220 | — | — | — | 9.984 | 9.98 | 0.0 | 5.00 | 0.004±0.000 | -0.043 | -0.023 | -0.042 |
| SpikeDreamer | 89 | 0.000 | 1.000 | 0.000 | — | — | — | — | — | — | — | — | — | — | — | — |

## Table 2. Per-experiment detail links

| Experiment | File |
|---|---|
| B1/B1Fix event-align | `results/journal_prep/B1_event_align_5m/summary_fixed.md` |
| B2 3-seed | `results/journal_prep/B2_multiseed/summary.md` |
| B3 probe fix | `results/journal_prep/B3_probe_fix/probe_table_fixed.md` |
| B4 ablation | `results/journal_prep/B4_ablation/summary.md` |
| P11 energy | `results/journal_prep/P11_energy/energy_summary.md` |
| P12 synthetic | `results/journal_prep/P12_synthetic/SUMMARY.md` |
| P13 multi-epoch | `results/journal_prep/P13_multi_epoch/summary.md` |
| P22 cheetah | `results/journal_prep/P22_cheetah/verdict_60eps.md` |

## Notes

- **env-SR = 0 for all**: 5-step CEM vs 25-step goal is a pipeline ceiling (v0.7.13 bug #3), not a model property.
- **LeWM@0.05 is the falsified metric**: MLP 0.948 with div=0.0002 — included for the falsification narrative only.
- **SLT-trace n=72** (vs 89): 17 eval cells missing in the original 5m run (F2/F3 partial).
- **event-ρ**: SNN family (STJEWM×4 + SLT) ≥ 0.9987; GRU -0.111 → event alignment is spike-based, not recurrence.
- **event-AUROC**: all ≈ 0.5 (chance) at 1 epoch; P13 shows LeWM recovers to 0.63 at 3 epochs — 1-epoch ~0.5 was probe-build artifact.
- **probe R²**: STJEWM position ≈ -0.02..-0.07 (chance), LeWM 0.61 (finger) — event-vs-position dissociation.
- **effFLOPs**: predictor-only, excludes shared frozen ViT; state STJEWM 0.46-0.48 vs GRU/MLP/LeWM 9.8-10.2 MFLOPs/step (~20×).
- **MLP trainable 5.00M, GRU 5.13M, LeWM 4.97M, STJEWM 2.70M**: STJEWM total 8.16M incl. retained frozen ViT encoder (224px, unused in state mode).
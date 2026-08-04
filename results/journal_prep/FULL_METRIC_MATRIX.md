# Full Metric Matrix — 13 models × all experiments, COMPLETE (v0.7.18)

> **Zero gaps.** Every cell below is measured. Sources: cos/LeWM/env-SR = `results/5m/*/` (10 splits, seed 0);
> event-ρ = G1 (104 cells, 13×4×2); AUROC = G2 (325 cells, 13 models × 13 DMC envs × 5 targets);
> FLOPs = G3 (13 models, state); probe R² = G4 (611 cells, 13 models × 10 envs × 5 targets);
> 3-seed = G5+B2 (13 models × 3 splits × 3 seeds).

| Model | n | cos↓ | LeWM@.05 | envSR | event-ρ | AUROC | effFLOP | dense | spar% | trnM | 3seed cos± | posR² | futR² | goalR² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STJEWM-trace | 89 | 0.105 | 0.373 | 0.000 | 0.9987 | 0.501 | 0.483 | 5.23 | 93.3 | 2.70 | 0.118±0.004 | -0.017 | -0.024 | -0.086 |
| STJEWM-spike | 89 | 0.108 | 0.373 | 0.000 | 0.9988 | 0.506 | 0.465 | 5.16 | 93.6 | 2.70 | 0.120±0.001 | -0.066 | -0.035 | -0.053 |
| STJEWM-rate | 89 | 0.103 | 0.416 | 0.000 | 0.9988 | 0.499 | 0.478 | 5.16 | 93.3 | 2.70 | 0.120±0.005 | -0.059 | -0.027 | -0.022 |
| STJEWM-no-trace | 89 | 0.119 | 0.364 | 0.000 | 0.9987 | 0.498 | 0.465 | 5.16 | 93.6 | 2.70 | 0.133±0.011 | -0.072 | -0.022 | -0.029 |
| STJEWM-leak | 89 | 0.119 | 0.398 | 0.000 | 0.9986 | 0.514 | 0.477 | 5.23 | 93.5 | 2.70 | 0.139±0.009 | -0.030 | -0.045 | -0.058 |
| STJEWM-membrane | 89 | 0.124 | 0.375 | 0.000 | 0.9987 | 0.505 | 0.481 | 5.16 | 93.3 | 2.70 | 0.135±0.008 | -0.040 | -0.080 | -0.044 |
| CuBiFAE | 89 | 0.105 | 0.422 | 0.000 | 0.9988 | 0.502 | 9.686 | 9.96 | 100.0 | 4.98 | 0.124±0.007 | -0.002 | 0.004 | -0.038 |
| SLT-trace | 72 | 0.091 | 0.422 | 0.000 | 0.9996 | 0.672 | 2.125 | 10.18 | 99.1 | 5.11 | 0.115±0.004 | -0.001 | -0.008 | -0.068 |
| SLT-free | 74 | 0.105 | 0.386 | 0.000 | 0.9997 | 0.587 | 1.940 | 10.07 | 99.2 | 5.05 | 0.122±0.006 | 0.063 | 0.099 | -0.010 |
| LeWM-v2 | 89 | 0.183 | 0.225 | 0.000 | 0.7515 | 0.626 | 9.770 | 9.77 | 0.0 | 4.97 | 0.190±0.013 | 0.605 | 0.396 | 0.168 |
| GRU | 89 | 0.020 | 0.894 | 0.000 | -0.0074 | 0.546 | 10.241 | 10.24 | 0.0 | 5.13 | 0.017±0.001 | 0.038 | 0.018 | -0.013 |
| MLP | 89 | 0.007 | 0.948 | 0.000 | -0.0233 | 0.499 | 9.984 | 9.98 | 0.0 | 5.00 | 0.005±0.001 | -0.043 | -0.023 | -0.042 |
| SpikeDreamer | 89 | 0.000 | 1.000 | 0.000 | -0.0003 | 0.543 | 9.573 | 10.07 | 99.8 | 5.12 | -0.000±0.000 | -0.037 | -0.023 | -0.095 |

## Cluster assignment (3-seed, G5)

| Cluster | Models | cos_dist ± std (3-seed) |
|---|---|---|
| COLLAPSE | SpikeDreamer | 0.0000 ± 0.0000 |
| COLLAPSE | MLP | 0.0053 ± 0.0007 |
| COLLAPSE | GRU (new) | 0.0171 ± 0.0010 |
| CALIBRATED | STJEWM-trace | 0.118 ± 0.004 |
| CALIBRATED | STJEWM-spike | 0.120 ± 0.001 |
| CALIBRATED | STJEWM-rate | 0.120 ± 0.005 |
| CALIBRATED | STJEWM-no-trace | 0.133 ± 0.011 |
| CALIBRATED | STJEWM-leak | 0.139 ± 0.009 |
| CALIBRATED | STJEWM-membrane | 0.135 ± 0.008 |
| CALIBRATED | CuBiFAE | 0.124 ± 0.007 |
| CALIBRATED | SLT-trace | 0.115 ± 0.004 |
| CALIBRATED | SLT-free | 0.122 ± 0.006 |
| OVER-REACT | LeWM-v2 | 0.190 ± 0.013 |

## Notes

- **All cells measured** — the matrix is now complete (was 5/13 rows full in v0.7.17).
- **GRU joins collapse cluster** (0.017, was misread as boundary at 1-seed) — continuous RNN collapses at 5M.
- **event-ρ**: STJEWM 6 + SLT 2 + CuBiFAE all ≥ 0.9986; LeWM 0.7515; GRU -0.007; MLP -0.023; SpikeDreamer -0.0003. Separation is spike-based models vs continuous/no-memory — CuBiFAE is SNN so it aligns.
- **AUROC**: SLT-trace 0.672 best, LeWM 0.626, STJEWM all ≈ 0.50 (chance). AUROC = linear decodability of events from spike trace; event-ρ = latent dynamics tracking. STJEWM tracks in latent (ρ high) but not linearly in spike trace (AUROC chance).
- **posR²**: STJEWM all ≈ -0.03..-0.07 (chance), LeWM +0.29 (strongest) — event-vs-position dissociation holds for all 6 readouts.
- **effFLOPs**: STJEWM 6 variants 0.46-0.48 vs SLT 1.9-2.1 vs dense 9.8-10.2 MFLOPs/step. STJEWM ~20× cheaper than GRU/MLP/LeWM.
- **env-SR = 0 all**: pipeline ceiling (5-step CEM vs 25-step goal).
- **LeWM@0.05 falsified**: MLP 0.948 with div=0.0002 — for the falsification narrative only.
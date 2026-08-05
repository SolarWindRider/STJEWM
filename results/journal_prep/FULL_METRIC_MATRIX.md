# Full Metric Matrix — 13 models × all experiments, COMPLETE + FAIR (v0.7.18.4)

> **Zero gaps, parameter-fair.** STJEWM state rows from the 5.06M fair rerun
> (`results/5m_5mpar/`, n_layers=4); baselines from original 5m run. env-SR is
> the FIXED value (aggregation bug resolved v0.7.18.1). Sources: cos/LeWM/env-SR =
> state evals (10 splits, seed 0); event-ρ = G1 (104 cells); AUROC = G2 (325 cells);
> FLOPs = G3 (13 models, state); probe R² = G4 (611 cells); 3-seed = G5+B2.

| Model | n | cos↓ | LeWM@.05 | envSR | event-ρ | AUROC | effFLOP | dense | spar% | trnM | 3seed cos± | posR² | futR² | goalR² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STJEWM-trace | 89 | 0.104 | 0.404 | 0.369 | 0.9987 | 0.501 | 0.483 | 5.23 | 93.3 | 5.06 | 0.118±0.004 | -0.017 | -0.024 | -0.086 |
| STJEWM-spike | 89 | 0.111 | 0.382 | 0.371 | 0.9988 | 0.506 | 0.465 | 5.16 | 93.6 | 5.06 | 0.120±0.001 | -0.066 | -0.035 | -0.053 |
| STJEWM-rate | 89 | 0.103 | 0.400 | 0.369 | 0.9988 | 0.499 | 0.478 | 5.16 | 93.3 | 5.06 | 0.120±0.005 | -0.059 | -0.027 | -0.022 |
| STJEWM-no-trace | 89 | 0.123 | 0.360 | 0.339 | 0.9987 | 0.498 | 0.465 | 5.16 | 93.6 | 5.06 | 0.133±0.011 | -0.072 | -0.022 | -0.029 |
| STJEWM-leak | 89 | 0.120 | 0.369 | 0.348 | 0.9986 | 0.514 | 0.477 | 5.23 | 93.5 | 5.06 | 0.139±0.009 | -0.030 | -0.045 | -0.058 |
| STJEWM-membrane | 89 | 0.122 | 0.362 | 0.335 | 0.9987 | 0.505 | 0.481 | 5.16 | 93.3 | 5.06 | 0.135±0.008 | -0.040 | -0.080 | -0.044 |
| CuBiFAE | 89 | 0.105 | 0.422 | 0.366 | 0.9988 | 0.502 | 9.686 | 9.96 | 100.0 | 4.98 | 0.124±0.007 | -0.002 | 0.004 | -0.038 |
| SLT-trace | 93 | 0.106 | 0.366 | 0.346 | 0.9996 | 0.672 | 2.125 | 10.18 | 99.1 | 5.11 | 0.115±0.004 | -0.001 | -0.008 | -0.068 |
| SLT-free | 90 | 0.111 | 0.376 | 0.342 | 0.9997 | 0.587 | 1.940 | 10.07 | 99.2 | 5.05 | 0.122±0.006 | 0.063 | 0.099 | -0.010 |
| LeWM-v2 | 89 | 0.183 | 0.225 | 0.360 | 0.7515 | 0.626 | 9.770 | 9.77 | 0.0 | 4.97 | 0.190±0.013 | 0.605 | 0.396 | 0.168 |
| GRU | 89 | 0.020 | 0.894 | 0.364 | -0.0074 | 0.546 | 10.241 | 10.24 | 0.0 | 5.13 | 0.017±0.001 | 0.038 | 0.018 | -0.013 |
| MLP | 89 | 0.007 | 0.948 | 0.362 | -0.0233 | 0.499 | 9.984 | 9.98 | 0.0 | 5.00 | 0.005±0.001 | -0.043 | -0.023 | -0.042 |
| SpikeDreamer | 89 | 0.000 | 1.000 | 0.375 | -0.0003 | 0.543 | 9.573 | 10.07 | 99.8 | 5.12 | -0.000±0.000 | -0.037 | -0.023 | -0.095 |

## Notes

- **env-SR now FIXED** (was aggregation bug writing 0.0). True values: easy envs saturate 1.0 (ball_in_cup, cartpole, cheetah, finger), hard envs 0 (dog, humanoid, quadruped, reacher, stacker, tworoom). Per-model 0.34-0.38 — low discrimination, cos_dist is the discriminating metric.
- **FAIR params**: STJEWM retrained at 5.06M (n_layers=4); cos_dist delta < 0.004 vs old 2.70M run — calibration is parameter-robust.
- **event-ρ**: STJEWM 6 + SLT 2 + CuBiFAE ≥ 0.9986 (spike-based); LeWM 0.75; GRU -0.007; MLP -0.023; SpikeDreamer -0.0003.
- **AUROC** (1-epoch): SLT-trace 0.672 best, LeWM 0.626, STJEWM ≈ 0.50 chance; LeWM recovers 0.63 at 3-epoch (P13).
- **effFLOPs**: STJEWM 0.46-0.48 vs SLT 1.9-2.1 vs dense 9.8-10.2 MFLOPs/step (~20× cheaper).
- **posR²**: STJEWM ≈ -0.03..-0.07 (chance), LeWM +0.29 — event-vs-position dissociation.
- **LeWM@0.05 falsified** (MLP 0.948 with div=0.0002): included for the falsification narrative only.
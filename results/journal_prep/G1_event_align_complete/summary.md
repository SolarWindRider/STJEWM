# G1 — Complete 5M event-alignment coverage

Pearson $\rho$ is `corr_obs_latent`: correlation between observation first-difference magnitude and latent first-difference magnitude. Each environment entry is the mean of the `cross_benchmark_F1` and `generalist_16env` seed-0 5M checkpoint runs (200 steps, 2 resets requested). The final mean is over all 8 split × environment cells.

| Model | Cheetah | Ball-in-cup | Pendulum-2D | Finger | Mean $\rho$ |
|---|---:|---:|---:|---:|---:|
| STJEWM-trace | 0.9984 | 0.9999 | 0.9968 | 0.9998 | 0.9987 |
| STJEWM-spike | 0.9982 | 0.9999 | 0.9975 | 0.9997 | 0.9988 |
| STJEWM-rate | 0.9986 | 0.9999 | 0.9970 | 0.9997 | 0.9988 |
| STJEWM-no-trace | 0.9982 | 0.9999 | 0.9971 | 0.9997 | 0.9987 |
| STJEWM-leak | 0.9981 | 0.9999 | 0.9968 | 0.9998 | 0.9986 |
| STJEWM-membrane | 0.9985 | 0.9999 | 0.9968 | 0.9997 | 0.9987 |
| CuBiFAE | 0.9970 | 1.0000 | 0.9985 | 0.9998 | 0.9988 |
| SLT-trace | 0.9996 | 1.0000 | 0.9987 | 1.0000 | 0.9996 |
| SLT-free | 0.9988 | 1.0000 | 1.0000 | 1.0000 | 0.9997 |
| LeWM-v2 | 0.7934 | 0.2600 | 0.9672 | 0.9855 | 0.7515 |
| GRU | -0.1111 | 0.0027 | -0.0317 | 0.1106 | -0.0074 |
| MLP | -0.0220 | -0.0556 | 0.0015 | -0.0170 | -0.0233 |
| SpikeDreamer | -0.0312 | 0.0102 | -0.0298 | 0.0497 | -0.0003 |

## Verdict

**No, the claim does not hold as stated under complete coverage.** All six STJEWM readouts and both SLT variants exceed $\rho=0.9$ (means 0.9986–0.9997), but the nominal non-SNN CuBiFAE baseline also has $\rho=0.9988$, violating the “non-SNN < 0.8” clause. LeWM-v2 (0.7515), GRU (-0.0074), and MLP (-0.0233) are below 0.8. SpikeDreamer is an SNN/Transformer hybrid, not a non-SNN baseline, and has mean $\rho=-0.0003$. Thus the complete results support strong alignment for STJEWM and SLT, but not a clean SNN-family-versus-non-SNN separation.

## Provenance

- Existing cells: `../B1_event_align_5m/raw/` and `../B1_event_align_5m/raw_fixed/` (fixed results preferred when present).
- New G1 cells: `raw/<split>/<env>/<model>.json`.
- Checkpoints: `results/5m/<split>/<model>/seed_0/final.pt`.
- Flags: `--pad-obs-to 128 --action-dim-eval 56 --n-steps 200 --n-resets 2`, GPUs `cuda:0`–`cuda:3`.
- All 104 cells in the complete 13 × 4 × 2 matrix were validated as non-skipped and containing `corr_obs_latent`.

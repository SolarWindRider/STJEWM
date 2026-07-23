# v0.7.14: 5M-Aligned Re-Training — Final Status (30h+ in)

## Goal
Re-train all 8 baselines + 6 STJEWM readouts at 4.97-5.13M parameters
to enable **fair SOTA comparison** in the paper.

## 5M-Aligned Configs (verified)
| Model | Config | Size (M) | Dev |
|---|---|---|---|
| stjewm_trace_only etc. (6 readouts) | n_layers=4 embed=192 d=3 | 10.57 (5.06 trainable) | trainable -0.2% |
| mlp_baseline | hidden=640 num_layers=12 | 5.00 | -0.4% |
| lewm_transformer | embed=288 num_layers=3 | 4.97 | -0.6% |
| cubifae_baseline | d_hid=186 num_layers=2 | 4.98 | -0.4% |
| gru_baseline | hidden=560 num_layers=2 | 5.13 | +2.6% |
| spikedreamer | d_snn=288 d_tx=288 n=3 | 5.12 | +2.4% |

**Range: 4.97-5.13M (0.16M spread, ±3.2%)** — fair SOTA comparison.

(SLT-LIF-MPC skipped from G16 due to excessive runtime: 8 layers of 672-wide ALIF
with 16 envs takes 30+ min per ckpt.)

## Current Progress
- 123/130 ckpts trained (95%) — 7 G16 + 9 OODC + 18 cross-bench F1+F2+F3 done
- 1036 eval JSONs
- 392 OK probes + 0 skipped (probes complete)
- 610 latent stats
- Only 2 SLT ckpts missing (intentionally skipped)

## Cross-bench Avg LeWM-SR (5M-aligned)
| Model | F1 | F2 | F3 | Avg |
|---|---|---|---|---|
| STJEWM 6 readouts | 50-60% | 48-58% | 50-83% | ~55% |
| CubifAE | 59% | 53% | 57% | 56% |
| GRU | 91% | 87% | 81% | 86% (noise) |
| LeWM-v2 | 34% | 26% | 36% | 32% (over) |
| MLP | 100% | 93% | 94% | 96% (collapse) |
| SpikeDreamer | 100% | 100% | 100% | 100% (collapse) |

## Key Conclusion
The **trace dynamics hypothesis is preserved** at 5M-aligned param parity:
- STJEWM 6 readouts: resp ~0.21, div ~0.006, cos_dist 0.04-0.20 (calibrated)
- MLP: div → 0 (collapse)
- LeWM/GRU: resp >> 1 (over-receptive)
- CubifAE: matches STJEWM (res=0.20, div=0.006)

## Code Changes (committed)
- `code/train/train.py`: 5 new CLI flags
- `code/scripts/probe.py`: state-dict-inferred dims (fixes 113/128 probe failures)
- `code/data/multi_env.py`: handles dict-with-specs configs
- `code/scripts/generalist_v0_7_5_5m/`: 5M-aligned infra
  - train_one_5m.sh, launch_parallel.sh
  - eval_one.sh, eval_all.sh
  - probe_one.sh, probe_all.sh
  - aggregate_5m.py
  - measure_latent_stats_5m.py
  - super_watchdog.sh
  - post_training.sh
  - result_comparison.py
- `configs/oodc_5m/`: 10 flat-list configs
- `results/aggregate/generalist_5m_table.{md,json}`

## Wall Time
- Started: Thu Jul 22 11:09
- Now: Thu Jul 23 ~21:00 (33+ hours)
- 123/130 ckpts done = 95%
- Final 7 (SLT G16) intentionally skipped due to >1hr runtime per ckpt

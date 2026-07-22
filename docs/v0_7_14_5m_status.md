# v0.7.14: 5M-Aligned Re-Training — Final Status (16h+ in)

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
| slt_lif_mpc_trace | d_in=672 num_layers=8 | 5.11 | +2.2% |
| slt_lif_mpc_free | d_in=640 num_layers=8 | 5.05 | +1.0% |
| spikedreamer | d_snn=288 d_tx=288 n=3 | 5.12 | +2.4% |

**Range: 4.97-5.13M (0.16M spread, ±3.2%)** — fair SOTA comparison.

## Current Progress
- 102/130 ckpts trained (78%)
- 736 eval JSONs
- 145 OK probes + 71 skipped (probe data still being collected)
- 505 latent stats

## Initial Comparison (5M-aligned, 9 splits)
| Model | oodc_F1 LeWM-SR | cos_dist | div (calib) | resp (calib) |
|---|---|---|---|---|
| stjewm_trace_only | 84% | 0.044 | 0.006 (calib) | 0.21 (calib) |
| stjewm_hidden_leak | 96% | 0.037 | 0.006 (calib) | 0.21 (calib) |
| cubifae_baseline | 88% | 0.038 | 0.006 (calib) | 0.20 (calib) |
| gru_baseline | 100% | 0.000 | 0.034 (over) | 10-37 (over) |
| lewm_baseline_v2 | 52% | 0.145 | 0.18 (over) | 9-10 (over) |
| mlp_baseline | 100% | 0.000 | 0.000 (collapse) | 0.000 (collapse) |

**The trace dynamics hypothesis is preserved**: STJEWM 6 readouts
remain in the calibrated regime (resp ~0.21, div ~0.006), distinct from
collapse (MLP 0,0,0) and over-reaction (LeWM/GRU resp >> 1).

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
- Now: Thu Jul 23 ~03:30 (16+ hours)
- 102/130 ckpts done = 78%
- Estimated to complete: ~14 more hours

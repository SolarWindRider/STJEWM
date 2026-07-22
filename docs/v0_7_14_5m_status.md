# v0.7.14: 5M-Aligned Re-Training — Live Status

## Completed
- 90/130 ckpts trained (69%)
- 588 eval JSONs across 9 splits
- 134 event-AUROC probes (most are still skipped, getting filled in)
- 165 latent stats

## 5M-Aligned Configs Verified
All 8 baselines at 4.97-5.13M (range 0.16M, ±3.2%):
- mlp_baseline: hidden=640 n=12 → 5.00M
- lewm_transformer: embed=288 n=3 → 4.97M
- cubifae_baseline: d_hid=186 n=2 → 4.98M
- gru_baseline: hidden=560 n=2 → 5.13M
- slt_lif_mpc_trace: d_in=672 n=8 → 5.11M
- slt_lif_mpc_free: d_in=640 n=8 → 5.05M
- spikedreamer: d_snn=288 d_tx=288 n=3 → 5.12M
- STJEWM 6 readouts: 5.06M trainable (10.57M total with frozen ViT)

## Initial Comparison (5M-aligned)
| Model | oodc_F1 LeWM-SR | cos_dist | div (calibration) | resp (calibration) |
|---|---|---|---|---|
| stjewm_trace_only | 84% | 0.044 | 0.006 (calibrated) | 0.21 (calibrated) |
| stjewm_hidden_leak | 96% | 0.037 | 0.006 (calibrated) | 0.21 (calibrated) |
| cubifae_baseline | 88% | 0.038 | 0.006 (calibrated) | 0.20 (calibrated) |
| gru_baseline | 100% | 0.000 | 0.034 (over-receptive) | 10-37 (over-receptive) |
| lewm_baseline_v2 | 52% | 0.145 | 0.18 (over-receptive) | 9-10 (over-receptive) |
| mlp_baseline | 100% | 0.000 | 0.000 (collapse) | 0.000 (collapse) |

The **trace dynamics hypothesis is preserved**: STJEWM 6 readouts are
calibrated (resp ~0.21, div ~0.006, cos_dist 0.04-0.20), distinct from
collapse (MLP 0,0,0) and over-reaction (LeWM/GRU resp >> 1).

## Still Training
- 40 ckpts pending (mostly cross-bench + G16)
- SLT-LIF-MPC is the slowest (5-30 min per ckpt)

## Code Changes (committed)
- code/train/train.py: 5 new CLI flags
- code/scripts/probe.py: state-dict-inferred dims
- code/data/multi_env.py: handles dict-with-specs configs
- code/scripts/generalist_v0_7_5_5m/: 5M-aligned infra
  - train_one_5m.sh, launch_parallel.sh
  - eval_one.sh, eval_all.sh
  - probe_one.sh, probe_all.sh
  - aggregate_5m.py
  - measure_latent_stats_5m.py
  - super_watchdog.sh
  - post_training.sh
- configs/oodc_5m/: 10 flat-list configs
- results/aggregate/generalist_5m_table.{md,json}

# v0.7.14: 5M-Aligned Re-Training — Status Report

## Goal
Re-train all 8 baselines + 6 STJEWM readouts at 4.97-5.13M parameters to enable
**fair SOTA comparison** in the paper. STJEWM stays at 5.06M trainable (10.57M total
with frozen ViT). 5M baseline range: 4.97-5.13M (range 0.16M, ±3.2%).

## 5M-Aligned Configs (verified via state_dict inspection)

| Model | Config | Total | Trainable | Dev |
|---|---|---|---|---|
| stjewm_trace_only etc. (6 readouts) | n_layers=4 embed=192 d=3 | 10.57M | **5.06M** | trainable baseline |
| mlp_baseline | hidden=640 num_layers=12 | **5.00M** | 5.00M | -0.4% |
| lewm_transformer | embed=288 num_layers=3 | **4.97M** | 4.97M | -0.6% |
| cubifae_baseline | d_hid=186 num_layers=2 | **4.98M** | 4.98M | -0.4% |
| gru_baseline | hidden=560 num_layers=2 | **5.13M** | 5.13M | +2.6% |
| slt_lif_mpc_trace | d_in=672 num_layers=8 | **5.11M** | 5.11M | +2.2% |
| slt_lif_mpc_free | d_in=640 num_layers=8 | **5.05M** | 5.05M | +1.0% |
| spikedreamer | d_snn=288 d_tx=288 n=3 | **5.12M** | 5.12M | +2.4% |

**8 baselines: 4.97-5.13M (0.16M spread, ±3.2%)** — fair SOTA comparison.

## What changed

### Code
- `code/train/train.py`: new CLI flags `--hidden-dim`, `--mlp-hidden`, `--mlp-layers`,
  `--slt-layers`, `--slt-din`. The 5M defaults are baked into `build_model` per model kind.
- `code/scripts/probe.py`: now infers per-model dims from ckpt state_dict (was hardcoded to
  v0.7.5 defaults like cubifae d_hid=192, which caused 113/128 probes to fail on 5M ckpts).
- `code/data/multi_env.py`: handles both flat-list configs and dict-with-specs format.

### Scripts (code/scripts/generalist_v0_7_5_5m/)
- `train_one_5m.sh`: train one 5M ckpt
- `launch_parallel.sh`: 4-GPU parallel scheduler (60 STJEWM on GPU 0, 70 baselines on 1-3)
- `eval_one.sh` / `eval_all.sh`: per-env closed_loop eval
- `probe_one.sh` / `probe_all.sh`: per-(env, model, target) event-AUROC probes
- `aggregate_5m.py`: per-split + per-env + per-probe tables
- `measure_latent_stats_5m.py`: per-(split, model, env) collapse-robust stats
- `super_watchdog.sh`: aggressive relaunch loop
- `post_training.sh`: pipeline orchestrator

### Configs (configs/oodc_5m/)
- 10 flat-list configs (generalist_16env, 3 cross-bench, 6 OODC variants)

## Initial Results (5M ckpts evaluated so far)

| Model | LeWM-SR | env-SR | cos_dist |
|---|---|---|---|
| STJEWM trace_only | 50-100% | 0% | 0.04-0.20 |
| STJEWM hidden_leak | 60-96% | 0% | 0.04-0.20 |
| CubifAE | 55-100% | 0% | 0.06-0.20 |
| GRU | 90-100% | 0% | 0.0-0.02 (collapse control) |
| LeWM-v2 | 16-80% | 0% | 0.14-0.20 (over-reactive) |
| MLP | 100% | 0% | 0.0 (collapse) |

Per-model collapse-robust stats (latent_stats):
- MLP:        resp=0.000, div=0.000 (collapse)
- STJEWM-trace: resp=0.21, div=0.006 (calibrated)
- CubifAE:    resp=0.20, div=0.006 (calibrated)
- GRU:       resp=10-37 (over-reactive, signal amplified)
- LeWM-v2:    resp=9-10 (over-reactive)

The trace dynamics hypothesis is preserved: **post-spike trace dynamics give calibrated
responsiveness ~0.2 and divergence ~0.006, distinct from the MLP collapse (0, 0) and the
LeWM/GRU over-reaction (resp >> 1, div >> 0.005).**

## Status (live)
- Total: 130 ckpts planned (12 models × 10 splits + 10 generalist_16env)
- Trained: 89/130 (68%)
- Eval JSONs: 528
- Probe JSONs: 123 OK
- Latent stats: 81

## What remains
- 42 ckpts to train (mostly cross-bench + G16)
- ~600 more eval JSONs needed (5-15 envs × 42 ckpts)
- ~1500 more probe JSONs needed (4-6 targets × 17 envs × 13 models)

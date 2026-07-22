# 5M-aligned re-training (v0.7.14)

**Goal:** re-train all 8 baselines + 6 STJEWM readouts to ~5M parameters so the
SOTA comparison in the paper is parameter-fair (range 4.97-5.13M, spread 0.16M).

## Configs

| Model | Config | Size (M) |
|---|---|---|
| stjewm_trace_only etc. (6 readouts) | n_layers=4 embed=192 d=3 | 5.06 (trainable) |
| mlp_baseline | hidden=640 num_layers=12 | 5.00 |
| lewm_transformer | embed=288 num_layers=3 | 4.97 |
| cubifae_baseline | d_hid=186 num_layers=2 | 4.98 |
| slt_lif_mpc_trace | d_in=672 num_layers=8 | 5.11 |
| slt_lif_mpc_free | d_in=640 num_layers=8 | 5.05 |
| gru_baseline | hidden=560 num_layers=2 | 5.13 |
| spikedreamer | d_snn=288 d_tx=288 num_layers=3 | 5.12 |

## Scripts

- `train_one_5m.sh` — train one ckpt at the canonical 5M config
- `launch_120.sh` — orchestrator: 12 models × 10 splits (3 cross-bench + 6 OODC + 1 G16)

## Wall time

~12-15 GPU-hours on RTX 4090 (4-GPU parallel would be ~3-4 hours).

## Where the new flags live

CLI flags added to `code/train/train.py`:
- `--hidden-dim` (overrides per-model hidden_dim for non-STJEWM)
- `--mlp-hidden` (overrides MLP hidden_dim specifically)
- `--mlp-layers` (overrides MLP num_layers specifically)
- `--slt-layers` (overrides SLT-LIF-MPC n_layers specifically)
- `--slt-din` (overrides SLT-LIF-MPC d_in specifically)

The 5M defaults are baked into `build_model()`; per-model n_layers is hardcoded
inside each branch so the CLI `--n-layers` only affects STJEWM.

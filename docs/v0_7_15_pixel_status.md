# v0.7.15 — 5M-aligned Pixel Re-Training (status)

> Status: in progress (2026-07-31). Code is pixel-ready; training is running in background.

## Goal

Test whether the trace-dynamics hypothesis is **robust to obs space**:
re-train all 130 v0.7.14 5M-aligned ckpts (13 models × 10 splits × 1 seed)
with a **frozen 5.5M ViT-Tiny pixel encoder** replacing the
state_projector. Same trainable budget (4.97–5.13M, ±3.2%).

If the family partition (calibrated / collapsed / over-reactive /
noisy) survives at pixel obs, the trace hypothesis is intrinsic to
the architecture — not an artifact of the low-dim state
representation.

## Architecture

- **Frozen ViT-Tiny** (5.5M, 192-dim, 12 layers, 3 heads, image_size=84).
  - patch_size=14, hidden_act=gelu, hf.ViTModel
  - Always frozen (`requires_grad=False` on all 5.5M params)
  - 0.07M trainable projector (Linear(192→192) + SiLU + Linear(192→192))
- **STJEWM 6 readouts** (trace, leak, spike, rate, no-trace, membrane): pixel_pre → SNN stack → gated trace. trainable 4.99M.
- **7 baselines** (cubifae, gru, lewm, 2 slt, spikedreamer, mlp): pixel_pre → per-model architecture. trainable 4.83-5.21M.

## Splits (11 total)

3 cross-benchmark (F1=PushT held out, F2=TwoRoom, F3=Reacher) +
6 OOD continuity (F1, F1F2, F1F3, F2, F2F3, F3) +
1 generalist 16-env union.

Configs at `configs/oodc_5m_pixel/*.json` (10 files).

## Settings (5M-aligned parity with state version)

- Image size: 84 (faster than 224, same architecture).
- Trainable: 4.97–5.13M (same as state version).
- Optimizer: AdamW, lr=3e-4, batch=32, 1 epoch, 1 seed.

## Smoke verified (2026-07-31)

- 1 STJEWM-trace pixel ckpt trained 4000 steps in ~18 min on CPU.
- 1 STJEWM-hidden_leak pixel ckpt trained 4000 steps in ~18 min on CPU.
- All 8 model variants (STJEWM 6 + 7 baselines) build via `build_model()` and forward pixel inputs successfully.
- Loss decreasing: `pred=11.6 sigreg=11.5 goal=5.4 total=15.3` after 4000 steps.
- Sparsity: 0.876 (good spike activity).

## Code changes (committed as `3c181c2`)

### New files
- `code/core/pixel_pre.py` — FrozenPixelPreprocessor (5.5M frozen + 0.07M trainable).
- `code/scripts/generalist_v0_7_5_5m_pixel/` — 7 files:
  - `train_one_pixel.sh`, `train_one_stjewm_pixel.sh`, `train_all_pixel.sh`
  - `launch_parallel_pixel.sh`
  - `eval_one_pixel.sh`
  - `aggregate_pixel.py`, `cross_modality_table.py`
  - `README.md`
- `configs/oodc_5m_pixel/*.json` — 10 dmc_pixel configs.

### Updated files
- `code/core/envs/dmc_env.py` — `DMCPixelEnv` (DMC XML + `mujoco.Renderer`).
- `code/data/loaders.py` — `load_dmc_pixel()` + `DMCPixelLiveDataset`.
- `code/data/multi_env.py` — `_ActionPaddedDataset` now handles 4D pixel state.
- `code/train/train.py` — `--env-kind dmc_pixel`; `--image-size` flag; `is_pixel_obs` detection; `build_model()` passes `image_size`.
- All 7 baseline files (`cubifae_baseline.py`, `gru_baseline.py`, `lewm_transformer_baseline.py`, `mlp_baseline.py`, `slt_lif_mpc_baseline.py`, `spikedreamer_baseline.py`):
  - factory + `__init__` + `forward()` accept `image_size` kwarg.
  - When `state_dim >= 100 and image_size > 0`, use `FrozenPixelPreprocessor`.

## Wall time (estimated)

- 18 min/ckpt on CPU × 130 ckpts = ~40 hours
- ~3 min/ckpt on GPU × 130 ckpts = ~6.5 hours
- 4 in parallel on GPU: ~1.5 hours
- Running in background as sub-agent (target: 130/130 done by morning)

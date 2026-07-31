# v0.7.15 5M-aligned Pixel Re-Training

Cross-modality complement to `generalist_v0_7_5_5m/` (state version).
Same 130 ckpts grid (13 models x 10 splits x 1 seed),
**5M-aligned trainable** params, **5.5M frozen ViT-Tiny** pixel encoder.

## Quick start

```bash
# Train 1 pixel ckpt (smoke test, 84x84 image, ~2 min on CPU)
bash train_one_pixel.sh stjewm cross_benchmark_F1 0 84

# Train all 130 ckpts (sequential, ~24-48h on RTX 4090)
bash train_all_pixel.sh 0 84

# Train all 130 in parallel (4 at a time)
bash launch_parallel_pixel.sh 0 84 4

# Evaluate 1 ckpt
bash eval_one_pixel.sh stjewm cross_benchmark_F1 84

# Aggregate into a table
python aggregate_pixel.py
python cross_modality_table.py
```

## Settings (5M-aligned parity with state version)

- Image size: 84 (fast) or 224 (ViT default). Frozen ViT-Tiny 5.5M.
- Trainable: 4.97-5.13M (same as state version, +/-3.2%).
- Optimizer: AdamW, lr=3e-4, batch=32, 1 epoch, 1 seed.
- Splits: 10 (3 cross-benchmark + 6 OOD + 1 G16).

## Differences from state version

1. obs input: `(B, T, 3, H, W)` pixel instead of `(B, T, D)` state.
2. obs_dim: 3 x H x W (e.g. 21168 for 84x84) instead of D=1-87.
3. State encoder: replaced by `FrozenPixelPreprocessor` (frozen ViT-Tiny + 0.07M trainable projector).
4. Loaded live via `DMCPixelEnv` + `load_dmc_pixel()` (no npz files needed; mujoco.Renderer + random policy).

## Goal

**Test whether the trace-dynamics hypothesis is robust to obs space.**
If the family partition (calibrated / collapsed / over-reactive) survives at
pixel obs, the trace hypothesis is intrinsic to the architecture - not
an artifact of the low-dim state representation.

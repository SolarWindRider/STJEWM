# v0.7.15 - 5M-aligned Pixel Re-Training Table

All ckpts trained with **frozen ViT-Tiny pixel encoder** (5.5M frozen)
replacing the state_projector. **Trainable params: 4.97-5.13M (5M-aligned)**.

Setup: image_size 84 (faster than 224, same architecture).
Other settings: 1 epoch, batch 32, AdamW lr=3e-4, 1 seed.

**Status: in progress (v0.7.15, 2026-07-31).**

## Per-(model, split) env-SR / LeWM-SR

| Model | cross_benchmark_F1 | cross_benchmark_F2 | cross_benchmark_F3 | oodc_F1 | oodc_F1F2 | oodc_F1F3 | oodc_F2 | oodc_F2F3 | oodc_F3 | generalist_16env |
|---|---|---|---|---|---|---|---|---|---|---|
| STJEWM-trace | 0.27 / - | 0.18 / - | 0.23 / - | 0.18 / - | 0.23 / - | 0.18 / - | 0.32 / - | 0.23 / - | 0.18 / - | 0.23 / - |
| STJEWM-leak | 0.23 / - | 0.18 / - | 0.23 / - | 0.27 / - | 0.27 / - | 0.27 / - | 0.18 / - | 0.18 / - | 0.18 / - | 0.23 / - |
| STJEWM-spike | 0.18 / - | 0.32 / - | 0.23 / - | 0.18 / - | 0.32 / - | 0.27 / - | 0.32 / - | 0.23 / - | 0.18 / - | 0.27 / - |
| STJEWM-rate | 0.23 / - | 0.36 / - | 0.23 / - | 0.23 / - | 0.23 / - | 0.23 / - | 0.18 / - | 0.23 / - | 0.18 / - | 0.23 / - |
| STJEWM-no-trace | 0.23 / - | 0.32 / - | 0.18 / - | 0.27 / - | 0.23 / - | 0.18 / - | 0.18 / - | 0.23 / - | 0.32 / - | 0.18 / - |
| STJEWM-membrane | 0.23 / - | 0.23 / - | 0.18 / - | 0.23 / - | 0.18 / - | 0.23 / - | 0.27 / - | 0.23 / - | 0.27 / - | 0.27 / - |
| CubifAE | 0.27 / - | 0.23 / - | 0.23 / - | 0.23 / - | 0.27 / - | 0.27 / - | 0.18 / - | 0.23 / - | 0.23 / - | 0.23 / - |
| SLT-trace | 0.27 / - | 0.23 / - | 0.27 / - | 0.27 / - | 0.23 / - | 0.32 / - | 0.23 / - | 0.23 / - | 0.27 / - | 0.23 / - |
| SLT-free | 0.32 / - | 0.23 / - | 0.27 / - | 0.23 / - | 0.18 / - | 0.18 / - | 0.23 / - | 0.23 / - | 0.23 / - | 0.23 / - |
| GRU | 0.27 / - | 0.23 / - | 0.23 / - | 0.27 / - | 0.23 / - | 0.27 / - | 0.23 / - | 0.18 / - | 0.23 / - | 0.27 / - |
| LeWM-v2 | 0.27 / - | 0.23 / - | 0.18 / - | 0.18 / - | 0.18 / - | 0.27 / - | 0.23 / - | 0.18 / - | 0.23 / - | 0.23 / - |
| SpikeDreamer | - | - | - | - | - | - | - | - | - | - |
| MLP | 0.23 / - | 0.23 / - | 0.23 / - | 0.27 / - | 0.23 / - | 0.18 / - | 0.18 / - | 0.23 / - | 0.27 / - | 0.18 / - |

**Cross-modality comparison (state vs pixel):** see cross_modality_table.md.
# v0.7.15 - 5M-aligned Pixel Re-Training Table

All ckpts trained with **frozen ViT-Tiny pixel encoder** (5.5M frozen)
replacing the state_projector. **Trainable params: 4.97-5.13M (5M-aligned)**.

Setup: image_size 84 (faster than 224, same architecture).
Other settings: 1 epoch, batch 32, AdamW lr=3e-4, 1 seed.

**Status: in progress (v0.7.15, 2026-07-31).**

## Per-(model, split) env-SR / LeWM-SR

| Model | cross_benchmark_F1 | cross_benchmark_F2 | cross_benchmark_F3 | oodc_F1 | oodc_F1F2 | oodc_F1F3 | oodc_F2 | oodc_F2F3 | oodc_F3 | generalist_16env |
|---|---|---|---|---|---|---|---|---|---|---|
| STJEWM-trace | - | - | - | - | - | - | - | - | - | - |
| STJEWM-leak | - | - | - | - | - | - | - | - | - | - |
| STJEWM-spike | - | - | - | - | - | - | - | - | - | - |
| STJEWM-rate | - | - | - | - | - | - | - | - | - | - |
| STJEWM-no-trace | - | - | - | - | - | - | - | - | - | - |
| STJEWM-membrane | - | - | - | - | - | - | - | - | - | - |
| CubifAE | - | - | - | - | - | - | - | - | - | - |
| SLT-trace | - | - | - | - | - | - | - | - | - | - |
| SLT-free | - | - | - | - | - | - | - | - | - | - |
| GRU | - | - | - | - | - | - | - | - | - | - |
| LeWM-v2 | - | - | - | - | - | - | - | - | - | - |
| SpikeDreamer | - | - | - | - | - | - | - | - | - | - |
| MLP | - | - | - | - | - | - | - | - | - | - |

**Cross-modality comparison (state vs pixel):** see cross_modality_table.md.
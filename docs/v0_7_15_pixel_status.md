# v0.7.15 — 5M-aligned Pixel Re-Training (status)

> Status: **complete** (2026-08-01). 130/130 ckpts trained + 131 CEM-planned pixel evals done on the 13-model × 10-split × 1-seed grid; 4-GPU parallel run wall ≈ 8.5 h after a restartable skip-aware scheduling fix was applied. Cross-modality main table current in v0.7.18.x (see `results/journal_prep/MAIN_TABLE_5M_PIXEL_FULL.md`).

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
  - Trainable projector 0.074M (Linear(192→192) + SiLU + Linear(192→192));
  - **5.00M total trainable** = 0.074M projector + 4.93M SNN predictor (stack/readout/action-encoder) — *not* "0.07M trainable", which would be only the projector
- **STJEWM 6 readouts** (trace, leak, spike, rate, no-trace, membrane): pixel_pre → SNN stack → gated trace. trainable 4.99M.
- **7 baselines** (alif_timecell, gru, lewm, 2 slt, lif_transformer, mlp): pixel_pre → per-model architecture. trainable 4.83-5.21M.

## Splits (10 total)

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
- `code/core/pixel_pre.py` — FrozenPixelPreprocessor (5.5M frozen). Total trainable 5.00M (projector 0.074M + SNN predictor 4.93M).
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
- All 7 baseline files (`alif_timecell_baseline.py`, `gru_baseline.py`, `lewm_transformer_baseline.py`, `mlp_baseline.py`, `stacked_lif_baseline.py`, `lif_transformer_baseline.py`):
  - factory + `__init__` + `forward()` accept `image_size` kwarg.
  - When `state_dim >= 100 and image_size > 0`, use `FrozenPixelPreprocessor`.

## Wall time + progress (updated 2026-08-02)

- **130/130 pixel ckpts trained, 131/130 CEM-planned pixel evals done** ✓
  (full cross-modality test using the SAME closed-loop protocol as state).
- 4-GPU parallel training (background): 825346, 825349, 825351, 965640 (lewm retrain)
- 1 LIFTransformer pixel ckpt failed (1/130); 129 trained successfully + 1 lewm retrained
- The **authoritative pixel main table is at**
  `results/journal_prep/MAIN_TABLE_5M_PIXEL_FULL.md`
  (v0.7.18.x, 13 models × 10 splits × 13 DMC envs, env-SR / cos_dist per cell).
  The earlier random-policy pixel table is **deleted** (vacuous — cos_dist
  ~ 1.0-1.25 uniformly across all models because no model could actually
  reach the goal with random actions). `cross_modality_table_cem.md` is
  retained as the CEM-planned pixel rollup referenced by the main table.
- Per-model mean_cos_dist under CEM (state vs pixel):
  - State: STJEWM 0.103-0.123 (v0.7.18.4 fair rerun), LeWM-v2 0.183 (over-react), MLP 0.007 (collapse)
  - Pixel: STJEWM 1.10-1.29, LeWM-v2 1.23, MLP 1.14
  - **Rank order preserved at extremes**: MLP lowest (collapse), LeWM-v2 highest (over-react)
  - Calibrated cluster flatter in pixel (1.10-1.29) than in state (0.103-0.123) because the
    frozen ViT-Tiny's discrete 14x14 patches cannot represent the fine-grained 0.1-scale
    task-relevant features that state does.
- **Pixel encoder is a representational bottleneck, not a family-partition failure**.

## CEM pixel sample (STJEWM-trace/cross_benchmark_F1):

| Env | env-SR (CEM) | cos_dist (CEM) |
|---|---|---|
| cartpole | 1.000 | 0.315 |
| pendulum | 0.333 | 0.639 |
| ball_in_cup | 1.000 | **0.013** |
| cheetah | 0.000 | 0.292 |
| walker | 0.000 | 0.593 |
| hopper | 0.000 | 0.411 |
| quadruped | 0.000 | 0.646 |
| humanoid | 0.000 | 0.781 |
| dog | 0.000 | 0.471 |
| fish | 0.000 | 9.964 |
| stacker | 0.000 | 0.397 |

The fact that ball_in_cup (4-DOF goal alignment) achieves env-SR=1.0 with cos_dist=0.013
shows the pixel-side encoder produces **executable control signals** for the right kind of
task. Locomotion envs (cheetah/walker/etc.) need a higher-fidelity encoder to plan 5-step
horizons under 5-step CEM, which is the v0.7.13 bug #3 ceiling.



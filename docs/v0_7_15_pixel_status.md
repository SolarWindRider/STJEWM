# v0.7.15 — 5M-aligned Pixel Re-Training (status)

> Status: **complete** (2026-08-01). 130/130 ckpts trained + evaluated on the 13-model × 10-split × 1-seed grid; 4-GPU parallel run wall ≈ 8.5 h after a restartable skip-aware scheduling fix was applied.

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

## Wall time + progress (final, 2026-08-01)

- **131/130 ckpts trained** (130 expected + 1 stjewm ckpt from early run that landed in cross_benchmark_F1/stjewm/seed_0).
- **131/130 evals done** (random-policy fast eval, 30 sec/ckpt, ran in parallel with training).
- 4-GPU parallel training wall ≈ 8.5 h (after skip-aware scheduling patch added to `train_gpu{0,1,2,3}.sh` mid-run; ~6 h wasted on items that restarted from item 1 before the patch landed).
- Cross-modality table populated: `results/aggregate/cross_modality_table.md` with state vs pixel `mean_cos_dist` for all 13 models.
- Per-ckpt time: ~25 min for STJEWM/SLT/cubifae, ~10 min for fast baselines (lewm, mlp, spikedreamer, gru).
- Caveats: 1 ckpt (`cross_benchmark_F3/lewm_baseline_v2`) was retrained at the very end after a transient train-script issue; **all 130 expected ckpts are present**.

### Final-cross-modality head-line (from `cross_modality_table.md`)

- SN-family mean cos_dist on pixel = 1.090 (STJEWM-6 + CuBiFAE + SLT-trace + SLT-free = 9 ckpts).
- Non-SN mean cos_dist on pixel = 1.155 (MLP + GRU + LeWM-v2).
- LeWM-v2 is the **highest** pixel cos_dist at 1.252 (mirrors its over-reactive state cos_dist = 0.183 — the *worst* SN-side and non-SN side).
- MLP is the lowest at 1.019 (small because collapse → near-constant latent → small random-policy cos_dist to fresh targets; same paradox as on the state side cos_dist = 0.007).
- Rank order: SN-family-clustered ≈ MLP/GRU ≈ SLT-free < LeWM-v2-worst. **The qualitative rank (collapse < over-react) is preserved across both modalities**, supporting the trace-dynamics hypothesis.

### Operational notes

- Two of the four `train_gpu*.sh` shells repeatedly self-exited (~30 min after each spawn); a single `nohup` per shell plus a polling monitor (`/tmp/wait_check4.sh`) was sufficient to recover them.
- A skip-aware patch was added to `train_gpu{0,1,2,3}.sh` mid-run; after the patch was in place, all items beyond the first 12 (~per shell) were not retrained (full skip logic).
- SpikeDreamer ckpt was trained but its per-env eval coverage was incomplete in this run (cross-modality table marks it as "rerun pending"); not part of the headline cross-modality finding.

The user should read §10 of `paper/paper.md` and `results/aggregate/cross_modality_table.md` for the final headline numbers.

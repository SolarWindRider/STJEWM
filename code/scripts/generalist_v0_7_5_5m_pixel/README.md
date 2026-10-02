# v0.7.15 5M-aligned Pixel Re-Training

Cross-modality complement to `generalist_v0_7_5_5m/` (state version).
Same 130 ckpts grid (13 models x 10 splits x 1 seed),
**5M-aligned trainable** params, **5.5M frozen ViT-Tiny** pixel encoder.

## Current manifest-gated commands

Run from `/home/lx/snn`. The final consolidated training manifest is the sole
repair-training driver; the historical one-off training commands are not
substitutes for its approved checkpoint/data/step budgets.

After its audited checkpoints are ready, evaluate into a fresh root:

```bash
PY=/home/lx/miniconda3/envs/snn/bin/python
export TRAINING_MANIFEST=/data/lx/tmp/results/_repair_archive/20260916T102154Z/training_final_repair_manifest.json
export CKPT_ROOT=/data/lx/tmp/results/5m_pixel
: "${EVAL_ROOT:?Set a fresh pixel evaluation output directory}"
export EVAL_ROOT
bash code/scripts/generalist_v0_7_5_5m_pixel/run_pixel_eval_all.sh
```

The canonical producer is `eval_pixel_ckpt.py`. The completed grid receipt
must cover 130 checkpoints × 13 environments (1690 cells), with successful
process exits and matching checkpoint/summary hashes. Both old
`eval_pixel_ckpt_cem.py` and `eval_pixel_ckpt_fast.py` were removed.
The source-only `run_highpower_cem.sh` sensitivity grid is not a primary-grid
substitute.

To summarize completed manifests without scanning old checkpoint directories:

```bash
: "${STATE_RUN:?Set a completed final-generation state evaluation directory}"
: "${REPORT_ROOT:?Set a fresh report output directory}"
$PY -m code.scripts.generalist_v0_7_5_5m_pixel.aggregate_pixel \
  --training-manifest "$TRAINING_MANIFEST" --pixel-run "$EVAL_ROOT" \
  --out "$REPORT_ROOT/pixel_summary.md"
$PY -m code.scripts.generalist_v0_7_5_5m_pixel.cross_modality_table \
  --training-manifest "$TRAINING_MANIFEST" --state-run "$STATE_RUN" \
  --pixel-run "$EVAL_ROOT" --out "$REPORT_ROOT/cross_modality_cells.md"
$PY -m code.scripts.generalist_v0_7_5_5m_pixel.cross_modality_table_cem \
  --training-manifest "$TRAINING_MANIFEST" --state-run "$STATE_RUN" \
  --pixel-run "$EVAL_ROOT" --out "$REPORT_ROOT/cross_modality_models.md"
```

Pixel checkpoints retain their actual `stjewm` / `lewm_baseline` identities,
mapped explicitly to state `stjewm_trace_only` / `lewm_baseline_v2` for paired
comparisons. Unpaired conditions are listed, not silently pooled. These are
descriptive comparisons: state and pixel goals, initialization, horizon and
replanning protocols differ. One training seed cannot provide a seed CI.

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

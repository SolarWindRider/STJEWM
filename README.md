# ST-JEWM: Learning Calibrated Event-Driven Predictive States for Generalizable World Models

> **Can the event history of a spiking dynamical system itself become a
> world-model predictive state that generalises across environments,
> when the downstream predictor and planner are forbidden from reading
> the continuous membrane potential?**

ST-JEWM is a **pure-SNN reconstruction-free world model** whose predictive
latent is a **post-spike trace** (bounded in [0,1] per dim, content-aware
forget gate, event-driven) rather than a continuous recurrent hidden state.
The planner reads only the trace — the **membrane-forbidden protocol**.

**Status: final repaired generation (2026-09-17) — 371-checkpoint retrain + full
grid re-evaluation complete; see `results/journal_prep/EXP_Report.md` (v4.0).**

---

## Headline results (final repaired generation, 2026-09-16)

> 371 affected checkpoints retrained (training manifest `4376bb14…`, protocol
> `causal_grad_sigreg_B_20260916`); all evaluation grids re-run under frozen
> sources (freeze v7) with per-cell provenance. All pre-repair numbers are void.
> Hub document: `results/journal_prep/EXP_Report.md` (v4.0).

### 1. LeWM-SR@0.05 is falsified — and inverted

The cos<0.05 hit-rate proxy is not just unreliable, it is **anti-correlated
with latent quality in the final generation**: collapsed / near-constant
latents score highest (LIF-Transformer 1.00 at cos 0.000, ALIF 0.95,
Stacked-LIF-trace 0.94, MLP 0.88) while STJEWM-trace — tied-best env-SR —
scores 0.37. Do not use it as a headline. The central diagnostic remains the
three-axis package (`div` + `resp` + event-ρ) plus closed-loop cos_dist.

### 2. Cluster structure is seed-stable; scale-invariance is not

13 models × 3 seeds (G5, paired): no seed flips collapse ↔ non-collapse
clusters. 3-seed pooled cos: STJEWM-trace 0.118±0.012, STJEWM-spike
0.213±0.016, LeWM 0.196±0.018 vs collapse cluster (LIF-Tx 0.001±0.002,
SLIF-trace 0.017±0.009, ALIF 0.028±0.021, SLIF-free 0.030±0.002).

On the scale axis (G4/G8/G16) the earlier "all 12 models scale-invariant"
claim **does not survive an explicit criterion** (div range <0.5 AND resp
max/min <5): 10 of 12 fail, with failure modes *switching across scale* —
ALIF resp 5.7/9.9 → **1937.9** at G16; MLP collapsed at G4/G8 → exploding at
G16 (div 1.4e8); SLIF-trace collapsed → non-collapsed. Only STJEWM-spike
(stable calibration) and LeWM (stable over-reaction) pass. Scale is a
failure-mode amplifier, not a preserved property.

### 3. Event-ρ erratum (final values)

The v3.x claim "SNN family ρ 0.95 vs continuous family chance" is void. Final
readout ρ (70 cells/model) spans **0.004–0.527 for all 13 models** — LeWM 0.527
is the highest, STJEWM-trace 0.034 among the lowest; same-batch obs-embedding ρ
averages 0.814 (no ≈1 ceiling). STJEWM-trace's profile is a near-constant trace
readout (div 1e-4, resp 0.26) with the best closed-loop cos (0.109) and
tied-best env-SR (0.31).

### 4. Efficiency is a ~3× hypothetical proxy, not 20×

Sparsity-weighted proxy (explicitly *not* measured hardware energy/speed):
STJEWM-trace 3.37 vs GRU/MLP/LeWM 9.76–10.24 MFLOP-equivalents/token (≈2.9–3.0×);
Stacked-LIF-trace/free 5.05/3.66 are the only other savers. Measured soma
sparsity (LIF-Tx 98.4%, ALIF 96.1%) does *not* translate into savings under the
weighted partition.

### 5. Honest negatives (reported, not hidden)

- **Trace causality rejected**: 0/2 envs show differential trace use
  (cartpole 0.60→0.70 when ablated; cheetah 0.00 baseline).
- **Pixel collapse is graded, not binary**: on the 1,690-cell grid
  (10-split pooling) STJEWM-trace/spike keep the highest cos (0.115/0.177),
  rate/no_trace/leak/membrane/LeWM/GRU sit mid-range (0.020–0.044), ALIF is
  near-zero (0.003), and Stacked-LIF×2/MLP/LIF-Tx are exactly 0.000 with
  trivial LeWM-SR = 1.00.
- **λ=0 collapse** (SIGReg ablation): cos 0.252→0.005 with LeWM-SR 0.37→0.99
  while env-SR stays flat — single-metric evaluation hides collapse.
- **External Spiking-WM**: the earlier external table evaluated a randomly
  initialized network (strict-load hit 0/82 weights). After strict subtree
  loading (82–96 tensors, 7.23–7.48M params, all 12 tasks), its within-episode
  event-ρ is 0.10–0.84 — same order as local models. No cross-protocol ranking
  is claimed.

### 6. Cross-modality (state → pixel)

Paired same-env comparison (1105 pairs, `agg_final/cross_modality_paired.json`):
state env-SR 0.252 / cos 0.080 / LeWM-SR 0.679 vs pixel 0.147 / 0.045 / 0.800.
Pixel eval protocol: horizon 5, replan every step; 13-env grid (humanoid_CMU
restored).

---

## Authoritative tables

All regenerated 2026-09-17 from the final repaired evidence (provenance headers
in every file): `results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md` (E1+E2+E4+E7,
2892 cells), `MAIN_TABLE_5M_PIXEL_FULL.md` (1690 cells),
`DIAG_RELOAD_SUMMARY.md`, `SCALE_INVARIANCE.md`, `SIGREG_SWEEP.md`,
`LATENT_ROLLOUT_SUMMARY.md`, `UTILITY_MPC_TABLE.md`, `FULL_METRIC_MATRIX.md`,
`g2_summary.json`, `pixel_stats_summary.json`. Headline digest:
`/data/lx/tmp/results/agg_final/HEADLINE_NUMBERS.md`.

## Paper

- `experiment_report_full_zh.{tex,pdf}` — Chinese experiment report。**OBS-only**：
  `obs://lixiang01/STJEWM_NMI/paper/`。**注意:OBS 上的 PDF 为 v0.7.19 旧代数字,需按
  EXP_Report v4.0 重写后再上传**。
- `intro_draft_v4.tex` — intro 草稿;个别数字待按本版更新(见 `~/fig_captions.md` 同步说明)。

## Experiments

| Experiment | Grid | Status (2026-09-17) |
|---|---|---|
| State 5M-aligned (E1) | 13 × 10, 1300 cells | **done** (`/data/lx/tmp/results/state_final_corrected_20260916/`) |
| Seed robustness (E2, B2/G5) | 13 × 3 splits × 3 seeds, 962 new cells | **done** (same root + `agg_final/g5_multiseed.json`) |
| Pixel 5M-aligned (E3) | 13 × 10 × 13 envs, 1690 cells | **done** (`/data/lx/tmp/results/5m_pixel_final_20260916/`) |
| Held-out (E4) | 9 specs, 546 cells | **done** (state root, `heldout_*`) |
| Scale axis (E5) | 36 ckpts, 252 stats | **done** (`agg_final/scaling_table.json`) |
| Three-axis diagnostics (E6) | state 910 + pixel 520 cells | **done** (aux grid, `_repair_auxiliary_grid/20260916_final_resume/`) |
| sigreg sweep (E7) | 4 λ × 2 splits, 84 cells | **done** (state root) |
| Position probe (E8) | 91 cells | **done** (`journal_prep/g2_summary.json`) |
| Trace ablation (E9) | 2 env × 6 modes | **done** (`/data/lx/tmp/results/trace_ablation_final/`) |
| Latent-goal MPC (E10) | 12 × 4 × 5 horizons | **done** (`/data/lx/tmp/results/utility_final/`) |
| External Spiking-WM (E11) | 12 tasks, strict-load | **done** (`/data/lx/tmp/results/spiking_wm_final_20260916/`) |
| Unified rollout (E12) | 68 runs | **done** (`/data/lx/tmp/results/latent_rollout/`) |
| Energy proxy (G3/P11) | 26 rows | **done** (`journal_prep/P11_energy_final/`) |
| Synthetic validation (P12) | reconstructed per documented spec | **done** (`/data/lx/tmp/results/p12_synthetic_reconstruction/`) |
| G2 event-AUROC, P13 multi-epoch, P22, utility 4 drivers | — | **not run** (no claims made) |

### Checkpoint recovery (2026-08)

A batch text-replacement accident (non-text files opened in text mode) corrupted a
generation of auxiliary checkpoints. Recovery status at `git HEAD f65e2d8`+:

- **Retrained from scratch with the original commands**: `5m_seed1` (39), `5m_pixel`
  (130 + 4 post-hoc: 3 lewm + 1 hidden_leak), `sigreg` (8). All pass
  `torch.load` integrity checks — the full tree (`results/**/*.pt`, 1174 files) is
  **0 corrupt**.
- **Permanently unrecoverable**: 16 old single-env checkpoints
  (`cheetah_velhidden/finger/stacker/dog/cartpole_2d/cartpole_flicker/fish/tworoom` ×
  `stacked_lif_trace`/`stacked_lif_free`) from early v0.7.x experiments — no training
  command was recorded for them. Their eval numbers are archived in
  `docs/single_env_historical_eval.md` (the legacy `results/<env>/` directories
  were removed in the 2026-08 legacy cleanup); the checkpoints themselves cannot
  be reproduced.
- **Bug fixed during retraining** (`code/train/train.py`): state-mode runs had
  `image_size` overwritten to 0 after dataset load (build fell back to 84px →
  ViT positional embeddings `(1,37,192)`), mismatching the 5M-main 224px layout
  `(1,257,192)` used by `event_align`. Now the CLI `--image-size` (default 84) is
  honored in state mode, and STJEWM's `state_dim` routing keys off pixel geometry
  (`obs_dim == 3·H²`) instead of `image_size > 0`. Event-alignment spot-checks after
  retraining: `corr_obs_latent` 0.9991 (seed1) / 0.9999 (sigreg), consistent with G1.

## Repository layout

```
code/                          # models, CEM, eval, train, scripts
configs/oodc_5m/               # state split configs (10)
configs/oodc_5m_pixel/         # pixel split configs (10)
/data/lx/tmp/results/          # final evidence trees (state/pixel/aux/external grids,
                               #   utility, energy, P12; see README Experiments table)
results/5m*, results/spiking_wm/, results/latent_rollout/   # canonical checkpoint + diagnostic roots
results/journal_prep/          # AUTHORITATIVE aggregated tables (regenerated 2026-09-17)
paper/                         # figs + 报告源文件（报告 PDF 仅 OBS: obs://lixiang01/STJEWM_NMI/paper/）
docs/                          # rebuttal letter + pixel status (current only)
```

## Data — download & placement

All training / validation / test data used by the experiments is archived on OBS:

```
obs://lixiang01/STJEWM_NMI/data/
```

### Where to put each file (so training/eval works out of the box)

| # | Download from OBS | Size | Place it at | Why this exact location |
|---|---|---|---|---|
| 1 | `STJEWM_data.tar` | 646 MB | **repo root**, then `tar -xf STJEWM_data.tar` | restores `data/…` relative to the repo root — this is what `configs/oodc_5m/*.json` reference as `data/dm_control/…` |
| 2 | `pusht_expert_train.h5.zst` | 13 GB | `/home/lx/LeWM/data/pusht_expert_train.h5` (after `zstd -d`) | the split configs hard-code the **absolute** path `/home/lx/LeWM/data/pusht_expert_train.h5` |
| 3 | `tworoom.h5` | 13 GB | `/home/lx/LeWM/data/tworoom_extract/tworoom.h5` | the split configs hard-code `/home/lx/LeWM/data/tworoom_extract/tworoom.h5` |
| 4 | `spiking_wm/` (dir) | ~9 GB | `results/spiking_wm/` | real external baseline Spiking-WM (PNAS 2025): 12 DMC task checkpoints (`logs_<task>/latest_model.pt`, one per task: cartpole_swingup, cheetah_run, walker_walk, finger_spin, pendulum_swingup, cup_catch, reacher_easy, hopper_hop, quadruped_walk, dog_walk, fish_swim, humanoid_run), training logs, and protocol metrics (`protocol_<task>.json`); note the OBS object prefix is `spiking_wm/spiking_wm/…` (nested folder), so after downloading place the inner `spiking_wm/` folder at `results/`; eval script `code/scripts/eval_spiking_wm_protocol.py`; upstream code is not vendored — see `code/scripts/run_spiking_wm.py` (clone `https://github.com/Brain-Cog-Lab/Spiking-WM` to `/home/lx/Spiking-WM`) |

After step 1 your repo root must look like this:

```
<repo>/
  data/
    dm_control/cartpole_250k.npz
    dm_control/pendulum_250k.npz
    dm_control/reacher_mujoco_rollouts_5x.npz
    dm_control/3d_rollouts_250k/{ball_in_cup,cheetah,dog,finger,fish,hopper,
                                 humanoid,humanoid_CMU,quadruped,reacher,
                                 stacker,walker}_250k.npz
    delayed_t_maze_30k.npz
    delayed_t_maze_30k_3d.npz
    event_window_50k.npz
  configs/…
```

> **pusht / tworoom note.** These two live outside the repo because the configs use
> absolute paths under `/home/lx/LeWM/data/` (the original author's machine layout).
> Two options:
> 1. Recreate that layout on your machine (simplest, nothing to edit):
>    `mkdir -p /home/lx/LeWM/data/tworoom_extract` and place the files as in the table.
> 2. Or point the configs at your own paths:
>    `sed -i 's#/home/lx/LeWM/data#<YOUR_DIR>#g' configs/oodc_5m/*.json configs/oodc_5m_pixel/*.json`

One-command download & placement (option 1):

```bash
obsutil cp obs://lixiang01/STJEWM_NMI/data/STJEWM_data.tar .
tar -xf STJEWM_data.tar          # restores data/ inside the repo root

mkdir -p /home/lx/LeWM/data/tworoom_extract
obsutil cp obs://lixiang01/STJEWM_NMI/data/pusht_expert_train.h5.zst /home/lx/LeWM/data/
zstd -d /home/lx/LeWM/data/pusht_expert_train.h5.zst
obsutil cp obs://lixiang01/STJEWM_NMI/data/tworoom.h5 /home/lx/LeWM/data/tworoom_extract/
```

> The DMC `_250k.npz` files are raw float32 rollouts (incompressible, hence the
> uncompressed tar). Pixel experiments need **no** data files: `--env-kind dmc_pixel`
> collects episodes live from DMC via the frozen ViT encoder.

## Reproducing the final repaired experiments

The final generation is manifest-driven: a single training manifest
(`/data/lx/tmp/results/_repair_archive/20260916T102154Z/training_final_repair_manifest.json`,
sha256 `4376bb14…`) lists all 371 affected checkpoints with original args,
approved deltas (incl. LeWM embed 288), expected steps and required data
protocols. Drivers validate protocol provenance and refuse stale generations:

```bash
# 1. Retrain (quarantines canonical dirs, trains staged, validates, promotes)
python -m code.scripts.generalist_v0_7_5_5m.repair_training \
  --manifest <manifest> --archive-root <archive> \
  --staging-root /data/lx/tmp/results/_repair_staging_final \
  --gpus 0,1,3 --workers-per-gpu 2 --execute

# 2. State evaluation grids (E1/E2/E4/E7, 2892 cells; plan-mode first)
python -m code.scripts.generalist_v0_7_5_5m.repair_eval_grid \
  --manifest <manifest> --out-root /data/lx/tmp/results/state_final_corrected_20260916 \
  --checkpoint-selection all --gpus 0,1,3 --execute --resume

# 3. Auxiliary diagnostics (div/resp/ρ, probes, scaling, rollouts; 1321 cells)
python -m code.scripts.repair_auxiliary_grid --training-manifest <manifest> ...

# 4. Pixel grid (130 ckpts × 13 envs = 1690 cells)
CKPT_ROOT=/data/lx/tmp/results/5m_pixel EVAL_ROOT=/data/lx/tmp/results/5m_pixel_final_20260916 \
TRAINING_MANIFEST=<manifest> GPU=2 WORKERS=8 \
  bash code/scripts/generalist_v0_7_5_5m_pixel/run_pixel_eval_all.sh

# 5. Aggregation (manifest-gated; refuses incomplete grids)
python -m code.scripts.generalist_v0_7_5_5m.aggregate_5m \
  --training-manifest <manifest> --state-run <state_root> --out state_cells.md
```

External Spiking-WM diagnostics (strict subtree loading):
`code/scripts/eval_spiking_wm_protocol.py` / `eval_all_spiking_wm.sh`.
Energy proxy: `code/scripts/generalist_v0_7_5_5m/measure_energy.py --training-manifest <manifest>`.
P12 reconstruction: `code/scripts/p12_synthetic_validation.py`.

Legacy per-run launchers (`eval_one.sh`, `train_one*.sh`, run_phase*.sh) are kept
for archaeology only; the manifest drivers above are the supported path.
`eval_one.sh` additionally documents the canonical single-cell protocol constants
(see "Frozen protocol" below).

## Environment

Python 3.10 (single conda env), CUDA GPU (≥24 GB for pixel training; state eval
runs on 8–16 GB). Version pins used for the final generation:

```
torch==2.6.0+cu126   numpy==2.2.6        mujoco==3.11.0
dm_control==1.0.44   transformers==5.11.0  h5py==3.16.0
matplotlib==3.10.9   imageio==2.37.3     ruamel.yaml==0.17.4   zstandard==0.25.0
```

`obsutil` (any recent build) is only needed for the OBS data download above.

## Conventions — read before running anything

1. **Run from the repo root with `python -m`.** The package dir `code/` shadows
   the stdlib `code` module; every driver starts with
   `sys.path.insert(0, <repo_root>)` and is launched as
   `python -m code.scripts.<name>` (or `-m code.eval.closed_loop`). Running a
   script by bare file path from another cwd breaks imports.
2. **`build_model()` does NOT load weights.**
   `code/scripts/event_align.py::build_model` only *constructs* the module
   (using the checkpoint dict to infer architecture). Every consumer MUST
   follow with `model.load_state_dict(ckpt["model"], strict=True)` — this is
   what `code/eval/closed_loop.py` does and prints
   (`[closed_loop] trained weights loaded strictly`). If you write your own
   analysis script, copy that pattern; a missing load silently evaluates a
   random-initialized network.
3. **Checkpoint layout** is `<results_root>/<split>/<model_name>/seed_<n>/final.pt`
   (e.g. `results/5m_seed1/cross_benchmark_F1/stjewm_trace_only/seed_0/final.pt`).
   Model names are functional identifiers
   (`stjewm_{trace_only,spike,rate,no_trace,hidden_leak,membrane}`, `alif_timecell`,
   `stacked_lif_{trace,free}`, `lewm_baseline_v2`, `gru_baseline`, `mlp_baseline`,
   `liftransformer_baseline`, `stjewm_state`, …).
4. **Frozen protocol (state closed-loop).** pad_obs 128, action_dim 56,
   history_size 1 (single current-state latent), horizon 25, eval_budget 50,
   goal_offset 25, CEM 300 samples / 30 elites / 10 iters (30 for pusht),
   goal = same-episode state at t+25, `in_dist` split. Encoded as CLI defaults in
   `code/eval/closed_loop.py` and env vars in
   `code/scripts/generalist_v0_7_5_5m/eval_one.sh`. Pixel closed-loop differs:
   horizon 5, replan every step (`eval_pixel_ckpt.py`).
5. **Out-of-repo evidence roots.** The final grids live under
   `/data/lx/tmp/results/` (see the Experiments table). Re-creating them locally
   is optional — every cell JSON carries its own provenance; the aggregated,
   self-contained tables are committed under `results/journal_prep/`.

## Experiment → code file map

Training single cell (any experiment): `code/scripts/generalist_v0_7_5_5m/train_one_5m.sh`
→ `python -m code.train.train` with a split config from `configs/oodc_5m/`
(state) or `configs/oodc_5m_pixel/` (pixel). G-scale training uses
`configs/generalist_G{4,8,16}_train.json`; sigreg sweep uses
`train_sigreg_sweep.sh` / `eval_sigreg_sweep.sh`.

| # | Experiment | Driver (code file) | Config / input | Output → table |
|---|---|---|---|---|
| E1 | State 5M grid, 13 models × 10 splits | `code/scripts/generalist_v0_7_5_5m/repair_eval_grid.py` (manifest; per-cell legacy: `eval_one.sh` → `code/eval/closed_loop.py`) | `configs/oodc_5m/cross_benchmark_F*.json`, `oodc_F*.json` | `eval_<env>.json` per cell → `MAIN_TABLE_5M_STATE_FULL.md` via `aggregate_5m.py` |
| E2 | 3-seed robustness (B2/G5) | `B2_multiseed_launcher.sh` / `G5_multiseed_launcher.sh` + `B2_multiseed_aggregate.py` / `G5_multiseed_aggregate.py` | same splits, seeds 0/1/2 | `results/journal_prep/{B2,G5}_multiseed/` |
| E3 | Pixel 5M grid, 130 ckpts × 13 envs | `code/scripts/generalist_v0_7_5_5m_pixel/run_pixel_eval_all.sh` → `eval_pixel_ckpt.py`; train: `train_one_pixel.sh` / `train_all_130_pixel.sh` | `configs/oodc_5m_pixel/*.json` (live DMC pixels, no npz) | `eval_summary.json` per cell → `MAIN_TABLE_5M_PIXEL_FULL.md` via `aggregate_pixel.py` |
| E4 | Held-out specs | `run_heldout_eval.sh` (same closed_loop CLI) | `configs/heldout_eval/` | `heldout_*` cells |
| E5 | Scale axis (G4/G8/G16), div/resp/ρ | training: `train_one_5m.sh` + `configs/generalist_G*_train.json`; diagnostic: `code/scripts/cls0917_scaling.py` | — | `agg_final/scaling_table.json`, `SCALE_INVARIANCE.md` |
| E6 | Three-axis diagnostics (div/resp/ρ cells) | `code/scripts/repair_auxiliary_grid.py` (manifest-gated); single cell: `measure_latent_stats_5m.py` → `event_align.py::collect_state_rollout` | manifest | aux grid JSONs → `FULL_METRIC_MATRIX.md` |
| E7 | SIGReg sweep (λ ∈ {0, 0.005, 0.01, 0.05}) | `train_sigreg_sweep.sh`, `eval_sigreg_sweep.sh` | — | `SIGREG_SWEEP.md` |
| E8 | Position probe (linear R²) | `code/scripts/probe.py` (loop: `probe_all.sh`, `probe_one.sh`) | state checkpoints | `journal_prep/g2_summary.json` |
| E9 | Trace/window ablation, 6 modes | `code/scripts/event_window_ablation.py` + `aggregate_event_window_ablation.py` | final trace ckpt × {cartpole_2d, cheetah} | `/data/lx/tmp/results/trace_ablation_final/`, report §10 |
| E10 | Latent-goal MPC horizon sweep | `code/scripts/utility/run_latent_goal_mpc.py` → `latent_goal_mpc.py` | state ckpts × 4 goal sets | `UTILITY_MPC_TABLE.md` |
| E11 | External Spiking-WM (PNAS 2025) | generate: `code/scripts/run_spiking_wm.py` (clones upstream); diagnose: `code/scripts/eval_spiking_wm_protocol.py` / `eval_all_spiking_wm.sh` | `results/spiking_wm/logs_<task>/` (12 tasks, OBS) | strict-subtree-load ρ + returns; report §11 |
| E12 | Unified latent rollout (div/resp/ρ + npz) | `code/scripts/latent_rollout.py` | 17 ckpts × 2 env × 2 split, 200-step random policy | `results/latent_rollout/`, `LATENT_ROLLOUT_SUMMARY.md` |
| E13 | Paired control (state) | `code/scripts/e13_paired_control.py` | — | `journal_prep/E13_paired_control/` |
| E14 | Scale-axis closed loop (G4/G8/G16) | `code/scripts/e14_scale_closed_loop.py` | in-union 4 envs, E1 protocol | `journal_prep/E14_scale_closed_loop/` |
| G1 | Event-alignment ρ | `code/scripts/event_align.py` (per-cell driver inside `repair_auxiliary_grid.py`) | 27-frame windows, `--pad-obs-to 128 --action-dim-eval 56` | per-cell `latent_stats*.json` |
| G3 | Energy/FLOPs proxy | `code/scripts/generalist_v0_7_5_5m/measure_energy.py --training-manifest <m>` | manifest | `journal_prep/P11_energy_final/` |
| P12 | Synthetic validation (public reconstruction) | `code/scripts/p12_synthetic_validation.py` | protocol spec in report §P12 | `p12_synthetic_reconstruction/` |
| Cross-modality | state↔pixel paired table | `cross_modality_table.py` / `cross_modality_table_cem.py` (pixel pkg) | 1105 pairs | `agg_final/cross_modality_paired.json` |

Mechanism probes (report §5, one command each, all strict-load; outputs under
`results/mech_0917/` and `/data/lx/tmp/results/accessibility_20260917/`):

| Probe | Script |
|---|---|
| S1 history-window latents | `code/scripts/mech0917_history.py` |
| S2 multistep rollout / K-sweep | `code/scripts/mech0917_multistep.py` (`load_model` reused by `access_eval.py`) |
| S3 gain (g_obs/g_a) | `code/scripts/stab0917_gain.py` |
| S4 obs-mask | `code/scripts/stab0917_maskobs.py` |
| S5 rollout slope | `code/scripts/stab0917_rollout.py` (repair: `stab0917_rollout_repair.py`) |
| M4 transient perturbation recovery | `code/scripts/mech0917_transient.py` |
| timescale | `code/scripts/mech0917_timescale.py` |
| nonlinear probe / transfer | `code/scripts/cls0917_nonlinear_probe.py`, `cls0917_transfer.py` |
| action-EMA control | `code/scripts/stab0917_action_ema.py` |

Figures: `code/scripts/make_fig1.py`, `make_fig2.py`, `make_nmi_figures.py`
(trajectory dumps from `collect_traj_for_figs.py`). Consistency re-audit of any
table against raw cells: `code/scripts/audited_results.py`.

Not run / no claims (do not cite): G2 event-type AUROC, P13 multi-epoch, P22
cheetah, and the utility drivers `sample_efficiency.py`, `latent_env_grad.py`,
`cross_env_gen.py`, `budget_scaling.py` (code present, marked pending in
EXP_Report §12).

## Verification checklist

After setting up data + environment, verify your setup reproduces the protocol
before launching grids. Point `--ckpt` at any retrained `final.pt` (paths below
are the author's out-of-repo evidence root; retrain locally or download
checkpoints to substitute your own):

```bash
# 1. Data integrity: every checkpoint .pt loads
python - <<'PY'
import torch, glob
bad = [p for p in glob.glob("results/**/*.pt", recursive=True)
       if not torch.load(p, map_location="cpu", weights_only=False)]
print("corrupt:", len(bad))          # expect 0
PY

# 2. Strict weight loading + single closed-loop cell (cartpole, ~8 s on GPU)
python -m code.eval.closed_loop --env cartpole \
  --ckpt /data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt \
  --data data/dm_control/cartpole_250k.npz --n-episodes 1 --n-seeds 1 \
  --history-size 1 --horizon 25 --eval-budget 50 --goal-offset 25 \
  --pad-obs-eval 128 --action-dim-eval 56 --out /tmp/verify_cell.json
# must print: [closed_loop] trained weights loaded strictly on <device>
# sanity (1 episode, final repaired ckpt): env-SR 1.0, mean_cos_dist ~0.04

# 3. Event-alignment spot check (obs<->latent jump correlation, ~1 min)
python -m code.scripts.event_align --env cartpole_2d \
  --model stjewm_trace_only \
  --ckpt /data/lx/tmp/results/5m_5mpar/oodc_F1/stjewm_trace_only/seed_0/final.pt \
  --out /tmp/verify_align.json --pad-obs-to 128 --action-dim-eval 56 --device cuda:0
# sanity: representations.observation_embedding.event_rho >= 0.99 (obs-embedding
# alignment); the READOUT rho (printed line) is a *finding*, not a check —
# final-generation values are 0.004-0.527 across all 13 models (report §G1 erratum)
```

## License

See the upstream LeWM / dmc_control / OGBench licenses.

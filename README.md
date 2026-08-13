# ST-JEWM: Learning Calibrated Event-Driven Predictive States for Generalizable World Models

> **Can the event history of a spiking dynamical system itself become a
> world-model predictive state that generalises across environments,
> when the downstream predictor and planner are forbidden from reading
> the continuous membrane potential?**

ST-JEWM is a **pure-SNN reconstruction-free world model** whose predictive
latent is a **post-spike trace** (bounded in [0,1] per dim, content-aware
forget gate, event-driven) rather than a continuous recurrent hidden state.
The planner reads only the trace — the **membrane-forbidden protocol**.

**Status: v0.7.19 — all experiments complete, paper current.**

---

## Headline results

### 1. LeWM-SR is falsified as a calibration metric (§2.3a)

A stateless MLP baseline (no recurrence, no event state) reaches
**LeWM-SR = 98.0%** on the 20-env suite while its latent has per-dim
standard deviation **0.0002** (a constant zero vector). A metric that a
constant latent passes trivially cannot diagnose planner quality.
**LeWM-SR is deprecated as a standalone headline**; the four-metric
package (`mean_cos_dist` + `div` + `resp` + event-ρ) is the paper's
central diagnostic. Its boundaries are quantitatively identifiable on
synthetic ground truth (P12): encoder gain k=0.3→calibrated, k=0.5→over-reactive;
noise σ=0.02→inliers, σ=0.05→noise.

### 2. Three-cluster partition (5M-aligned, parameter-fair, 3-seed)

13 models × 10 splits × 3 seeds, CEM 300×30×10 closed-loop eval:

| Cluster | Models | cos_dist (3-seed CI) |
|---|---|---|
| **Calibrated** | STJEWM 6 readouts + Stacked-LIF-trace/free + ALIF-timecell | 0.10–0.14, CIs pairwise overlap |
| **Collapse** | MLP, GRU, LIF-Transformer | ≈ 0.000–0.02 (constant latent) |
| **Over-reactive** | LeWM-v2 | 0.19 (CI disjoint, Cohen's d −7…−8.5 vs calibrated) |

- **Parameter-robust**: STJEWM retrained 2.70M→5.06M (n_layers=4),
  cos_dist delta < 0.004 — calibration is an architecture property,
  not a parameter-count artifact (v0.7.18.4 FAIR rerun).
- **env-SR** (aggregation-corrected): easy envs saturate 1.0
  (ball_in_cup, cartpole, cheetah, finger), hard envs 0 (dog, humanoid,
  quadruped, reacher, stacker, tworoom) — a CEM 5-step planning ceiling,
  not a model property.

### 3. Event alignment is spike-based, not recurrence

Event-ρ (obs-event ↔ latent first-difference), 13 models × 4 envs × 2 splits:

| Family | mean ρ |
|---|---|
| SNN (STJEWM 6 + Stacked-LIF 2 + ALIF-timecell) | **0.9989** |
| LeWM-v2 (Transformer) | 0.7515 |
| GRU (recurrent, continuous gating) | **−0.0074** |
| MLP / LIF-Transformer | ≈ 0 |

The **GRU reverse control** (same recurrent temporal aggregation, continuous
gating → chance alignment) shows the alignment comes from the **spike
representation**, not from recurrence.

### 4. Efficiency

Effective per-step FLOPs (event-driven discount, state obs):
**STJEWM 0.46–0.48 MFLOPs** vs Stacked-LIF 1.94–2.13 vs GRU/MLP/LeWM-v2 9.8–10.2.
≈ **20× cheaper than dense baselines**, ≈ 4.4× cheaper than Stacked-LIF.

### 5. Honest negatives (reported, not hidden)

- **Trace causality rejected**: event-window + CEM-rollout ablation →
  0/3 cells show differential trace use. Correlation (ρ) stands; the
  strong causal claim does not (B4).
- **Event-AUROC at 1-epoch**: Stacked-LIF-trace 0.672 (best), LeWM 0.626,
  STJEWM ≈ 0.50 (chance). Stacked-LIF-trace's edge comes from its *linear* readout
  (moving-avg→Linear); STJEWM's gated trace is nonlinear — event info
  is in the latent (ρ high) but not linearly decodable.
- **Cheetah edge is marginal**: 60 eps paired, pooled t=4.15 but
  4/10 splits flip → "split-dependent, not a strong edge".
- **Probe R²**: STJEWM position R² ≈ −0.03…−0.07 (chance),
  LeWM +0.29 — the event-vs-position dissociation.
- **sigreg sweep**: lambda_sigreg ∈ {0.09…0.0} does not change pred
  loss; the STJEWM-vs-Stacked-LIF cos_dist gap is within noise (hypothesis
  "sigreg hijacks optimization" rejected).

### 6. Cross-modality (state → pixel)

Pixel (frozen ViT-Tiny 5.46M + trainable 5.00M), 13 DMC envs × 10 splits,
CEM eval with static (reachable) goals:
- Cluster ordering preserved at the extremes (collapse lowest, over-react highest).
- **LeWM-v2 fails control on pixel**: env-SR 0.091, lowest of all models
  (STJEWM-trace 0.171, Stacked-LIF-trace 0.178, MLP 0.172).
- Frozen ViT is a representational bottleneck (cos scale 0.6–0.9 vs state 0.1);
  fish env is a ViT blind spot (cos 3.4–5.4).

---

## Authoritative tables

| Table | Content |
|---|---|
| `results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md` | **State headline**: 10 splits × 13 models × per-env (env-SR/cos_dist), fair 5.06M |
| `results/journal_prep/MAIN_TABLE_5M_PIXEL_FULL.md` | **Pixel headline**: 10 splits × 13 models × 13 envs |
| `results/journal_prep/FULL_METRIC_MATRIX.md` | 13 models × 14 metrics, zero gaps (event-ρ, AUROC, FLOPs, probe R², 3-seed) |
| `results/journal_prep/JOURNAL_STORY.md` | Evidence map (read first) |
| `results/journal_prep/GAP_LIST.md` | Coverage gaps (all closed) |
| `results/journal_prep/sigreg_sweep_summary.md` | sigreg weight sweep |

## Paper

- `paper/experiment_report_full_zh.tex` + `.pdf` — Chinese experiment report (current, v0.7.19)
- English Nature Machine Intelligence draft — superseded; current English narrative is the Chinese report structure (see `experiment_report_full_zh.tex`)
- Compile the PDF with Tectonic: `tectonic experiment_report_full_zh.tex`

## Experiments

| Experiment | Models × Splits | Protocol | Status |
|---|---|---|---|
| State 5M-aligned | 13 × 10 | CEM 300×30×10, H=5, budget 50, goal_offset 25, 5 eps | **done** (`results/5m/`, `results/5m_5mpar/` fair rerun) |
| Pixel 5M-aligned | 13 × 10 | same CEM, static goals, frozen ViT | **done** (`results/5m_pixel/`) |
| Event-ρ (G1) | 13 models × 4 envs × 2 splits | 200-step random policy | done |
| Event-AUROC (G2) | 13 × 13 DMC envs × 5 targets | B3-fixed probes | done |
| FLOPs (G3/P11) | 13 models | analytic + measured sparsity | done |
| Probe R² (G4/B3) | 13 × 10 envs × 5 targets | B3-fixed probes | done |
| 3-seed CIs (B2/G5) | 13 × 3 splits × 3 seeds | state, seed 1–2 retrained | done |
| Ablation (B4) | 3 cells × 6 modes | event-window + CEM-rollout | done |
| Synthetic validation (P12) | 6 encoders + sweeps | known ground truth | done |
| Multi-epoch (P13) | 3 models × 2 splits × {3,5} ep | stability check | done |
| Cheetah edge (P22) | 2 models × 10 splits × 60 eps | paired | done |
| sigreg sweep | 4 weights × 2 splits | 8 ckpts | done |

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
results/5m/                    # state evals (baselines)
results/5m_5mpar/              # state evals (fair STJEWM 5.06M)
results/5m_pixel/              # pixel evals (CEM summaries)
results/journal_prep/          # AUTHORITATIVE aggregated tables
paper/                         # experiment_report_full_zh.{tex,pdf} (中文实验报告, current)
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

## Reproducing the 5M-aligned experiments

State training (fair STJEWM, n_layers=4):

```bash
CUDA_VISIBLE_DEVICES=0 python -m code.train.train \
  --model stjewm --multi-env-spec configs/oodc_5m/<SPLIT>.json \
  --pad-obs-to 128 --action-dim 56 --embed-dim 192 --image-size 0 \
  --n-layers 4 --epochs 1 --batch 32 --lr 3e-4 \
  --history-size 1 --goal-offset 25 --seed 0 \
  --readout-mode <trace_only|spike_only|...> --out results/5m_5mpar/<SPLIT>/stjewm_<RO>/seed_0
```

State eval:

```bash
CUDA_VISIBLE_DEVICES=0 OUT_PARENT=results/5m_5mpar \
  bash code/scripts/generalist_v0_7_5_5m/eval_one.sh \
  stjewm_<RO> results/5m_5mpar/<SPLIT>/stjewm_<RO>/seed_0/final.pt \
  configs/oodc_5m/<SPLIT>.json 0
```

Pixel training uses `configs/oodc_5m_pixel/<SPLIT>.json` + `--image-size 84`; pixel eval via
`code/scripts/generalist_v0_7_5_5m_pixel/eval_pixel_ckpt_cem.py`.

## License

See the upstream LeWM / dmc_control / OGBench licenses.

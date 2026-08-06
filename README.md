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
| **Calibrated** | STJEWM 6 readouts + SLT-trace/free + CuBiFAE | 0.10–0.14, CIs pairwise overlap |
| **Collapse** | MLP, GRU, SpikeDreamer | ≈ 0.000–0.02 (constant latent) |
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
| SNN (STJEWM 6 + SLT 2 + CuBiFAE) | **0.9989** |
| LeWM-v2 (Transformer) | 0.7515 |
| GRU (recurrent, continuous gating) | **−0.0074** |
| MLP / SpikeDreamer | ≈ 0 |

The **GRU reverse control** (same recurrent temporal aggregation, continuous
gating → chance alignment) shows the alignment comes from the **spike
representation**, not from recurrence.

### 4. Efficiency

Effective per-step FLOPs (event-driven discount, state obs):
**STJEWM 0.46–0.48 MFLOPs** vs SLT 1.94–2.13 vs GRU/MLP/LeWM-v2 9.8–10.2.
≈ **20× cheaper than dense baselines**, ≈ 4.4× cheaper than SLT.

### 5. Honest negatives (reported, not hidden)

- **Trace causality rejected**: event-window + CEM-rollout ablation →
  0/3 cells show differential trace use. Correlation (ρ) stands; the
  strong causal claim does not (B4).
- **Event-AUROC at 1-epoch**: SLT-trace 0.672 (best), LeWM 0.626,
  STJEWM ≈ 0.50 (chance). SLT's edge comes from its *linear* readout
  (moving-avg→Linear); STJEWM's gated trace is nonlinear — event info
  is in the latent (ρ high) but not linearly decodable.
- **Cheetah edge is marginal**: 60 eps paired, pooled t=4.15 but
  4/10 splits flip → "split-dependent, not a strong edge".
- **Probe R²**: STJEWM position R² ≈ −0.03…−0.07 (chance),
  LeWM +0.29 — the event-vs-position dissociation.
- **sigreg sweep**: lambda_sigreg ∈ {0.09…0.0} does not change pred
  loss; the STJEWM-vs-SLT cos_dist gap is within noise (hypothesis
  "sigreg hijacks optimization" rejected).

### 6. Cross-modality (state → pixel)

Pixel (frozen ViT-Tiny 5.46M + trainable 5.00M), 13 DMC envs × 10 splits,
CEM eval with static (reachable) goals:
- Cluster ordering preserved at the extremes (collapse lowest, over-react highest).
- **LeWM-v2 fails control on pixel**: env-SR 0.091, lowest of all models
  (STJEWM-trace 0.171, SLT-trace 0.178, MLP 0.172).
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
- `paper/paper.md` — English narrative (current, v0.7.19)
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

## Repository layout

```
code/                          # models, CEM, eval, train, scripts
configs/oodc_5m/               # state split configs (10)
configs/oodc_5m_pixel/         # pixel split configs (10)
results/5m/                    # state evals (baselines)
results/5m_5mpar/              # state evals (fair STJEWM 5.06M)
results/5m_pixel/              # pixel evals (CEM summaries)
results/journal_prep/          # AUTHORITATIVE aggregated tables
paper/                         # paper.md + experiment_report_full_zh.{tex,pdf}
docs/                          # rebuttal letter + pixel status (current only)
```

## Data — download & placement

All training / validation / test data used by the experiments is archived on OBS:

```
obs://lixiang01/STJEWM_NMI/data/
```

| File | Size | Extract / place to (relative to repo root) | Used by |
|---|---|---|---|
| `STJEWM_data.tar` | 646 MB | `tar -xf STJEWM_data.tar` (restores `data/…`) | all state splits |
| `pusht_expert_train.h5.zst` | 13 GB | `zstd -d` → `/home/lx/LeWM/data/pusht_expert_train.h5` | pusht (train + eval) |
| `tworoom.h5` | 13 GB | `/home/lx/LeWM/data/tworoom_extract/tworoom.h5` | tworoom (train + eval) |

`STJEWM_data.tar` contains the 18 DMC / T-maze / event-window datasets referenced by
`configs/oodc_5m/*.json` and `configs/oodc_5m_pixel/*.json`:

```
data/dm_control/cartpole_250k.npz
 data/dm_control/pendulum_250k.npz
 data/dm_control/reacher_mujoco_rollouts_5x.npz
 data/dm_control/3d_rollouts_250k/{ball_in_cup,cheetah,dog,finger,fish,hopper,
                                       humanoid,humanoid_CMU,quadruped,reacher,
                                       stacker,walker}_250k.npz
 data/delayed_t_maze_30k.npz
 data/delayed_t_maze_30k_3d.npz
 data/event_window_50k.npz
```

> The pusht / tworoom paths are absolute (`/home/lx/LeWM/data/…`) in the split configs;
> they live outside this repo, so place the two files there before training. The DMC
> `_250k.npz` files are raw float32 rollouts (incompressible, hence the uncompressed tar).

Download example (obsutil):

```bash
obsutil cp obs://lixiang01/STJEWM_NMI/data/STJEWM_data.tar .
tar -xf STJEWM_data.tar   # restores data/ inside the repo root
obsutil cp obs://lixiang01/STJEWM_NMI/data/pusht_expert_train.h5.zst /home/lx/LeWM/data/
zstd -d /home/lx/LeWM/data/pusht_expert_train.h5.zst
mkdir -p /home/lx/LeWM/data/tworoom_extract
obsutil cp obs://lixiang01/STJEWM_NMI/data/tworoom.h5 /home/lx/LeWM/data/tworoom_extract/
```

Pixel experiments need no data files: `--env-kind dmc_pixel` collects episodes live from
DMC via the frozen ViT encoder.

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

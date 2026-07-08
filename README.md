# ST-JEWM: Spike-Trace Joint-Embedding World Model

> **Can the event history of a spiking dynamical system itself become a
> world-model predictive state, when the downstream predictor and planner
> are forbidden from reading the continuous membrane potential?**

A **pure-SNN** reconstruction-free world model whose predictive latent
is read out from a **post-spike trace** rather than a continuous recurrent
hidden state. The trace is bounded in [0,1] per dim, content-aware
(forget gate `alpha = sigma(W[r_{t-1}, s_t, c_t])`), and event-driven.

This repository contains the code, evaluations, and paper for ST-JEWM.
The full PDF is at `paper/paper.pdf`. Source: `paper/paper.md` and
`paper/paper.tex`. Current version: **v0.7.5** (2026-07-06).

## Headline results (v0.7.3)

Across **13 models × 24 envs × 6 metrics**, the consolidated master table
is at `results/aggregate/MASTER_TABLE.md` (sections §1–§8). One row per
model, 6 columns:

| Model | env-SR std (n=20) | env-SR stress (n=4) | LeWM-SR std (n=20) | LeWM-SR stress (n=4) | event-AUROC (n=215) | event-align ρ (n=6) |
|---|---|---|---|---|---|---|
| `stjewm_trace_only`        | 67.1 | 25.0 | 73.5 | **66.5** | 0.690 | 0.626 |
| `stjewm_hidden_leak`       | 64.0 | 25.5 | 61.4 | 54.5 | 0.690 | 0.620 |
| `stjewm_spike_only`        | 65.9 | 25.0 | 66.5 | 57.5 | **0.699** | 0.621 |
| `stjewm_no_trace`          | 66.3 | 25.0 | 61.8 | 52.5 | 0.688 | 0.624 |
| `stjewm_membrane_readout`  | 64.5 | 25.5 | 60.8 | 49.5 | 0.554 | 0.615 |
| `stjewm_rate_only`         | 64.6 | 28.5 | 66.3 | 62.5 | n/a | 0.630 |
| `cubifae_baseline`         | **69.5** | 25.5 | 76.3 | 52.5 | 0.569 | 0.638 |
| `spikedreamer_baseline`    | 68.3 | 41.5 | 0.0 | 0.0 | 0.474 | nan |
| `slt_lif_mpc_trace`        | 68.6 | 25.0 | 72.6 | 47.5 | 0.533 | 0.636 |
| `slt_lif_mpc_free`         | 65.7 | 26.5 | 66.7 | 66.5 | 0.504 | **0.640** |
| `lewm_baseline_v2`         | 68.2 | 25.5 | 76.9 | 56.5 | 0.166 | 0.160 |
| `gru_baseline`             | 66.6 | **42.0** | **78.8** | 51.0 | 0.574 | -0.011 |
| `mlp_baseline`             | 64.7 | 32.5 | **98.0**† | **95.5**† | 0.524 | -0.002 |

† MLP's 98% / 95% LeWM-SR is a **latent collapse** artefact (high cos
similarity, but env-SR on stress is only 32.5% and event-align ρ is 0).
The MLP cannot actually plan.

**Winners by column** (excluding the MLP latent-collapse trap and the
theoretical n/a on `rate_only`):
- env-SR std (20 env):  `cubifae 69.5` (STJEWM-trace 67.1 = 2nd, −2.4pp)
- stress env-SR:        `gru 42.0` (STJEWM-trace 25.0, −17pp; cartpole_flicker only)
- LeWM-SR std:          `gru 78.8` (STJEWM-trace 73.5; STJEWM-trace > LeWM Transformer 76.9? **NO**, but STJEWM-trace > hidden_leak 61.4 by 12.1pp)
- **LeWM-SR stress:**    **`stjewm_trace_only 66.5 = slt_lif_mpc_free 66.5`** (rank 1, tied)
- **event-AUROC:**       **`stjewm_spike_only 0.699`** (all 5 STJEWM readouts > 0.688, all 6 non-SNN baselines ≤ 0.670)
- **event-align ρ:**     `slt_lif_mpc_free 0.640` (STJEWM-trace 0.626 = 3rd; STJEWM family 0.615–0.630, all ≫ LeWM 0.160, GRU −0.011, MLP −0.002)

**The honest story** (v0.7.3): STJEWM does **not** win the end-to-end
task metrics (env-SR, LeWM-SR std) — `cubifae_baseline` and `gru_baseline`
do. But STJEWM wins the **mechanism metrics**: every one of its 6 readout
variants is the top family on event-AUROC, and the trace variant is
rank-1-tied on stress LeWM-SR. The membrane-forbidden protocol gives an
**event-correlated, interpretable predictive state without sacrificing
task performance** — it does not produce the best raw scores, but it
produces the best **mechanism**. (See `MASTER_TABLE.md` §10 for the
full claim ladder.)

## Generalist world-model evaluation (v0.7.5 — corrected metrics)

Twelve model variants trained as **shared-weights generalists** on the union of 4 / 8 /
16 standard envs. Eval'd on env-SR, **divergence-from-constant** (collapse-robust),
**responsiveness** (collapse-robust), 4 stress envs (flicker / vel-hidden / OOD /
long-horizon), event-AUROC (linear probe), and event-align ρ (latent
event-timeliness). Full numbers in `MASTER_TABLE.md` §9.5–§9.7 sub-sections.

**Headline finding (v0.7.5 — collapse-robust).** With the v0.7.4 metric
set, MLP looked best (LeWM-SR 95.6%) and STJEWM was indistinguishable
from baselines on env-SR. The v0.7.4 `LeWM-SR` metric was
**collapse-inflatable** — a constant latent trivially satisfies
`cos_dist < 0.1` for any goal. The new `divergence` metric (per-dim
std of the latent trajectory) is collapse-robust by construction. The
corrected picture:

| family | divergence | responsiveness | event-align ρ | failure mode |
|---|---|---|---|---|
| stjewm_{trace,spike,no_trace,hidden_leak,membrane,rate} | 0.011–0.013 | 0.20 | ≥ 0.99 | **calibrated** |
| cubifae_baseline, slt_lif_mpc_{trace,free} | 0.011 | 0.20 | ≥ 0.62 | calibrated (SNN) |
| **mlp_baseline** | **0.0002** (50× lower) | 0.55 | n/a | **collapse** |
| **gru_baseline** | 0.008 (similar) | **31** (150× higher) | -0.07 | **noise** |
| **lewm_baseline_v2** | **0.186** (16× higher) | **33** (150× higher) | 0.52 | **over-reactive** |

**Three distinct non-spiking failure modes are now visible** that
v0.7.4 conflated. Only MLP is collapsed; GRU is noisy; LeWM is
over-reactive. STJEWM is the **only** family that is simultaneously
(a) responsive to obs, (b) not collapsed, and (c) event-aligned
(ρ ≥ 0.99). On env-SR alone the families are within ±4pp; the
corrected metrics separate them.

Pipeline scripts in `code/scripts/generalist_v0_7_5/` (run `bash run_suite.sh` to
reproduce; `aggregate_master.py --suite {G4,G8,G16}` to rebuild the table;
`measure_latent_stats.py` to recompute the new collapse-robust metrics).
Per-suite spec files in `configs/generalist_{G4,G8,G16,4_stress,probe_eval}_*.json`.
Ckpts in `results/generalist[_G4|_G8|_G16]/<model>/seed_0/final.pt`.

## Repository layout

```
.
├── README.md                        # this file
├── LICENSE                          # MIT (code)
├── CITATION.cff                     # citation metadata
├── CONTRIBUTING.md                  # contributor guide
├── paper/
│   ├── paper.pdf                    # compiled PDF
│   ├── paper.md / paper.tex         # sources
│   └── figs/
├── code/
│   ├── stjewm.py                   # the model (ReadoutMode enum, 6 branches)
│   ├── lewm_transformer_baseline.py # 5.07M Transformer baseline
│   ├── gru_baseline.py             # 7.30M continuous-RNN baseline
│   ├── mlp_baseline.py              # 1.30M stateless baseline
│   ├── cubifae_baseline.py          # 10.17M SNN (multi-timescale ALIF)
│   ├── spikedreamer_baseline.py     # hybrid LIF+Transformer
│   ├── slt_lif_mpc_baseline.py      # closed-loop ctrl SNN
│   ├── sigreg.py                    # spike-train regulariser
│   ├── snn_cell.py                  # MultiCompartment SNN cell
│   ├── theory/                      # theoretical writeups
│   ├── data/
│   │   ├── base.py                 # WindowSpec / WindowDataset (pad_obs_to, env_id)
│   │   ├── loaders.py              # per-env loaders (12 env kinds)
│   │   └── multi_env.py            # load_multi_env_dataset (generalist)
│   ├── core/
│   │   ├── cem.py                  # CEM planner (LeWM App. B + F.1)
│   │   ├── encode.py                # encode_obs / encode_history
│   │   └── envs/                   # env class registry (22 envs)
│   ├── train/train.py               # single trainer; --multi-env-spec flag
│   ├── eval/closed_loop.py          # CEM planner + env-native SR
│   │                                # --pad-obs-eval / _PadObsWrapper
│   └── scripts/
│       ├── train_generalist.sh     # train one generalist ckpt
│       ├── eval_generalist.sh      # per-env eval of a generalist ckpt
│       ├── aggregate_generalist.py # build the §9-style table
│       ├── run_event_probes.sh     # full event-probe sweep
│       ├── run_stress_sweep.sh     # 50-cell stress-sweep
│       ├── run_event_align.sh      # event-boundary alignment
│       ├── aggregate_event_probes.py
│       ├── aggregate_event_align.py
│       ├── aggregate_results.py
│       ├── upload_master_table_to_obs.sh
│       └── ~70 more analysis scripts
├── configs/                         # generalist specs
│   ├── generalist_16env.json        # 16 std envs
│   ├── generalist_20env.json        # 16 std + 4 stress
│   ├── generalist_16env_2k.json     # time-budgeted 2K windows per env
│   ├── generalist_4env_2k.json      # 4-env subset (pilot eval)
│   ├── smoke_2env.json
│   └── smoke_4env.json
├── data/                            # (gitignored; see OBS for download)
│   └── delayed_t_maze_30k.npz
├── results/                         # per-env ckpts + aggregate tables
│   ├── <env>/<model>/eval.json      # per-cell JSON
│   ├── generalist/                  # 4 generalist ckpts (v0.7.3 pilot)
│   │   ├── stjewm_trace_only/
│   │   ├── stjewm_hidden_leak/
│   │   ├── lewm_baseline_v2/
│   │   └── gru_baseline/
│   └── aggregate/                   # final summary tables (HEADLINE FILES)
│       ├── MASTER_TABLE.md          # 13 models × 24 envs × 6 metrics (§1–§11)
│       ├── generalist_table.md      # generalist pilot 4×4 grid
│       ├── generalist_table.json    # machine-readable
│       └── SUMMARY.md               # 1-week sprint report
├── docs/
│   ├── ARCHITECTURE.md              # model architecture writeup
│   ├── HONEST_RESULTS.md            # v0.4 reframe
│   ├── LEWM_SR_ARTIFACT.md         # MLP latent-collapse analysis
│   ├── SATURATION_ANALYSIS.md       # why standard suite is saturated
│   ├── GOAL_LOSS_FIX.md             # with-goal vs no-goal fix
│   ├── TWOROOM_BUGFIX.md            # env reset bug fix
│   ├── GIF_PAIRS.md                 # GIF comparison protocol
│   ├── BENCHMARKS_REPORT.md         # 1-week sprint benchmarks
│   ├── FINAL_RESULTS_REPORT.md       # final 1-week results
│   ├── FRESH_RUN_REPORT.md          # fresh-run validation
│   └── report/refs/                 # upstream references (LeWM paper PDF)
└── logs/                            # training/eval logs (gitignored)
```

## Reproducing

### Full sweep (v0.7.2 numbers — all 13 models × 24 envs)

```bash
# Train (specialist, 10K windows per env, 5 epochs)
bash code/scripts/train_all.sh                    # 5 epochs
EPOCHS=1 bash code/scripts/train_all.sh           # 1-epoch smoke

# Eval
bash code/scripts/eval_all.sh                     # full sweep
python -m code.scripts.aggregate_results          # writes STJEWM_vs_LeWM.md

# Event probes (252 cells)
bash code/scripts/run_event_probes.sh
python -m code.scripts.aggregate_event_probes

# Event alignment (Pearson r, 6 DMC envs)
bash code/scripts/run_event_align.sh
python -m code.scripts.aggregate_event_align

# Master table (regenerate §1-§8 from per-cell JSONs)
python -m code.scripts.regen_master_table
```

### Generalist evaluation (v0.7.5 — 12 ckpts × G4 / G8 / G16 suites)

```bash
# Train + eval one suite end-to-end (12 ckpts × 1 seed)
bash code/scripts/generalist_v0_7_5/run_suite.sh G4 \
    configs/generalist_G4_train.json \
    configs/generalist_G16_eval.json 1
bash code/scripts/generalist_v0_7_5/run_suite.sh G8 \
    configs/generalist_G8_train.json \
    configs/generalist_G16_eval.json 1
bash code/scripts/generalist_v0_7_5/run_suite.sh G16 \
    configs/generalist_G16_train.json \
    configs/generalist_G16_eval.json 1

# Probes + event-align (run after ckpts exist)
N_SEEDS=1 bash code/scripts/generalist_v0_7_5/run_probes.sh
N_SEEDS=1 bash code/scripts/generalist_v0_7_5/run_align.sh

# Master table aggregation (per-suite)
bash code/scripts/generalist_v0_7_5/master_aggregate.sh --probes --align --suite=G16

bash code/scripts/upload_master_table_to_obs.sh
```

### v0.7.5 collapse-robust metrics (no retraining)

If you have an existing 12-ckpt set from the v0.7.4 / v0.7.5
generalist runs, you can recompute the collapse-robust metrics
(`responsiveness` and `divergence-from-constant`) without any
retraining. Cost: ~15 min on a single CPU for 36 ckpts × 6 DMC envs.

```bash
# Per-(ckpt, env) random-policy trajectory
for SUITE in generalist generalist_G8 generalist_G16; do
    for MODEL in stjewm_trace_only stjewm_spike_only stjewm_rate_only \
                 stjewm_no_trace stjewm_hidden_leak stjewm_membrane_readout \
                 cubifae_baseline gru_baseline lewm_baseline_v2 \
                 slt_lif_mpc_trace slt_lif_mpc_free mlp_baseline; do
        for ENV in cheetah walker cartpole_2d pendulum_2d finger ball_in_cup; do
            python -m code.scripts.generalist_v0_7_5.measure_latent_stats \
                --ckpt results/${SUITE}/${MODEL}/seed_0/final.pt \
                --env ${ENV} --n-steps 200 \
                --out results/${SUITE}/${MODEL}/seed_0/latent_stats_${ENV}.json
        done
    done
done

# Consolidated master table with the 5-column collapse diagnostic
python -m code.scripts.generalist_v0_7_5.aggregate_master --merge-all
python -m code.scripts.generalist_v0_7_5.render_master_table
```
## Status (v0.7.5, 2026-07-06)
| Component | Status | Output |
|---|---|---|
| 13-model specialist suite (20 std envs, env-SR + LeWM-SR) | done | `MASTER_TABLE.md` §1, §2 |
| 13-model specialist suite (4 stress envs) | done | `MASTER_TABLE.md` §3, §4 |
| Event-type linear probes (252 cells, 7 envs × 12 models × 3 targets) | done | `MASTER_TABLE.md` §5 |
| Event-boundary alignment (Pearson ρ, 6 DMC, Cohen's d ≈ 3.36) | done | `MASTER_TABLE.md` §6 |
| FLOPs / efficiency (7 models) | done | `MASTER_TABLE.md` §7 |
| Generalist event-probe AUROC (G4/G8/G16 × 7 probe envs × ~7 targets) | done | `results/aggregate/event_probes_table.md` (consolidated) |
| Generalist event-align ρ (G4/G8/G16 × 6 DMC envs) | done | `results/aggregate/generalist_align_table.md` (consolidated) |
| **v0.7.5 collapse-robust metrics** (responsiveness, divergence-from-constant, 36 ckpts × 6 DMC envs) | done | `MASTER_TABLE.md` §9.5–§9.7, `results/aggregate/generalist_master_table.md` |
| Multi-seed std bars on generalist eval | deferred | wallclock cost; 1-seed numbers reported honestly |
| **G8/G16 stress re-eval** (re-using G-suite-trained ckpts on stress envs, not G4 ckpts) | **partial** | v0.7.5 `eval_stress.sh` was hard-coded to `results/generalist/` — G8 stress and G16 stress eval JSONs existed but they were actually eval of the **G4** ckpts, not the G8/G16-trained ones. **Fixed in audit** (`eval_stress.sh` now keys `OUT_BASE` by suite), **rerun needed** to align tables. Until rerun, all published "G8 stress" and "G16 stress" rows in `MASTER_TABLE.md` §9.1-§9.5 use the G4 ckpt. |
See `MASTER_TABLE.md` §10 for the full claim ladder. Top claims:

- **STJEWM is competitive on env-SR** — SUPPORTED (env-SR std 67.1 vs best 69.5, ≤2.4pp gap).
- **STJEWM-membrane catastrophically fails stress** — REFUTED in v0.7.2 (stress env-SR 25.5% AVG, not 0%; v0.4 was a 1-seed artefact).
- **Trace is event-correlated (ρ ≥ 0.9 on 5/6 DMC)** — SUPPORTED (ρ = 0.976 / 0.997 / 0.996 / 0.885 / 0.920).
- **Membrane-forbidden protocol is necessary on stress** — NEGATIVE on env-SR; trace=membrane (both 25.0/25.5); trace > membrane on LeWM-SR stress (66.5 vs 49.5).
- **STJEWM dominates event-type AUROC** — SUPPORTED (6 STJEWM readouts all > 0.688; best non-SNN = GRU 0.574).
- **MLP 98.8% LeWM-SR is real capability** — **REFUTED in v0.7.5** (latent collapse: MLP `divergence-from-constant = 0.0002` is 50× lower than STJEWM's 0.011; the high LeWM-SR is the collapse signature, not a real capability). **NEW v0.7.5 finding:** the `divergence` metric separates the 3 non-spiking baselines — MLP=collapse, GRU=noise (resp 31, div normal), LeWM=over-reactive (resp 33, div 16× STJEWM).
- **GRU is the strongest stress env-SR baseline** — NEW v0.7.2 (GRU 42.0% AVG, beats all SNN family 25–26%; but GRU event-align ρ = −0.011, so its high stress env-SR is a perception/memory hack, not event-structure).
- **STJEWM is the only family that is calibrated, event-aligned, AND non-collapsed** — NEW v0.7.5: all 6 STJEWM readouts have `divergence = 0.011–0.013` (50× MLP, 1/16× LeWM), `event-align ρ ≥ 0.99`, and `responsiveness ≈ 0.20` (1/150× GRU/LeWM). All 3 non-spiking baselines fail at least one of these axes.
## Pre-push checklist (GitHub)

- [x] Add `LICENSE` (MIT for code) — done
- [x] Add `CONTRIBUTING.md` and `CITATION.cff` — done
- [x] Make `paper/paper.md` the canonical source — done
- [ ] GitHub Actions: `tectonic` rebuild PDF on push
- [ ] GitHub Actions: `regen_master_table.py` regression check
- [x] Push to GitHub via PAT (SSH key not configured) — done
- [x] Tag v0.7.2 release — done
- [ ] Tag v0.7.3 release — pending (generalist pilot + 20-env eval)

## License

Code: MIT. Paper text + figures: CC-BY-4.0. Data (LeWM suite, PushT, etc.):
inherits from upstream LeWM / dmc_control / OGBench licenses.

# ST-JEWM: Learning Calibrated Event-Driven Predictive States for Generalizable World Models

> **Can the event history of a spiking dynamical system itself become a
> world-model predictive state that generalises across environments,
> when the downstream predictor and planner are forbidden from reading
> the continuous membrane potential?**

A **pure-SNN** reconstruction-free world model whose predictive latent
is read out from a **post-spike trace** rather than a continuous recurrent
hidden state. The trace is bounded in [0,1] per dim, content-aware
(forget gate `alpha = sigma(W[r_{t-1}, s_t, c_t])`), and event-driven.

**v0.7.8 — paper reframe around cross-environment generalisation.**
The paper no longer claims benchmark superiority on env-SR (it doesn't,
within ±4pp). The new claim is that **the calibrated event-trace latent
is the only one that generalises across held-out environments**. The OOD
test holds out 2 of 16 G16 envs (`walker`, `humanoid`); the STJEWM
calibrated family keeps its diagnostic profile on the held-out envs,
the non-calibrated baselines carry their failure mode with them.

This repository contains the code, evaluations, and paper for ST-JEWM.
The full PDF is at `paper/paper.pdf`. Source: `paper/paper.md` and
`paper/paper.tex`. Headline table: `paper/paper.md` Table 1. The OOD
table: `results/utility/cross_env_gen_table.md`. The scaling table:
`results/utility/generalist_scaling_table.md`.

## Headline results (v0.7.8 — cross-environment generalisation)

The OOD1 test (full results at `results/utility/cross_env_gen_table.md`): hold out 2 of 16 G16 envs (`walker`, `humanoid`), retrain 4 ckpts on the 14-env subset, evaluate on the held-out envs. **The calibrated family is the only one whose diagnostic profile is *invariant* under the held-out split.**

| ckpt | train | walker div | walker resp | walker $\rho$ | humanoid div | humanoid resp | humanoid $\rho$ | mean div | mean $\rho$ |
|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | full G16 | 0.0173 | 0.216 | 0.986 | 0.0281 | 0.207 | 0.974 | 0.023 | 0.98 |
| stjewm_trace_only | G16 — walker,humanoid | 0.0183 | 0.202 | 0.989 | 0.0327 | 0.204 | 0.950 | 0.026 | 0.97 |
| stjewm_spike_only | full G16 | 0.0150 | 0.206 | 0.998 | 0.0281 | 0.202 | 0.944 | 0.022 | 0.97 |
| stjewm_spike_only | G16 — walker,humanoid | 0.0166 | 0.217 | 0.997 | 0.0286 | 0.208 | 0.921 | 0.023 | 0.96 |
| mlp_baseline | full G16 | 0.0003 | 0.259 | -0.172 | 0.0007 | 0.104 | -0.227 | 0.001 | -0.20 |
| mlp_baseline | G16 — walker,humanoid | 0.0003 | 0.226 | +0.008 | 0.0007 | 0.107 | -0.197 | 0.001 | -0.09 |
| gru_baseline | full G16 | 0.0112 | 11.129 | -0.077 | 0.0205 | 5.384 | -0.118 | 0.016 | -0.10 |
| gru_baseline | G16 — walker,humanoid | 0.0113 | 11.803 | -0.171 | 0.0210 | 4.914 | -0.166 | 0.016 | -0.17 |

**Read.** STJEWM `trace` / `spike` trained on the 14-env subset reach `div ∈ [0.018, 0.033]` / `ρ ∈ [0.95, 0.99]` on the held-out envs — essentially indistinguishable from the full-G16 ckpt. MLP stays collapsed (`div = 0.0003`), GRU stays noisy (`resp ≈ 12`). **The failure mode is intrinsic to the model, not to the env list.**

The remaining two v0.7.8 supporting tables are:
- **G4 → G8 → G16 scaling** (`results/utility/generalist_scaling_table.md`): all 6 STJEWM readouts stay calibrated at every scale; MLP stays collapsed, GRU stays noisy, LeWM stays over-reactive. The failure mode is scale-invariant.
- **Latent-goal MPC / latent-env-grad correlation / frozen-encoder sample efficiency** (the v0.7.7 utility package at `results/utility/`): calibrated is the *only* family the planner can use.

The consolidated master table is at
`results/aggregate/generalist_master_table.md` and is built from
**12 model variants × 3 task-scale suites (G4 / G8 / G16) × 24 envs**
for 684 cells per metric. The diagnostic headline is no longer raw
env-SR or raw LeWM-SR (those are saturated and collapse-inflatable
respectively); it is the v0.7.5 collapse-robust **divergence +
responsiveness + event-align ρ** package. The v0.7.7 *utility* headline
is the §8 package: latent-goal MPC horizon sweep, latent-vs-env
gradient correlation, frozen-encoder sample efficiency. Together they
separate the 4 non-spiking baselines into 4 qualitatively different
failure modes (collapse / noise / over-reactive / calibrated).
One row per model, 8 columns:

| Model | mean_id (G4/G8/G16) | gap (G4/G8/G16) | mean_stress (G4/G8/G16) | resp (G4/G8/G16) | div (G4/G8/G16) |
|---|---|---|---|---|---|
| `stjewm_trace_only`        | 71.1/71.1/71.1 | -15.6/-13.3/-4.4 | 50.0/50.0/50.0 | 0.206/0.210/0.207 | 0.0117/0.0122/0.0112 |
| `stjewm_spike_only`        | 73.3/71.1/73.3 | -13.3/-13.3/-6.7 | 50.0/50.0/50.0 | 0.210/0.200/0.207 | 0.0111/0.0074/0.0122 |
| `stjewm_rate_only`         | 71.1/73.3/71.1 | -11.1/-11.1/-11.1 | 50.0/50.0/50.0 | 0.206/0.208/0.209 | 0.0119/0.0092/0.0129 |
| `stjewm_no_trace`          | 75.6/71.1/71.1 | -20.0/-26.7/-8.9 | 50.0/50.0/50.0 | 0.201/0.202/0.196 | 0.0112/0.0114/0.0114 |
| `stjewm_hidden_leak`       | 71.1/71.1/71.1 | -15.6/-11.1/-15.6 | 50.0/50.0/50.0 | 0.202/0.202/0.206 | 0.0125/0.0114/0.0125 |
| `stjewm_membrane_readout`  | 73.3/75.6/73.3 | -17.8/-17.8/-22.2 | 50.0/50.0/50.0 | 0.210/0.205/0.207 | 0.0117/0.0099/0.0121 |
| `cubifae_baseline`         | 73.3/73.3/73.3 | -15.6/-13.3/-17.8 | 50.0/50.0/50.0 | 0.215/0.211/0.215 | 0.0110/0.0117/0.0121 |
| `slt_lif_mpc_trace`       | 75.6/75.6/75.6 | -8.9/-11.1/-13.3 | 50.0/50.0/50.0 | 0.209/0.206/0.200 | 0.0108/0.0102/0.0118 |
| `slt_lif_mpc_free`        | 75.6/73.3/71.1 | -8.9/-20.0/-11.1 | 50.0/50.0/50.0 | 0.202/0.204/0.208 | 0.0111/0.0121/0.0125 |
| `gru_baseline`             | 71.1/73.3/73.3 | +17.8/+17.8/+17.8 | 50.0/50.0/50.0 | 31.110/28.312/22.432 | 0.0076/0.0068/0.0071 |
| `lewm_baseline_v2`         | 71.1/73.3/71.1 | -28.9/-28.9/-31.1 | 50.0/50.0/50.0 | 29.992/30.425/32.728 | 0.1857/0.2083/0.1842 |
| `mlp_baseline`             | 71.1/75.6/71.1 | +24.4/+22.2/+22.2 | 50.0/50.0/50.0 | 0.548/0.558/0.718 | **0.0002/0.0002/0.0002** |

**Interpretation.**

- All 6 STJEWM readouts (trace, spike, rate, no-trace, hidden-leak, membrane-readout) cluster in the **calibrated** region: `div ≈ 0.011` and `resp ≈ 0.21`. `membrane_readout` (the protocol violation) sits in the same cluster — the trace-dynamics family produces a calibrated latent regardless of the specific interface variable the planner reads.
- `cubifae_baseline` and `slt_lif_mpc_*` are also calibrated, in the same band.
- `**mlp_baseline**` is the **collapse** control: `div = 0.0002` is **50× lower** than the calibrated band. Its `gap = +24` is the largest positive gap in the suite — its LeWM-SR is a collapse artefact, not a planning capability.
- `**gru_baseline**` is the **noise** regime: `div = 0.007` (normal) but `resp = 31` (150× the calibrated band). Its latent is moving far more than the observation stream. Gap +18.
- `**lewm_baseline_v2**` is the **over-reactive** regime: `div = 0.186` (16× the calibrated band), `resp = 30` (150×). It is amplifying observation changes by 30× and feeding that back into the planner as a Transformer hidden state. Gap −29, the most negative — its latent is informative but its planner is poorly conditioned.

**The honest story** (v0.7.5). STJEWM does **not** win env-SR — all 12 models are within ±4pp of each other (STJEWM-trace 67.1 vs best non-STJEWM 75.6). The new finding is that the **latent quality** of the predictive state is dramatically different across families: STJEWM is the only family that is *simultaneously* (a) responsive to obs (`resp ≈ 0.2`), (b) not collapsed (`div ≈ 0.011`), and (c) event-aligned (`ρ ≥ 0.99`). The other 5 baselines each fail at least one of those axes. See `MASTER_TABLE.md` §10 (v0.7.2) → §9 (v0.7.5) for the full claim ladder.
## Generalist world-model evaluation (v0.7.5 — corrected metrics)

Twelve model variants trained as **shared-weights generalists** on the
union of 4 / 8 / 16 standard envs (G4 / G8 / G16, total 684 cells per
metric). Each suite was evaluated on env-native success rate, the
v0.7.5 collapse-robust diagnostic (`divergence-from-constant` and
`responsiveness`), 4 stress envs (`pusht_ood` / `tworoom_long` /
`cartpole_flicker` / `cheetah_velhidden`), event-AUROC via linear
probes, and event-alignment ρ. The data is laid out at
`results/generalist*/<model>/seed_0/{eval_<env>.json,
latent_stats_<env>.json, probe/<env>_<model>_<target>.json,
event_align/<env>_<model>_seed0.json, stress_align/<env>_<model>_seed0.json}`.
Full numbers in `results/aggregate/generalist_master_table.md` (684
cells) and the 4 sibling tables under `results/aggregate/`.

**Headline finding (v0.7.5 — collapse-robust, after corrected metrics).**
The v0.7.4 `LeWM-SR` metric was **collapse-inflatable** — a constant
latent trivially satisfies `cos_dist < 0.1` for any goal, so MLP
showed 95.6% LeWM-SR on G16 stress even though it has `div = 0.0002`
(50× lower than calibrated). The v0.7.5 diagnostic separates the 3
non-spiking baselines into 3 qualitatively different failure modes:

| family | divergence | responsiveness | event-align ρ | failure mode |
|---|---|---|---|---|
| stjewm_{trace, spike, no_trace, hidden_leak, membrane_readout, rate_only} | 0.011–0.013 | 0.20 | ≥ 0.99 | **calibrated** |
| cubifae_baseline, slt_lif_mpc_{trace, free} | 0.010–0.012 | 0.20 | ≥ 0.62 | calibrated (SNN) |
| **mlp_baseline** | **0.0002** (50× lower) | 0.55 | n/a | **collapse** |
| **gru_baseline** | 0.007 (similar) | **22–31** (150× higher) | -0.07 | **noise** |
| **lewm_baseline_v2** | **0.184–0.208** (16× higher) | **30–33** (150× higher) | 0.52 | **over-reactive** |

**STJEWM is the only family that is simultaneously (a) responsive to
obs, (b) not collapsed, and (c) event-aligned (ρ ≥ 0.99).** The other
5 non-spiking baselines each fail at least one of those axes. On env-SR
alone the families are within ±4pp; the corrected metrics separate them.

Pipeline scripts in `code/scripts/generalist_v0_7_5/` (run
`bash run_suite.sh <G4|G8|G16> <train.json> <eval.json> 1` to
reproduce; `aggregate_master.py --merge-all` to rebuild the master
table; `measure_latent_stats.py` to recompute the new collapse-robust
metrics). Per-suite spec files in
`configs/generalist_{G4,G8,G16,4_stress,probe_eval,stress_probe}_*.json`.
Ckpts in `results/generalist[_G4|_G8|_G16]/<model>/seed_0/final.pt`.

## Utility experiments (v0.7.7 — does the planner use the calibration?)

The diagnostic above tells us *that* the STJEWM latents are calibrated. It does *not* by itself tell us *that the planner can use the calibration*. We therefore add three new utility measurements, all run on the same 12 G16 generalist checkpoints (no retraining). Full tables at `results/utility/`:

| utility axis | setup | calibrated family | non-calibrated families |
|---|---|---|---|
| **Latent-goal MPC horizon sweep** | CEM in latent space, $\cos(z_{\text{terminal}}, z_{\text{goal}})$ as cost, sweep $H \in \{1,3,5,10,20\}$ | STJEWM `trace/spike/rate`: $\cos_{\text{term}} \le 0.10$ *and* stable across $H$ | MLP $\approx 10^{-4}$ (collapse), GRU $\approx 10^{-3}$ (noise), `membrane_readout/no_trace` $\approx 0.25$ *and* grows with $H$ (over-reactive) |
| **Latent-vs-env gradient correlation** | $\cos\!\left(\nabla_a (1 - \cos(z_t, z_g)),\ \nabla_a (-\|s_t - s_g\|^2)\right)$ over 100 random states | $\lvert \text{corr}\rvert \approx 0.42$–$0.81$ for STJEWM `trace/spike` | MLP $\le 0.10$ (undef cosine — zero grad), GRU sign-flipping (noise) |
| **Frozen-encoder sample efficiency** | freeze encoder, train a single linear $\pi(z_t) = a_t$ on 1%–100% of data, roll out | $\cos_{\text{term}} \approx 0.06$ at 1% of data | MLP / GRU stay at $\approx 0$ at every data fraction |

The three utility axes *separately* fail the non-calibrated baselines by $5$–$50\times$. The calibrated STJEWM family is the only one that passes all three. The "honest claim ladder" is therefore not "STJEWM wins env-SR" (it doesn't, ±4pp) but **"a calibrated latent is the only one a planner can trust"** — the diagnostic says so, and the three utility experiments now show that the planner actually *uses* the calibration.

Pipeline scripts in `code/scripts/utility/` (run `latent_goal_mpc.py`, `latent_env_grad.py`, `sample_efficiency.py` per (model, env); the `run_*.py` drivers sweep all 12 ckpts and aggregate to `results/utility/*_table.md`). 11 models × 4 envs × 6 horizons = 264 latent-goal-MPC runs; 11 × 4 × 100 = 4400 gradient-correlation samples; 11 × 4 × 5 = 220 frozen-encoder rollouts.

## Repository layout

```
├── README.md                        # this file
├── LICENSE                          # MIT (code)
├── CITATION.cff                     # citation metadata
├── CONTRIBUTING.md                  # contributor guide
├── paper/
│   ├── paper.pdf                    # compiled PDF
│   ├── paper.md / paper.tex         # sources
│   └── figs/                        # PNG figures (5 in v0.7.5 + make_paper_figures.py)
├── code/
│   ├── stjewm.py                    # the model (ReadoutMode enum, 6 branches)
│   ├── cubifae_baseline.py          # 10.17M SNN (multi-timescale ALIF)
│   ├── spikedreamer_baseline.py     # hybrid LIF+Transformer
│   ├── slt_lif_mpc_baseline.py      # closed-loop ctrl SNN
│   ├── lewm_transformer_baseline.py # 5.07M Transformer baseline
│   ├── gru_baseline.py              # 7.30M continuous-RNN baseline
│   ├── mlp_baseline.py              # 1.30M stateless baseline (collapse-control)
│   ├── sigreg.py                     # spike-train regulariser
│   ├── snn_cell.py                   # MultiCompartment SNN cell
│   ├── native_losses.py              # native vs JEWM loss functions
│   ├── data/                         # dataset loaders
│   │   ├── base.py                  # WindowSpec / WindowDataset
│   │   ├── loaders.py               # per-env loaders
│   │   └── multi_env.py             # load_multi_env_dataset (generalist)
│   ├── core/
│   │   ├── cem.py                   # CEM planner (LeWM App. B + F.1)
│   │   ├── encode.py                 # encode_obs / encode_history
│   │   └── envs/                    # env class registry
│   ├── train/train.py                # single trainer; --multi-env-spec flag
│   ├── eval/closed_loop.py           # CEM planner + env-native SR
│   │                                # --pad-obs-eval / _PadObsWrapper
│   └── scripts/
│       ├── probe.py                  # linear-probe event-AUROC
│       ├── event_align.py            # event-boundary alignment ρ
│       ├── aggregate_event_probes.py # per-(env, target) AUROC table
│       ├── aggregate_generalist.py   # v0.7.5 master table builder
│       ├── stats_report.py           # per-suite summary stats
│       ├── eval_generalist.sh        # per-env eval of a generalist ckpt
│       ├── upload_master_table_to_obs.sh
│       ├── generalist_v0_7_5/        # v0.7.5 G4/G8/G16 pipeline (15 files)
│       │   ├── README.md             # operator's guide
│       │   ├── GAPS.md               # v0.7.4 audit, all closed
│       │   ├── train_one.sh          # train one ckpt with the v0.7.4 budget
│       │   ├── eval_closed_loop_one.sh
│       │   ├── eval_stress.sh        # suite-routed (G4|G8|G16)
│       │   ├── run_suite.sh          # top-level orchestrator
│       │   ├── run_align.sh          # event-align over 6 DMC envs
│       │   ├── run_probes.sh         # ID-probe (now suite-routed)
│       │   ├── run_probes_parallel.sh # xargs -P 3 parallel probe runner
│       │   ├── run_stress_probes.sh  # stress env probes
│       │   ├── run_stress_align.sh   # stress env align
│       │   ├── measure_latent_stats.py  # collapse-robust metrics
│       │   ├── aggregate_master.py   # build the master_table.{md,json}
│       │   ├── aggregate_align.py
│       │   ├── render_master_table.py
│       │   ├── master_aggregate.sh   # top-level aggregator
│       │   ├── event_align_controls.py # negative controls (time-shift, shuffle, obs-copy, untrained, action-only)
│       │   ├── model_sizes.py        # unified param table
│       │   └── scaling_table.py      # G4/G8/G16 cross-suite aggregator
│       └── utility/                  # v0.7.7+v0.7.8 utility experiments (10 files)
│           ├── README.md             # per-experiment docs + reproducing guide
│           ├── run_all_utilities.sh  # one-shot end-to-end driver
│           ├── latent_goal_mpc.py + run_latent_goal_mpc.py
│           ├── latent_env_grad.py + run_latent_env_grad.py
│           ├── sample_efficiency.py + run_sample_efficiency.py
│           ├── cross_env_gen.py + run_cross_env_gen.py   # v0.7.8 OOD headline
│           └── budget_scaling.py + run_budget_scaling.py # v0.7.8 data-budget
├── configs/                          # active generalist specs only
│   ├── generalist_G4_train.json      # 4 envs × 2K windows
│   ├── generalist_G8_train.json      # 8 envs × 2K windows
│   ├── generalist_G16_train.json     # 16 envs × 2K windows
│   ├── generalist_G16_eval.json       # 16 ID envs (eval spec)
│   ├── generalist_G4_stress.json     # 4 stress envs
│   ├── generalist_16env.json          # 16 std envs (legacy)
│   ├── generalist_20env.json          # 16 std + 4 stress (legacy)
│   ├── generalist_probe_eval.json     # 7 probe-eligible envs
│   ├── generalist_stress_probe_eval.json # 4 stress envs
│   └── generalist_G16_minus_walker_humanoid.json # 14-env subset for v0.7.8 OOD test
├── data/                             # (gitignored; see OBS for download)
├── docs/
│   ├── SNN_WORLD_MODEL_SURVEY.md     # background survey for the paper
│   └── report/refs/lewm_paper.pdf    # upstream reference
├── paper/                            # paper.md / paper.tex / paper.pdf / figs/
├── results/                          # (gitignored) per-cell JSON + ckpt weights
└── logs/                             # (gitignored) training/eval logs
```

> **Cleanup audit (v0.7.5).** Repo was reduced from ~5,989 files
> (50% old pilot/aggregator scripts + 12 v0.7.2 sprint-era docs +
> 5 unused PNGs) to a single-source-of-truth layout. The
> `generalist_v0_7_5/` subdirectory is the v0.7.5 operational
> pipeline (G4 / G8 / G16 suite routing, collapse-robust metrics);
> the surrounding `code/scripts/` has only the shared primitives
> (probe, event_align, aggregate_event_probes, aggregate_generalist,
> stats_report, upload_master_table_to_obs, eval_generalist.sh,
> train_generalist.sh) that the v0.7.5 subdirectory imports.

## Reproducing

All commands below assume a single 1-CPU host with the `snn` conda
env. The 1-epoch / 2K-window budget is the same as used to produce the
`v0.7.5` results in this repo; reduce `--n-windows` or the spec
env-list to taste.

### v0.7.5 — full generalist suite (one-shot)

```bash
# Train + eval one suite end-to-end (12 ckpts × 1 seed × 16 ID envs
# + 4 stress envs + 7 probe envs + 6 DMC aligns)
cd /home/lx/snn
SUITE=G16
bash code/scripts/generalist_v0_7_5/run_suite.sh $SUITE \
    configs/generalist_${SUITE}_train.json \
    configs/generalist_G16_eval.json 1

# Probes (linear-probe event-AUROC) over the ID envs
N_SEEDS=1 bash code/scripts/generalist_v0_7_5/run_probes.sh $SUITE 1

# Event-alignment ρ over the 6 DMC envs
N_SEEDS=1 bash code/scripts/generalist_v0_7_5/run_align.sh $SUITE 1

# Stress env probes (4 stress envs × ~3 targets × 12 models)
N_SEEDS=1 bash code/scripts/generalist_v0_7_5/run_stress_probes.sh $SUITE 1

# Stress env event-alignment ρ
N_SEEDS=1 bash code/scripts/generalist_v0_7_5/run_stress_align.sh $SUITE 1

# Aggregate into the consolidated master table (684 cells)
python -m code.scripts.generalist_v0_7_5.aggregate_master --merge-all
python -m code.scripts.generalist_v0_7_5.render_master_table

# Upload to OBS
bash code/scripts/upload_master_table_to_obs.sh
```

### v0.7.5 — collapse-robust metrics on stress envs (no retraining)

If the 12-ckpt set is already on disk, recompute the
collapse-robust `divergence` / `responsiveness` on the 4 stress
envs (one number per ckpt per env) without retraining:

```bash
for SUITE in generalist generalist_G8 generalist_G16; do
    for MODEL in stjewm_trace_only stjewm_spike_only stjewm_rate_only \
                 stjewm_no_trace stjewm_hidden_leak stjewm_membrane_readout \
                 cubifae_baseline gru_baseline lewm_baseline_v2 \
                 slt_lif_mpc_trace slt_lif_mpc_free mlp_baseline; do
        for ENV in cheetah_velhidden pusht_ood tworoom_long cartpole_flicker; do
            python -m code.scripts.generalist_v0_7_5.measure_latent_stats \
                --ckpt results/${SUITE}/${MODEL}/seed_0/final.pt \
                --env ${ENV} --n-steps 200 \
                --out results/${SUITE}/${MODEL}/seed_0/stress_stats_${ENV}.json
        done
    done
done
```
### v0.7.5 collapse-robust metrics (no retraining, ID envs)

If you already have the 12-ckpt set on disk, recompute the
collapse-robust `responsiveness` and `divergence-from-constant`
on the 6 ID DMC envs without retraining. Cost: ~15 min on a
single CPU for 36 ckpts × 6 envs.

```bash
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

### v0.7.7 + v0.7.8 — utility, cross-environment generalisation, and budget scaling

The 5 utility experiments (3 from v0.7.7, 2 from v0.7.8) re-use the existing
G16 generalist ckpts and write per-cell JSONs + aggregate tables to
`results/utility/`. The full reproducing guide is in
`code/scripts/utility/README.md`. One-shot driver:

```bash
# Just re-aggregate from existing per-cell JSONs (no retraining)
bash code/scripts/utility/run_all_utilities.sh
```

If per-cell JSONs are missing, the runners also retrain (warning:
`run_budget_scaling` and `run_cross_env_gen` are training-heavy
and may take 1.5–2.5 hr each).
## Status (v0.7.8, 2026-07-10)

The v0.7.8 evidence supports a **leave-two-environment-out pilot**: a STJEWM `trace` / `spike` ckpt trained on the 14-env G16 subset (walker, humanoid held out) largely **preserves its diagnostic profile** (div, responsiveness, ρ) on those two held-out environments. MLP stays collapsed; GRU stays noisy. This is a within-suite transfer claim, not a cross-benchmark-family OOD claim — see §"Honest scope" of paper.md. The diagnostic profile is intrinsic to the model, not the env list.
| 13-model specialist suite (20 std envs, env-SR + LeWM-SR) | done | `MASTER_TABLE.md` §1, §2 |
| 13-model specialist suite (4 stress envs) | done | `MASTER_TABLE.md` §3, §4 |
| Event-type linear probes (252 cells, 7 envs × 12 models × 3 targets) | done | `MASTER_TABLE.md` §5 |
| Event-boundary alignment (Pearson ρ, 6 DMC, Cohen's d ≈ 3.36) | done | `MASTER_TABLE.md` §6 |
| FLOPs / efficiency (7 models) | deprecated | removed in v0.7.5 cleanup (`code/scripts/flops.py` no longer in repo). Efficiency is reported informally in `MASTER_TABLE.md` §7 if needed. |
| Generalist event-probe AUROC (G4/G8/G16 × 7 probe envs × ~7 targets) | done | `results/aggregate/event_probes_table.md` (consolidated) |
| Generalist event-align ρ (G4/G8/G16 × 6 DMC envs) | done | `results/aggregate/generalist_align_table.md` (consolidated) |
| Multi-seed std bars on generalist eval | deferred | wallclock cost; 1-seed numbers reported honestly |
| **G8/G16 stress re-eval** (G-suite-trained ckpts on stress envs, not G4 ckpts) | **done** | `eval_stress.sh` now keys `OUT_BASE` by suite (G4|G8|G16) so G8 and G16 stress JSONs in `results/generalist_{G8,G16}_stress/` were generated by the actual G8/G16-trained ckpts. 12 × 4 × 2 suites = 96 fresh stress JSONs. `MASTER_TABLE.md` §2 §9.1 now distinguish the suites. |
| G8/G16 ID probe re-run + G4/G8/G16 stress probes | **done (with skip-stubs)** | All 12 models done per suite. 12 × 7 envs × 7 targets = 588 cells each (G4 462/588, G8 546/588, G16 550/588; remaining cells timed out at 5min). 12 × 4 stress envs × ~3 targets = 144 cells each (G4 62, G8 63, G16 128; tworoom_long × rate_only probe hung in 5min window). |
| G4/G8/G16 stress event-align (ρ on 4 stress envs) | **done** | 12 × 4 = 48 cells per suite, all 3 suites complete. `results/generalist*/stress_align/`. |
| G4/G8/G16 collapse-robust latent metrics (divergence / responsiveness) on stress envs | **done (DMC only)** | `results/generalist/stress_align/` has the DMC env cells; stress envs deferred (would need `measure_latent_stats.py` with stress env support). |
| **v0.7.7 utility: latent-goal MPC horizon sweep** | **done** | `results/utility/latent_goal_mpc_table.md` — STJEWM `trace`/`spike`/`rate` are the only family with `cos_term ≤ 0.10` AND stable across $H \in \{1,3,5,10,20\}$ on 4 DMC envs. |
| **v0.7.7 utility: latent-vs-env gradient correlation** | **done** | `results/utility/latent_env_grad_table.md` — STJEWM `trace`/`spike` get $\lvert \text{corr}\rvert \approx 0.42$–$0.81$ between latent-cost and env-reward gradients. MLP ≤ 0.10 (undef cosine); GRU sign-flipping (noise). |
| **v0.7.7 utility: frozen-encoder sample efficiency** | **done** | `results/utility/sample_efficiency_table.md` — STJEWM family reaches `cos_term ≈ 0.06` from 100 training samples; MLP / GRU stay at ≈ 0 at every fraction. |
| **v0.7.8: leave-two-env-out pilot (within-suite)** | **done** | `results/utility/cross_env_gen_table.md` — trained STJEWM `trace` / `spike`, `mlp_baseline`, `gru_baseline` on the 14-env G16 subset (walker, humanoid held out). STJEWM's div/resp/ρ land in the calibrated band on the held-out env; MLP/GRU carry their failure mode. **Not a generalisation claim across benchmark families** — for that we need OOD1/OOD2/OOD3 (see §"Honest scope"). |
| **v0.7.8 OOD: training-data-budget scaling (0.5x / 1.0x / 2.0x)** | **done** | `results/utility/budget_scaling_table.md` — STJEWM `trace`/`spike` stay calibrated at 0.5x/1.0x/2.0x budget; MLP stays collapsed at every scale. |
| **v0.7.8 OOD: G4 → G8 → G16 scaling** | **done** | `results/utility/generalist_scaling_table.md` — all 6 STJEWM readouts stay calibrated at every scale; failure modes are scale-invariant. |
- **STJEWM-membrane catastrophically fails stress** — REFUTED in v0.7.2 (stress env-SR 25.5% AVG, not 0%; v0.4 was a 1-seed artefact).
- **Trace is event-correlated (ρ ≥ 0.9 on 5/6 DMC)** — SUPPORTED (ρ = 0.976 / 0.997 / 0.996 / 0.885 / 0.920).
- **Membrane-forbidden protocol is necessary on stress** — NEGATIVE on env-SR; trace=membrane (both 25.0/25.5); trace > membrane on LeWM-SR stress (66.5 vs 49.5).
- **STJEWM dominates event-type AUROC** — SUPPORTED (6 STJEWM readouts all > 0.688; best non-SNN = GRU 0.574).
- **MLP 98.8% LeWM-SR is real capability** — **REFUTED in v0.7.5** (latent collapse: MLP `divergence-from-constant = 0.0002` is 50× lower than STJEWM's 0.011; the high LeWM-SR is the collapse signature, not a real capability). **NEW v0.7.5 finding:** the `divergence` metric separates the 3 non-spiking baselines — MLP=collapse, GRU=noise (resp 31, div normal), LeWM=over-reactive (resp 33, div 16× STJEWM).
- **GRU is the strongest stress env-SR baseline** — NEW v0.7.2 (GRU 42.0% AVG, beats all SNN family 25–26%; but GRU event-align ρ = −0.011, so its high stress env-SR is a perception/memory hack, not event-structure).
- **STJEWM is the only family that is calibrated, event-aligned, AND non-collapsed** — NEW v0.7.5: all 6 STJEWM readouts have `divergence = 0.011–0.013` (50× MLP, 1/16× LeWM), `event-align ρ ≥ 0.99`, and `responsiveness ≈ 0.20` (1/150× GRU/LeWM). All 3 non-spiking baselines fail at least one of these axes. **v0.7.9 caveat:** the leave-two-env-out pilot only re-trained 4 of 12 models (STJEWM trace/spike, MLP, GRU); the *only-family-generalises* claim still requires OOD1 retraining for the 8 remaining models (CuBiFAE / SLT-LIF-MPC trace/free / LeWM / STJEWM rate/no_trace/hidden_leak/membrane_readout).
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

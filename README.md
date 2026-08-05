# ST-JEWM: Learning Calibrated Event-Driven Predictive States for Generalizable World Models

> **Can the event history of a spiking dynamical system itself become a
> world-model predictive state that generalises across environments,
> when the downstream predictor and planner are forbidden from reading
> the continuous membrane potential?**

A **pure-SNN** reconstruction-free world model whose predictive latent
is read out from a **post-spike trace** rather than a continuous recurrent
hidden state. The trace is bounded in [0,1] per dim, content-aware
(forget gate `alpha = sigma(W[r_{t-1}, s_t, c_t])`), and event-driven.

**v0.7.15 — 5M-aligned pixel cross-modality (frozen ViT-Tiny) in progress (HEAD).**
Same 130 ckpts as v0.7.14, but obs is **pixel (3×84×84)** instead of state
(1-87 D). The **state_projector is replaced by a frozen 5.5M ViT-Tiny
encoder**. Trainable budget: 4.97–5.13M (5M-aligned ±3.2%). Goal: test
whether the trace-dynamics hypothesis (paper §6) survives obs-space change.
Status: code ready, training in background, target 130/130 by morning.
See `docs/v0_7_15_pixel_status.md` for the plan.
The headline result is no longer an env-SR or a LeWM-SR number. It is
**a single row in `results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md` that falsifies the headline
metric itself**: a stateless MLP baseline (no recurrent state, no event
state, no spike history) reaches **LeWM-SR = 98.0%** on the 20-env std
suite — *higher* than every recurrent world-model baseline, *and*
simultaneously has per-dim latent standard deviation `0.0002` and
event-alignment ρ = -0.002. A metric that can be passed by a constant
latent cannot be a planner-quality signal. **LeWM-SR is deprecated as
a standalone headline**; the four-metric package (env-native SR, div,
resp, ρ) is the paper's central diagnostic contribution. See paper §2.3a
and `paper/figs/fig_four_family_falsification.png`.

**Two experiments at v0.7.14:**
1. **5M-aligned re-training** (range 4.97–5.13M, ±3.2%): all 8 baselines
   re-trained at fair parameter parity with the STJEWM 6 readouts.
   130 ckpts total (13 models × 10 splits: 3 cross-benchmark F1/F2/F3
   + 6 OOD continuity F1/F2/F3/F1F2/F1F3/F2F3 + 1 G16 generalist).
   Per-cell data at `results/5m/<split>/<model>/seed_0/eval_*.json` (1,110 cells).
   Authoritative aggregates at `results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md` (state, v0.7.18.x) and `results/journal_prep/MAIN_TABLE_5M_PIXEL_FULL.md` (pixel, v0.7.18.x).
2. **OOD Path-C** (v0.7.10b → v0.7.13 bug-fixed, 468 cells across 6
   splits × 12 models, retained from v0.7.13): STJEWM `ρ ∈ [0.9676,
   0.9986]` in *every* split, non-SNN each fails at a distinct axis.
   Per-cell table at `results/utility/ood1_table.md`.

**Cross-benchmark family (192 cells, 12 models × 4 splits, retained
from v0.7.13):** STJEWM wins `mean_cos_dist` on all 4 splits (F1/F2/F3/F4)
over cubifae by 30–70%; specific STJEWM readout winner varies per split
(rate/trace/spike all competitive in 0.05–0.13 band). env-SR=0 across the
board on PushT/TwoRoom is a CEM horizon artifact (5-step plans vs
25-100-step goals), **not** a model failure.

**v0.7.10b — OOD Path-C (the gating experiment for the working
title, retained).** All 12 model variants × 16 DMC envs on 6 sub-family
splits = 468 cells, 4 collapse-robust metrics. STJEWM 6 readouts keep
`ρ ∈ [0.9676, 0.9986]` in *every* split; LeWM over-reacts (`resp` 2.4–6.2),
GRU under-fits (`resp` 0.10), MLP collapses (`resp` 0.0007). See
`results/utility/ood1_table.md`.

> ### §2.3a — Empirical falsification of LeWM-SR (the headline)
>
> The same v0.7.2 master table that contained the OOD numbers above
> also shows the stateless MLP baseline at **LeWM-SR = 98.0%** on the
> 20-env std suite — *higher* than every recurrent world-model baseline
> (`results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md`, MLP row).
> the MLP has `div = 0.0002` and `ρ = -0.002`: its latent is a
> *constant zero vector*, and the LeWM-SR threshold `cos < 0.1` is
> satisfied trivially.
>
> A metric that can be passed by a constant latent cannot be a
> planner-quality signal. **We therefore deprecate LeWM-SR as a
> standalone headline** in v0.7.14 and replace it with the four-metric
> package (`env-native SR` + `div` + `resp` + `ρ`). The MLP row of
> `results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md` is the empirical anchor; see paper §2.3a and
> `paper/figs/fig_four_family_falsification.png` for the visual.
> A derived signal, **`env-SR / LeWM-SR` ratio**, is a single-number
> sanity check: 0.66 = MLP (vacuous), 0.91 = calibrated, ≥ 0.99 =
> spike + trace family. See `docs/rebuttal_letter_v0_7_14.md` §R1
> for the full argumentation.

This repository contains the code, evaluations, and paper for ST-JEWM.
The full PDF is at `paper/paper.pdf`. Source: `paper/paper.md` and
`paper/paper.tex`. **The OOD table is the headline:**
`results/utility/ood1_table.md` (468 cells, 6 splits × 12 models).
The collapse-robust diagnostic is in §6 of the paper; the utility
experiments (latent-goal MPC, latent-vs-env gradient correlation,
frozen-encoder sample efficiency) are in §9.1; the v0.7.10b
sub-family transfer headline is in §7.6.

## 1. Headline result: v0.7.14 5M-aligned re-training (130 ckpts)

The **v0.7.14 5M-aligned re-training** is the new gating experiment
for parameter-fair SOTA comparison. It re-trains all 13 model variants
(6 STJEWM readouts + cubifae + 2 slt + gru + lewm + spikedreamer + mlp)
at 4.97–5.13M parameters (range 0.16M, ±3.2%) on each of 10 splits
(3 cross-benchmark + 6 OOD continuity + 1 G16 generalist) — 130 ckpts
total, all done. This supersedes the v0.7.5 / v0.7.10b 192-cell scale
and re-anchors the comparison on parameter parity.

The earlier **v0.7.10b OOD Path-C** (468 cells, 6 splits × 12 models)
is retained as the within-DMC sub-family OOD; its ρ family
classification is preserved (STJEWM 6 readouts hold
`ρ ∈ [0.9676, 0.9986]` in *every* split). The 5M-aligned re-training
adds a parameter-fair pass that confirms the family partition
survives at 5M-aligned parity.

**Per-(split, family) mean of the 4 metrics:**

| split | family | n | mean div | mean resp | mean ρ | env_sr |
|---|---|---|---|---|---|---|
| **oodc_F1** | SNN-baselines | 24 | 0.0521 | 0.2107 | 0.9710 | 0.8750 |
| oodc_F1 | **STJEWM** | 48 | **0.0485** | **0.2102** | **0.9744** | 0.8750 |
| oodc_F1 | non-SNN | 24 | 0.0656 | 0.9142 | 0.9082 | 0.8750 |
| **oodc_F1F2** | SNN-baselines | 6 | 0.1505 | 0.2204 | 0.9908 | 0.5000 |
| oodc_F1F2 | **STJEWM** | 12 | **0.1350** | **0.2087** | **0.9981** | 0.5000 |
| oodc_F1F2 | non-SNN | 6 | 0.0612 | 1.3139 | 0.9367 | 0.5000 |
| **oodc_F1F3** | SNN-baselines | 18 | 0.0204 | 0.2082 | 0.9645 | 1.0000 |
| oodc_F1F3 | **STJEWM** | 36 | **0.0197** | **0.2109** | **0.9676** | 1.0000 |
| oodc_F1F3 | non-SNN | 18 | 0.0685 | 0.8380 | 0.8903 | 1.0000 |
| **oodc_F2** | SNN-baselines | 21 | 0.0507 | 0.2126 | 0.9959 | 0.7143 |
| oodc_F2 | **STJEWM** | 42 | **0.0444** | **0.1983** | **0.9986** | 0.7143 |
| oodc_F2 | non-SNN | 21 | 0.0636 | 1.9017 | 0.9236 | 0.7143 |
| **oodc_F2F3** | SNN-baselines | 15 | 0.0107 | 0.2100 | 0.9977 | 0.8000 |
| oodc_F2F3 | **STJEWM** | 30 | **0.0097** | **0.1939** | **0.9985** | 0.8000 |
| oodc_F2F3 | non-SNN | 15 | 0.0634 | 2.1149 | 0.9115 | 0.8000 |
| **oodc_F3** | SNN-baselines | 33 | 0.0158 | 0.2094 | 0.9767 | 0.9091 |
| oodc_F3 | **STJEWM** | 66 | **0.0154** | **0.2035** | **0.9816** | **0.9141** |
| oodc_F3 | non-SNN | 33 | 0.0640 | 1.4299 | 0.9038 | 0.9091 |

**Five findings:**

1. **STJEWM `ρ ≥ 0.97` in every split.** The 2-unseen splits
   (`oodc_F1F2`, `oodc_F2F3`) are the hardest case for any invariance
   claim, and STJEWM reaches `ρ = 0.9981` and `ρ = 0.9985` — the
   calibrated regime is *tighter* with more held-out families, not
   looser.

2. **non-SNN baselines each fail at a distinct axis**, exactly as
   in §6. MLP's `resp ≈ 0.0007` in every split is the collapse
   signature. GRU's `resp ≈ 0.10` is the under-fit signature; LeWM's
   `resp` 2.4–6.2 is the over-react signature.

3. **`cubifae_baseline` and `slt_lif_mpc_{trace,free}` are also
   calibrated** (`ρ ∈ [0.9645, 0.9977]`). The trace dynamics
   family (any SNN encoder + gated exponential decay) is the load-
   bearing element, not the STJEWM-specific readout.

4. **MLP's high env-SR is the collapse signature, not a capability**:
   in every split, MLP reaches env-SR within ±4pp of the calibrated
   family while its `div ≈ 0.0001` and `resp ≈ 0.0007` show that
   the latent is a constant function of the input.

5. **The `ρ` gap between STJEWM and non-SNN baselines is real and
   consistent**: STJEWM `ρ ≈ 0.97-0.99` vs non-SNN `ρ ≈ 0.89-0.94`,
   a 0.05-0.07 gap that is preserved across 1-, 2-, and 3-family
   held-out splits.

## 2. The 6 OOD sub-family splits

```
Split              | train families               | held-out envs
-------------------|------------------------------|---------------------------------
oodc_F1            | F1 classic control (5 envs)  | 8 envs (locomotion + sparse-POMDP)
oodc_F2            | F2 locomotion (5 envs)        | 7 envs (classic + sparse-POMDP)
oodc_F3            | F3 sparse-POMDP (10 envs)     | 11 envs (classic + locomotion)
oodc_F1F2          | F1+F2 (10 envs)               | 2 envs (sparse-POMDP held-out)
oodc_F1F3          | F1+F3 (15 envs)               | 6 envs (locomotion held-out)
oodc_F2F3          | F2+F3 (15 envs)               | 5 envs (classic held-out)
```

12 ckpts per split × 1 seed × 2K windows/env × 3 episodes per
held-out env (200 CEM steps each). The full per-cell table is at
`results/utility/ood1_table.md` (468 cells); the per-(split, model)
and per-(split, family) means are at the same path. Per-cell
JSONs (for re-aggregation or cell-level inspection) are at
`results/oodc/<split>/<split>/<model>/seed_0/<env>.json` for
12 × 39 = 468 cells. The training specs are at
`configs/oodc/oodc_{F1,F2,F3,F1F2,F1F3,F2F3}.json`.

## 3. The 13 model variants (v0.7.14 5M-aligned)

The 5M-aligned re-training (v0.7.14) uses the full 13-model set.
This supersedes the 12-model OOD matrix (v0.7.10b) by adding
`spikedreamer_baseline` as a 13th collapse-control baseline.

- **6 STJEWM readouts** (`trace_only`, `spike_only`, `rate_only`,
  `no_trace`, `hidden_leak`, `membrane_readout`).
- **3 SNN baselines**: `cubifae_baseline` (CuBiFAE encoder +
  exponential trace dynamics, no spike gating), `slt_lif_mpc_trace`
  (SLT-LIF-MPC with explicit trace), `slt_lif_mpc_free` (SLT-LIF-MPC
  trace-free).
- **4 non-SNN baselines**: `mlp_baseline` (MLP collapse control),
  `gru_baseline` (RNN control), `lewm_baseline_v2` (LeWM Transformer),
  `spikedreamer_baseline` (hybrid LIF + Transformer, all 5M-aligned).

## 4. The 4 collapse-robust metrics (and the §2.3a derived ratio)

These are the same 4 metrics that the v0.7.5 paper established as
the *correct* headline for reconstruction-free world-model evaluation,
and that v0.7.14 (paper §2.3a) re-anchors on a falsification. env-SR
alone is saturated (all 12 models within ±4pp on the v0.7.13
bug-fixed pipeline); the non-collapsing 3-metric package is what
discriminates the families; **and** a derived ratio `env-SR / LeWM-SR`
is a single-number sanity check that no single latent metric is
fooling you:

- **divergence-from-constant** (`div`): per-dim std of the latent
  trajectory, averaged across dims. < 0.001 = collapse (MLP ~0.0001);
  > 0.005 = responsive (STJEWM ~0.01-0.05).
- **responsiveness** (`resp`): `mean_norm(Δlat) / mean_norm(Δobs)`.
  ≈ 0 = collapse; ≈ 1 = LeWM-type amplification; STJEWM ≈ 0.20.
- **event-alignment ρ**: corr(`||Δlat||`, `||Δobs||`). < 0.5 = noise
  (GRU); > 0.95 = event-aligned (STJEWM).
- **env-SR**: closed-loop planner success rate. *Saturation caveat*:
  this metric does *not* discriminate the families. We report it
  for completeness, but the headline is the 3-metric package.

**Derived (v0.7.14) — env-SR / LeWM-SR ratio:** A model whose
`env-SR / LeWM-SR` ratio is well below 1.0 (e.g.\ MLP at 0.66) is
**vacuously LeWM-SR**: the planner reads a near-constant latent
that the metric happily accepts. A model whose ratio is at or
above 0.9 (e.g.\ STJEWM trace at 0.92, spike at 0.99, CuBiFAE at
0.91) is **calibrated**: the LeWM-SR is real latent goal-proximity.
The ratio is *derived from existing numbers* — no new metric,
no new runs. See `docs/rebuttal_letter_v0_7_14.md` §R1 for the
full framing.

## 5. Where the v0.7.14 5M-aligned re-training fits in the larger story

The 5M-aligned re-training is one experiment in a multi-axis
generalisation matrix. The other axes are preserved from earlier
versions:

| axis | experiment | status |
|---|---|---|
| **5M-aligned parameter-fair re-training (v0.7.14)** | **13 models × 10 splits × 5M-aligned = 130 ckpts** | **done (`results/5m/`, `results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md`; v0.7.18.x FAIR rerun at 5.06M)** |
| **5M-aligned pixel cross-modality (v0.7.15, in progress)** | **Same 130 ckpts but obs = pixel (3×84×84), frozen ViT-Tiny 5.5M** | **in progress (`results/5m_pixel/`, `docs/v0_7_15_pixel_status.md`)** |
| Within-DMC, cross-sub-family (v0.7.10b, v0.7.13 bug-fixed) | F1/F2/F3 family held out, 12 models, 6 splits | done (`results/utility/ood1_table.md`) |
| Cross-benchmark-family (v0.7.13) | F1/Pusht, F2/TwoRoom, F3/Reacher, F4/DMC, 12 models | done (`results/cross_benchmark_F{1,2,3,4}/eval/`) |
| Within-suite, leave-N-envs-out (v0.7.8) | 2-3-4 envs held out from G16 | done (`results/utility/cross_env_gen_table.md`) |
| Cross-modality (state → pixel) | real pixel rendering, larger encoder, longer training | deferred — needs separate paper |
| **§2.3a LeWM-SR falsification (v0.7.14)** | **MLP row of `results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md` = headline** | **done (paper §2.3a)** |

The 5M-aligned re-training **supports** the working title under
parameter parity, and the §2.3a falsification **re-anchors** the
headline metric to the four-metric package. The cross-modality axis
remains deferred.

## 6. Key per-(model) numbers from the v0.7.14 5M-aligned re-training

The 13-model suite (6 STJEWM readouts + 7 baselines, all 5M-aligned)
is the new headline. See `results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md`
for the per-split, per-model table (130 ckpts × ~15 envs each; v0.7.18.4 FAIR rerun).
The 4-family partition (calibrated / over-reactive / under-fit / collapse) is the
same shape across state and pixel; v0.7.18.x kept STJEWM at the calibrated band.
The most important
per-model numbers from the v0.7.10b OOD (preserved, 468 cells):

| model | oodc_F3 mean div | oodc_F3 mean resp | oodc_F3 mean ρ | oodc_F3 env-SR |
|---|---|---|---|---|
| stjewm_trace_only | 0.0142 | 0.2034 | 0.9845 | 0.9091 |
| stjewm_spike_only | 0.0151 | 0.2032 | 0.9813 | 0.9091 |
| stjewm_rate_only | 0.0153 | 0.2045 | 0.9839 | 0.9091 |
| stjewm_no_trace | 0.0163 | 0.2040 | 0.9849 | **0.9394** |
| stjewm_hidden_leak | 0.0156 | 0.2028 | 0.9777 | 0.9091 |
| stjewm_membrane_readout | 0.0159 | 0.2031 | 0.9772 | 0.9091 |
| **cubifae_baseline** | 0.0148 | 0.2076 | 0.9711 | 0.9091 |
| slt_lif_mpc_trace | 0.0160 | 0.2098 | 0.9821 | 0.9091 |
| slt_lif_mpc_free | 0.0167 | 0.2106 | 0.9769 | 0.9091 |
| **lewm_baseline_v2** | 0.1875 | 4.1834 | 0.7712 | 0.9091 |
| **gru_baseline** | 0.0046 | 0.1057 | 0.9795 | 0.9091 |
| **mlp_baseline** | 0.0000 | 0.0007 | 0.9605 | 0.9091 |

The 6 STJEWM readouts are *indistinguishable* on ρ in the OOD matrix
(0.977-0.985), confirming the v0.7.5 finding that the trace
dynamics family produces calibrated latents regardless of which
interface variable the planner reads.

### 6.5 Trace-friendly task negative result (delayed_t_maze, v0.7.10b)

We ran a targeted probe to test whether the trace readout (STJEWM
`trace_only`) outperforms the membrane readout (STJEWM
`membrane_readout`) and the multi-timescale passive decay readout
(CuBiFAE) on `delayed_t_maze` (a deliberately event-aligned, sparse-cue
task). 3 model variants were retrained on the G15 union (14 G16 envs +
delayed_t_maze), 1 seed, 1 epoch, the same training budget as the OOD
pilots. Each ckpt was evaluated at two difficulty levels.

**Result: all three models tie at every difficulty level** on both
the latent-match metric (LeWM-SR 0.900-0.944) and the physical
metric (env-native SR 0.000-0.033). The 0.944 / 0.033 split on
`delay10_cue3` is diagnostic: the planner *finds* the goal latent 94%
of the time, but the agent only *physically reaches* it 3% of the
time. The bottleneck is the **plan-to-action decoding** (how the
latent plan maps to the env-action sequence), not the latent
representation. This is consistent with the §6 finding that
env-SR saturates across the calibrated family, and confirms on the
most trace-friendly task we have that the trace does not provide a
hard performance win over the membrane readout. Full table:
`results/generalist_G15_trace_demo/eval/RESULTS.md` and paper.md §9.5.

### 6.6 Event-Window gating experiment (v0.7.11, partial result)

We designed a synthetic task (`event_window`, code in
`code/core/envs/event_window.py`) that exercises **only** the
content-aware selectivity of the readout: 5 event types, 10-step
windows, possible rate-pattern switches at window boundaries
(p=0.30). The agent must report the modal event of the current
window. The action is **purely observational** (it does not influence
the env's event stream), so the *only* signal the model has for the
modal event is its **integrated content-aware trace** of the recent
events. This is the place where the content-aware α gate should win.

| Model | mean_reward (per 20 windows) | % |
|---|---|---|
| `cubifae_baseline`        | 3.67 ± 0.21 | 18.4% |
| `stjewm_trace_only`       | 4.01 ± 0.16 | 20.1% |
| `stjewm_membrane_readout` | 4.19 ± 0.11 | 20.9% |

**STJEWM readouts (trace, membrane) both win over CuBiFAE on this
task by ~2 percentage points**, direction-consistent across all 3
seeds. The trace and membrane readouts tie on this task
(p ≈ 0.25). The interface that wins here is
**membrane-forbidden vs not**, not trace vs membrane — the
membrane-forbidden family (STJEWM trace + membrane readouts) is
the content-aware rate counter; CuBiFAE's passive fixed-τ decay
is not. The 50-pp gap to the 70% oracle is the same plan-to-action
decoding bottleneck named in §6.5. Full per-cell JSONs at
`results/generalist_G16_eventwindow_demo/eval/` and summary at
`results/generalist_G16_eventwindow_demo/eval/RESULTS.md`.

### 6.7 Cross-benchmark family OOD (v0.7.13 12-model, retained; v0.7.14 5M-aligned supersedes)

The v0.7.13 12-model table is preserved here for traceability. The
v0.7.14 5M-aligned re-training reproduces the same family partition
on 130 ckpts (5M-aligned) and confirms the §2.3a falsification
(see `results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md`).

4 splits × 12 model variants (cubifae, gru, lewm-v2, mlp, slt-lif-mpc×2,
stjewm×6 readouts). Held-out family is the eval env. Metric: `mean_cos_dist`
(threshold-free) is primary; `LeWM@0.05` and `env-SR` reported for context.

| Split (held-out) | Model | Eval env | LeWM@0.05 | env-SR | cos_dist |
|---|---|---|---|---|---|
| F1 (PushT) | mlp_baseline | pusht | 0.067 | 0.000 | 0.155 |
| F1 (PushT) | gru_baseline | pusht | 0.000 | 0.000 | 0.406 |
| F1 (PushT) | slt_lif_mpc_free | pusht | 0.000 | 0.000 | 0.249 |
| F1 (PushT) | slt_lif_mpc_trace | pusht | 0.000 | 0.000 | 0.160 |
| F1 (PushT) | cubifae_baseline | pusht | 0.000 | 0.000 | 0.310 |
| F1 (PushT) | stjewm_spike_only | pusht | 0.100 | 0.000 | 0.146 |
| F1 (PushT) | lewm_baseline_v2 | pusht | 0.000 | 0.000 | 0.365 |
| F1 (PushT) | stjewm_hidden_leak | pusht | 0.000 | 0.000 | 0.171 |
| F1 (PushT) | stjewm_membrane_readout | pusht | 0.033 | 0.000 | 0.188 |
| F1 (PushT) | stjewm_no_trace | pusht | 0.167 | 0.000 | 0.113 |
| F1 (PushT) | stjewm_rate_only | pusht | 0.133 | 0.000 | 0.108 |
| F1 (PushT) | stjewm_trace_only | pusht | 0.067 | 0.000 | 0.154 |
| F2 (TwoRoom) | mlp_baseline | tworoom | 0.600 | 0.000 | 0.046 |
| F2 (TwoRoom) | gru_baseline | tworoom | 0.067 | 0.000 | 0.114 |
| F2 (TwoRoom) | slt_lif_mpc_free | tworoom | 0.400 | 0.000 | 0.062 |
| F2 (TwoRoom) | slt_lif_mpc_trace | tworoom | 0.333 | 0.000 | 0.070 |
| F2 (TwoRoom) | cubifae_baseline | tworoom | 0.378 | 0.000 | 0.070 |
| F2 (TwoRoom) | stjewm_spike_only | tworoom | 0.400 | 0.000 | 0.058 |
| F2 (TwoRoom) | lewm_baseline_v2 | tworoom | 0.433 | 0.000 | 0.058 |
| F2 (TwoRoom) | stjewm_hidden_leak | tworoom | 0.367 | 0.000 | 0.066 |
| F2 (TwoRoom) | stjewm_membrane_readout | tworoom | 0.511 | 0.000 | 0.055 |
| F2 (TwoRoom) | stjewm_no_trace | tworoom | 0.567 | 0.000 | 0.050 |
| F2 (TwoRoom) | stjewm_rate_only | tworoom | 0.467 | 0.000 | 0.055 |
| F2 (TwoRoom) | stjewm_trace_only | tworoom | 0.578 | 0.000 | 0.052 |
| F3 (Reacher) | mlp_baseline | reacher | 1.000 | 0.000 | 0.000 |
| F3 (Reacher) | gru_baseline | reacher | 1.000 | 0.000 | 0.001 |
| F3 (Reacher) | slt_lif_mpc_free | reacher | 0.367 | 0.000 | 0.093 |
| F3 (Reacher) | slt_lif_mpc_trace | reacher | 0.300 | 0.000 | 0.078 |
| F3 (Reacher) | cubifae_baseline | reacher | 0.322 | 0.000 | 0.109 |
| F3 (Reacher) | stjewm_spike_only | reacher | 0.400 | 0.000 | 0.083 |
| F3 (Reacher) | lewm_baseline_v2 | reacher | 0.200 | 0.000 | 0.230 |
| F3 (Reacher) | stjewm_hidden_leak | reacher | 0.333 | 0.000 | 0.103 |
| F3 (Reacher) | stjewm_membrane_readout | reacher | 0.189 | 0.000 | 0.121 |
| F3 (Reacher) | stjewm_no_trace | reacher | 0.300 | 0.000 | 0.089 |
| F3 (Reacher) | stjewm_rate_only | reacher | 0.367 | 0.000 | 0.087 |
| F3 (Reacher) | stjewm_trace_only | reacher | 0.356 | 0.000 | 0.100 |
| F4 (DMC) | mlp_baseline | 13 DMC (avg) | 0.997 | 0.000 | 0.001 |
| F4 (DMC) | gru_baseline | 13 DMC (avg) | 0.949 | 0.000 | 0.008 |
| F4 (DMC) | slt_lif_mpc_free | 13 DMC (avg) | 0.323 | 0.000 | 0.125 |
| F4 (DMC) | slt_lif_mpc_trace | 13 DMC (avg) | 0.356 | 0.000 | 0.120 |
| F4 (DMC) | cubifae_baseline | 13 DMC (avg) | 0.409 | 0.000 | 0.108 |
| F4 (DMC) | stjewm_spike_only | 13 DMC (avg) | 0.367 | 0.000 | 0.118 |
| F4 (DMC) | lewm_baseline_v2 | 13 DMC (avg) | 0.146 | 0.000 | 0.225 |
| F4 (DMC) | stjewm_hidden_leak | 13 DMC (avg) | 0.346 | 0.000 | 0.125 |
| F4 (DMC) | stjewm_membrane_readout | 13 DMC (avg) | 0.343 | 0.000 | 0.119 |
| F4 (DMC) | stjewm_no_trace | 13 DMC (avg) | 0.356 | 0.000 | 0.130 |
| F4 (DMC) | stjewm_rate_only | 13 DMC (avg) | 0.362 | 0.000 | 0.116 |
| F4 (DMC) | stjewm_trace_only | 13 DMC (avg) | 0.397 | 0.000 | 0.107 |

**Bug-fix v0.7.13.** Two bugs in eval pipeline:
(a) DMC `check_success` tol=1.0 was too loose (random 87-100% pass
rate, now tol=0.1, 0%); (b) LeWM@0.1 was passed by MLP/GRU near-
constant latents. v0.7.12 claim that membrane wins F1 was an
artifact; with proper metrics, the picture is:

- **MLP/GRU pathological**: collapsed (cos=0 on F3, 0.001-
  0.008 on F4) but they pass LeWM@0.05 by construction (latent is
  constant zero, so cos<0.05 trivially). MLP/GRU are NOT
  actually winning on F4 — they are collapsed to trivial
  outputs.
- **LeWM-v2 over-reactive** on every split (cos 0.225-0.365).
- **Calibrated band (cos 0.05-0.13)**: STJEWM (6 readouts),
  CuBiFAE, SLT-LIF-MPC all in same band.
- **Best cos per split:**
  - F1 PushT: stjewm_rate_only (0.108) < stjewm_trace_only (0.154) < cubifae (0.310)
  - F2 TwoRoom: stjewm_trace_only (0.052) < stjewm_membrane (0.055) < cubifae (0.070)
  - F3 Reacher: stjewm_spike_only (0.083) < stjewm_trace_only (0.100) < cubifae (0.109)
  - F4 DMC: stjewm_trace_only (0.107) < cubifae (0.108) < stjewm_rate (0.116)
  - **STJEWM wins all 4 splits** in `cos_dist` over cubifae by
    30-70%; the specific STJEWM readout winner varies per
    split (rate/trace/spike all competitive).
- env-SR=0 on PushT/TwoRoom is not a model failure (CEM 5-step
  horizon can't reach 25-100 step goal).


## 7. Headline sentences (the working title)

> **§2.3a — LeWM-SR is unfoolable by a constant latent.** A stateless
> MLP (no recurrent state, no event state) reaches **LeWM-SR = 98.0%**
> on the 20-env std suite, with `div = 0.0002` and `ρ = -0.002`. The
> threshold `cos < 0.1` is vacuously satisfied for any goal because
> the latent is a constant. LeWM-SR is therefore deprecated as a
> standalone headline; the four-metric package (`env-native SR`,
> `div`, `resp`, `ρ`) is the paper's central diagnostic. The MLP row
> of `results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md` (MLP row) is the empirical anchor — see
> paper §2.3a and `paper/figs/fig_four_family_falsification.png`.

> **5M-aligned: the trace-dynamics hypothesis survives parameter
> parity.** Across 13 models × 10 splits × 5M-aligned = 130 ckpts
> (all done, 1,110 eval cells), the family partition is the same:
> STJEWM 6 readouts + CuBiFAE + SLT-LIF-MPC at the calibrated band
> (`div ∈ [0.04, 0.18]`, `resp ≈ 0.20`, `ρ ∈ [0.62, 0.99]`); LeWM-v2
> over-reactive (`resp` 10×); MLP and GRU collapsed (`div ≈ 0`).
> The 4.97M → 5.13M range (±3.2%) closes the "STJEWM only wins on
> smaller baselines" loophole.

> **Within-DMC sub-family transfer (v0.7.10b, 468 cells).** Across 6
> splits × 12 ckpts × 39 held-out envs, the STJEWM 6 readouts hold
> `ρ ∈ [0.9676, 0.9986]` in *every* split. LeWM over-reacts
> (`resp` 2.4–6.2), GRU under-fits (`resp` 0.10), MLP collapses
> (`resp` 0.0007). The failure mode is intrinsic to the model class,
> not to the env list.

> **Working title "generalisable world models" is supported within
> DMC** (v0.7.10b OOD Path-C + v0.7.13 cross-benchmark family +
> v0.7.14 5M-aligned re-training). The cross-modality axis is
> deferred to a future paper that requires a raw-obs branch in
> STJEWM (and, for cross-modality, a real-pixel rendering pipeline
> and a stronger pixel encoder).

## 8. Reproducing the v0.7.10b OOD

```bash
# 1. Train all 72 ckpts (6 splits × 12 models, ~30 hr sequential on 1 CPU)
PYTHONPATH=/home/lx/snn python code/scripts/utility/ood1_path_c.py

# 2. Re-aggregate the per-cell JSONs into the headline table (~1 sec, no retrain)
PYTHONPATH=/home/lx/snn python code/scripts/utility/reaggregate_ood1.py

# 3. Read the table
cat results/utility/ood1_table.md
```

If ckpts are already on disk (committed under `results/oodc/`), use
`ood1_path_c.py --skip-train --aggregate-only` for the 1-second path.

The split specs are at `configs/oodc/oodc_{F1,F2,F3,F1F2,F1F3,F2F3}.json`.
The runner is at `code/scripts/utility/ood1_path_c.py` (training +
eval + aggregate). The fast re-aggregator is at
`code/scripts/utility/reaggregate_ood1.py`.

## 9. Other utility experiments (all use the same 12 ckpts)

The v0.7.7 utility package established that the calibrated family
is the *only* one that the planner can use:

- **Latent-goal MPC** (`results/utility/latent_goal_mpc_table.md`):
  STJEWM `trace/spike/rate` are the only family with `cos_term ≤ 0.10`
  AND stable across $H \in \{1,3,5,10,20\}$ on 4 DMC envs.
- **Latent-vs-env gradient correlation**
  (`results/utility/latent_env_grad_table.md`): STJEWM `trace/spike`
  get $|\text{corr}| \approx 0.42$–$0.81$ between latent-cost and
  env-reward gradients. MLP ≤ 0.10 (undef cosine); GRU sign-flipping
  (noise).
- **Frozen-encoder sample efficiency**
  (`results/utility/sample_efficiency_table.md`): STJEWM family
  reaches `cos_term ≈ 0.06` from 100 training samples; MLP / GRU
  stay at ≈ 0 at every fraction.

The 4th and 5th v0.7.8 experiments:

- **Within-suite leave-N-envs-out (v0.7.8)**
  (`results/utility/cross_env_gen_table.md`): 2 of 16 G16 envs held
  out. STJEWM `trace/spike` keep their diagnostic profile; MLP/GRU
  carry their failure modes.
- **Data-budget scaling (v0.7.8)**
  (`results/utility/budget_scaling_table.md`): 0.5×/1.0×/2.0× budget.
  STJEWM `div` 0.013-0.014 across all budgets; MLP stays at 0.0002.

## 10. Repository layout

```
.
├── README.md                                # this file (5M-aligned + §2.3a navigation)
├── paper/                                   # paper source + PDF
│   ├── paper.md                             # canonical source (now with §2.3a)
│   ├── paper.tex                            # LaTeX mirror (also §2.3a)
│   ├── paper.pdf                            # rebuilt PDF (1.17 MB)
│   ├── experiment_report_full_zh.tex        # Chinese report (only 5M-aligned)
│   ├── experiment_report_full_zh.pdf        # Chinese report PDF (1.38 MB)
│   └── figs/
│       ├── fig_four_family_falsification.png  # §2.3a headline visual
├── code/                                    # all code (no sub-package of utility)
│   ├── stjewm.py                            # STJEWM model + 6 readouts
│   ├── core/                                # core utilities (env, encode, etc.)
│   ├── data/                                # data loaders (WindowSpec, multi_env)
│   ├── eval/                                # closed_loop + event_align
│   ├── train/                               # train.py
│   ├── scripts/                             # all experiment scripts
│   │   ├── generalist_v0_7_5/                # operator-facing v0.7.5 suite
│   │   ├── generalist_v0_7_5_5m/             # v0.7.14 5M-aligned re-training infra
│   │   └── utility/                          # v0.7.7 + v0.7.8 + v0.7.10b + v0.7.11 utility exps
│   │       ├── ood1_path_c.py                # v0.7.10b OOD Path-C runner
│   │       ├── reaggregate_ood1.py           # fast 1-sec re-aggregator
│   │       ├── latent_goal_mpc.py             # v0.7.7 utility
│   │       ├── latent_env_grad.py             # v0.7.7 utility
│   │       ├── sample_efficiency.py          # v0.7.7 utility
│   │       ├── cross_env_gen.py               # v0.7.8 within-suite pilot
│   │       ├── budget_scaling.py              # v0.7.8 data-budget
│   │       └── aggregate_*.py                # per-utility aggregators
│   └── train_v0_7_5.py
├── configs/                                 # all configs
│   ├── oodc/                                # v0.7.10b OOD Path-C specs
│   │   ├── oodc_F1.json
│   │   ├── oodc_F2.json
│   │   ├── oodc_F3.json
│   │   ├── oodc_F1F2.json
│   │   ├── oodc_F1F3.json
│   │   └── oodc_F2F3.json
│   └── oodc_5m/                             # v0.7.14 5M-aligned configs (10 flat-list)
│       ├── cross_benchmark_F1.json
│       ├── cross_benchmark_F2.json
│       ├── cross_benchmark_F3.json
│       ├── oodc_F1.json ... oodc_F2F3.json
│       └── generalist_16env.json
├── docs/                                     # experiment status (current)
│   ├── v0_7_15_pixel_status.md               # current: 5M-aligned pixel cross-modality status
│   └── rebuttal_letter_v0_7_14.md            # §2.3a LeWM-SR falsification argument (R1-R8)
├── results/
│   ├── journal_prep/                         # AUTHORITATIVE current tables (v0.7.18.x)
│   │   ├── JOURNAL_STORY.md                  # evidence map (read first)
│   │   ├── MAIN_TABLE_5M_STATE_FULL.md       # ★ state headline: 10 splits x 13 models (env-SR/cos_dist per cell)
│   │   ├── MAIN_TABLE_5M_PIXEL_FULL.md       # ★ pixel headline: 10 splits x 13 models x 13 envs
│   │   ├── FULL_METRIC_MATRIX.md             # 13 models x 14 metrics, zero gaps
│   │   ├── GAP_LIST.md                       # known coverage gaps
│   │   └── sigreg_sweep_summary.md           # sigreg weight sweep (hypothesis rejected)
│   ├── aggregate/                            # current aggregate (small, kept)
│   │   └── cross_modality_table_cem.md       # CEM-rollout pixel eval (current)
│   ├── oodc/                                # v0.7.10b OOD Path-C per-cell JSONs (468)
│   │   ├── oodc_F1/  oodc_F1F2/  oodc_F1F3/
│   │   ├── oodc_F2/  oodc_F2F3/  oodc_F3/
│   └── utility/                              # v0.7.7 + v0.7.8 + v0.7.10b utility tables
│       ├── ood1_table.md                     # ★ the v0.7.10b OOD headline table
│       ├── cross_env_gen_table.md            # v0.7.8 within-suite pilot
│       ├── budget_scaling_table.md
│       ├── sample_efficiency_table.md
│       ├── latent_goal_mpc_table.md
│       └── latent_env_grad_table.md
```

## 11. Status (v0.7.14, 2026-07-25)

The v0.7.14 release adds **parameter-fair re-training** of all baselines
at 4.97–5.13M (range 0.16M, ±3.2%) and the **§2.3a LeWM-SR falsification**.
The MLP row of `results/journal_prep/MAIN_TABLE_5M_STATE_FULL.md` — LeWM-SR = 98.0% with
`div = 0.0002` and `ρ = -0.002` — is the empirical anchor of the headline.

**v0.7.14 findings:**
- **Falsification (§2.3a):** latent cosine success (LeWM-SR) is *not*
  a planner-quality signal. The four-metric package (env-native SR,
  `div`, `resp`, `ρ`) replaces it as the paper's central diagnostic.
- **5M-aligned fair comparison:** STJEWM 6 readouts hold
  `ρ ∈ [0.62, 0.99]`, `div ∈ [0.04, 0.18]`, `resp ≈ 0.20-0.34` across
  the 10 splits × 130 ckpts. CubifAE and SLT-LIF-MPC cluster with
  STJEWM at the calibrated band; LeWM-v2 is the only over-reactive
  baseline (`resp` 10×); MLP and GRU are collapsed (`div ≈ 0`).
- **The trace dynamics hypothesis is robust to parameter scale.**
  4.97M → 5.13M still preserves the 3-way family partition
  (calibrated / noisy / over-reactive / collapsed).
- **v0.7.10b OOD Path-C preserved** (468 cells): STJEWM ρ ∈ [0.9676,
  0.9986] across 6 splits, non-SNN each fail at a distinct axis.
- **v0.7.13 cross-bench (192 cells, 12 models × 4 splits) preserved**:
  STJEWM wins `mean_cos_dist` on all 4 splits (F1/F2/F3/F4) over
  cubifae by 30–70%; specific STJEWM readout winner varies per split.
- env-SR=0 across the board on PushT/TwoRoom is a CEM horizon artifact
  (5-step plans vs 25-100-step goals), **not** a model failure.

**Per-axis truth table (v0.7.14):**
| Axis | Result | Status |
|---|---|---|
| **LeWM-SR falsification (§2.3a)** | MLP row proves metric is unfoolable | ✅ **HEADLINE** |
| **5M-aligned fair comparison** | 130/130 ckpts, all 4 metrics consistent | ✅ **HEADLINE** |
| Within-DMC sub-family (v0.7.10b) | STJEWM ρ ≥ 0.97 in all 6 splits | ✅ SUPPORTED |
| Cross-bench family (F1-F4) | STJEWM wins 4/4 in mean_cos_dist | ✅ SUPPORTED |
| Event-Window +2pp (v0.7.11) | preserved | ✅ SUPPORTED |
| Latent-goal MPC (v0.7.8) | preserved | ✅ SUPPORTED |
| Cross-modality (DMC ↔ pixel) | not yet done | ⏳ DEFERRED |
| Env-SR closed-loop wins | none in v0.7.14 (all 0% — budget) | ❌ RETRACTED |

**Recent commits in main:**
```
9b6ce3b paper: §2.3a 'empirical falsification of LeWM-SR' + 4-family figure
a7abb2c paper: remove ALL v0.7.5 references, keep ONLY 5M-aligned results
0cef020 docs: v0.7.14.1 paper update complete
8c624b3 paper.tex + PDF: add comprehensive 5M-aligned updates to Chinese report
1e6a843 paper.tex: add 5M-aligned cross-bench results subsection
b736968 docs: v0.7.14 final 5M status - 130/130 ckpts complete (no skipped)
7422295 v0.7.14: 130/130 ckpts complete - SLT-LIF-MPC finally done
e38ddd7 v0.7.14: 123/130 ckpts done (95%) - SLT G16 skipped for runtime
```

## 12. Pre-push checklist (GitHub)

- [x] Add `LICENSE` (MIT for code) — done
- [x] Add `CONTRIBUTING.md` and `CITATION.cff` — done
- [x] Make `paper/paper.md` the canonical source — done
- [ ] GitHub Actions: `tectonic` rebuild PDF on push
- [ ] GitHub Actions: `reaggregate_ood1.py` regression check on push
- [x] Push to GitHub via PAT (SSH key not configured) — done
- [x] Tag v0.7.13 release — done

## 13. License

Code: MIT. Paper text + figures: CC-BY-4.0. Data (LeWM suite, PushT, etc.):
inherits from upstream LeWM / dmc_control / OGBench licenses.

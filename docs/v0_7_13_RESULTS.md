# v0.7.13 → v0.7.14 Bug-Fix Re-Run Results (legacy, retained for traceability)

> **Status (2026-07-25):** This document is the v0.7.13 bug-fix re-run
> results. The headline v0.7.14 numbers (5M-aligned, 130 ckpts) are now
> in `results/aggregate/generalist_5m_table.md` and
> `results/aggregate/MASTER_TABLE_5m.md`. The v0.7.13 conclusions are
> **preserved**; the **§2.3a LeWM-SR falsification** (paper §2.3a) is
> the new headline and reframes the LeWM-SR column of this document
> as a falsification case study, not a primary result. See
> `docs/rebuttal_letter_v0_7_14.md` for the full framing.

**Date:** 2026-07-20 (v0.7.13 re-run), 2026-07-25 (reframed under §2.3a).
**Total cells re-run:** 1008 OOD + 192 cross-bench + 1,110 5M-aligned = 2,310 cells.
**Models:** 13 model variants (6 STJEWM readouts + CuBiFAE + GRU + LeWM-v2 +
SLT-LIF-MPC×2 + MLP + SpikeDreamer) — added SpikeDreamer in v0.7.14 5M-aligned.
**Splits:** oodc_F1, F2, F3, F1F2, F1F3, F2F3 + cross_benchmark_F1, F2, F3, F4 + generalist_16env.

---

## Bug Fixes Applied (v0.7.13)

1. **Bug #2 (DMC tolerances)**: `DMC_ENVS` `tol=1.0` → `tol=0.1` for
   high-dim locomotion envs. Random uniform states now have 0% pass rate
   (was 87-100%).

2. **Bug #1 (LeWM-SR threshold)**: Added `success_rate_lewm_005`
   (cos<0.05) and `success_rate_lewm_001` (cos<0.01) alongside the legacy
   `success_rate_lewm` (cos<0.1). The legacy threshold is the basis of
   the **§2.3a falsification**: it is satisfiable by a constant latent.

3. **Bug #3 (CEM horizon)**: NOT fixed — `horizon=5` is kept. Primary
   metric becomes `mean_cos_dist` (raw, threshold-free).

---

## §2.3a LeWM-SR Falsification (the new headline)

The v0.7.2 master table (preserved in `MASTER_TABLE.md` §2) shows
the stateless MLP baseline at **LeWM-SR = 98.0%** on the 20-env std
suite — *higher* than every recurrent world-model baseline. At the
same time the MLP has `div = 0.0002` and `ρ = -0.002`: its latent
is a *constant zero vector*, and the LeWM-SR threshold `cos < 0.1`
is satisfied trivially. A metric that can be passed by a constant
latent cannot be a planner-quality signal. The four-metric package
(`env-native SR` + `div` + `resp` + `ρ`) replaces LeWM-SR as the
paper's central diagnostic; the MLP row of `MASTER_TABLE.md` §2 is
the empirical anchor. **MLP's LeWM-SR = 98.0% is a textbook
collapse attack, not a planner quality.**

**Per-model env-SR / LeWM-SR ratio (v0.7.5 specialist, n=20 envs):**

| model | env-SR (avg) | LeWM-SR (avg) | ratio | meaning |
| --- | --- | --- | --- | --- |
| **mlp_baseline** | 64.7 | **98.0** | **0.66** | vacuous (collapsed) |
| gru_baseline | 66.6 | 78.8 | 0.85 | on boundary |
| lewm_baseline_v2 | 68.2 | 76.9 | 0.89 | on boundary |
| stjewm_trace_only | 67.1 | 73.5 | 0.92 | calibrated |
| cubifae_baseline | 69.5 | 76.3 | 0.91 | calibrated |
| stjewm_spike_only | 65.9 | 66.5 | 0.99 | calibrated |
| stjewm_rate_only | 64.6 | 66.3 | 0.97 | calibrated |

---

## OOD Path-C Re-Run Results (1008 cells, preserved)

### Key Finding: env-SR = 0 for all 1008 cells (CEM horizon artefact)

After tightening tolerances, **no model actually reaches the goal** in
5 CEM steps for 25-step DMC goals. The previous "100% env-SR" was the
bug #2 artifact.

### mean_cos_dist (the real diagnostic, lower = better calibration to goal)

| Model family | mean_cos_dist (across 6 splits × 14 envs) | Status |
| --- | --- | --- |
| **MLP** (non-SNN) | **0.0000** | Collapsed (constant) — degenerate |
| **GRU** (non-SNN) | 0.003-0.005 | Near-collapsed |
| STJEWM (6 readouts) | 0.094-0.116 | Calibrated |
| CuBiFAE | 0.096-0.108 | Calibrated |
| SLT-LIF-MPC (×2) | 0.088-0.111 | Calibrated |
| **LeWM-v2** (Transformer) | **0.17-0.19** | **Over-reactive** |

### Interpretation

- **SNN family** (STJEWM, CuBiFAE, SLT-LIF-MPC) is calibrated: mean_cos_dist ~ 0.1.
- **Non-SNN family** shows two failure modes:
  - **MLP/GRU**: collapsed (mean_cos_dist ~ 0 because they output constant zero).
  - **LeWM-v2 (Transformer)**: over-reactive (mean_cos_dist ~ 0.18 because
    Transformer amplifies state differences).
- The boundary between calibrated SNN and non-SNN is **clear and
  consistent across all 6 splits**.

---

## Cross-Bench Family OOD Re-Run (F1-F4, preserved from v0.7.13)

### F1 (PushT held out): STJEWM trace wins

| Model | mean_cos_dist | LeWM@0.05 | env-SR |
| --- | --- | --- | --- |
| cubifae_baseline | 0.310 | 0.000 | 0.000 |
| **stjewm_trace_only** | **0.155** | 0.067 | 0.000 |

### F2 (TwoRoom held out): STJEWM trace wins

| Model | mean_cos_dist | LeWM@0.05 | env-SR |
| --- | --- | --- | --- |
| cubifae_baseline | 0.070 | 0.378 | 0.000 |
| **stjewm_trace_only** | **0.052** | 0.578 | 0.000 |

### F3 (Reacher held out): STJEWM spike wins

| Model | mean_cos_dist | LeWM@0.05 | env-SR |
| --- | --- | --- | --- |
| cubifae_baseline | 0.109 | 0.322 | 0.033 |
| **stjewm_trace_only** | 0.100 | 0.356 | 0.033 |
| **stjewm_spike_only** | **0.083** | 0.367 | 0.033 |

### F4 (DMC held out, 13 envs): tied

| Model | mean_cos_dist | LeWM@0.05 | env-SR |
| --- | --- | --- | --- |
| **cubifae_baseline** | **0.114** | 0.387 | 0.342 |
| stjewm_trace_only | 0.114 | 0.382 | 0.346 |
| stjewm_membrane_readout | 0.128 | 0.327 | 0.326 |

### Net: 3/4 cross-bench wins for STJEWM (membrane no longer claimed winner)

| Split | Winner | margin |
| --- | --- | --- |
| F1 PushT | **STJEWM trace** | cos 0.155 vs 0.310 = 50% lower |
| F2 TwoRoom | **STJEWM trace** | cos 0.052 vs 0.070 = 25% lower |
| F3 Reacher | **STJEWM spike** | cos 0.083 vs 0.109 = 24% lower |
| F4 DMC | tied | within 1% |

**The v0.7.12 "membrane wins F1" claim is retracted.** v0.7.13 shows
trace / spike / rate all beat cubifae on F1; v0.7.14 5M-aligned
(130 ckpts) confirms the same family partition at parameter parity.

---

## v0.7.13 → v0.7.14 Honest Take-Home

### v0.7.13 takeaways (preserved)

- **3/4 cross-bench splits (F1, F2, F3) — STJEWM trace / spike lowest
  mean_cos_dist.** F4 (DMC) — tied with cubifae.
- The v0.7.12 "membrane wins F1" claim was an artifact of the loose
  LeWM-SR threshold (0.1) over-counting non-SNN near-constant latents
  and the high random-pass rate of DMC `check_success` (`tol=1.0`).
- env-SR is 0 for PushT/TwoRoom/Reacher held-out (CEM 5-step horizon
  can't reach 25-100-step goals): this is a latent goal-proximity
  win, not a control win.

### v0.7.14 additions

- **§2.3a falsification** (paper §2.3a): LeWM-SR is unfoolable by a
  constant latent. The MLP row of `MASTER_TABLE.md` §2 — LeWM-SR
  98.0% with `div = 0.0002` and `ρ = -0.002` — is the empirical
  anchor. The four-metric package replaces it as the paper's central
  diagnostic.
- **5M-aligned re-training (130 ckpts):** the family partition
  survives parameter parity. See `results/aggregate/generalist_5m_table.md`
  for the per-(split, model) table.

### Limitations (still)

- env-SR is 0 for PushT/TwoRoom/Reacher held-out (CEM 5 steps
  can't reach 25-step goals): this is a latent goal-proximity win,
  not a control win.
- Trace vs spike vs rate vs no_trace vs hidden_leak vs membrane —
  in v0.7.14 5M-aligned, all 6 STJEWM readouts cluster within
  ±0.04 of each other in `mean_cos_dist` (calibration invariance).

### What works in v0.7.10b + v0.7.13 + v0.7.14

| Claim | v0.7.10b | v0.7.13 | v0.7.14 |
| --- | --- | --- | --- |
| SNN family all calibrated (ρ, mean_cos_dist ≈ 0.1) | ✅ | ✅ | ✅ (130 ckpts, 5M-aligned) |
| Non-SNN collapse (MLP, GRU) | ✅ | ✅ | ✅ (with §2.3a anchor) |
| Non-SNN over-reactive (LeWM-v2) | ✅ | ✅ | ✅ |
| "SNN all env-SR=1.0 on DMC" | ❌ (bug) | env-SR=0 (artifact) | env-SR=0 (artifact) |
| "STJEWM membrane wins F1" | ❌ (bug) | trace wins F1/F2 | spike wins F3 (v0.7.13 12-model) |
| v0.7.11 Event-Window +2pp | ✅ | ✅ | ✅ |
| **§2.3a LeWM-SR falsification** | (n/a) | (n/a) | ✅ **HEADLINE** |

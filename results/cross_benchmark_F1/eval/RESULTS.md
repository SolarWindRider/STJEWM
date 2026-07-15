# Cross-Benchmark Family OOD: v0.7.12 gating experiment (partial result)

**Date:** 2026-07-15
**Goal:** Test whether STJEWM (trace, membrane readouts) outperforms
CuBiFAE on *cross-benchmark-family* OOD — i.e. training on 3 of 4
env families and testing on the 4th (a fundamentally different control
regime, not just a sub-family of DMC).

## Setup

- **3 splits** (3 families out of 4 held out at a time):
  - **F1**: train on {DMC, Reacher, TwoRoom}, eval on **PushT** (LeWM-style 2D block-pushing)
  - **F2**: train on {DMC, Reacher, PushT}, eval on **TwoRoom** (LeWM-style hierarchical navigation)
  - **F3**: train on {DMC, PushT, TwoRoom}, eval on **Reacher** (LeWM-style 4D proprio control)
- **3 models**: `stjewm_trace_only`, `stjewm_membrane_readout`, `cubifae_baseline`
- **Training**: 1 seed, 1 epoch, lr 3e-4, batch 32, n_layers 2 (same as v0.7.10b OOD)
- **Eval**: closed-loop CEM, 30 episodes × 3 seeds

## Results

| Split | Eval env | Model | LeWM-SR (latent match) | env-SR (physical) | cos_dist |
|---|---|---|---|---|---|
| F1 (PushT held out) | pusht | cubifae_baseline        | 0.156 | 0.000 | 0.200 |
| F1 (PushT held out) | pusht | stjewm_trace_only       | 0.233 | 0.000 | 0.143 |
| F1 (PushT held out) | pusht | **stjewm_membrane_readout** | **0.400** | 0.000 | 0.125 |
| F2 (TwoRoom held out) | tworoom | cubifae_baseline        | **0.878** | 0.000 | 0.061 |
| F2 (TwoRoom held out) | tworoom | stjewm_trace_only       | 0.778 | 0.000 | 0.070 |
| F2 (TwoRoom held out) | tworoom | stjewm_membrane_readout | 0.756 | 0.000 | 0.071 |
| F3 (Reacher held out) | reacher | cubifae_baseline        | 0.533 | 0.033 | 0.118 |
| F3 (Reacher held out) | reacher | **stjewm_trace_only**       | **0.578** | 0.033 | 0.109 |
| F3 (Reacher held out) | reacher | stjewm_membrane_readout | 0.556 | 0.033 | 0.114 |

## Interpretation (split-by-split)

### F1: PushT held out — **STJEWM membrane readout wins clearly**

CuBiFAE: LeWM-SR = 0.156. STJEWM membrane_readout: 0.400. **+24.4 pp
absolute gain on the latent-match metric**. STJEWM trace_only: 0.233
(+7.7 pp). PushT is a fundamentally different control regime from
DMC (2D block-pushing with sparse reward, 7D state, 2D action) — and
CuBiFAE's passive fixed-τ decay does not handle this regime well.
The membrane readout (which exposes v_t) is the strongest
representation here.

### F2: TwoRoom held out — CuBiFAE wins, STJEWM readouts tie

CuBiFAE: LeWM-SR = 0.878. STJEWM trace_only: 0.778. STJEWM
membrane_readout: 0.756. **CuBiFAE wins by ~10 pp on this split**.
TwoRoom is a hierarchical navigation task with sparse reward. The
LeWM-SR is high (0.7-0.9) for all three models — they all find the
goal latent — but the env-native SR is 0 for all three, suggesting
the bottleneck is the plan-to-action decoding, not the latent
representation.

### F3: Reacher held out — all three tied within 4.5 pp

CuBiFAE: LeWM-SR = 0.533. STJEWM trace_only: 0.578 (+4.5 pp).
STJEWM membrane_readout: 0.556 (+2.3 pp). All three env-native SR
are 0.033 (1/30 episodes — the 1/30 success is consistent with
random). The differences are within sampling noise.

## Net result

- **1/3 splits shows STJEWM clearly wins (F1, +24.4 pp on membrane)**
- **1/3 splits shows CuBiFAE wins (F2, +10 pp)**
- **1/3 splits shows all three tied within noise (F3)**

**STJEWM does not have a hard universal win on cross-benchmark-family
OOD.** The story is split — STJEWM wins on PushT (the most different
control regime from DMC), loses on TwoRoom (hierarchical navigation),
and ties on Reacher (low-dim proprio).

## Honest scope

- This is **one seed of training, three seeds of evaluation** on
  each held-out env (so per-cell is on 90 episodes).
- The cross-benchmark family axis is *not* uniformly favorable to
  STJEWM. The v0.7.10b OOD path-C result (DMC sub-family transfer,
  where STJEWM ties CuBiFAE) is consistent with this — the trace
  and membrane interfaces do not provide a universal hard performance
  win over CuBiFAE on environments that are structurally different
  from the training distribution.
- **What STJEWM provides is the membrane-forbidden protocol** — a
  methodological framework for asking *which interface* the planner
  reads. On F1, the membrane readout is dramatically better than
  the trace readout (0.400 vs 0.233), suggesting the *best interface
  choice* depends on the held-out env family.

## What this means for the working title

The cross-benchmark-family axis **does not** support "generalisable
world models" at v0.7.10b. The honest scope is: STJEWM *can*
generalise across DMC sub-families (F1/F2/F3 DMC, v0.7.10b), but
*cannot* uniformly generalise across benchmark families. The working
title is now **only supported within DMC sub-families**, not across
benchmark families.

## Files

- 3 ckpt configs: `configs/oodc/cross_benchmark_F{1,2,3}.json`
  (full split with held-out annotation) +
  `configs/oodc/cross_benchmark_F{1,2,3}_train_only.json` (list-only
  for `--multi-env-spec`)
- 9 ckpts: `results/cross_benchmark_F{1,2,3}/{stjewm_trace_only,
  stjewm_membrane_readout, cubifae_baseline}/seed_0/final.pt`
- 9 eval JSONs: `results/cross_benchmark_F{1,2,3}/eval/*.json`

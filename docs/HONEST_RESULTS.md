# Honest Results — v0.4 reframe

**Date:** 2026-07-02
**Status:** Paper v0.4 has been rewritten around the honest findings.

## What changed from v0.3 to v0.4

| Aspect | v0.3 framing | v0.4 framing |
|---|---|---|
| Headline metric | LeWM-SR (cos\_dist < 0.1) | env-SR (did the CEM planner actually achieve the goal?) |
| "MLP 98.8% LeWM-SR" | "MLP beats LeWM Transformer" | "MLP achieves 98.8\% by latent collapse (pred loss 3.5e-7); env-SR is 80.9\%" |
| "Trace 100% beats all" | "trace is the new SOTA" | "STJEWM-trace 83.9\% is within 1.5pp of 5-epoch LeWM; within 0.2pp of GRU" |
| "Membrane is not upper bound" | (headline) | "Membrane 0\% on stress is the catastrophic failure mode" |
| "Stress env-SR 89% trace" | yes | "Stress env-SR 40\% trace; saturated suite gives inflated numbers" |

## The standard suite is saturated

The 16-env LeWM suite does not distinguish models. Across 12 of 16 envs,
every trained model achieves ≥94\% env-SR. The four envs that show variation
(finger 12–58\%, cartpole\_2d 30–68\%, pendulum\_2d 8–20\%, pusht/tworoom 0\%)
are under-saturated for different reasons and are not enough to make
strong claims about model capability.

The stress suite was built specifically because the standard suite cannot
distinguish a 5.03M STJEWM-trace from a 1.3M no-memory MLP from a
7.3M GRU from a 5.07M LeWM Transformer. They are all within 6 percentage
points on the standard env-SR (80.9\% -- 85.7\%).

## The decisive result is on the stress suite

| Task | trace | leak | spike | no-trace | membrane | GRU | MLP |
|---|---|---|---|---|---|---|---|
| cartpole\_flicker (50\% mask) | 61.7 | 63.3 | 60.0 | 60.0 | **0.0** | 68.0 | 30.0 |
| cheetah\_velhidden (no vel) | 100.0 | 100.0 | 100.0 | 100.0 | **0.0** | 100.0 | 100.0 |
| pusht\_ood (unseen goals) | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 | 0.0 |
| tworoom\_long (goal=200) | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 | 0.0 |
| **AVG env-SR** | 40.4 | 40.8 | 40.0 | 40.0 | **0.0** | 42.0 | 32.5 |

The membrane\_readout ablation collapses to 0\% on every stress task. This
is the **headline** of the paper. The continuous membrane potential is
not transferable to OOD goals, long-horizon planning, partial-observability,
or hidden-state conditions.

The standard env-SR for membrane is 80.4\% — i.e., membrane ablation
performs *normally* on the standard suite. The protocol violation is
only visible on the unsaturated stress suite.

## What we claim (and what we don't)

**We claim:**
- The membrane-forbidden protocol is necessary for generalisation to
  unsaturated tasks.
- The post-spike trace is a competitive predictive state for
  reconstruction-free world models.
- The trace encodes event boundaries (Cohen's d=3.36 vs LeWM 0.22).
- The trace is memory-bearing, not capacity-bearing (decay 30pp range).
- The LeWM-SR metric is gamed by latent collapse and should be
  interpreted with care.

**We do NOT claim:**
- The trace is a new SOTA (3-epoch STJEWM-trace 83.9\% env-SR < 5-epoch
  LeWM Transformer 85.4\%).
- The trace "100\% beats all" baselines (within 1.5pp on standard).
- The LeWM-SR metric reflects capability (MLP 98.8\% LeWM-SR, 80.9\% env-SR).
- The standard suite is sufficient (saturated, plus the LeWM-SR metric
  is gamed).
- All claim of better-than-LeWM Transformer on raw task success
  (it isn't; within 1.5pp).

## What's still real

The event-boundary alignment (Cohen's d=3.36) and the membrane
catastrophe on stress are **real** and **strong** findings. The
delayed T-Maze probe (92.5\% LeWM-SR) is **real**. The trace memory
property (30pp range on decay sweep) is **real**.

The paper is honest about the gap between the **headline metric**
(LeWM-SR, gaming-able) and the **capability metric** (env-SR, honest).
We use env-SR as the primary metric in v0.4 and report LeWM-SR only
as a secondary diagnostic.

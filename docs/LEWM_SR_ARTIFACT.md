# The LeWM-SR metric is gaming-able — analysis

**Date:** 2026-07-02
**Status:** This is the analysis that motivated the v0.4 paper reframe.

## TL;DR

The 1.3M MLP (no memory) achieves **98.8\% LeWM-SR** on the 16-env
suite, beating every other model including the 5.07M LeWM Transformer.
Its env-SR is **80.9\%** — the third lowest. **The 98.8\% is a metric
artifact.**

## What is the LeWM-SR metric?

`LeWM-SR = fraction of CEM plans where cos\_dist(encode\_obs(final\_state),
encode\_obs(goal\_state)) < 0.1`

The cosine distance is in the model's latent space. If the model can
map any state to a fixed 192-dim point, then every (final, goal) pair
has cos\_dist close to 0, and LeWM-SR is 100\%.

## Why MLP achieves 98.8\%

For the MLP, `encode\_obs(s) = state\_proj(s) + FFN(state\_proj(s), 0)` is
a **stateless deterministic function of the input state only**. After
training, the MLP's JEPA self-distillation loss drops to:

| Model | pred loss (cheetah) |
|---|---|
| **MLP** | **3.5 × 10⁻⁷** |
| STJEWM-trace | 6.7 × 10⁻⁴ |

MLP is **1900x better** at the JEPA prediction task. But what is the
MLP actually learning? It is learning a **near-perfect state→latent
mapping** that maps similar states to similar latents. The MLP has no
memory, no recurrence, no trace — so it can only do this.

The result: `cos\_dist ≈ 5 × 10⁻⁶` between final and goal states.
**This is below the 0.1 LeWM-SR threshold by a factor of 20000.** Trivially
satisfied.

## What this means for the LeWM-SR metric

The LeWM-SR metric measures **latent similarity** between final and goal
states, not **plan quality**. A model that collapses the latent
representation to a single point will trivially satisfy the threshold.

**Implication for the LeWM paper (and follow-up work):**

The LeWM paper reports LeWM-SR 79.1\% for its main model. This is
**higher** than STJEWM-trace (71.6\% on the same 3-epoch retrain
budget). But the 3-epoch LeWM budget is not the 5-epoch reference;
STJEWM-trace was retrained to isolate the readout protocol effect.

The MLP's 98.8\% shows that the metric can be **gamed by stateless
models**. Future work should:

1. Report **env-SR** (env-native success rate) as the primary metric.
2. Use LeWM-SR only as a training-loss proxy.
3. Adopt the unsaturated stress suite for benchmarking.

## What is the env-SR metric?

`env-SR = fraction of CEM plans that achieve env-native success`

This measures **actual plan quality**: did the CEM-chosen action
sequence result in the agent reaching the goal in the real
environment? The model is only used to **encode** states into a
latent space where the CEM searches for actions; the env is run
directly (not through the model's predict).

On env-SR:

| Model | LeWM-SR (16 env) | env-SR (16 env) |
|---|---|---|
| MLP | 98.8 (gamed) | 80.9 (3rd lowest) |
| GRU | 87.5 | 83.7 |
| STJEWM-trace | 71.6 | 83.9 |

The MLP is good at latent collapse (LeWM-SR) but bad at planning (env-SR).
The opposite is true for the LeWM Transformer: it's the most balanced.

## Recommendation

| Use case | Recommended metric |
|---|---|
| Training loss proxy | LeWM-SR (any latent distance is fine) |
| Headline benchmark | env-SR (actual task success) |
| Architecture comparison | env-SR + stress env-SR (need unsaturated tasks) |
| Latent quality | LeWM-SR (with caveat about collapse) |
| Mechanistic analysis | linear probe, event alignment (not LeWM-SR) |

The membrane-forbidden protocol's value is in env-SR, not LeWM-SR.
The stress suite env-SR is where the membrane ablation collapses to
0\% — the protocol is **necessary** for generalisation to unsaturated
conditions, not for fitting saturated benchmarks.

## Cross-reference

- `paper/v0_draft.md` §5.1.2 — the "LeWM-SR metric can be gamed" section
- `paper/stjewm-paper.pdf` — the v0.4 paper with this analysis incorporated
- `results/aggregate/lewm_sr_vs_env_sr.md` — the side-by-side comparison
- `results/aggregate/env_sr_table.md` — the env-SR is the new primary metric

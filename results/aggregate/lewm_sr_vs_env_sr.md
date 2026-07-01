# LeWM-SR vs Env-SR: Why MLP "wins" on LeWM-SR but is a LeWM-SR artifact

## Key finding: MLP achieves 98.8% LeWM-SR by collapsing the latent space

### Evidence

| Model | LeWM-SR (16 env) | Env-SR (16 env) | Pred loss (cheetah) |
|---|---|---|---|
| STJEWM-trace | 71.6% | **73.4%** | 6.7e-4 |
| STJEWM-spike | 65.8% | 72.0% | (similar) |
| STJEWM-leak | 60.9% | 69.8% | (similar) |
| STJEWM-no-trace | 61.3% | 70.0% | (similar) |
| STJEWM-membrane | 61.0% | 70.4% | (similar) |
| **MLP** | **98.8%** | 70.8% | **3.5e-7** |
| GRU | 87.5% | 73.2% | (similar) |
| LeWM | 79.1% | 74.8% | (similar) |

### Why MLP "wins" on LeWM-SR but not on Env-SR

LeWM-SR = `cos_dist(encode_obs(final_state), encode_obs(goal_state)) < 0.1`

For MLP:
- `encode_obs(s) = state_proj(s) + FFN(state_proj(s), 0)` — a deterministic function of state only
- pred loss 3.5e-7 means the model fits the JEPA self-distillation target **trivially well**
- Because MLP is **stateless** (no recurrence), it cannot model how states *evolve* — but it can model the *distribution* of valid states
- When cheetah/ball_in_cup/fish have goal state geometrically close to typical state space, MLP maps them all to the same latent region
- cos_dist ≈ 5e-6 (essentially zero) — LeWM-SR trivially satisfied

For STJEWM:
- `encode_obs(s) = forward(s, 0)` passes through the SNN stack + gated trace + readout
- The trace accumulates, but with a=0 and 1 timestep, the trace is `r_0 = α * 0 + (1-α) * s_0 = (1-α) * s_0` — a deterministic function of s_0 too
- BUT: the SNN stack has nonlinearity + sparsity noise, so the latent is more "spread out" across state space
- pred loss 0.00067 — 1900x worse than MLP, because the model must use all its 5M params to learn a meaningful representation
- Result: cos_dist ≈ 0.08 — LeWM-SR satisfied but with less margin

### Why this matters for the paper

**The LeWM-SR metric is gamed by models that minimize latent sensitivity to action input.** MLP's 98.8% is an evaluation artifact, not a real capability.

**The honest metric is Env-SR (whether the CEM planner actually achieves the goal in the environment):**

- **Standard 16 env**: STJEWM-trace 73.4% > GRU 73.2% > MLP 70.8% > STJEWM-spike 72.0% > STJEWM-leak 69.8%
- **Stress 4 task**: GRU 42% > STJEWM-leak 40.8% > STJEWM-trace 40.4% > MLP 32.5%

**The fact that STJEWM-trace and GRU are roughly tied on env-SR shows the protocol is comparable but not magic on these tests.** The real differentiation is on **the membrane readout catastrophe (0% on stress)**.

## Recommendation for paper

1. **Replace the "MLP 98.8% wins!" framing** with "MLP achieves 98.8% LeWM-SR but only 70.8% Env-SR — the LeWM-SR metric can be gamed by latent collapse, so the real comparison is env-SR."

2. **Use env-SR as the primary metric** for both standard and stress suites.

3. **Explain the cause** of MLP's LeWM-SR in supplementary materials: pred loss 3.5e-7 shows the model has perfect JEPA prediction but the model is stateless, so the latent collapses to a low-variance region.

4. **Drop the "trace beats all" overstatement** for the standard suite — env-SR shows trace and GRU are tied. The differentiation is on:
   - **membrane catastrophe (0% on stress)** — the protocol has real value
   - **OOD pusht_ood** (env-SR 0% across the board, but LeWM-SR 0%/50%/82% shows who collapses)
   - **trace correlates with event boundaries** (0.87 vs 0.22)

5. **Keep the stress results** as the primary differentiator — that's where the membrane fails.

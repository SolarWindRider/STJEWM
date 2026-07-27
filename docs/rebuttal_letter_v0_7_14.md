# Rebuttal Letter — v0.7.14 (5M-aligned + LeWM-SR falsification)

This document holds rebuttal paragraphs for the most likely critical questions.
Each paragraph is self-contained and can be dropped into a response letter unchanged.

---

## R1. "Why not just use the MLP baseline? It's simpler."

> We agree the stateless MLP baseline is simpler; we **deliberately include it as a diagnostic probe**.
> Across the 20-env std suite, the MLP scores env-native SR = 64.7% — within 2.4pp of
> STJEWM-trace (67.1%) and within 4pp of every other world-model baseline. On LeWM-SR, the
> MLP scores **98.0%**, *higher* than every recurrent baseline (line 99,
> `results/aggregate/MASTER_TABLE.md` §2).
>
> The MLP's near-100% LeWM-SR is a measurement artefact, not a capability: its per-dim
> latent standard deviation is `0.0002` (i.e.\ the latent is a constant zero vector),
> and `cos_dist(z, z_goal) ≈ 0` for any goal, so the threshold `cos_dist < 0.1`
> is vacuously satisfied. This is the falsification that motivates §2.3a — see the paper's
> central diagnostic claim.
>
> The MLP's inclusion in our baseline set is therefore not a perf comparison; it is a
> **crash test**. The take-home is that a single latent metric (LeWM-SR alone)
> cannot diagnose calibration, and the four-metric package
> (`env-native SR`, `div`, `resp`, ρ) is what reveals MLP for what it is.

## R2. "env-SR is a saturated/leaky metric — no model has a real advantage."

> We agree. The 5-step CEM planner vs 25-step DMC goal gives env-SR = 0% for all trained
> models after bug-fix (see §2.4 bug #1 and §7.6). The env-SR column in
> `MASTER_TABLE.md` §1 reflects an *artefactual* performance floor of
> the old DMC tolerances, not a controllable task.
>
> For this reason env-SR was never the headline metric in v0.7.13 or v0.7.14. The
> *real* headline is the four-metric package — `mean_cos_dist` (raw, threshold-free)
> plus the three collapse-robust diagnostics (`div`, `resp`, ρ). On these the families
> cluster cleanly: SNN family plus CuBiFAE plus SLT-LIF-MPC at
> `mean_cos_dist ∈ [0.094, 0.116]`; MLP/GRU at `mean_cos_dist ∈ [0.000, 0.004]`
> (collapse); LeWM-v2 at `mean_cos_dist = 0.1825` (over-react).
> Env-SR remains in the table only as a sanity check, never as a claim.

## R3. "Why didn't prior SNN world models report the failure modes you report?"

> Most prior SNN world-model work (CuBiFAE, SpikeDreamer, SLT-LIF-MPC) reports
> **only** env-SR plus a planner-side metric such as LeWM-SR or its variants. They do
> not test whether the latent representation is meaningful — they measure whether
> the planner succeeds, not whether the latent actually tracks observations. Our
> contribution is to add the *representation-side* metrics (`div`, `resp`, ρ) and
> to demonstrate that the representation-side partition (collapsed vs noisy vs
> over-reactive vs calibrated) is the partition that generalises across suites
> and scaling. Prior work was measuring the wrong axis.

## R4. "Why is MLP a fair baseline if it has no event state? Isn't that stacking the deck?"

> We grant the deck-stacking concern. The point of including MLP is precisely
> **to prove that the planner-side metric is broken**. A model with no event state
> cannot, by construction, ever have non-trivial event alignment. If a planner-side
> metric rewards it the highest of any baseline, that metric is reporting on
> something other than latent quality. This is the contribution of §2.3a —
> the metric package, not any single architecture, is the unit of analysis.

## R5. "Why did you keep the headline 'competitive but not dominant on env-SR' — that reads as failure."

> The "competitive but not dominant" hedge applies to env-native success only,
> and is a **measured property of the eval pipeline**, not of the model:
> 5-step CEM plans vs 25-100 step goals do not solve DMC, regardless of what
> the latent looks like. We retain the hedge to be honest about the limitation;
> we have *not* removed it. The cost of removing it would be reporting LeWM-SR
> as a successor to env-SR — but a metric package whose headline cell can be
> satisfied by `div = 0.0002` is not a success indicator (see R1, §2.3a).
>
> We invite the reviewer to consider the four-metric package as the new headline.
> On the package, the SNN family plus CuBiFAE plus SLT-LIF-MPC form a
> calibrated cluster, MLP and GRU a collapse cluster, and LeWM-v2 alone an
> over-react cluster. This 4-family partition is the paper's central empirical claim,
> and it survives all 6 OOD splits in the v0.7.10b → v0.7.13 bug-fix re-run.

## R6. "Why is `div = 0.0002` for the MLP and not lower? What's stopping it being 0?"

> `div` is the per-dim standard deviation of the latent over a 200-step random-policy
> trajectory; it is an empirical estimate. A truly constant latent would have
> `div = 0`; the MLP at 0.0002 is *three orders of magnitude* below every other
> model (STJEWM ≈ 0.10, LeWM-v2 ≈ 0.18), so the estimate is not subtle.
> Doubling the random-policy length would shift the empirical estimate by O(1/n),
> not by an order of magnitude.

## R7. "Why is ρ almost constant (0.99) across all SNN readouts? Doesn't that wash out the protocol?"

> This is the protocol's signature, not its failure. All five forbidden readouts
> read from the **same trace dynamics**; only the variable handed to the
> predictor/planner changes. The trace is event-correlated, the planner uses it,
> and ρ ≈ 0.99 is the *calibration invariance* result (§5.5). What varies between
> readouts is the planner-side utility (§9); what is invariant is the event-latent
> alignment. We report both side of the package — invariance under protocol-tested
> readouts (strong evidence the protocol is well-posed) and divergence on
> utility axes (strong evidence the readouts are not interchangeable).

## R8. "How could MLP have 98% LeWM-SR on cartpole_2d / finger / pusht when those envs *require* event state?"

> It can't — and the env-SR column for MLP on those three envs is `0%` (cartpole_2d),
> `8%` (finger), `0%` (pusht). The MLP *gets credit* on LeWM-SR only because
> the metric is satisfied by constant latents; the env-SR reveals no such shortcut.
> The strength of the four-metric package is that the contradiction between
> LeWM-SR = 98% and env-SR = 0%/8%/0% becomes visible — a single-number
> report would have hidden it.

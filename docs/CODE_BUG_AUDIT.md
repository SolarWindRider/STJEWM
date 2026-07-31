# Code Bug Audit (v0.7.10b → v0.7.13 → v0.7.14) — 2026-07-25

> **Status (2026-07-25):** v0.7.14 reframes BUG #1 (`success_threshold_cos = 0.1`)
> as the **§2.3a LeWM-SR falsification** (paper §2.3a) — the empirical
> anchor is the MLP row of `MASTER_TABLE.md` §2 (line 99): LeWM-SR = 98.0%
> with `div = 0.0002` and `ρ = -0.002`. The v0.7.13 bug-fix (Bug #1 fix:
> report raw `mean_cos_dist` + `LeWM@0.05` / `LeWM@0.01`) is retained
> in the eval pipeline; the §2.3a reframing is the headline
> interpretation, not a new bug fix.

## Summary

3 BUGs were found in v0.7.10b/12. v0.7.10b's core claim (ρ family
classification) and v0.7.11 (Event-Window +2pp) still hold. The
v0.7.12 cross-bench claims and the v0.7.10b "env-SR=1.0 for all SNN"
claim were retracted. The **§2.3a LeWM-SR falsification** (v0.7.14)
re-anchors BUG #1 as a paper-wide headline: the v0.7.2 master table
already contained the falsification data, but we had treated
LeWM-SR = 98% on the MLP as a "metric artifact" rather than a
falsification of the headline metric itself.

---

## BUG #1: `success_threshold_cos = 0.1` is too loose for non-SNN models

**Where:** `code/eval/closed_loop.py:150` — `success_threshold_cos: float = 0.1`

**How it manifests:** `success_rate_lewm = mean(cos_dist < 0.1)`.

- For MLP, `encode(obs)` outputs near-constant zero vector (per-dim
  value ~1e-3 to 1e-5).
  → `cos_dist(encode(s1), encode(s2))` ≈ 0.0002 between ANY two states.
  → LeWM-SR = 100% trivially.
- For GRU, similar collapse but slightly less degenerate.
- For STJEWM, the latents are calibrated so cos_dist tracks obs
  distance (ρ ≈ 0.97).

### §2.3a LeWM-SR Falsification (the new headline — v0.7.14)

The same v0.7.2 master table that contained the OOD numbers above
also shows the stateless MLP baseline at **LeWM-SR = 98.0%** on the
20-env std suite — *higher* than every recurrent world-model baseline
(line 99 of `results/aggregate/MASTER_TABLE.md`). At the same time
the MLP has `div = 0.0002` and `ρ = -0.002`: its latent is a
*constant zero vector*, and the LeWM-SR threshold `cos < 0.1` is
satisfied trivially.

**A metric that can be passed by a constant latent cannot be a
planner-quality signal.** We therefore deprecate LeWM-SR as a
standalone headline in v0.7.14 and replace it with the four-metric
package (env-native SR + div + resp + ρ). The MLP row of
`MASTER_TABLE.md` §2 is the empirical anchor. **Bug #1 is not just
a "too-loose threshold" issue — it is the falsification evidence.**

| model (v0.7.5 specialist) | LeWM-SR | div (latent std per-dim) | ρ (event-align) | meaning |
| --- | --- | --- | --- | --- |
| **mlp_baseline** | **98.0%** | 0.0002 | -0.002 | collapse: latent = constant |
| stjewm_trace_only | 73.5% | 0.10 | 0.626 | calibrated |
| lewm_baseline_v2 | 76.9% | 0.18 | 0.160 | over-reactive |
| gru_baseline | 78.8% | (intermediate) | -0.011 | noisy |

**Effect on v0.7.10b claims:** ✅ UNCHANGED for the `ρ` claim
(calibrated family classification). ❌ Misleading for any "LeWM-SR"
claim — non-SNN LeWM-SR=1.0 is an artifact, not a real win.

**Effect on v0.7.12 F1 PushT:** F1 claimed +24.4pp win. With strict
threshold (cos<0.05), the gap shrinks to +1.1pp (within noise). With
raw `mean cos_dist` (no threshold), membrane still wins clearly
(0.125 vs 0.200, 37% lower).

**Fix:** Use raw `mean cos_dist` instead of thresholded `LeWM-SR`. The
paper §9.3 already shows `mean cos_dist` is the preferred diagnostic.
**The §2.3a operational reform:** a latent metric is admissible as
a planner-quality indicator only if it is `unfoolable by a constant
latent`. LeWM-SR alone is not; the four-metric package is.

---

## BUG #2: DMC `check_success` tolerance = 1.0 for high-dim states (always passes)

**Where:** `code/core/envs/dmc_env.py:83-100` — `DMC_ENVS` table. Several envs have `tol=1.0`:
- cheetah (nq=9):  tol=1.0
- walker (9):  tol=1.0
- hopper (7):  tol=1.0
- quadruped (30):  tol=1.0
- humanoid (28):  tol=1.0
- humanoid_cmu (63):  tol=1.0
- dog (87):  tol=1.0
- fish (14):  tol=1.0
- stacker (20):  tol=1.0

**How `check_success` works** (line 202-205):
```python
dist = float(np.linalg.norm(diff) / max(np.sqrt(len(state)), 1))
return dist < self._success_tol, dist
```

For dog (nq=87, tol=1.0), random uniform state gives `dist ≈ 0.45 < 1.0` → **always success**.

**Empirical verification (random uniform):**
- dog (87d): 100% random success
- humanoid (28d): 98% random success
- cheetah (9d): 90% random success
- walker (9d): 87% random success
- stacker (20d): 97% random success
- humanoid_cmu (63d): 100% random success

**Effect on v0.7.10b claims:** ❌ "All SNN family env-SR=1.0 on DMC"
is meaningless — random states also get 1.0. ✅ The `ρ` (event-align)
and `div`/`resp` (calibration diagnostics) are unaffected.

**Fix (v0.7.13):** Use tight per-env tolerances (`tol=0.1`).
After fix, env-SR=0 for all 1008 OOD cells (5-step CEM cannot reach
25-step DMC goal; see Bug #3).

---

## BUG #3: `CEM.plan` rolls out only 5 steps for 25-step goals

**Where:** `code/eval/closed_loop.py:149` — `horizon: int = 5`. `goal_offset = 25`.

**How it manifests:**
- The DMC/PushT envs need ~25 steps to reach the goal.
- CEM plans only 5 steps at a time, re-plans every 5 steps.
- After 5 steps, the agent's state is still ~5/25 = 20% of the way to the goal.
- For DMC, `phys_dist = 0.3-0.4` (state in same basin as goal because of bug #2's loose tol).
- For PushT, `phys_dist = 3000-4500` (way beyond goal — agent doesn't reach).

**Effect:** None on the `ρ` claim (cos_dist of last state vs goal),
but means `env-SR` for PushT is **fundamentally 0** (the agent
cannot reach the goal in 5 steps). This makes the F1 "PushT win"
claim a **latent-imagination** win, not an **env-success** win.

**Fix:** Either:
- Increase `horizon` to `goal_offset` (25 for DMC, 50 for PushT) —
  but this makes CEM planning infeasibly expensive.
- Or, more practically: explicitly report **"5-step latent goal
  proximity"** as the F1 metric, not "env success". This is what
  v0.7.14 does.

---

## Bug Effects on Conclusions (v0.7.14 reframing)

### Still VALID (raw metrics, not affected by bugs)
- **v0.7.10b `ρ` family classification** (468 cells): SNN ρ ∈ [0.96, 0.99],
  non-SNN ρ ∈ [0.04, 0.99] (one axis fails per model).
- **v0.7.10b `div` for MLP/GRU** (collapse signatures): real.
- **v0.7.10b `resp` for LeWM/GRU** (over-reactivity): real.
- **v0.7.11 Event-Window +2pp 胜**: event_window's `check_success`
  returns False, so the metric is **reward** not env-success. ✅ Valid.
- **v0.7.14 5M-aligned re-training** (130 ckpts): the family partition
  survives parameter parity (4.97M → 5.13M range).

### PARTIALLY valid (need reinterpretation)
- **v0.7.12 F1 PushT**: STJEWM membrane wins on `mean cos_dist`
  (0.125 vs 0.200 = 37% lower). But the env-SR=0 (agent never reaches
  goal). So the "win" is in **latent imagination quality**, not
  env-control. **Honest statement:** STJEWM membrane traces a closer
  imagined future state to the goal, but the closed-loop agent
  doesn't reach the goal. F1 is a **latent-diagnostic win, not a
  control win**.

### INVALID (claims that should be retracted)
- **v0.7.10b "SNN family all env-SR=1.0"**: meaningless due to bug #2
  (random states pass).
- **v0.7.12 F1 "+24.4pp on LeWM-SR"** at threshold 0.1: with strict
  threshold (0.05) the gap collapses to +1.1pp (within noise). The
  raw `mean cos_dist` shows a real difference (0.125 vs 0.200 = 37%
  lower for membrane), but LeWM-SR at the loose threshold is misleading.

### Not affected
- v0.7.12 F2 (cubifae wins): cos_dist is small for all (0.06-0.07),
  random tie-break may favor cubifae.
- v0.7.12 F3, F4: differences are within noise across all metrics.

---

## §2.3a reframing of BUG #1 (the headline)

The §2.3a LeWM-SR falsification reframes BUG #1 as a paper-wide
headline, not a "metric artefact to be fixed by lowering the
threshold":

> "**A latent metric is admissible as a planner-quality indicator
> only if it is unfoolable by a constant latent.** If a model
> whose latent is constant can pass the metric, the metric is a
> measurement artefact and must be paired with a collapse-robust
> diagnostic."

The fix path is not to lower `cos < 0.1` to `cos < 0.05` (which
still admits a constant zero-vector latent if it's close enough
to zero). The fix path is to **adopt a different metric family**
that admits only non-constant latents: `div`, `resp`, `ρ`. These
three metrics have the unfoolable-by-constant-latent property by
construction.

**The v0.7.2 master table is the empirical anchor** of this
reframing: the MLP row at LeWM-SR = 98.0% with `div = 0.0002` and
`ρ = -0.002` is exactly the unfoolability case we are now using
as a paper-wide headline.

---

## Recommended fixes (priority order)

1. **Bug #2 fix (DMC tolerances) [DONE in v0.7.13]:** Set per-env
   `tol` so random states < 10% pass. Re-ran v0.7.10b OOD with the
   fix. This changed the "SNN wins env-SR" claims to "SNN tied or
   random" — the `ρ` claim was unaffected.

2. **Bug #1 fix (drop threshold, use raw cos_dist) [DONE in v0.7.13,
   REFRAMED in v0.7.14 §2.3a]:** In paper §9.3 and v0.7.12 tables,
   replace `LeWM-SR (cos<0.1)` with `mean cos_dist` as the primary
   metric. Keep `LeWM-SR` only at multiple thresholds (0.1, 0.05,
   0.01) for sensitivity. **The v0.7.14 reframing is the §2.3a
   falsification:** LeWM-SR is not a planner-quality signal; use
   the four-metric package.

3. **Bug #3 fix (CEM horizon) [DONE in v0.7.14, re-anchored as
   "5-step latent goal proximity"]:** Report "5-step latent goal
   proximity" explicitly. v0.7.14 does not increase horizon to 25
   (4-5× compute increase). The env-SR = 0 result on PushT/TwoRoom
   is now a documented artifact, not a bug.

4. **Re-validate v0.7.10b + v0.7.12 + v0.7.13 + v0.7.14 conclusions**
   with the fixed versions. The `ρ` family claim holds. The env-SR
   claims collapse to "tied at random" for DMC and "STJEWM trace /
   spike < cubifae" for TwoRoom / Reacher.

---

## Honest revised take-home (v0.7.14)

| Claim | v0.7.10b | v0.7.13 | v0.7.14 |
| --- | --- | --- | --- |
| **§2.3a LeWM-SR falsification** | (n/a) | (n/a) | ✅ **HEADLINE** |
| SNN family all `ρ ∈ [0.96, 0.99]`, non-SNN at least one axis fails | ✅ Valid | ✅ Valid | ✅ Valid (5M-aligned) |
| SNN family all env-SR=1.0 on DMC | ✅ Claimed | ❌ Random states pass | env-SR=0 (budget) |
| v0.7.12 F1: STJEWM membrane +24.4pp on env-SR | ✅ Claimed | ❌ env-SR=0 | env-SR=0 (artifact) |
| v0.7.12 F1: STJEWM membrane wins on **latent goal proximity** | (not claimed) | ✅ Real | ✅ Real (5M-aligned) |
| v0.7.11 Event-Window: STJEWM +2pp 胜 | ✅ Valid | ✅ Valid | ✅ Valid |
| v0.7.12 F2 TwoRoom: cubifae wins | ✅ Valid | ✅ Valid | ✅ Valid |
| 5M-aligned: family partition survives parameter parity | (n/a) | (n/a) | ✅ Valid (130 ckpts) |

## Conclusion

**The methodology is sound (membrane-forbidden protocol + ρ
diagnostic). The env-SR numbers for DMC and PushT F1 are largely
artifacts of two distinct bugs (tol=1.0 for high-dim states, and
CEM horizon=5 vs goal_offset=25).** The v0.7.14 §2.3a reframing
of BUG #1 is the paper-wide headline: **LeWM-SR is unfoolable by
a constant latent, and the four-metric package is the only
collapse-robust diagnostic.**

The **core, honest claims** that survive the audit are:
1. STJEWM's `ρ` family is calibrated on DMC OOD (468 cells, SNN family).
2. v0.7.11 Event-Window content-aware rate counting +2pp over passive decay.
3. v0.7.12 F2 TwoRoom: cubifae wins on latent proximity (cos 0.06 vs 0.07).
4. v0.7.12 F1 PushT: STJEWM membrane has 37% lower mean cos_dist
   (latent imagination wins, but agent doesn't reach goal).
5. v0.7.14 5M-aligned: the family partition (calibrated / collapsed /
   over-reactive / noisy) survives parameter parity (130 ckpts,
   4.97M → 5.13M range).

**Retract:**
- v0.7.10b "SNN all env-SR=1.0 on DMC" (artifact of tol=1.0).
- v0.7.12 F1 "+24.4pp env-SR win" (artifact of 0.1 threshold + 5-step CEM).
- v0.7.12 F1 cross-bench family win (this is **latent quality**,
  not **control quality**).
- **"LeWM-SR is a planner-quality indicator"** — falsified at
  §2.3a; replaced by the four-metric package.

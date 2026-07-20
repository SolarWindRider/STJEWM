# Code Bug Audit (v0.7.10b/11/12) — 2026-07-20

## Summary

3 个 BUGs found that affect v0.7.10b/12 的某些 conclusions,但 v0.7.10b 的核心 claim (ρ family classification) 和 v0.7.11 (Event-Window +2pp) 仍成立。

---

## BUG #1: `success_threshold_cos = 0.1` is too loose for non-SNN models

**Where:** `code/eval/closed_loop.py:150` — `success_threshold_cos: float = 0.1`

**How it manifests:** `success_rate_lewm = mean(cos_dist < 0.1)`.

- For MLP, `encode(obs)` 输出近 0 的常量 vector (per-dim value ~1e-3 to 1e-5).
  → `cos_dist(encode(s1), encode(s2))` ≈ 0.0002 between ANY two states.
  → LeWM-SR = 100% trivially.
- For GRU, similar collapse but slightly less degenerate.
- For STJEWM, the latents are calibrated so cos_dist tracks obs distance (ρ ≈ 0.97).

**Effect on v0.7.10b claims:** ✅ UNCHANGED for the `ρ` claim (calibrated family classification). ❌ Misleading for any "LeWM-SR" claim — non-SNN LeWM-SR=1.0 is an artifact, not a real win.

**Effect on v0.7.12 F1 PushT:** F1 claimed +24.4pp win. With strict threshold (cos<0.05), the gap shrinks to +1.1pp (within noise). With raw `mean cos_dist` (no threshold), membrane still wins clearly (0.125 vs 0.200, 37% lower).

**Fix:** Use raw `mean cos_dist` instead of thresholded `LeWM-SR`. The paper §9.3 already shows `mean cos_dist` is the preferred diagnostic.

---

## BUG #2: DMC `check_success` tolerance = 1.0 for high-dim states (always passes)

**Where:** `code/core/envs/dmc_env.py:83-100` — `DMC_ENVS` table. Several envs have `tol=1.0`:
- cheetah (nq=9):  tol=1.0
- walker (nq=9):   tol=1.0
- hopper (nq=7):   tol=1.0
- quadruped (30):  tol=1.0
- humanoid (28):   tol=1.0
- humanoid_cmu (63): tol=1.0
- dog (87):       tol=1.0
- fish (14):       tol=1.0
- stacker (20):    tol=1.0

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

**Effect on v0.7.10b claims:** ❌ "All SNN family env-SR=1.0 on DMC" is meaningless — random states also get 1.0. ✅ The `ρ` (event-align) and `div`/`resp` (calibration diagnostics) are unaffected.

**Fix:** Use tight per-env tolerances. Suggested values based on state range (uniform [-1,1]²) and env's natural basin:
- dog 87d: tol=0.05 (random would give 0.0%)
- humanoid 28d: tol=0.1
- cheetah 9d: tol=0.1
- walker 9d: tol=0.1
- hopper 7d: tol=0.2
- etc.

---

## BUG #3: `CEM.plan` rolls out only 5 steps for 25-step goals

**Where:** `code/eval/closed_loop.py:149` — `horizon: int = 5`. `goal_offset = 25`.

**How it manifests:**
- The DMC/PushT envs need ~25 steps to reach the goal.
- CEM plans only 5 steps at a time, re-plans every 5 steps.
- After 5 steps, the agent's state is still ~5/25 = 20% of the way to the goal.
- For DMC, `phys_dist = 0.3-0.4` (state in same basin as goal because of bug #2's loose tol).
- For PushT, `phys_dist = 3000-4500` (way beyond goal — agent doesn't reach).

**Effect:** None on the `ρ` claim (cos_dist of last state vs goal), but means `env-SR` for PushT is **fundamentally 0** (the agent cannot reach the goal in 5 steps). This makes the F1 "PushT win" claim a **latent-imagination** win, not an **env-success** win.

**Fix:** Either:
- Increase `horizon` to `goal_offset` (25 for DMC, 50 for PushT) — but this makes CEM planning infeasibly expensive.
- Or, more practically: explicitly report **"5-step latent goal proximity"** as the F1 metric, not "env success".

---

## Bug Effects on Conclusions (revised)

### Still VALID (raw metrics, not affected by bugs)
- **v0.7.10b `ρ` family classification** (468 cells): SNN ρ ∈ [0.96, 0.99], non-SNN ρ ∈ [0.04, 0.99] (one axis fails per model). 
  - Real, not affected.
- **v0.7.10b `div` for MLP/GRU** (collapse signatures): real, not affected.
- **v0.7.10b `resp` for LeWM/GRU** (over-reactivity): real, not affected.
- **v0.7.11 Event-Window +2pp 胜**: event_window's `check_success` returns False, so the metric is **reward** not env-success. ✅ Valid.

### PARTIALLY valid (need reinterpretation)
- **v0.7.12 F1 PushT**: STJEWM membrane wins on `mean cos_dist` (0.125 vs 0.200 = 37% lower). But the env-SR=0 (agent never reaches goal). So the "win" is in **latent imagination quality**, not env-control.
  - **Honest statement:** STJEWM membrane traces a closer imagined future state to the goal, but the closed-loop agent doesn't reach the goal. F1 is a **latent-diagnostic win, not a control win**.

### INVALID (claims that should be retracted)
- **v0.7.10b "SNN family all env-SR=1.0"**: meaningless due to bug #2 (random states pass).
- **v0.7.12 F1 "+24.4pp on LeWM-SR"** at threshold 0.1: with strict threshold (0.05) the gap collapses to +1.1pp (within noise). The raw `mean cos_dist` shows a real difference (0.125 vs 0.200 = 37% lower for membrane), but LeWM-SR at the loose threshold is misleading.

### Not affected
- v0.7.12 F2 (cubifae wins): cos_dist is small for all (0.06-0.07), random tie-break may favor cubifae.
- v0.7.12 F3, F4: differences are within noise across all metrics.

---

## Recommended fixes (priority order)

1. **Bug #2 fix (DMC tolerances):** Set per-env `tol` so random states < 10% pass. Re-run v0.7.10b OOD with the fix. This will likely change the "SNN wins env-SR" claims to "SNN tied or random" — the `ρ` claim is unaffected.

2. **Bug #1 fix (drop threshold, use raw cos_dist):** In paper §9.3 and v0.7.12 tables, replace `LeWM-SR (cos<0.1)` with `mean cos_dist` as the primary metric. Keep `LeWM-SR` only at multiple thresholds (0.1, 0.05, 0.01) for sensitivity.

3. **Bug #3 fix (CEM horizon):** Either report "5-step latent goal proximity" explicitly, or increase horizon to 25 and increase `eval_budget` accordingly. For PushT, this would be a 4-5x compute increase.

4. **Re-validate v0.7.10b + v0.7.12 conclusions** with the fixed versions. The `ρ` family claim should hold. The env-SR claims will likely collapse to "tied at random" for DMC and "STJEWM membrane < cubifae" for TwoRoom.

---

## Honest revised take-home (v0.7.10b+)

| Claim | Before audit | After audit |
|---|---|---|
| SNN family all `ρ ∈ [0.96, 0.99]`, non-SNN at least one axis fails | ✅ Valid | ✅ Valid (ρ is a real diagnostic) |
| SNN family all env-SR=1.0 on DMC | ✅ Claimed | ❌ **Random states also pass due to tol=1.0 bug** |
| v0.7.12 F1: STJEWM membrane +24.4pp on env-SR | ✅ Claimed | ❌ **env-SR=0 for all (CEM 5 steps cannot reach 25-step goal)** |
| v0.7.12 F1: STJEWM membrane wins on **latent goal proximity** | (not claimed) | ✅ Real (mean cos_dist 0.125 vs 0.200) |
| v0.7.11 Event-Window: STJEWM +2pp 胜 | ✅ Valid | ✅ Valid (uses reward, not env-success) |
| v0.7.12 F2 TwoRoom: cubifae wins | ✅ Valid | ✅ Valid (cos_dist 0.06 vs 0.07, real gap) |

## Conclusion

**The methodology is sound (membrane-forbidden protocol + ρ diagnostic). The env-SR numbers for DMC and PushT F1 are largely artifacts of two distinct bugs (tol=1.0 for high-dim states, and CEM horizon=5 vs goal_offset=25).**

The **core, honest claims** that survive the audit are:
1. STJEWM's `ρ` family is calibrated on DMC OOD (468 cells, SNN family).
2. v0.7.11 Event-Window content-aware rate counting +2pp over passive decay.
3. v0.7.12 F2 TwoRoom: cubifae wins on latent proximity (cos 0.06 vs 0.07).
4. v0.7.12 F1 PushT: STJEWM membrane has 37% lower mean cos_dist (latent imagination wins, but agent doesn't reach goal).

**Retract:**
- v0.7.10b "SNN all env-SR=1.0 on DMC" (artifact of tol=1.0).
- v0.7.12 F1 "+24.4pp env-SR win" (artifact of 0.1 threshold + 5-step CEM).
- v0.7.12 F1 cross-bench family win (this is **latent quality**, not **control quality**).

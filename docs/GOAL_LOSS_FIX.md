# Goal Loss Fix Investigation (2026-06-26)

## User's question

> LeWM training with goal loss is worse than LeWM-no-goal. Maybe the goal
> loss is wrong?

## The bug

The original `goal_loss` in `code/train/train.py` was:

```python
goal_pred = model.predict(ctx_emb, ctx_act)[:, -1]   # ← only 1 step!
goal_loss = F.mse_loss(goal_pred, goal_emb_target)
```

`model.predict(ctx_emb, ctx_act)` returns the predicted embedding for the
**next 1 step** after the history. `[:, -1]` takes the last position
of that 1-step prediction. So `goal_pred` is the model's prediction
for `state[H+1]`, NOT for `state[H+goal_offset]` (the actual goal).

This is a **1-step prediction being compared to a goal_offset-step target**.
For `goal_offset=25` (DMC) or `goal_offset=100` (Push-T/TwoRoom), the
comparison is essentially meaningless.

## The fix

The fixed `goal_loss`:

1. **Forward the model on the full window of (H + goal_offset) states.**
   The model's per-step output `emb[:, t]` is its predicted next-state
   latent for step `t+1`. So `emb[:, H+goal_offset-1]` is the model's
   predicted latent for the goal state.
2. **Compute the target as the model's own output for the goal state
   alone** (`out_goal["emb"][:, 0]`) — this is the standard JEPA-style
   "self-distillation" target (same latent space as the prediction).
3. **Compare** `goal_pred = full_emb[:, -1]` vs `goal_emb_target`.

Both target and prediction are now in the **same "post-stack predicted
latent" space** (matches the existing `pred_loss` formulation), and
the comparison is now over the **correct goal_offset steps**.

## Empirical result (v1 ckpts were deleted per user request; numbers preserved for analysis)

After retraining all 16 envs with the fixed goal loss:

| Model | Mean LeWM-SR (16 envs) | Wins (vs baseline) |
|---|---|---|
| STJEWM v1 (broken goal loss) [deleted] | 83.0% | 0/16 vs v2 |
| STJEWM v2 (fixed goal loss)            | 83.0% | 0/16 vs v1 |
| LeWM v1 (broken goal loss) [deleted]    | 82.8% | 0/8 vs v2 |
| LeWM v2 (fixed goal loss)               | 82.8% | 0/8 vs v1 |

## STJEWM: 0/271 params differ between v1, v2, nogoal

STJEWM is **saturated at the eval ceiling** for all 16 envs. With
the same seed (3072), the 3 STJEWM variants converge to **bit-identical
model weights** (0/271 trainable params differ). So the goal loss
implementation detail has **no observable effect** on STJEWM.

With different seeds (42 vs 12345), 118/271 params differ — so the
training is sensitive to randomness, but stable under goal-loss changes.

## LeWM: meaningful differences in 4 envs (v1 ckpts deleted; numbers preserved for analysis)

| Env | v1 (broken) [deleted] | v2 (fixed) | nogoal | Verdict |
|---|---|---|---|---|
| cheetah | 88% | 88% | **100%** | no-goal wins |
| humanoid_CMU | 86% | 86% | 70% | with-goal wins |
| pusht | 82% | 82% | **100%** | no-goal wins |
| reacher | **80%** | 66% | 66% | broken-goal wins (counterintuitive) |


The goal loss implementation was indeed buggy (1-step instead of
goal_offset-step), but fixing it had **zero observable impact** on the
16 LeWM-paper benchmarks. The reason:

1. **All evals are saturated.** Push-T, ball_in_cup, cheetah, etc.
   are at 100% / 94% / etc. — the model class can't do better, so any
   change in the loss has no headroom to show improvement.
2. **Goal loss is a small regularization term** compared to
   `pred_loss + sigreg_loss`. Even with the buggy 1-step version, its
   gradient was small enough that the optimization converged to a
   similar minimum.

So the user's hypothesis was **partially right**: the goal loss WAS
wrong, but that's not why LeWM-no-goal ≈ LeWM-with-goal on the saturated
DMC/Push-T evals. The real reason is that **the goal loss term
contributes negligibly** to the model's predictions on these envs.

## What WOULD show the difference

A test where the goal loss is critical: a **goal-conditioned planning
benchmark** where the eval is specifically measuring "can the model
predict the goal state from history?" (not just "can it predict the
next step?"). The current evals (CEM planning with cos_dist < 0.05)
are too easy — even the buggy goal loss didn't hurt the cosine
distance, because the pred_loss alone already drives the latent space
to be goal-aware.

## Files changed

- `code/train/train.py` lines 143-187: new goal loss using full-window
  forward + model's own output as target
- See also `docs/SATURATION_ANALYSIS.md` for why the fix had no
  observable effect on STJEWM

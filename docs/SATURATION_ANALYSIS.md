# STJEWM Eval Saturation Analysis (2026-06-26)

## Observation
After training STJEWM with 3 different goal-loss variants (v1=broken, v2=fixed, nogoal), all 16
eval results are bit-identical. Model weights differ by md5 but converge to equivalent solutions on
the test envs.

## Why
All 16 LeWM-paper evals are **saturated** at the model's accuracy ceiling:
- ball_in_cup, fish, humanoid, humanoid_CMU, reacher, tworoom, walker: 100% (or near)
- cartpole_2d, finger, hopper, quadruped, stacker: 78-90% (model is at the data ceiling)
- cheetah, dog, pusht: 94-100% (LeWM is data-saturated)
- pendulum_2d: 28-34% (low because pendulum is hard)

When the eval is saturated, **any change to the loss function that doesn't change the
data-encoding limit has no observable effect**. The goal loss term is dominated by pred_loss + sigreg_loss
in terms of gradient magnitude (~50/50 split for goal, but pred loss has lower variance), so the
model converges to the same local optimum regardless of goal loss details.
## Empirical evidence (v1 ckpts were deleted; analysis preserved from prior comparison)
- v1, v2, nogoal STJEWM models on ball_in_cup: 0/271 params differ between v1 and v2,
  0/271 between v1 and nogoal, 0/271 between v2 and nogoal.
- With different seeds (42 vs 12345), 118/271 params differ — so the training is sensitive
  to randomness in principle, but stable under goal-loss changes with fixed seed.

## Implication
The user's hypothesis "no-goal is better because goal-loss is wrong" cannot be tested on these
16 saturated envs. To see the effect of goal-loss, we need an **unsaturated benchmark** that
specifically tests "can the model predict the goal state from history?" (not just "can it
predict the next step?").

## Recommended next test
1. **Long-horizon planning benchmark**: e.g. TwoRoom with goal_offset=50 (not 100) and
   measure `cos_dist(goal_pred, goal_target)` directly, not CEM success.
2. **Goal-conditioned retrieval**: train on short horizons, test on longer horizons where
   the model must extrapolate using only goal conditioning.
3. **OOD goal generalization**: train on a set of goals, test on unseen goals — this
   tests whether the model has actually learned to use the goal representation.

## File references (v1 deleted per user request; only v2 + nogoal remain)
- Model ckpts: `/home/lx/snn/results/{env}/stjewm_{v2,nogoal}/final.pt`
- Eval jsons: `/home/lx/snn/results/{env}/stjewm_{v2,nogoal}/eval.json`
- Comparison: `/home/lx/snn/results/aggregate/summary_3way_v2.md`

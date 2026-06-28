# 4-Condition Comparison (post-fix only)

Only valid (post-fix) variants. v1 (broken goal) deleted per user request.

Four metrics per model × env. Lower is better for cos_dist/phys_dist; higher is better for SR.

Note: phys_dist AVG uses **median** (not mean) because pusht (~1000) and tworoom (~100) would dominate the scale and hide other-env differences.

## LeWM-SR (%)

| Env |STJEWM (with goal) | STJEWM (no goal) | LeWM (with goal) | LeWM (no goal) |
|---|---|---|---|---|
| ball_in_cup |   100% |   100% |   100% |   100% |
| cartpole_2d |    86% |    86% |    86% |    86% |
| cheetah |    94% |    94% |    88% |   100% |
| dog |    78% |    78% |    68% |    68% |
| finger |    78% |    78% |    78% |    78% |
| fish |    98% |    98% |    98% |    98% |
| hopper |    90% |    90% |    88% |    88% |
| humanoid |    70% |    70% |    56% |    56% |
| humanoid_CMU |    86% |    86% |    86% |    70% |
| pendulum_2d |    34% |    34% |    28% |    28% |
| pusht |    96% |    96% |    82% |   100% |
| quadruped |    78% |    78% |    86% |    86% |
| reacher |    64% |    64% |    66% |    66% |
| stacker |    86% |    86% |    88% |    88% |
| tworoom |    94% |    94% |    74% |    74% |
| walker |    90% |    90% |    94% |    94% |
| **AVG** | **   83%** | **   83%** | **   79%** | **   80%** |

## Env-SR (%)

| Env |STJEWM (with goal) | STJEWM (no goal) | LeWM (with goal) | LeWM (no goal) |
|---|---|---|---|---|
| ball_in_cup |   100% |   100% |   100% |   100% |
| cartpole_2d |    38% |    38% |    36% |    36% |
| cheetah |   100% |   100% |   100% |   100% |
| dog |   100% |   100% |   100% |   100% |
| finger |    58% |    58% |    58% |    58% |
| fish |    98% |    98% |    98% |    98% |
| hopper |    96% |    96% |    96% |    96% |
| humanoid |   100% |   100% |   100% |   100% |
| humanoid_CMU |   100% |   100% |   100% |   100% |
| pendulum_2d |    20% |    20% |    20% |    20% |
| pusht |     0% |     0% |     0% |     0% |
| quadruped |    96% |    96% |    96% |    96% |
| reacher |   100% |   100% |   100% |   100% |
| stacker |    94% |    94% |    94% |    94% |
| tworoom |     0% |     0% |     0% |     0% |
| walker |    98% |    98% |    98% |    98% |
| **AVG** | **   75%** | **   75%** | **   75%** | **   75%** |

## cos_dist

| Env |STJEWM (with goal) | STJEWM (no goal) | LeWM (with goal) | LeWM (no goal) |
|---|---|---|---|---|
| ball_in_cup | 0.000 | 0.000 | 0.000 | 0.001 |
| cartpole_2d | 0.068 | 0.068 | 0.060 | 0.060 |
| cheetah | 0.030 | 0.030 | 0.053 | 0.075 |
| dog | 0.071 | 0.071 | 0.110 | 0.110 |
| finger | 0.108 | 0.108 | 0.095 | 0.095 |
| fish | 0.013 | 0.013 | 0.012 | 0.012 |
| hopper | 0.031 | 0.031 | 0.033 | 0.033 |
| humanoid | 0.091 | 0.091 | 0.118 | 0.118 |
| humanoid_CMU | 0.038 | 0.038 | 0.034 | 0.072 |
| pendulum_2d | 0.215 | 0.215 | 0.254 | 0.254 |
| pusht | 0.028 | 0.028 | 0.052 | 0.047 |
| quadruped | 0.073 | 0.073 | 0.053 | 0.053 |
| reacher | 0.156 | 0.156 | 0.159 | 0.159 |
| stacker | 0.043 | 0.043 | 0.042 | 0.042 |
| tworoom | 0.050 | 0.050 | 0.078 | 0.078 |
| walker | 0.032 | 0.032 | 0.026 | 0.026 |
| **AVG** | **0.065** | **0.065** | **0.074** | **0.077** |

## phys_dist

| Env |STJEWM (with goal) | STJEWM (no goal) | LeWM (with goal) | LeWM (no goal) |
|---|---|---|---|---|
| ball_in_cup |   0.02 |   0.02 |   0.01 |   0.02 |
| cartpole_2d |   1.11 |   1.11 |   1.12 |   1.12 |
| cheetah |   0.15 |   0.15 |   0.15 |   0.17 |
| dog |   0.26 |   0.26 |   0.26 |   0.26 |
| finger |   0.55 |   0.55 |   0.54 |   0.54 |
| fish |   0.47 |   0.47 |   0.47 |   0.47 |
| hopper |   0.35 |   0.35 |   0.34 |   0.34 |
| humanoid |   0.49 |   0.49 |   0.47 |   0.47 |
| humanoid_CMU |   0.19 |   0.19 |   0.18 |   0.31 |
| pendulum_2d |   1.57 |   1.57 |   1.59 |   1.59 |
| pusht | 1053.87 | 1053.87 | 1053.87 | 707.36 |
| quadruped |   0.35 |   0.35 |   0.35 |   0.35 |
| reacher |   0.00 |   0.00 |   0.00 |   0.00 |
| stacker |   0.15 |   0.15 |   0.15 |   0.15 |
| tworoom | 100.69 | 100.69 | 101.01 | 101.01 |
| walker |   0.30 |   0.30 |   0.29 |   0.29 |
| **AVG** | **  0.35 (median)** | **  0.35 (median)** | **  0.35 (median)** | **  0.35 (median)** |

## Metric definitions
- **LeWM-SR (%)**: Fraction of CEM plans whose final latent is within cos_dist < 0.1 of goal latent (LeWM paper primary metric).
- **Env-SR (%)**: Fraction of plans that achieve env-native goal (env-specific success criterion).
- **cos_dist**: Mean (1-cos_sim)/2 between final latent and goal latent. Lower is better. 0 = identical, 1 = orthogonal.
- **phys_dist**: Mean physical distance between plan trajectory and goal. Lower is better. Scale varies by env (DMC ~0-2, pusht ~0-1000, tworoom ~100).

## Analysis
Compare with-goal vs no-goal for both STJEWM and LeWM:
- If no-goal ≈ with-goal: goal loss term is negligible on these evals.
- If no-goal < with-goal: goal loss DOES help (improves SR / reduces distance).
- If no-goal > with-goal: goal loss HURTS on this env.

v1 (broken goal) was deleted per user request — see `docs/GOAL_LOSS_FIX.md` for the bug analysis.
Tworoom was previously NaN due to a missing env.reset() call (see `docs/TWOROOM_BUGFIX.md`); now fixed.
# 5-Condition Comparison (post-bugfix)

STJEWM under 4 readout modes + LeWM baseline.

Note: STJEWM-{trace,leak,spike,no_trace} are retrained on the same 16-env suite
with the same hyper-params (3 epochs, batch 64, max_windows=10000).
Their avg LeWM-SR is the membrane-forbidden protocol headline (Sec. 4.1, Table 1).

## LeWM-SR (%)

| Env | STJEWM-trace | STJEWM-leak | STJEWM-spike | STJEWM-no-trace | LeWM |
|---|---|---|---|---|---|
| ball_in_cup |  100.0% |  100.0% |  100.0% |    — |  100.0% |
| cartpole_2d |   82.0% |   82.0% |   86.0% |    — |   86.0% |
| cheetah |   98.0% |   90.0% |   98.0% |    — |   88.0% |
| dog |   26.0% |    2.0% |   20.0% |    — |   68.0% |
| finger |   44.0% |   48.0% |   46.0% |    — |   78.0% |
| fish |   98.0% |   98.0% |   98.0% |    — |   98.0% |
| hopper |   88.0% |   78.0% |   88.0% |    — |   88.0% |
| humanoid |   38.0% |    4.0% |   10.0% |    — |   56.0% |
| humanoid_CMU |   86.0% |   86.0% |   86.0% |    — |   86.0% |
| pendulum_2d |   26.0% |   24.0% |   26.0% |    — |   28.0% |
| pusht |   74.0% |   10.0% |   42.0% |    — |   82.0% |
| quadruped |   80.0% |   74.0% |    — |    — |   86.0% |
| reacher |   54.0% |   28.0% |   14.0% |    — |   66.0% |
| stacker |   86.0% |   86.0% |   86.0% |    — |   88.0% |
| tworoom |   92.0% |   94.0% |   90.0% |    — |   74.0% |
| walker |   74.0% |   70.0% |   82.0% |    — |   94.0% |
| **AVG** | **  71.6%** | **  60.9%** | **  64.8%** | **   —** | **  79.1%** |

## Env-SR (%)

| Env | STJEWM-trace | STJEWM-leak | STJEWM-spike | STJEWM-no-trace | LeWM |
|---|---|---|---|---|---|
| ball_in_cup |  100.0% |  100.0% |  100.0% |    — |  100.0% |
| cartpole_2d |   60.0% |   42.0% |   64.0% |    — |   36.0% |
| cheetah |  100.0% |  100.0% |  100.0% |    — |  100.0% |
| dog |  100.0% |  100.0% |  100.0% |    — |  100.0% |
| finger |   18.0% |    8.0% |    4.0% |    — |   58.0% |
| fish |   98.0% |   98.0% |   98.0% |    — |   98.0% |
| hopper |   96.0% |   92.0% |   94.0% |    — |   96.0% |
| humanoid |  100.0% |   84.0% |   98.0% |    — |  100.0% |
| humanoid_CMU |  100.0% |  100.0% |  100.0% |    — |  100.0% |
| pendulum_2d |   14.0% |    8.0% |    8.0% |    — |   20.0% |
| pusht |    0.0% |    0.0% |    0.0% |    — |    0.0% |
| quadruped |   96.0% |   96.0% |    — |    — |   96.0% |
| reacher |  100.0% |  100.0% |  100.0% |    — |  100.0% |
| stacker |   94.0% |   94.0% |   94.0% |    — |   94.0% |
| tworoom |    0.0% |    0.0% |    0.0% |    — |    0.0% |
| walker |   98.0% |   94.0% |   96.0% |    — |   98.0% |
| **AVG** | **  73.4%** | **  69.8%** | **  70.4%** | **   —** | **  74.8%** |

## cos_dist

| Env | STJEWM-trace | STJEWM-leak | STJEWM-spike | STJEWM-no-trace | LeWM |
|---|---|---|---|---|---|
| ball_in_cup |  0.000 |  0.001 |  0.001 |    — |  0.000 |
| cartpole_2d |  0.070 |  0.073 |  0.059 |    — |  0.060 |
| cheetah |  0.040 |  0.058 |  0.038 |    — |  0.053 |
| dog |  0.145 |  0.182 |  0.138 |    — |  0.110 |
| finger |  0.185 |  0.174 |  0.177 |    — |  0.095 |
| fish |  0.014 |  0.032 |  0.030 |    — |  0.012 |
| hopper |  0.042 |  0.057 |  0.039 |    — |  0.033 |
| humanoid |  0.117 |  0.214 |  0.183 |    — |  0.118 |
| humanoid_CMU |  0.039 |  0.062 |  0.042 |    — |  0.034 |
| pendulum_2d |  0.208 |  0.221 |  0.209 |    — |  0.254 |
| pusht |  0.083 |  0.230 |  0.121 |    — |  0.052 |
| quadruped |  0.076 |  0.100 |    — |    — |  0.053 |
| reacher |  0.201 |  0.202 |  0.272 |    — |  0.159 |
| stacker |  0.043 |  0.044 |  0.044 |    — |  0.042 |
| tworoom |  0.047 |  0.050 |  0.054 |    — |  0.078 |
| walker |  0.070 |  0.077 |  0.057 |    — |  0.026 |
| **AVG** | ** 0.086** | ** 0.111** | ** 0.098** | **   —** | ** 0.074** |

## phys_dist

| Env | STJEWM-trace | STJEWM-leak | STJEWM-spike | STJEWM-no-trace | LeWM |
|---|---|---|---|---|---|
| ball_in_cup |    0.02 |    0.02 |    0.03 |    — |    0.01 |
| cartpole_2d |    0.95 |    1.03 |    0.89 |    — |    1.12 |
| cheetah |    0.18 |    0.24 |    0.18 |    — |    0.15 |
| dog |    0.40 |    0.49 |    0.43 |    — |    0.26 |
| finger |    0.89 |    0.96 |    0.91 |    — |    0.54 |
| fish |    0.48 |    0.57 |    0.55 |    — |    0.47 |
| hopper |    0.38 |    0.48 |    0.39 |    — |    0.34 |
| humanoid |    0.60 |    0.88 |    0.76 |    — |    0.47 |
| humanoid_CMU |    0.22 |    0.35 |    0.24 |    — |    0.18 |
| pendulum_2d |    1.53 |    1.60 |    1.55 |    — |    1.59 |
| pusht |  983.37 | 4572.33 | 4600.72 |    — | 1053.87 |
| quadruped |    0.36 |    0.42 |    — |    — |    0.35 |
| reacher |    0.00 |    0.00 |    0.00 |    — |    0.00 |
| stacker |    0.15 |    0.17 |    0.15 |    — |    0.15 |
| tworoom |   94.74 |  100.69 |  110.86 |    — |  101.01 |
| walker |    0.44 |    0.54 |    0.42 |    — |    0.29 |
| **AVG** | **   0.44 (med)** | **   0.54 (med)** | **   0.43 (med)** | **   —** | **   0.35 (med)** |

## Metric definitions
- **LeWM-SR (%)**: Fraction of CEM plans whose final latent is within cos_dist < 0.1 of goal latent (LeWM paper primary metric).
- **Env-SR (%)**: Fraction of plans that achieve env-native goal.
- **cos_dist**: Mean (1-cos_sim)/2. Lower is better. 0 = identical, 1 = orthogonal.
- **phys_dist**: Mean physical distance. Lower is better. AVG uses median to avoid pusht/tworoom dominating the scale.

## Reading the table
- **STJEWM-trace** is the only model that respects the membrane-forbidden protocol (Sec. 2.1).
- **STJEWM-leak** is the legacy default (hidden + trace).
- **STJEWM-spike** masks the hidden state by the post-spike activation.
- **STJEWM-no-trace** drops the trace branch entirely (ablation).
- **LeWM** is a 4-layer Transformer + AdaLN-zero with no trace.

# Linear Probing Results (R² score, higher = better)

Envs: ball_in_cup, cartpole_2d, cheetah, finger, hopper, humanoid, pendulum_2d, pusht, quadruped, reacher, tworoom, walker
Targets: future_k, goal_direction, position

| env | lewm_baseline_no_goal\future_k | lewm_baseline_no_goal\goal_direction | lewm_baseline_no_goal\position | lewm_baseline_v2\future_k | lewm_baseline_v2\goal_direction | lewm_baseline_v2\position | stjewm_nogoal\future_k | stjewm_nogoal\goal_direction | stjewm_nogoal\position | stjewm_v2\future_k | stjewm_v2\goal_direction | stjewm_v2\position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ball_in_cup | 0.017 | 0.032 | 0.078 | 0.004 | 0.045 | 0.048 | -0.032 | -0.005 | -0.028 | -0.036 | -0.004 | -0.029 |
| cartpole_2d | 0.428 | -0.012 | 0.661 | 0.467 | -0.107 | 0.633 | 0.725 | -0.168 | 0.926 | 0.748 | -0.157 | 0.834 |
| cheetah | 0.068 | 0.372 | 0.721 | 0.072 | 0.367 | 0.703 | -0.044 | 0.057 | 0.062 | -0.036 | 0.055 | 0.068 |
| finger | -117.553 | -20.062 | -125.967 | -94.299 | -18.980 | -137.652 | -214.473 | -20.293 | -240.205 | -213.346 | -19.005 | -262.414 |
| hopper | 0.385 | -0.184 | 0.430 | 0.292 | -0.278 | 0.408 | 0.152 | -0.059 | 0.257 | 0.136 | -0.058 | 0.287 |
| humanoid | — | — | — | — | — | -0.735 | -0.677 | -0.038 | -0.765 | -0.714 | -0.031 | -0.802 |
| pendulum_2d | 0.543 | 0.407 | 0.993 | 0.528 | 0.400 | 0.992 | 0.443 | 0.366 | 0.887 | 0.442 | 0.367 | 0.929 |
| pusht | 0.049 | 0.406 | 0.109 | 0.052 | 0.054 | 0.105 | -0.007 | -0.060 | -0.009 | 0.077 | 0.120 | -0.033 |
| quadruped | -0.140 | -0.123 | -0.006 | -0.202 | -0.149 | -0.036 | -0.420 | -0.117 | -0.350 | -0.344 | -0.132 | -0.357 |
| reacher | 0.517 | 0.059 | 0.769 | 0.493 | 0.045 | 0.761 | 0.403 | 0.034 | 0.594 | 0.405 | 0.034 | 0.581 |
| tworoom | — | — | — | — | — | — | -16.069 | -0.703 | -10.464 | -11.535 | -0.198 | -0.835 |
| walker | 0.403 | -0.040 | 0.569 | 0.451 | -0.048 | 0.578 | 0.099 | -0.009 | 0.162 | 0.097 | -0.010 | 0.190 |

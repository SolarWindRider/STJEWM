# S3 action sensitivity + S6 EMA control (stab0917_b)

**SMOKE RUN** (4 contexts x 2 pairs, 1 closed-loop episode).

Contexts/env: 4 x 2 action pairs; persistence horizon h=1..10.

## Part A: one-step action sensitivity S1step

S1step = ||dz|| / ||(a1-a2)/act_std||; `rel` additionally normalizes dz by the prediction norm (cross-model comparable).

| model | S1step raw (pooled) | S1step rel (pooled) | latent norm |
|---|---|---|---|
| stjewm_trace_only | 0.0042 | 0.565 | 0.00749 |
| lewm_baseline_v2 | 0.0969 | 0.00718 | 13.6 |

## Part A: Impulse persistence (action differs only at step 1)

Curves pooled over envs. rel_to_h1 = ||dz_h||/||dz_1|| (wash-out shape); rel_to_latent = ||dz_h||/||z_base_h|| (cross-model comparable).

### rel_to_h1

| model | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 | h=8 | h=9 | h=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 1 | 0.497 | 0.697 | 0.745 | 0.879 | 1.47 | 1.21 | 1.1 | 1.31 | 0.814 |
| lewm_baseline_v2 | 1 | 0.705 | 0.569 | 0.498 | 0.464 | 0.45 | 0.446 | 0.445 | 0.439 | 0.427 |

### rel_to_latent

| model | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 | h=8 | h=9 | h=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.387 | 0.163 | 0.263 | 0.262 | 0.333 | 0.528 | 0.45 | 0.411 | 0.458 | 0.308 |
| lewm_baseline_v2 | 0.00493 | 0.00348 | 0.0028 | 0.00246 | 0.00229 | 0.00222 | 0.0022 | 0.00219 | 0.00217 | 0.0021 |


## Part A: Sustained (center+-d at every step)

Curves pooled over envs. rel_to_h1 = ||dz_h||/||dz_1|| (wash-out shape); rel_to_latent = ||dz_h||/||z_base_h|| (cross-model comparable).

### rel_to_h1

| model | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 | h=8 | h=9 | h=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 1 | 0.942 | 1.14 | 1.29 | 1.54 | 1.69 | 1.64 | 1.7 | 1.91 | 2.16 |
| lewm_baseline_v2 | 1 | 1.61 | 2.08 | 2.47 | 2.8 | 3.09 | 3.33 | 3.51 | 3.63 | 3.66 |

### rel_to_latent

| model | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 | h=8 | h=9 | h=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.387 | 0.309 | 0.431 | 0.454 | 0.582 | 0.608 | 0.611 | 0.635 | 0.665 | 0.817 |
| lewm_baseline_v2 | 0.00493 | 0.00792 | 0.0103 | 0.0122 | 0.0138 | 0.0152 | 0.0164 | 0.0173 | 0.0179 | 0.018 |

## Part B(a): EMA response matching (LeWM vs trace)

Response = ||dz|| of the one-step predicted latent under last-frame obs perturbation (+0.1*obs std), with EMA z' = alpha*p + (1-alpha)*z_anchor. Primary table: SHARED pre-perturbation anchor (resp = alpha * resp(1)).

| alpha | LeWM resp raw | LeWM resp rel | trace resp raw | trace resp rel |
|---|---|---|---|---|
| 0.1 | 0.766 | 0.055 | 1.04e-08 | 1.12e-06 |
| 0.3 | 2.3 | 0.165 | 3.1e-08 | 4.04e-06 |
| 0.5 | 3.83 | 0.276 | 5.15e-08 | 7.84e-06 |
| 0.7 | 5.36 | 0.386 | 7.21e-08 | 1.16e-05 |
| 0.9 | 6.89 | 0.495 | 9.27e-08 | 1.37e-05 |
| 1.0 | 7.66 | 0.55 | 1.03e-07 | 1.4e-05 |

Perturbed-anchor diagnostic (encoder pathway, EMA-immune), LeWM: 7.58..7.66 raw across alpha; trace: 0.00207..1.03e-07. Encoder-only response: LeWM 7.58 raw / 0.544 rel; trace 0.0023 raw / 0.226 rel.

**alpha\* raw match = 0.1** (exact ratio required: 1.35e-08); **alpha\* rel match = 0.1** (exact ratio required: 2.55e-05).

## Part B(b): closed-loop env-SR with EMA in the CEM planner

Standard budget: CEM 300/30/10, horizon 25, budget 50, goal_offset 25, history 3; 5 episodes, 1 seed.

| condition | env | env-SR | mean cos dist | mean phys dist |
|---|---|---|---|---|
| lewm_baseline_v2@alpha=0.1 | cartpole_2d | 0.00 | 0.0382 | 0.7663 |
| lewm_baseline_v2@alpha=0.5 | cartpole_2d | 0.00 | 0.0512 | 0.8869 |
| lewm_baseline_v2@alpha=1.0 | cartpole_2d | 0.00 | 0.0512 | 0.8869 |
| stjewm_trace_only@alpha=1.0 (anchor) | cartpole_2d | 1.00 | 0.0799 | 0.0606 |

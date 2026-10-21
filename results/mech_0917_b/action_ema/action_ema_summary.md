# S3 action sensitivity + S6 EMA control (stab0917_b)

Contexts/env: 32 x 16 action pairs; persistence horizon h=1..10.

## Part A: one-step action sensitivity S1step

S1step = ||dz|| / ||(a1-a2)/act_std||; `rel` additionally normalizes dz by the prediction norm (cross-model comparable).

| model | S1step raw (pooled) | S1step rel (pooled) | latent norm |
|---|---|---|---|
| stjewm_trace_only | 0.00191 | 0.239 | 0.00813 |
| stjewm_spike_only | 0.418 | 0.277 | 1.68 |
| stjewm_membrane_readout | 0.516 | 0.0251 | 26.4 |
| stjewm_hidden_leak | 0.506 | 0.0198 | 31.8 |
| lewm_baseline_v2 | 0.112 | 0.00891 | 12.8 |
| gru_baseline | 0.0517 | 0.00227 | 26.8 |
| mlp_baseline | 9.81e-08 | 3.13e-07 | 0.313 |
| stacked_lif_trace | 0.0654 | 0.00249 | 26.3 |

## Part A: Impulse persistence (action differs only at step 1)

Curves pooled over envs. rel_to_h1 = ||dz_h||/||dz_1|| (wash-out shape); rel_to_latent = ||dz_h||/||z_base_h|| (cross-model comparable).

### rel_to_h1

| model | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 | h=8 | h=9 | h=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 1 | 1.12 | 1.08 | 1.41 | 1.55 | 1.74 | 1.98 | 2.11 | 2.12 | 2.03 |
| stjewm_spike_only | 1 | 1.52 | 1.84 | 2.11 | 2.41 | 2.57 | 2.87 | 3.08 | 3.19 | 3.43 |
| stjewm_membrane_readout | 1 | 1.59 | 2.02 | 2.5 | 3.15 | 3.57 | 4.16 | 4.92 | 5.49 | 6.03 |
| stjewm_hidden_leak | 1 | 1.43 | 1.92 | 2.44 | 2.9 | 3.4 | 3.84 | 4.3 | 4.79 | 5.3 |
| lewm_baseline_v2 | 1 | 0.8 | 0.701 | 0.642 | 0.611 | 0.596 | 0.593 | 0.595 | 0.602 | 0.615 |
| gru_baseline | 1 | 0.811 | 0.736 | 0.721 | 0.494 | 0.534 | 0.687 | 1.01 | 1.63 | 2.12 |
| mlp_baseline | 1 | 0.261 | 0.262 | 0.262 | 0.264 | 0.262 | 0.263 | 0.261 | 0.262 | 0.264 |
| stacked_lif_trace | 1 | 0.563 | 1.23 | 1.69 | 1.74 | 1.67 | 3.03 | 2.63 | 2.73 | 2.95 |

### rel_to_latent

| model | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 | h=8 | h=9 | h=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.25 | 0.276 | 0.273 | 0.344 | 0.388 | 0.432 | 0.487 | 0.53 | 0.535 | 0.524 |
| stjewm_spike_only | 0.27 | 0.408 | 0.44 | 0.481 | 0.539 | 0.546 | 0.547 | 0.623 | 0.657 | 0.674 |
| stjewm_membrane_readout | 0.0255 | 0.0348 | 0.0402 | 0.0459 | 0.0522 | 0.0555 | 0.0608 | 0.0673 | 0.071 | 0.0752 |
| stjewm_hidden_leak | 0.0208 | 0.0251 | 0.0281 | 0.0309 | 0.0336 | 0.0349 | 0.0361 | 0.0376 | 0.0385 | 0.0398 |
| lewm_baseline_v2 | 0.0114 | 0.00976 | 0.00892 | 0.00836 | 0.00799 | 0.00775 | 0.00761 | 0.00751 | 0.00747 | 0.0075 |
| gru_baseline | 0.00299 | 0.00299 | 0.00285 | 0.0028 | 0.00178 | 0.0019 | 0.00245 | 0.00366 | 0.00612 | 0.00827 |
| mlp_baseline | 3.71e-07 | 8.14e-08 | 8.17e-08 | 8.15e-08 | 8.19e-08 | 8.16e-08 | 8.2e-08 | 8.12e-08 | 8.18e-08 | 8.21e-08 |
| stacked_lif_trace | 0.00244 | 0.00138 | 0.003 | 0.00413 | 0.00425 | 0.00409 | 0.0074 | 0.00644 | 0.00667 | 0.00722 |


## Part A: Sustained (center+-d at every step)

Curves pooled over envs. rel_to_h1 = ||dz_h||/||dz_1|| (wash-out shape); rel_to_latent = ||dz_h||/||z_base_h|| (cross-model comparable).

### rel_to_h1

| model | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 | h=8 | h=9 | h=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 1 | 1.37 | 1.56 | 1.82 | 2.2 | 2.35 | 2.61 | 2.79 | 2.93 | 3.05 |
| stjewm_spike_only | 1 | 1.72 | 2.21 | 2.57 | 2.82 | 2.97 | 3.25 | 3.45 | 3.55 | 3.77 |
| stjewm_membrane_readout | 1 | 2.12 | 3.07 | 3.97 | 5.09 | 5.92 | 6.71 | 7.93 | 8.6 | 9.38 |
| stjewm_hidden_leak | 1 | 1.91 | 2.94 | 3.83 | 4.72 | 5.61 | 6.34 | 7.09 | 8.11 | 8.86 |
| lewm_baseline_v2 | 1 | 1.74 | 2.37 | 2.93 | 3.43 | 3.9 | 4.36 | 4.81 | 5.26 | 5.7 |
| gru_baseline | 1 | 1.67 | 2.14 | 2.47 | 2.21 | 2.53 | 3.3 | 4.28 | 5.11 | 5.62 |
| mlp_baseline | 1 | 1 | 1 | 1 | 1 | 0.999 | 0.999 | 0.998 | 1 | 1 |
| stacked_lif_trace | 1 | 0.713 | 1.43 | 1.99 | 1.94 | 2.09 | 3.28 | 2.97 | 3.09 | 3.43 |

### rel_to_latent

| model | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 | h=8 | h=9 | h=10 |
|---|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.25 | 0.336 | 0.39 | 0.442 | 0.54 | 0.572 | 0.638 | 0.69 | 0.723 | 0.76 |
| stjewm_spike_only | 0.27 | 0.465 | 0.528 | 0.587 | 0.631 | 0.634 | 0.622 | 0.7 | 0.736 | 0.742 |
| stjewm_membrane_readout | 0.0255 | 0.0474 | 0.065 | 0.0794 | 0.091 | 0.101 | 0.111 | 0.12 | 0.125 | 0.132 |
| stjewm_hidden_leak | 0.0208 | 0.0345 | 0.0446 | 0.0518 | 0.0581 | 0.0621 | 0.0648 | 0.0671 | 0.0704 | 0.072 |
| lewm_baseline_v2 | 0.0114 | 0.0206 | 0.0288 | 0.0361 | 0.0426 | 0.0486 | 0.0543 | 0.0598 | 0.0651 | 0.0702 |
| gru_baseline | 0.00299 | 0.00553 | 0.00717 | 0.00806 | 0.00658 | 0.00784 | 0.011 | 0.0148 | 0.0177 | 0.0198 |
| mlp_baseline | 3.71e-07 | 3.71e-07 | 3.71e-07 | 3.71e-07 | 3.71e-07 | 3.71e-07 | 3.71e-07 | 3.71e-07 | 3.71e-07 | 3.71e-07 |
| stacked_lif_trace | 0.00244 | 0.00174 | 0.0035 | 0.00486 | 0.00474 | 0.00511 | 0.00801 | 0.00726 | 0.00757 | 0.00838 |

## Part B(a): EMA response matching (LeWM vs trace)

Response = ||dz|| of the one-step predicted latent under last-frame obs perturbation (+0.1*obs std), with EMA z' = alpha*p + (1-alpha)*z_anchor. Primary table: SHARED pre-perturbation anchor (resp = alpha * resp(1)).

| alpha | LeWM resp raw | LeWM resp rel | trace resp raw | trace resp rel |
|---|---|---|---|---|
| 0.1 | 0.275 | 0.0203 | 7.9e-06 | 0.000582 |
| 0.3 | 0.824 | 0.0611 | 2.37e-05 | 0.00218 |
| 0.5 | 1.37 | 0.102 | 3.95e-05 | 0.00456 |
| 0.7 | 1.92 | 0.143 | 5.53e-05 | 0.00762 |
| 0.9 | 2.47 | 0.184 | 7.11e-05 | 0.0104 |
| 1.0 | 2.75 | 0.204 | 7.9e-05 | 0.0109 |

Perturbed-anchor diagnostic (encoder pathway, EMA-immune), LeWM: 2.77..2.75 raw across alpha; trace: 0.00179..7.9e-05. Encoder-only response: LeWM 2.77 raw / 0.205 rel; trace 0.00199 raw / 0.216 rel.

**alpha\* raw match = 0.1** (exact ratio required: 2.87e-05); **alpha\* rel match = 0.1** (exact ratio required: 0.0532).

## Part B(b): closed-loop env-SR with EMA in the CEM planner

Standard budget: CEM 300/30/10, horizon 25, budget 50, goal_offset 25, history 3; 5 episodes, 1 seed.

| condition | env | env-SR | mean cos dist | mean phys dist |
|---|---|---|---|---|
| lewm_baseline_v2@alpha=0.1 | cartpole_2d | 0.00 | 0.2353 | 1.0613 |
| lewm_baseline_v2@alpha=0.5 | cartpole_2d | 0.20 | 0.3272 | 0.8209 |
| lewm_baseline_v2@alpha=1.0 | cartpole_2d | 0.00 | 0.2727 | 0.9142 |
| stjewm_trace_only@alpha=1.0 (anchor) | cartpole_2d | 1.00 | 0.1865 | 0.2122 |
| lewm_baseline_v2@alpha=0.1 | cheetah | 0.00 | 0.1625 | 0.2415 |
| lewm_baseline_v2@alpha=0.5 | cheetah | 0.00 | 0.1255 | 0.2551 |
| lewm_baseline_v2@alpha=1.0 | cheetah | 0.00 | 0.1503 | 0.2443 |
| stjewm_trace_only@alpha=1.0 (anchor) | cheetah | 0.40 | 0.1120 | 0.1180 |

# Event-Type Linear Probes (per-step)

**Setup.** Linear probe on the *gated spike trace* (pre-projection)
of each model. Targets are per-step event-type binary labels
extracted from the state trajectory. Metric: AUROC (calibration-free,
robust to class imbalance). AUPRC is reported alongside.

**Models.** STJEWM-{trace,leak,spike,no-trace,membrane}, LeWM, GRU, MLP.

**Coverage.** 8 envs × 10 models × avg 6.0 targets/env.


## Env: `ball_in_cup`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.655 | 0.612 | 0.602 | 0.584 | 0.573 | 0.555 | 0.564 | 0.595 | 0.598 | 0.617 |
| event_future_k10 | 0.546 | 0.586 | 0.494 | 0.525 | 0.529 | 0.524 | 0.533 | 0.502 | 0.545 | 0.511 |
| event_future_k5 | 0.540 | 0.530 | 0.525 | 0.573 | 0.527 | 0.551 | 0.531 | 0.547 | 0.545 | 0.396 |
| event_high_motion | 0.655 | 0.636 | 0.590 | 0.620 | 0.577 | 0.592 | 0.594 | 0.576 | 0.573 | 0.547 |
| event_low_motion | 0.584 | 0.614 | 0.555 | 0.555 | 0.550 | 0.559 | 0.568 | 0.557 | 0.557 | 0.574 |
| event_persistent | 0.665 | 0.662 | 0.573 | 0.613 | 0.599 | 0.612 | 0.617 | 0.595 | 0.608 | 0.526 |
| event_vel_above_median | 0.631 | 0.634 | 0.557 | 0.579 | 0.598 | 0.589 | 0.583 | 0.595 | 0.592 | 0.512 |

## Env: `cartpole_2d`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.474 | 0.176 | 0.378 |
| event_future_k10 | n/a | n/a | n/a | n/a | n/a | n/a | 0.689 | 0.539 | 0.533 | 0.524 |
| event_future_k5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.629 | 0.426 |
| event_high_motion | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.214 | 0.236 |
| event_low_motion | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.296 | 0.285 |
| event_persistent | n/a | n/a | n/a | n/a | 0.563 | n/a | n/a | 0.164 | 0.223 | 0.228 |
| event_vel_above_median | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.196 |

## Env: `cheetah`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.544 | 0.491 | 0.493 | 0.573 | 0.561 | 0.564 | 0.483 | 0.512 | 0.484 | 0.474 |
| event_future_k10 | 0.417 | 0.505 | 0.458 | 0.494 | 0.534 | 0.487 | 0.523 | 0.621 | 0.523 | 0.505 |
| event_future_k5 | 0.512 | 0.524 | 0.488 | 0.497 | 0.453 | 0.503 | 0.498 | 0.494 | n/a | 0.547 |
| event_high_motion | 0.502 | 0.459 | 0.496 | 0.520 | 0.545 | 0.538 | 0.532 | 0.442 | 0.426 | 0.506 |
| event_low_motion | 0.523 | 0.484 | 0.495 | 0.474 | 0.548 | 0.516 | 0.509 | 0.526 | 0.573 | 0.396 |
| event_persistent | 0.541 | 0.460 | 0.506 | 0.525 | 0.486 | 0.529 | 0.493 | 0.542 | 0.543 | 0.500 |
| event_vel_above_median | 0.525 | 0.476 | 0.504 | 0.518 | 0.544 | 0.527 | 0.499 | 0.523 | n/a | 0.476 |

## Env: `finger`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.416 | 0.395 | 0.505 | 0.484 | 0.561 | 0.569 | 0.585 | 0.624 | n/a | 0.494 |
| event_future_k10 | 0.496 | 0.456 | 0.484 | 0.452 | 0.507 | 0.472 | 0.469 | 0.471 | 0.472 | 0.503 |
| event_future_k5 | 0.476 | 0.507 | 0.482 | 0.484 | 0.447 | 0.466 | 0.457 | 0.446 | 0.471 | 0.491 |
| event_high_motion | 0.424 | 0.424 | 0.500 | 0.482 | 0.548 | 0.542 | 0.541 | 0.526 | 0.613 | 0.623 |
| event_low_motion | 0.492 | 0.576 | 0.503 | 0.499 | 0.525 | 0.528 | 0.532 | 0.524 | 0.547 | 0.536 |
| event_persistent | 0.509 | 0.472 | 0.506 | 0.498 | 0.580 | 0.561 | 0.550 | 0.552 | 0.578 | 0.590 |
| event_vel_above_median | 0.424 | 0.503 | 0.512 | 0.502 | 0.551 | 0.544 | 0.540 | 0.544 | 0.537 | 0.593 |

## Env: `pendulum_2d`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.422 |
| event_future_k10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.699 | 0.612 |
| event_future_k5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.699 | 0.625 |
| event_high_motion | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.665 | 0.485 |
| event_low_motion | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.617 | 0.526 |
| event_persistent | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.691 | 0.469 |
| event_vel_above_median | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.653 | 0.516 |

## Env: `pusht`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_block_near_target | **0.996** | **1.000** | n/a | **0.987** | **0.990** | **0.990** | **0.990** | **0.990** | **0.990** | n/a |
| event_contact | **0.923** | **0.950** | **0.792** | **0.887** | **0.834** | **0.836** | **0.837** | **0.837** | **0.837** | n/a |
| event_future_k10 | **0.889** | **0.910** | **0.778** | **0.874** | **0.783** | **0.780** | **0.782** | **0.781** | **0.780** | n/a |
| event_future_k5 | **0.917** | **0.952** | **0.764** | **0.833** | **0.831** | **0.832** | **0.832** | **0.833** | **0.828** | n/a |
| event_persistent | **0.870** | **0.914** | **0.715** | **0.818** | **0.785** | **0.785** | **0.784** | **0.784** | **0.784** | n/a |

## Env: `tworoom`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_high_motion | n/a | n/a | n/a | n/a | n/a | n/a | 0.500 | n/a | n/a | n/a |

## Env: `walker`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | **0.715** | 0.530 | 0.487 | 0.514 | 0.562 | 0.614 | 0.552 | 0.543 | 0.537 | 0.489 |
| event_future_k10 | 0.605 | 0.516 | 0.494 | 0.524 | 0.630 | 0.619 | 0.642 | 0.604 | 0.571 | 0.474 |
| event_future_k5 | 0.530 | 0.599 | 0.458 | 0.523 | 0.505 | 0.554 | 0.548 | 0.502 | n/a | 0.402 |
| event_high_motion | 0.643 | 0.608 | 0.532 | 0.527 | 0.629 | 0.643 | 0.628 | 0.656 | 0.538 | 0.500 |
| event_low_motion | 0.637 | 0.649 | 0.551 | 0.624 | 0.637 | 0.634 | 0.618 | 0.623 | 0.533 | 0.265 |
| event_persistent | 0.635 | 0.576 | 0.512 | 0.522 | 0.625 | 0.650 | 0.641 | 0.625 | n/a | 0.284 |
| event_vel_above_median | 0.687 | 0.656 | 0.544 | 0.592 | 0.672 | 0.648 | 0.659 | 0.665 | 0.528 | 0.276 |

## Headline comparison: event probes vs position probes

**Key claim.** STJEWM-trace is event-specialized: it ties or wins on
event-type targets even when its position-probe R² is moderate.

### Mean event-probe AUROC per model

| model | n_cells | mean AUROC | median AUROC |
|---|---|---|---|
| cubifae_baseline | 33 | 0.610 | 0.546 |
| gru_baseline | 33 | 0.602 | 0.576 |
| slt_lif_mpc_free | 32 | 0.545 | 0.506 |
| slt_lif_mpc_trace | 33 | 0.584 | 0.525 |
| stjewm_hidden_leak | 34 | 0.600 | 0.561 |
| stjewm_membrane_readout | 33 | 0.603 | 0.561 |
| stjewm_no_trace | 35 | 0.597 | 0.552 |
| stjewm_rate_only | 36 | 0.581 | 0.549 |
| stjewm_spike_only | 40 | 0.570 | 0.564 |
| stjewm_trace_only | 42 | 0.465 | 0.497 |

### Per-target winners (per env, model with highest AUROC)

| env | target | winner | AUROC | runner-up | AUROC |
|---|---|---|---|---|---|
| ball_in_cup | event_contact | cubifae_baseline | 0.655 | stjewm_trace_only | 0.617 |
| ball_in_cup | event_future_k10 | gru_baseline | 0.586 | cubifae_baseline | 0.546 |
| ball_in_cup | event_future_k5 | slt_lif_mpc_trace | 0.573 | stjewm_membrane_readout | 0.551 |
| ball_in_cup | event_high_motion | cubifae_baseline | 0.655 | gru_baseline | 0.636 |
| ball_in_cup | event_low_motion | gru_baseline | 0.614 | cubifae_baseline | 0.584 |
| ball_in_cup | event_persistent | cubifae_baseline | 0.665 | gru_baseline | 0.662 |
| ball_in_cup | event_vel_above_median | gru_baseline | 0.634 | cubifae_baseline | 0.631 |
| cartpole_2d | event_contact | stjewm_rate_only | 0.474 | stjewm_trace_only | 0.378 |
| cartpole_2d | event_future_k10 | stjewm_no_trace | 0.689 | stjewm_rate_only | 0.539 |
| cartpole_2d | event_future_k5 | stjewm_spike_only | 0.629 | stjewm_trace_only | 0.426 |
| cartpole_2d | event_high_motion | stjewm_trace_only | 0.236 | stjewm_spike_only | 0.214 |
| cartpole_2d | event_low_motion | stjewm_spike_only | 0.296 | stjewm_trace_only | 0.285 |
| cartpole_2d | event_persistent | stjewm_hidden_leak | 0.563 | stjewm_trace_only | 0.228 |
| cartpole_2d | event_vel_above_median | stjewm_trace_only | 0.196 | - | 0.000 |
| cheetah | event_contact | slt_lif_mpc_trace | 0.573 | stjewm_membrane_readout | 0.564 |
| cheetah | event_future_k10 | stjewm_rate_only | 0.621 | stjewm_hidden_leak | 0.534 |
| cheetah | event_future_k5 | stjewm_trace_only | 0.547 | gru_baseline | 0.524 |
| cheetah | event_high_motion | stjewm_hidden_leak | 0.545 | stjewm_membrane_readout | 0.538 |
| cheetah | event_low_motion | stjewm_spike_only | 0.573 | stjewm_hidden_leak | 0.548 |
| cheetah | event_persistent | stjewm_spike_only | 0.543 | stjewm_rate_only | 0.542 |
| cheetah | event_vel_above_median | stjewm_hidden_leak | 0.544 | stjewm_membrane_readout | 0.527 |
| finger | event_contact | stjewm_rate_only | 0.624 | stjewm_no_trace | 0.585 |
| finger | event_future_k10 | stjewm_hidden_leak | 0.507 | stjewm_trace_only | 0.503 |
| finger | event_future_k5 | gru_baseline | 0.507 | stjewm_trace_only | 0.491 |
| finger | event_high_motion | stjewm_trace_only | 0.623 | stjewm_spike_only | 0.613 |
| finger | event_low_motion | gru_baseline | 0.576 | stjewm_spike_only | 0.547 |
| finger | event_persistent | stjewm_trace_only | 0.590 | stjewm_hidden_leak | 0.580 |
| finger | event_vel_above_median | stjewm_trace_only | 0.593 | stjewm_hidden_leak | 0.551 |
| pendulum_2d | event_contact | stjewm_trace_only | 0.422 | - | 0.000 |
| pendulum_2d | event_future_k10 | stjewm_spike_only | 0.699 | stjewm_trace_only | 0.612 |
| pendulum_2d | event_future_k5 | stjewm_spike_only | 0.699 | stjewm_trace_only | 0.625 |
| pendulum_2d | event_high_motion | stjewm_spike_only | 0.665 | stjewm_trace_only | 0.485 |
| pendulum_2d | event_low_motion | stjewm_spike_only | 0.617 | stjewm_trace_only | 0.526 |
| pendulum_2d | event_persistent | stjewm_spike_only | 0.691 | stjewm_trace_only | 0.469 |
| pendulum_2d | event_vel_above_median | stjewm_spike_only | 0.653 | stjewm_trace_only | 0.516 |
| pusht | event_block_near_target | gru_baseline | 1.000 | cubifae_baseline | 0.996 |
| pusht | event_contact | gru_baseline | 0.950 | cubifae_baseline | 0.923 |
| pusht | event_future_k10 | gru_baseline | 0.910 | cubifae_baseline | 0.889 |
| pusht | event_future_k5 | gru_baseline | 0.952 | cubifae_baseline | 0.917 |
| pusht | event_persistent | gru_baseline | 0.914 | cubifae_baseline | 0.870 |
| tworoom | event_high_motion | stjewm_no_trace | 0.500 | - | 0.000 |
| walker | event_contact | cubifae_baseline | 0.715 | stjewm_membrane_readout | 0.614 |
| walker | event_future_k10 | stjewm_no_trace | 0.642 | stjewm_hidden_leak | 0.630 |
| walker | event_future_k5 | gru_baseline | 0.599 | stjewm_membrane_readout | 0.554 |
| walker | event_high_motion | stjewm_rate_only | 0.656 | stjewm_membrane_readout | 0.643 |
| walker | event_low_motion | gru_baseline | 0.649 | cubifae_baseline | 0.637 |
| walker | event_persistent | stjewm_membrane_readout | 0.650 | stjewm_no_trace | 0.641 |
| walker | event_vel_above_median | cubifae_baseline | 0.687 | stjewm_hidden_leak | 0.672 |

### Win counts (event-type targets)

| model | wins |
|---|---|
| cubifae_baseline | 5 |
| gru_baseline | 12 |
| slt_lif_mpc_free | 0 |
| slt_lif_mpc_trace | 2 |
| stjewm_hidden_leak | 4 |
| stjewm_membrane_readout | 1 |
| stjewm_no_trace | 3 |
| stjewm_rate_only | 4 |
| stjewm_spike_only | 10 |
| stjewm_trace_only | 7 |

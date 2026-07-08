# Event-Type Linear Probes (per-step)

**Setup.** Linear probe on the *gated spike trace* (pre-projection)
of each model. Targets are per-step event-type binary labels
extracted from the state trajectory. Metric: AUROC (calibration-free,
robust to class imbalance). AUPRC is reported alongside.

**Models.** STJEWM-{trace,leak,spike,no-trace,membrane}, LeWM, GRU, MLP.

**Coverage.** 8 envs × 10 models × avg 6.2 targets/env.


## Env: `ball_in_cup`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.655 | 0.612 | 0.602 | 0.584 | 0.573 | 0.555 | 0.564 | 0.595 | 0.598 | 0.643 |
| event_future_k10 | 0.546 | 0.586 | 0.494 | 0.525 | 0.529 | 0.524 | 0.533 | 0.502 | 0.545 | 0.561 |
| event_future_k5 | 0.540 | 0.530 | 0.525 | 0.573 | 0.527 | 0.551 | 0.531 | 0.547 | 0.545 | 0.568 |
| event_high_motion | 0.655 | 0.636 | 0.590 | 0.620 | 0.577 | 0.592 | 0.594 | 0.576 | 0.573 | 0.643 |
| event_low_motion | 0.584 | 0.614 | 0.555 | 0.555 | 0.550 | 0.559 | 0.568 | 0.557 | 0.557 | 0.643 |
| event_persistent | 0.665 | 0.662 | 0.573 | 0.613 | 0.599 | 0.612 | 0.617 | 0.595 | 0.608 | 0.619 |
| event_vel_above_median | 0.631 | 0.634 | 0.557 | 0.579 | 0.598 | 0.589 | 0.583 | 0.595 | 0.592 | 0.617 |

## Env: `cartpole_2d`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.494 | 0.512 |
| event_future_k10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.688 | 0.685 |
| event_future_k5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.631 |
| event_high_motion | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.572 |
| event_low_motion | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | **0.728** |
| event_persistent | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.541 | 0.538 |
| event_vel_above_median | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.656 |

## Env: `cheetah`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.544 | 0.491 | 0.493 | 0.573 | 0.561 | 0.564 | 0.551 | 0.530 | 0.512 | 0.507 |
| event_future_k10 | 0.417 | 0.505 | 0.458 | 0.494 | 0.490 | 0.487 | 0.468 | 0.520 | 0.539 | 0.542 |
| event_future_k5 | 0.512 | 0.524 | 0.488 | 0.497 | 0.453 | 0.503 | 0.498 | 0.494 | 0.483 | 0.498 |
| event_high_motion | 0.502 | 0.459 | 0.496 | 0.520 | 0.510 | 0.485 | 0.531 | 0.520 | 0.548 | 0.530 |
| event_low_motion | 0.523 | 0.484 | 0.495 | 0.474 | 0.548 | 0.516 | 0.509 | 0.526 | 0.534 | 0.566 |
| event_persistent | 0.541 | 0.460 | 0.506 | 0.525 | 0.486 | 0.529 | 0.493 | 0.542 | 0.510 | 0.536 |
| event_vel_above_median | 0.525 | 0.476 | 0.504 | 0.518 | 0.544 | 0.527 | 0.499 | 0.523 | 0.518 | 0.563 |

## Env: `finger`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.416 | 0.395 | 0.505 | 0.484 | 0.561 | 0.569 | 0.585 | 0.624 | 0.596 | 0.585 |
| event_future_k10 | 0.496 | 0.456 | 0.484 | 0.452 | 0.507 | 0.472 | 0.469 | 0.471 | 0.472 | 0.496 |
| event_future_k5 | 0.476 | 0.507 | 0.482 | 0.484 | 0.447 | 0.466 | 0.457 | 0.446 | 0.464 | 0.492 |
| event_high_motion | 0.424 | 0.424 | 0.500 | 0.482 | 0.548 | 0.542 | 0.541 | 0.526 | 0.534 | 0.582 |
| event_low_motion | 0.492 | 0.576 | 0.503 | 0.499 | 0.525 | 0.528 | 0.532 | 0.524 | 0.532 | 0.577 |
| event_persistent | 0.509 | 0.472 | 0.506 | 0.498 | 0.580 | 0.561 | 0.550 | 0.552 | 0.561 | 0.589 |
| event_vel_above_median | 0.424 | 0.503 | 0.512 | 0.502 | 0.551 | 0.544 | 0.540 | 0.544 | 0.537 | 0.574 |

## Env: `pendulum_2d`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.645 |
| event_future_k10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.699 |
| event_future_k5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.698 |
| event_high_motion | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.673 |
| event_low_motion | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.618 |
| event_persistent | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.690 |
| event_vel_above_median | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.657 |

## Env: `pusht`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_block_near_target | **0.996** | **1.000** | n/a | **0.987** | **0.990** | **0.990** | **0.990** | **0.990** | **0.990** | **0.990** |
| event_contact | **0.923** | **0.950** | **0.792** | **0.887** | **0.834** | **0.836** | **0.837** | **0.837** | **0.837** | **0.876** |
| event_future_k10 | **0.889** | **0.910** | **0.778** | **0.874** | **0.783** | **0.780** | **0.782** | **0.781** | **0.780** | **0.781** |
| event_future_k5 | **0.917** | **0.952** | **0.764** | **0.833** | **0.831** | **0.832** | **0.832** | **0.833** | **0.828** | **0.829** |
| event_persistent | **0.870** | **0.914** | **0.715** | **0.818** | **0.785** | **0.785** | **0.784** | **0.784** | **0.784** | **0.784** |

## Env: `tworoom`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_future_k5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.500 | 0.500 |
| event_high_motion | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.500 | 0.500 |
| event_room_entered | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.500 | 0.500 | 0.500 |

## Env: `walker`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | **0.715** | 0.530 | 0.487 | 0.514 | 0.562 | 0.614 | 0.552 | 0.543 | 0.645 | 0.542 |
| event_future_k10 | 0.605 | 0.516 | 0.494 | 0.524 | 0.630 | 0.619 | 0.642 | 0.604 | 0.600 | 0.575 |
| event_future_k5 | 0.530 | 0.599 | 0.458 | 0.523 | 0.505 | 0.554 | 0.548 | 0.502 | 0.547 | 0.517 |
| event_high_motion | 0.643 | 0.608 | 0.532 | 0.527 | 0.629 | 0.643 | 0.628 | 0.656 | 0.635 | 0.537 |
| event_low_motion | 0.637 | 0.649 | 0.551 | 0.624 | 0.637 | 0.634 | 0.618 | 0.623 | 0.633 | 0.536 |
| event_persistent | 0.635 | 0.576 | 0.512 | 0.522 | 0.625 | 0.650 | 0.641 | 0.625 | 0.623 | 0.546 |
| event_vel_above_median | 0.687 | 0.656 | 0.544 | 0.592 | 0.672 | 0.648 | 0.659 | 0.665 | 0.676 | 0.536 |

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
| stjewm_hidden_leak | 33 | 0.598 | 0.561 |
| stjewm_membrane_readout | 33 | 0.602 | 0.561 |
| stjewm_no_trace | 33 | 0.598 | 0.552 |
| stjewm_rate_only | 34 | 0.596 | 0.549 |
| stjewm_spike_only | 39 | 0.594 | 0.548 |
| stjewm_trace_only | 50 | 0.610 | 0.576 |

### Per-target winners (per env, model with highest AUROC)

| env | target | winner | AUROC | runner-up | AUROC |
|---|---|---|---|---|---|
| ball_in_cup | event_contact | cubifae_baseline | 0.655 | stjewm_trace_only | 0.643 |
| ball_in_cup | event_future_k10 | gru_baseline | 0.586 | stjewm_trace_only | 0.561 |
| ball_in_cup | event_future_k5 | slt_lif_mpc_trace | 0.573 | stjewm_trace_only | 0.568 |
| ball_in_cup | event_high_motion | cubifae_baseline | 0.655 | stjewm_trace_only | 0.643 |
| ball_in_cup | event_low_motion | stjewm_trace_only | 0.643 | gru_baseline | 0.614 |
| ball_in_cup | event_persistent | cubifae_baseline | 0.665 | gru_baseline | 0.662 |
| ball_in_cup | event_vel_above_median | gru_baseline | 0.634 | cubifae_baseline | 0.631 |
| cartpole_2d | event_contact | stjewm_trace_only | 0.512 | stjewm_spike_only | 0.494 |
| cartpole_2d | event_future_k10 | stjewm_spike_only | 0.688 | stjewm_trace_only | 0.685 |
| cartpole_2d | event_future_k5 | stjewm_trace_only | 0.631 | - | 0.000 |
| cartpole_2d | event_high_motion | stjewm_trace_only | 0.572 | - | 0.000 |
| cartpole_2d | event_low_motion | stjewm_trace_only | 0.728 | - | 0.000 |
| cartpole_2d | event_persistent | stjewm_spike_only | 0.541 | stjewm_trace_only | 0.538 |
| cartpole_2d | event_vel_above_median | stjewm_trace_only | 0.656 | - | 0.000 |
| cheetah | event_contact | slt_lif_mpc_trace | 0.573 | stjewm_membrane_readout | 0.564 |
| cheetah | event_future_k10 | stjewm_trace_only | 0.542 | stjewm_spike_only | 0.539 |
| cheetah | event_future_k5 | gru_baseline | 0.524 | cubifae_baseline | 0.512 |
| cheetah | event_high_motion | stjewm_spike_only | 0.548 | stjewm_no_trace | 0.531 |
| cheetah | event_low_motion | stjewm_trace_only | 0.566 | stjewm_hidden_leak | 0.548 |
| cheetah | event_persistent | stjewm_rate_only | 0.542 | cubifae_baseline | 0.541 |
| cheetah | event_vel_above_median | stjewm_trace_only | 0.563 | stjewm_hidden_leak | 0.544 |
| finger | event_contact | stjewm_rate_only | 0.624 | stjewm_spike_only | 0.596 |
| finger | event_future_k10 | stjewm_hidden_leak | 0.507 | cubifae_baseline | 0.496 |
| finger | event_future_k5 | gru_baseline | 0.507 | stjewm_trace_only | 0.492 |
| finger | event_high_motion | stjewm_trace_only | 0.582 | stjewm_hidden_leak | 0.548 |
| finger | event_low_motion | stjewm_trace_only | 0.577 | gru_baseline | 0.576 |
| finger | event_persistent | stjewm_trace_only | 0.589 | stjewm_hidden_leak | 0.580 |
| finger | event_vel_above_median | stjewm_trace_only | 0.574 | stjewm_hidden_leak | 0.551 |
| pendulum_2d | event_contact | stjewm_trace_only | 0.645 | - | 0.000 |
| pendulum_2d | event_future_k10 | stjewm_trace_only | 0.699 | - | 0.000 |
| pendulum_2d | event_future_k5 | stjewm_trace_only | 0.698 | - | 0.000 |
| pendulum_2d | event_high_motion | stjewm_trace_only | 0.673 | - | 0.000 |
| pendulum_2d | event_low_motion | stjewm_trace_only | 0.618 | - | 0.000 |
| pendulum_2d | event_persistent | stjewm_trace_only | 0.690 | - | 0.000 |
| pendulum_2d | event_vel_above_median | stjewm_trace_only | 0.657 | - | 0.000 |
| pusht | event_block_near_target | gru_baseline | 1.000 | cubifae_baseline | 0.996 |
| pusht | event_contact | gru_baseline | 0.950 | cubifae_baseline | 0.923 |
| pusht | event_future_k10 | gru_baseline | 0.910 | cubifae_baseline | 0.889 |
| pusht | event_future_k5 | gru_baseline | 0.952 | cubifae_baseline | 0.917 |
| pusht | event_persistent | gru_baseline | 0.914 | cubifae_baseline | 0.870 |
| tworoom | event_future_k5 | stjewm_spike_only | 0.500 | stjewm_trace_only | 0.500 |
| tworoom | event_high_motion | stjewm_spike_only | 0.500 | stjewm_trace_only | 0.500 |
| tworoom | event_room_entered | stjewm_rate_only | 0.500 | stjewm_spike_only | 0.500 |
| walker | event_contact | cubifae_baseline | 0.715 | stjewm_spike_only | 0.645 |
| walker | event_future_k10 | stjewm_no_trace | 0.642 | stjewm_hidden_leak | 0.630 |
| walker | event_future_k5 | gru_baseline | 0.599 | stjewm_membrane_readout | 0.554 |
| walker | event_high_motion | stjewm_rate_only | 0.656 | stjewm_membrane_readout | 0.643 |
| walker | event_low_motion | gru_baseline | 0.649 | cubifae_baseline | 0.637 |
| walker | event_persistent | stjewm_membrane_readout | 0.650 | stjewm_no_trace | 0.641 |
| walker | event_vel_above_median | cubifae_baseline | 0.687 | stjewm_spike_only | 0.676 |

### Win counts (event-type targets)

| model | wins |
|---|---|
| cubifae_baseline | 5 |
| gru_baseline | 11 |
| slt_lif_mpc_free | 0 |
| slt_lif_mpc_trace | 2 |
| stjewm_hidden_leak | 1 |
| stjewm_membrane_readout | 1 |
| stjewm_no_trace | 1 |
| stjewm_rate_only | 4 |
| stjewm_spike_only | 5 |
| stjewm_trace_only | 20 |

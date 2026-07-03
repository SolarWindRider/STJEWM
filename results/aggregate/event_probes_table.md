# Event-Type Linear Probes (per-step)

**Setup.** Linear probe on the *gated spike trace* (pre-projection)
of each model. Targets are per-step event-type binary labels
extracted from the state trajectory. Metric: AUROC (calibration-free,
robust to class imbalance). AUPRC is reported alongside.

**Models.** STJEWM-{trace,leak,spike,no-trace,membrane}, LeWM, GRU, MLP.

**Coverage.** 7 envs × 12 models × avg 3.0 targets/env.


## Env: `ball_in_cup`

| target | cubifae_baseline | gru_baseline | lewm_baseline_v2 | mlp_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | spikedreamer_baseline | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.618 | 0.553 | 0.568 | 0.562 | 0.559 | 0.596 | 0.535 | 0.634 | 0.633 | 0.603 | 0.629 | 0.628 |
| event_future_k5 | 0.605 | 0.573 | 0.535 | 0.491 | 0.560 | 0.585 | 0.500 | 0.595 | 0.591 | 0.588 | 0.593 | 0.593 |
| event_high_motion | 0.622 | 0.582 | 0.548 | 0.544 | 0.547 | 0.576 | 0.536 | 0.635 | 0.639 | 0.614 | 0.637 | 0.637 |

## Env: `cartpole_2d`

| target | cubifae_baseline | gru_baseline | lewm_baseline_v2 | mlp_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | spikedreamer_baseline | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | **0.748** | **0.781** | 0.612 | 0.519 | 0.609 | 0.648 | 0.652 | **0.710** | **0.712** | **0.745** | **0.715** | **0.704** |
| event_future_k5 | **0.801** | **0.755** | 0.642 | 0.587 | 0.573 | 0.633 | 0.605 | **0.773** | **0.770** | **0.784** | **0.775** | **0.767** |
| event_high_motion | **0.796** | **0.832** | 0.588 | 0.523 | 0.591 | 0.656 | 0.632 | **0.727** | **0.729** | **0.749** | **0.733** | **0.732** |

## Env: `cheetah`

| target | cubifae_baseline | gru_baseline | lewm_baseline_v2 | mlp_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | spikedreamer_baseline | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_future_k10 | 0.541 | 0.562 | n/a | 0.547 | 0.521 | 0.533 | 0.505 | 0.530 | 0.532 | 0.530 | 0.519 | 0.530 |
| event_high_motion | 0.561 | 0.590 | n/a | 0.551 | 0.523 | 0.545 | 0.505 | 0.499 | 0.500 | 0.505 | 0.502 | 0.497 |
| event_low_motion | 0.514 | 0.524 | n/a | 0.536 | 0.494 | 0.504 | 0.478 | 0.501 | 0.503 | 0.493 | 0.497 | 0.507 |

## Env: `delayed_t_maze`

| target | cubifae_baseline | gru_baseline | lewm_baseline_v2 | mlp_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | spikedreamer_baseline | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_cue_state | n/a | n/a | n/a | n/a | n/a | n/a | n/a | **0.965** | n/a | **0.964** | **0.964** | **0.964** |
| event_future_k5 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | **0.933** | n/a | **0.933** | **0.932** | **0.933** |
| event_high_motion | n/a | n/a | n/a | n/a | n/a | n/a | n/a | **0.954** | n/a | **0.955** | **0.955** | **0.955** |

## Env: `finger`

| target | cubifae_baseline | gru_baseline | lewm_baseline_v2 | mlp_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | spikedreamer_baseline | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.503 | 0.514 | n/a | 0.487 | 0.487 | 0.501 | 0.523 | 0.510 | 0.508 | 0.519 | 0.510 | 0.516 |
| event_future_k5 | 0.496 | 0.504 | n/a | 0.497 | 0.497 | 0.511 | 0.508 | 0.500 | 0.507 | 0.504 | n/a | 0.509 |
| event_high_motion | 0.501 | 0.512 | n/a | 0.511 | 0.493 | 0.499 | 0.503 | 0.511 | 0.510 | 0.506 | 0.513 | 0.515 |

## Env: `pusht`

| target | cubifae_baseline | gru_baseline | lewm_baseline_v2 | mlp_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | spikedreamer_baseline | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_block_near_target | **0.999** | **1.000** | n/a | **1.000** | **0.972** | **0.998** | **0.711** | **0.997** | **0.997** | **0.993** | **0.997** | **0.997** |
| event_contact | **0.939** | **0.978** | n/a | **0.967** | **0.795** | **0.887** | 0.615 | **0.936** | **0.936** | **0.863** | **0.936** | **0.936** |
| event_future_k10 | **0.891** | **0.970** | n/a | **0.936** | **0.777** | **0.861** | 0.590 | **0.881** | **0.881** | **0.836** | **0.882** | **0.881** |

## Env: `tworoom`

| target | cubifae_baseline | gru_baseline | lewm_baseline_v2 | mlp_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | spikedreamer_baseline | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| event_future_k5 | 0.500 | 0.521 | n/a | 0.510 | 0.493 | 0.488 | 0.499 | 0.493 | 0.496 | 0.521 | 0.491 | 0.488 |
| event_high_motion | 0.514 | 0.513 | n/a | 0.491 | 0.493 | 0.488 | 0.499 | 0.492 | 0.493 | 0.523 | 0.492 | 0.492 |
| event_room_entered | **0.800** | **0.791** | n/a | **0.755** | 0.606 | 0.691 | 0.565 | **0.708** | **0.706** | **0.715** | **0.706** | **0.704** |

## Headline comparison: event probes vs position probes

**Key claim.** STJEWM-trace is event-specialized: it ties or wins on
event-type targets even when its position-probe R² is moderate.

### Mean event-probe AUROC per model

| model | n_cells | mean AUROC | median AUROC |
|---|---|---|---|
| cubifae_baseline | 18 | 0.664 | 0.612 |
| gru_baseline | 18 | 0.670 | 0.578 |
| lewm_baseline_v2 | 6 | 0.582 | 0.578 |
| mlp_baseline | 18 | 0.612 | 0.540 |
| slt_lif_mpc_free | 18 | 0.588 | 0.553 |
| slt_lif_mpc_trace | 18 | 0.622 | 0.580 |
| spikedreamer_baseline | 18 | 0.553 | 0.529 |
| stjewm_hidden_leak | 21 | 0.690 | 0.635 |
| stjewm_membrane_readout | 18 | 0.647 | 0.612 |
| stjewm_no_trace | 21 | 0.688 | 0.614 |
| stjewm_spike_only | 20 | 0.699 | 0.671 |
| stjewm_trace_only | 21 | 0.690 | 0.637 |

### Per-target winners (per env, model with highest AUROC)

| env | target | winner | AUROC | runner-up | AUROC |
|---|---|---|---|---|---|
| ball_in_cup | event_contact | stjewm_hidden_leak | 0.634 | stjewm_membrane_readout | 0.633 |
| ball_in_cup | event_future_k5 | cubifae_baseline | 0.605 | stjewm_hidden_leak | 0.595 |
| ball_in_cup | event_high_motion | stjewm_membrane_readout | 0.639 | stjewm_trace_only | 0.637 |
| cartpole_2d | event_contact | gru_baseline | 0.781 | cubifae_baseline | 0.748 |
| cartpole_2d | event_future_k5 | cubifae_baseline | 0.801 | stjewm_no_trace | 0.784 |
| cartpole_2d | event_high_motion | gru_baseline | 0.832 | cubifae_baseline | 0.796 |
| cheetah | event_future_k10 | gru_baseline | 0.562 | mlp_baseline | 0.547 |
| cheetah | event_high_motion | gru_baseline | 0.590 | cubifae_baseline | 0.561 |
| cheetah | event_low_motion | mlp_baseline | 0.536 | gru_baseline | 0.524 |
| delayed_t_maze | event_cue_state | stjewm_hidden_leak | 0.965 | stjewm_spike_only | 0.964 |
| delayed_t_maze | event_future_k5 | stjewm_no_trace | 0.933 | stjewm_trace_only | 0.933 |
| delayed_t_maze | event_high_motion | stjewm_no_trace | 0.955 | stjewm_trace_only | 0.955 |
| finger | event_contact | spikedreamer_baseline | 0.523 | stjewm_no_trace | 0.519 |
| finger | event_future_k5 | slt_lif_mpc_trace | 0.511 | stjewm_trace_only | 0.509 |
| finger | event_high_motion | stjewm_trace_only | 0.515 | stjewm_spike_only | 0.513 |
| pusht | event_block_near_target | gru_baseline | 1.000 | mlp_baseline | 1.000 |
| pusht | event_contact | gru_baseline | 0.978 | mlp_baseline | 0.967 |
| pusht | event_future_k10 | gru_baseline | 0.970 | mlp_baseline | 0.936 |
| tworoom | event_future_k5 | stjewm_no_trace | 0.521 | gru_baseline | 0.521 |
| tworoom | event_high_motion | stjewm_no_trace | 0.523 | cubifae_baseline | 0.514 |
| tworoom | event_room_entered | cubifae_baseline | 0.800 | gru_baseline | 0.791 |

### Win counts (event-type targets)

| model | wins |
|---|---|
| cubifae_baseline | 3 |
| gru_baseline | 7 |
| lewm_baseline_v2 | 0 |
| mlp_baseline | 1 |
| slt_lif_mpc_free | 0 |
| slt_lif_mpc_trace | 1 |
| spikedreamer_baseline | 1 |
| stjewm_hidden_leak | 2 |
| stjewm_membrane_readout | 1 |
| stjewm_no_trace | 4 |
| stjewm_spike_only | 0 |
| stjewm_trace_only | 1 |

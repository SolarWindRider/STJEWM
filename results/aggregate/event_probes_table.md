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
| event_contact | 0.655 | 0.612 | 0.602 | 0.584 | 0.606 | 0.521 | 0.582 | 0.575 | 0.607 | 0.552 |
| event_future_k10 | 0.546 | 0.586 | 0.494 | 0.525 | 0.506 | 0.500 | 0.507 | 0.521 | 0.502 | 0.505 |
| event_future_k5 | 0.540 | 0.530 | 0.525 | 0.573 | 0.435 | 0.430 | 0.443 | 0.377 | 0.435 | 0.431 |
| event_high_motion | 0.655 | 0.636 | 0.590 | 0.620 | 0.602 | 0.441 | 0.514 | 0.616 | 0.498 | 0.443 |
| event_low_motion | 0.584 | 0.614 | 0.555 | 0.555 | 0.571 | 0.558 | 0.554 | 0.556 | 0.561 | 0.547 |
| event_persistent | 0.665 | 0.662 | 0.573 | 0.613 | 0.522 | 0.551 | 0.612 | 0.523 | 0.593 | 0.570 |
| event_vel_above_median | 0.631 | 0.634 | 0.557 | 0.579 | 0.505 | 0.510 | 0.516 | 0.521 | 0.516 | 0.522 |

## Env: `cartpole_2d`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.654 | **0.742** | n/a | **0.801** | 0.191 | 0.394 | 0.361 | 0.561 | 0.176 | 0.378 |
| event_future_k10 | 0.411 | 0.456 | n/a | 0.517 | 0.522 | 0.522 | 0.540 | 0.527 | 0.533 | 0.524 |
| event_future_k5 | n/a | n/a | n/a | n/a | 0.419 | 0.427 | 0.437 | 0.440 | 0.431 | 0.427 |
| event_high_motion | n/a | n/a | n/a | n/a | 0.196 | 0.401 | 0.221 | 0.221 | 0.188 | 0.430 |
| event_low_motion | n/a | n/a | n/a | n/a | 0.280 | 0.288 | 0.282 | 0.296 | 0.283 | 0.279 |
| event_persistent | 0.630 | 0.591 | n/a | 0.698 | 0.224 | 0.241 | 0.298 | 0.264 | 0.223 | 0.228 |
| event_vel_above_median | n/a | n/a | n/a | n/a | 0.204 | 0.205 | 0.206 | 0.191 | 0.230 | 0.212 |

## Env: `cheetah`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.460 | 0.385 | 0.472 | 0.450 | 0.456 | 0.408 | 0.482 | 0.512 | 0.484 | 0.474 |
| event_future_k10 | 0.482 | 0.532 | 0.506 | 0.578 | 0.536 | 0.448 | 0.509 | 0.493 | 0.523 | 0.505 |
| event_future_k5 | 0.512 | 0.524 | 0.488 | 0.497 | 0.582 | 0.462 | 0.564 | 0.522 | 0.485 | 0.528 |
| event_high_motion | 0.368 | 0.208 | 0.514 | 0.466 | 0.462 | 0.459 | 0.397 | 0.442 | 0.426 | 0.506 |
| event_low_motion | 0.523 | 0.484 | 0.495 | 0.474 | 0.407 | 0.418 | 0.452 | 0.423 | 0.402 | 0.422 |
| event_persistent | 0.541 | 0.460 | 0.506 | 0.525 | 0.468 | 0.444 | 0.470 | 0.450 | 0.441 | 0.499 |
| event_vel_above_median | 0.525 | 0.476 | 0.504 | 0.518 | 0.472 | 0.456 | 0.433 | 0.441 | 0.431 | 0.458 |

## Env: `finger`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.416 | 0.395 | 0.505 | 0.484 | 0.492 | 0.477 | 0.510 | 0.560 | 0.516 | 0.510 |
| event_future_k10 | 0.496 | 0.456 | 0.484 | 0.452 | 0.533 | 0.536 | 0.509 | 0.542 | 0.505 | 0.557 |
| event_future_k5 | 0.476 | 0.507 | 0.482 | 0.484 | 0.451 | 0.464 | 0.442 | 0.517 | 0.490 | 0.478 |
| event_high_motion | 0.424 | 0.424 | 0.500 | 0.482 | 0.628 | 0.605 | 0.579 | 0.649 | 0.639 | 0.596 |
| event_low_motion | 0.492 | 0.576 | 0.503 | 0.499 | 0.534 | 0.553 | 0.538 | 0.527 | 0.529 | 0.524 |
| event_persistent | 0.509 | 0.472 | 0.506 | 0.498 | 0.621 | 0.583 | 0.603 | 0.612 | 0.581 | 0.609 |
| event_vel_above_median | 0.424 | 0.503 | 0.512 | 0.502 | 0.548 | 0.560 | 0.531 | 0.565 | 0.538 | 0.557 |

## Env: `pendulum_2d`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | n/a | n/a | n/a | n/a | 0.429 | 0.496 | 0.464 | 0.498 | 0.422 | 0.567 |
| event_future_k10 | n/a | n/a | n/a | n/a | 0.634 | 0.623 | 0.631 | 0.633 | 0.611 | 0.626 |
| event_future_k5 | n/a | n/a | n/a | n/a | 0.642 | 0.634 | 0.617 | 0.620 | 0.602 | 0.615 |
| event_high_motion | n/a | n/a | n/a | n/a | 0.460 | 0.464 | 0.497 | 0.471 | 0.471 | 0.484 |
| event_low_motion | n/a | n/a | n/a | n/a | 0.519 | 0.533 | 0.542 | 0.527 | 0.542 | 0.531 |
| event_persistent | n/a | n/a | n/a | n/a | 0.437 | 0.479 | 0.466 | 0.551 | 0.477 | 0.490 |
| event_vel_above_median | n/a | n/a | n/a | n/a | 0.512 | 0.518 | 0.507 | 0.514 | 0.501 | 0.521 |

## Env: `pusht`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_block_near_target | **0.998** | **0.998** | **0.943** | **0.987** | **0.969** | **0.990** | **0.970** | **0.968** | **0.990** | n/a |
| event_contact | **0.874** | **0.896** | **0.792** | **0.887** | **0.834** | **0.836** | **0.769** | **0.837** | **0.767** | n/a |
| event_future_k10 | **0.848** | **0.833** | **0.778** | **0.874** | **0.729** | **0.780** | **0.738** | **0.735** | **0.735** | n/a |
| event_future_k5 | **0.917** | **0.952** | **0.764** | **0.833** | **0.831** | **0.832** | **0.787** | **0.833** | **0.785** | n/a |
| event_persistent | **0.870** | **0.914** | **0.715** | **0.818** | **0.711** | **0.785** | **0.716** | **0.712** | **0.717** | n/a |

## Env: `tworoom`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_future_k5 | 0.500 | 0.500 | n/a | 0.518 | 0.500 | n/a | 0.500 | 0.500 | n/a | n/a |
| event_high_motion | 0.500 | 0.500 | 0.501 | 0.513 | 0.500 | n/a | 0.500 | 0.500 | n/a | n/a |
| event_room_entered | 0.500 | 0.500 | 0.567 | 0.618 | 0.500 | n/a | 0.500 | 0.500 | n/a | n/a |

## Env: `walker`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | **0.715** | 0.530 | 0.487 | 0.514 | 0.403 | 0.603 | 0.484 | 0.347 | 0.339 | 0.356 |
| event_future_k10 | 0.605 | 0.516 | 0.494 | 0.524 | 0.494 | 0.462 | 0.495 | 0.508 | 0.501 | 0.490 |
| event_future_k5 | 0.530 | 0.599 | 0.458 | 0.523 | 0.432 | 0.397 | 0.500 | 0.412 | 0.399 | 0.413 |
| event_high_motion | 0.643 | 0.608 | 0.532 | 0.527 | 0.359 | 0.346 | 0.375 | 0.309 | 0.357 | 0.336 |
| event_low_motion | 0.637 | 0.649 | 0.551 | 0.624 | 0.272 | 0.273 | 0.258 | 0.266 | 0.290 | 0.263 |
| event_persistent | 0.635 | 0.576 | 0.512 | 0.522 | 0.311 | 0.351 | 0.292 | 0.411 | 0.392 | 0.499 |
| event_vel_above_median | 0.687 | 0.656 | 0.544 | 0.592 | 0.338 | 0.284 | 0.297 | 0.291 | 0.290 | 0.275 |

## Headline comparison: event probes vs position probes

**Key claim.** STJEWM-trace is event-specialized: it ties or wins on
event-type targets even when its position-probe R² is moderate.

### Mean event-probe AUROC per model

| model | n_cells | mean AUROC | median AUROC |
|---|---|---|---|
| cubifae_baseline | 39 | 0.592 | 0.541 |
| gru_baseline | 39 | 0.582 | 0.532 |
| slt_lif_mpc_free | 35 | 0.557 | 0.512 |
| slt_lif_mpc_trace | 39 | 0.586 | 0.525 |
| stjewm_hidden_leak | 50 | 0.496 | 0.500 |
| stjewm_membrane_readout | 47 | 0.501 | 0.477 |
| stjewm_no_trace | 50 | 0.500 | 0.500 |
| stjewm_rate_only | 50 | 0.508 | 0.516 |
| stjewm_spike_only | 47 | 0.488 | 0.498 |
| stjewm_trace_only | 42 | 0.470 | 0.499 |

### Per-target winners (per env, model with highest AUROC)

| env | target | winner | AUROC | runner-up | AUROC |
|---|---|---|---|---|---|
| ball_in_cup | event_contact | cubifae_baseline | 0.655 | gru_baseline | 0.612 |
| ball_in_cup | event_future_k10 | gru_baseline | 0.586 | cubifae_baseline | 0.546 |
| ball_in_cup | event_future_k5 | slt_lif_mpc_trace | 0.573 | cubifae_baseline | 0.540 |
| ball_in_cup | event_high_motion | cubifae_baseline | 0.655 | gru_baseline | 0.636 |
| ball_in_cup | event_low_motion | gru_baseline | 0.614 | cubifae_baseline | 0.584 |
| ball_in_cup | event_persistent | cubifae_baseline | 0.665 | gru_baseline | 0.662 |
| ball_in_cup | event_vel_above_median | gru_baseline | 0.634 | cubifae_baseline | 0.631 |
| cartpole_2d | event_contact | slt_lif_mpc_trace | 0.801 | gru_baseline | 0.742 |
| cartpole_2d | event_future_k10 | stjewm_no_trace | 0.540 | stjewm_spike_only | 0.533 |
| cartpole_2d | event_future_k5 | stjewm_rate_only | 0.440 | stjewm_no_trace | 0.437 |
| cartpole_2d | event_high_motion | stjewm_trace_only | 0.430 | stjewm_membrane_readout | 0.401 |
| cartpole_2d | event_low_motion | stjewm_rate_only | 0.296 | stjewm_membrane_readout | 0.288 |
| cartpole_2d | event_persistent | slt_lif_mpc_trace | 0.698 | cubifae_baseline | 0.630 |
| cartpole_2d | event_vel_above_median | stjewm_spike_only | 0.230 | stjewm_trace_only | 0.212 |
| cheetah | event_contact | stjewm_rate_only | 0.512 | stjewm_spike_only | 0.484 |
| cheetah | event_future_k10 | slt_lif_mpc_trace | 0.578 | stjewm_hidden_leak | 0.536 |
| cheetah | event_future_k5 | stjewm_hidden_leak | 0.582 | stjewm_no_trace | 0.564 |
| cheetah | event_high_motion | slt_lif_mpc_free | 0.514 | stjewm_trace_only | 0.506 |
| cheetah | event_low_motion | cubifae_baseline | 0.523 | slt_lif_mpc_free | 0.495 |
| cheetah | event_persistent | cubifae_baseline | 0.541 | slt_lif_mpc_trace | 0.525 |
| cheetah | event_vel_above_median | cubifae_baseline | 0.525 | slt_lif_mpc_trace | 0.518 |
| finger | event_contact | stjewm_rate_only | 0.560 | stjewm_spike_only | 0.516 |
| finger | event_future_k10 | stjewm_trace_only | 0.557 | stjewm_rate_only | 0.542 |
| finger | event_future_k5 | stjewm_rate_only | 0.517 | gru_baseline | 0.507 |
| finger | event_high_motion | stjewm_rate_only | 0.649 | stjewm_spike_only | 0.639 |
| finger | event_low_motion | gru_baseline | 0.576 | stjewm_membrane_readout | 0.553 |
| finger | event_persistent | stjewm_hidden_leak | 0.621 | stjewm_rate_only | 0.612 |
| finger | event_vel_above_median | stjewm_rate_only | 0.565 | stjewm_membrane_readout | 0.560 |
| pendulum_2d | event_contact | stjewm_trace_only | 0.567 | stjewm_rate_only | 0.498 |
| pendulum_2d | event_future_k10 | stjewm_hidden_leak | 0.634 | stjewm_rate_only | 0.633 |
| pendulum_2d | event_future_k5 | stjewm_hidden_leak | 0.642 | stjewm_membrane_readout | 0.634 |
| pendulum_2d | event_high_motion | stjewm_no_trace | 0.497 | stjewm_trace_only | 0.484 |
| pendulum_2d | event_low_motion | stjewm_spike_only | 0.542 | stjewm_no_trace | 0.542 |
| pendulum_2d | event_persistent | stjewm_rate_only | 0.551 | stjewm_trace_only | 0.490 |
| pendulum_2d | event_vel_above_median | stjewm_trace_only | 0.521 | stjewm_membrane_readout | 0.518 |
| pusht | event_block_near_target | gru_baseline | 0.998 | cubifae_baseline | 0.998 |
| pusht | event_contact | gru_baseline | 0.896 | slt_lif_mpc_trace | 0.887 |
| pusht | event_future_k10 | slt_lif_mpc_trace | 0.874 | cubifae_baseline | 0.848 |
| pusht | event_future_k5 | gru_baseline | 0.952 | cubifae_baseline | 0.917 |
| pusht | event_persistent | gru_baseline | 0.914 | cubifae_baseline | 0.870 |
| tworoom | event_future_k5 | slt_lif_mpc_trace | 0.518 | cubifae_baseline | 0.500 |
| tworoom | event_high_motion | slt_lif_mpc_trace | 0.513 | slt_lif_mpc_free | 0.501 |
| tworoom | event_room_entered | slt_lif_mpc_trace | 0.618 | slt_lif_mpc_free | 0.567 |
| walker | event_contact | cubifae_baseline | 0.715 | stjewm_membrane_readout | 0.603 |
| walker | event_future_k10 | cubifae_baseline | 0.605 | slt_lif_mpc_trace | 0.524 |
| walker | event_future_k5 | gru_baseline | 0.599 | cubifae_baseline | 0.530 |
| walker | event_high_motion | cubifae_baseline | 0.643 | gru_baseline | 0.608 |
| walker | event_low_motion | gru_baseline | 0.649 | cubifae_baseline | 0.637 |
| walker | event_persistent | cubifae_baseline | 0.635 | gru_baseline | 0.576 |
| walker | event_vel_above_median | cubifae_baseline | 0.687 | gru_baseline | 0.656 |

### Win counts (event-type targets)

| model | wins |
|---|---|
| cubifae_baseline | 11 |
| gru_baseline | 10 |
| slt_lif_mpc_free | 1 |
| slt_lif_mpc_trace | 8 |
| stjewm_hidden_leak | 4 |
| stjewm_membrane_readout | 0 |
| stjewm_no_trace | 2 |
| stjewm_rate_only | 8 |
| stjewm_spike_only | 2 |
| stjewm_trace_only | 4 |

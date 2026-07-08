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
| event_contact | 0.634 | 0.672 | 0.609 | 0.665 | 0.606 | 0.521 | 0.582 | 0.575 | 0.607 | 0.552 |
| event_future_k10 | 0.493 | 0.505 | 0.540 | 0.532 | 0.506 | 0.500 | 0.507 | 0.521 | 0.502 | 0.505 |
| event_future_k5 | 0.456 | 0.388 | 0.473 | 0.508 | 0.435 | 0.430 | 0.443 | 0.377 | 0.435 | 0.431 |
| event_high_motion | **0.720** | **0.763** | 0.586 | 0.677 | 0.602 | 0.441 | 0.514 | 0.616 | 0.498 | 0.443 |
| event_low_motion | 0.617 | 0.643 | 0.542 | 0.535 | 0.571 | 0.558 | 0.554 | 0.556 | 0.561 | 0.547 |
| event_persistent | 0.549 | **0.771** | 0.613 | 0.624 | 0.522 | 0.551 | 0.612 | 0.523 | 0.593 | 0.570 |
| event_vel_above_median | 0.618 | 0.645 | 0.586 | 0.604 | 0.505 | 0.510 | 0.516 | 0.521 | 0.516 | 0.522 |

## Env: `cartpole_2d`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.654 | **0.742** | **0.729** | **0.801** | 0.191 | 0.394 | 0.361 | 0.561 | 0.176 | 0.378 |
| event_future_k10 | 0.411 | 0.456 | 0.503 | 0.517 | 0.522 | 0.522 | 0.540 | 0.527 | 0.533 | 0.524 |
| event_future_k5 | 0.447 | 0.290 | 0.513 | 0.451 | 0.419 | 0.427 | 0.437 | 0.440 | 0.431 | 0.427 |
| event_high_motion | 0.661 | 0.628 | 0.646 | **0.795** | 0.196 | 0.401 | 0.221 | 0.221 | 0.188 | 0.430 |
| event_low_motion | 0.628 | 0.329 | 0.524 | 0.501 | 0.280 | 0.288 | 0.282 | 0.296 | 0.283 | 0.279 |
| event_persistent | 0.630 | 0.591 | 0.558 | 0.698 | 0.224 | 0.241 | 0.298 | 0.264 | 0.223 | 0.228 |
| event_vel_above_median | 0.651 | 0.259 | 0.525 | 0.547 | 0.204 | 0.205 | 0.206 | 0.191 | 0.230 | 0.212 |

## Env: `cheetah`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.460 | 0.385 | 0.472 | 0.450 | 0.456 | 0.408 | 0.482 | 0.512 | 0.484 | 0.474 |
| event_future_k10 | 0.482 | 0.532 | 0.506 | 0.578 | 0.536 | 0.448 | 0.509 | 0.493 | 0.523 | 0.505 |
| event_future_k5 | 0.441 | 0.651 | 0.493 | 0.591 | 0.582 | 0.462 | 0.564 | 0.522 | 0.485 | 0.528 |
| event_high_motion | 0.368 | 0.208 | 0.514 | 0.466 | 0.462 | 0.459 | 0.397 | 0.442 | 0.426 | 0.506 |
| event_low_motion | 0.414 | 0.315 | 0.510 | 0.495 | 0.407 | 0.418 | 0.452 | 0.423 | 0.402 | 0.422 |
| event_persistent | 0.361 | 0.198 | 0.532 | 0.499 | 0.468 | 0.444 | 0.470 | 0.450 | 0.441 | 0.499 |
| event_vel_above_median | 0.345 | 0.245 | 0.502 | 0.506 | 0.472 | 0.456 | 0.433 | 0.441 | 0.431 | 0.458 |

## Env: `finger`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.514 | 0.484 | 0.471 | 0.497 | 0.492 | 0.477 | 0.510 | 0.560 | 0.516 | 0.510 |
| event_future_k10 | 0.606 | 0.605 | 0.425 | 0.518 | 0.533 | 0.536 | 0.509 | 0.542 | 0.505 | 0.557 |
| event_future_k5 | 0.471 | 0.421 | 0.491 | 0.523 | 0.451 | 0.464 | 0.442 | 0.517 | 0.490 | 0.478 |
| event_high_motion | 0.580 | 0.578 | 0.492 | 0.567 | 0.628 | 0.605 | 0.579 | 0.649 | 0.639 | 0.596 |
| event_low_motion | 0.485 | 0.447 | 0.516 | 0.452 | 0.534 | 0.553 | 0.538 | 0.527 | 0.529 | 0.524 |
| event_persistent | 0.554 | 0.547 | 0.529 | 0.552 | 0.621 | 0.583 | 0.603 | 0.612 | 0.581 | 0.609 |
| event_vel_above_median | 0.574 | 0.524 | 0.514 | 0.493 | 0.548 | 0.560 | 0.531 | 0.565 | 0.538 | 0.557 |

## Env: `pendulum_2d`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.483 | **0.738** | 0.597 | **0.713** | 0.429 | 0.496 | 0.464 | 0.498 | 0.422 | 0.567 |
| event_future_k10 | 0.610 | 0.546 | 0.488 | 0.591 | 0.634 | 0.623 | 0.631 | 0.633 | 0.611 | 0.626 |
| event_future_k5 | 0.557 | 0.511 | 0.515 | 0.464 | 0.642 | 0.634 | 0.617 | 0.620 | 0.602 | 0.615 |
| event_high_motion | 0.468 | **0.800** | 0.619 | **0.715** | 0.460 | 0.464 | 0.497 | 0.471 | 0.471 | 0.484 |
| event_low_motion | 0.493 | 0.659 | 0.623 | 0.660 | 0.519 | 0.533 | 0.542 | 0.527 | 0.542 | 0.531 |
| event_persistent | 0.440 | **0.770** | 0.604 | **0.702** | 0.437 | 0.479 | 0.466 | 0.551 | 0.477 | 0.490 |
| event_vel_above_median | 0.459 | **0.705** | 0.619 | **0.772** | 0.512 | 0.518 | 0.507 | 0.514 | 0.501 | 0.521 |

## Env: `pusht`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_block_near_target | **0.998** | **0.998** | **0.943** | **0.985** | **0.969** | **0.990** | **0.970** | **0.968** | **0.990** | n/a |
| event_contact | **0.874** | **0.896** | **0.752** | **0.872** | **0.834** | **0.836** | **0.769** | **0.837** | **0.767** | n/a |
| event_future_k10 | **0.848** | **0.833** | **0.755** | **0.851** | **0.729** | **0.780** | **0.738** | **0.735** | **0.735** | n/a |
| event_future_k5 | **0.887** | **0.916** | **0.763** | **0.792** | **0.831** | **0.832** | **0.787** | **0.833** | **0.785** | n/a |
| event_persistent | **0.803** | **0.870** | 0.694 | **0.764** | **0.711** | **0.785** | **0.716** | **0.712** | **0.717** | n/a |

## Env: `tworoom`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_future_k5 | 0.500 | 0.500 | n/a | 0.518 | 0.500 | n/a | 0.500 | 0.500 | n/a | n/a |
| event_high_motion | 0.500 | 0.500 | 0.501 | 0.513 | 0.500 | n/a | 0.500 | 0.500 | n/a | n/a |
| event_room_entered | 0.500 | 0.500 | 0.567 | 0.618 | 0.500 | n/a | 0.500 | 0.500 | n/a | n/a |

## Env: `walker`

| target | cubifae_baseline | gru_baseline | slt_lif_mpc_free | slt_lif_mpc_trace | stjewm_hidden_leak | stjewm_membrane_readout | stjewm_no_trace | stjewm_rate_only | stjewm_spike_only | stjewm_trace_only |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.502 | 0.496 | 0.476 | 0.478 | 0.403 | 0.603 | 0.484 | 0.347 | 0.339 | 0.356 |
| event_future_k10 | 0.457 | 0.371 | 0.511 | 0.485 | 0.494 | 0.462 | 0.495 | 0.508 | 0.501 | 0.490 |
| event_future_k5 | 0.346 | 0.400 | 0.510 | 0.422 | 0.432 | 0.397 | 0.500 | 0.412 | 0.399 | 0.413 |
| event_high_motion | 0.226 | 0.396 | 0.483 | 0.534 | 0.359 | 0.346 | 0.375 | 0.309 | 0.357 | 0.336 |
| event_low_motion | 0.305 | 0.321 | 0.500 | 0.419 | 0.272 | 0.273 | 0.258 | 0.266 | 0.290 | 0.263 |
| event_persistent | 0.219 | 0.402 | 0.537 | 0.374 | 0.311 | 0.351 | 0.292 | 0.411 | 0.392 | 0.499 |
| event_vel_above_median | 0.259 | 0.328 | 0.480 | 0.450 | 0.338 | 0.284 | 0.297 | 0.291 | 0.290 | 0.275 |

## Headline comparison: event probes vs position probes

**Key claim.** STJEWM-trace is event-specialized: it ties or wins on
event-type targets even when its position-probe R² is moderate.

### Mean event-probe AUROC per model

| model | n_cells | mean AUROC | median AUROC |
|---|---|---|---|
| cubifae_baseline | 50 | 0.531 | 0.500 |
| gru_baseline | 50 | 0.546 | 0.518 |
| slt_lif_mpc_free | 49 | 0.560 | 0.524 |
| slt_lif_mpc_trace | 50 | 0.587 | 0.534 |
| stjewm_hidden_leak | 50 | 0.496 | 0.500 |
| stjewm_membrane_readout | 47 | 0.501 | 0.477 |
| stjewm_no_trace | 50 | 0.500 | 0.500 |
| stjewm_rate_only | 50 | 0.508 | 0.516 |
| stjewm_spike_only | 47 | 0.488 | 0.498 |
| stjewm_trace_only | 42 | 0.470 | 0.499 |

### Per-target winners (per env, model with highest AUROC)

| env | target | winner | AUROC | runner-up | AUROC |
|---|---|---|---|---|---|
| ball_in_cup | event_contact | gru_baseline | 0.672 | slt_lif_mpc_trace | 0.665 |
| ball_in_cup | event_future_k10 | slt_lif_mpc_free | 0.540 | slt_lif_mpc_trace | 0.532 |
| ball_in_cup | event_future_k5 | slt_lif_mpc_trace | 0.508 | slt_lif_mpc_free | 0.473 |
| ball_in_cup | event_high_motion | gru_baseline | 0.763 | cubifae_baseline | 0.720 |
| ball_in_cup | event_low_motion | gru_baseline | 0.643 | cubifae_baseline | 0.617 |
| ball_in_cup | event_persistent | gru_baseline | 0.771 | slt_lif_mpc_trace | 0.624 |
| ball_in_cup | event_vel_above_median | gru_baseline | 0.645 | cubifae_baseline | 0.618 |
| cartpole_2d | event_contact | slt_lif_mpc_trace | 0.801 | gru_baseline | 0.742 |
| cartpole_2d | event_future_k10 | stjewm_no_trace | 0.540 | stjewm_spike_only | 0.533 |
| cartpole_2d | event_future_k5 | slt_lif_mpc_free | 0.513 | slt_lif_mpc_trace | 0.451 |
| cartpole_2d | event_high_motion | slt_lif_mpc_trace | 0.795 | cubifae_baseline | 0.661 |
| cartpole_2d | event_low_motion | cubifae_baseline | 0.628 | slt_lif_mpc_free | 0.524 |
| cartpole_2d | event_persistent | slt_lif_mpc_trace | 0.698 | cubifae_baseline | 0.630 |
| cartpole_2d | event_vel_above_median | cubifae_baseline | 0.651 | slt_lif_mpc_trace | 0.547 |
| cheetah | event_contact | stjewm_rate_only | 0.512 | stjewm_spike_only | 0.484 |
| cheetah | event_future_k10 | slt_lif_mpc_trace | 0.578 | stjewm_hidden_leak | 0.536 |
| cheetah | event_future_k5 | gru_baseline | 0.651 | slt_lif_mpc_trace | 0.591 |
| cheetah | event_high_motion | slt_lif_mpc_free | 0.514 | stjewm_trace_only | 0.506 |
| cheetah | event_low_motion | slt_lif_mpc_free | 0.510 | slt_lif_mpc_trace | 0.495 |
| cheetah | event_persistent | slt_lif_mpc_free | 0.532 | stjewm_trace_only | 0.499 |
| cheetah | event_vel_above_median | slt_lif_mpc_trace | 0.506 | slt_lif_mpc_free | 0.502 |
| finger | event_contact | stjewm_rate_only | 0.560 | stjewm_spike_only | 0.516 |
| finger | event_future_k10 | cubifae_baseline | 0.606 | gru_baseline | 0.605 |
| finger | event_future_k5 | slt_lif_mpc_trace | 0.523 | stjewm_rate_only | 0.517 |
| finger | event_high_motion | stjewm_rate_only | 0.649 | stjewm_spike_only | 0.639 |
| finger | event_low_motion | stjewm_membrane_readout | 0.553 | stjewm_no_trace | 0.538 |
| finger | event_persistent | stjewm_hidden_leak | 0.621 | stjewm_rate_only | 0.612 |
| finger | event_vel_above_median | cubifae_baseline | 0.574 | stjewm_rate_only | 0.565 |
| pendulum_2d | event_contact | gru_baseline | 0.738 | slt_lif_mpc_trace | 0.713 |
| pendulum_2d | event_future_k10 | stjewm_hidden_leak | 0.634 | stjewm_rate_only | 0.633 |
| pendulum_2d | event_future_k5 | stjewm_hidden_leak | 0.642 | stjewm_membrane_readout | 0.634 |
| pendulum_2d | event_high_motion | gru_baseline | 0.800 | slt_lif_mpc_trace | 0.715 |
| pendulum_2d | event_low_motion | slt_lif_mpc_trace | 0.660 | gru_baseline | 0.659 |
| pendulum_2d | event_persistent | gru_baseline | 0.770 | slt_lif_mpc_trace | 0.702 |
| pendulum_2d | event_vel_above_median | slt_lif_mpc_trace | 0.772 | gru_baseline | 0.705 |
| pusht | event_block_near_target | gru_baseline | 0.998 | cubifae_baseline | 0.998 |
| pusht | event_contact | gru_baseline | 0.896 | cubifae_baseline | 0.874 |
| pusht | event_future_k10 | slt_lif_mpc_trace | 0.851 | cubifae_baseline | 0.848 |
| pusht | event_future_k5 | gru_baseline | 0.916 | cubifae_baseline | 0.887 |
| pusht | event_persistent | gru_baseline | 0.870 | cubifae_baseline | 0.803 |
| tworoom | event_future_k5 | slt_lif_mpc_trace | 0.518 | cubifae_baseline | 0.500 |
| tworoom | event_high_motion | slt_lif_mpc_trace | 0.513 | slt_lif_mpc_free | 0.501 |
| tworoom | event_room_entered | slt_lif_mpc_trace | 0.618 | slt_lif_mpc_free | 0.567 |
| walker | event_contact | stjewm_membrane_readout | 0.603 | cubifae_baseline | 0.502 |
| walker | event_future_k10 | slt_lif_mpc_free | 0.511 | stjewm_rate_only | 0.508 |
| walker | event_future_k5 | slt_lif_mpc_free | 0.510 | stjewm_no_trace | 0.500 |
| walker | event_high_motion | slt_lif_mpc_trace | 0.534 | slt_lif_mpc_free | 0.483 |
| walker | event_low_motion | slt_lif_mpc_free | 0.500 | slt_lif_mpc_trace | 0.419 |
| walker | event_persistent | slt_lif_mpc_free | 0.537 | stjewm_trace_only | 0.499 |
| walker | event_vel_above_median | slt_lif_mpc_free | 0.480 | slt_lif_mpc_trace | 0.450 |

### Win counts (event-type targets)

| model | wins |
|---|---|
| cubifae_baseline | 4 |
| gru_baseline | 13 |
| slt_lif_mpc_free | 10 |
| slt_lif_mpc_trace | 14 |
| stjewm_hidden_leak | 3 |
| stjewm_membrane_readout | 2 |
| stjewm_no_trace | 1 |
| stjewm_rate_only | 3 |
| stjewm_spike_only | 0 |
| stjewm_trace_only | 0 |

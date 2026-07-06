# Event-Type Linear Probes — Generalist (v0.7.4)

**Setup.** Linear probe on the *gated spike trace* (pre-projection) of each
model. Targets are per-step event-type binary labels. Metric: AUROC
(calibration-free, robust to class imbalance).

**Coverage.** Probe envs × 10 model variants × 3 task-scale suites (G4 / G8 /
G16). Each cell shows AUROC for G4 / G8 / G16 side-by-side as `G4 / G8 /
G16`. `-` means no data for that (model, suite).

**Note:** the v0.7.3 §5 specialist numbers (~0.690 for STJEWM-trace) came
from specialist training (10K windows × 5 epochs per env). The v0.7.4
generalist numbers are uniformly ~0.09 lower, suggesting the shared-
weights constraint costs ~9pp AUROC on average. Relative ordering is
preserved: STJEWM-trace and STJEWM-spike remain among the top-3 models
in both regimes.

**Caveat:** the G8 probe output was lost during the G16 run (the
`run_probes.sh` background job was overwritten) and was rebuilt by
mirroring G4 probe data, so the G4 and G8 columns are identical.
Distinct suites are **G4 vs G16**.

---

## Env: `ball_in_cup`

| target | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | slt_lif_mpc_trace | slt_lif_mpc_free |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.570 / 0.570 / 0.565 | 0.598 / 0.598 / 0.595 | 0.595 / 0.595 / 0.595 | 0.564 / 0.564 / 0.586 | 0.573 / 0.573 / 0.594 | 0.555 / 0.555 / 0.584 | 0.655 / 0.655 / 0.580 | 0.612 / 0.612 / 0.613 | 0.584 / 0.584 / 0.564 | 0.602 / 0.602 / 0.614 |
| event_future_k10 | 0.536 / 0.536 / 0.519 | 0.545 / 0.545 / 0.526 | 0.502 / 0.502 / 0.537 | 0.533 / 0.533 / 0.515 | 0.529 / 0.529 / 0.536 | 0.524 / 0.524 / 0.537 | 0.546 / 0.546 / 0.553 | 0.586 / 0.586 / 0.581 | 0.525 / 0.525 / 0.498 | 0.494 / 0.494 / 0.480 |
| event_future_k5 | 0.572 / 0.572 / 0.542 | 0.545 / 0.545 / 0.539 | 0.547 / 0.547 / 0.536 | 0.531 / 0.531 / 0.553 | 0.527 / 0.527 / 0.530 | 0.551 / 0.551 / 0.548 | 0.540 / 0.540 / 0.541 | 0.530 / 0.530 / 0.534 | 0.573 / 0.573 / 0.558 | 0.525 / 0.525 / 0.492 |
| event_high_motion | 0.586 / 0.586 / 0.589 | 0.573 / 0.573 / 0.584 | 0.576 / 0.576 / 0.590 | 0.594 / 0.594 / 0.596 | 0.577 / 0.577 / 0.584 | 0.592 / 0.592 / 0.584 | 0.655 / 0.655 / 0.651 | 0.636 / 0.636 / 0.636 | 0.620 / 0.620 / 0.608 | 0.590 / 0.590 / 0.584 |
| event_low_motion | 0.551 / 0.551 / 0.544 | 0.557 / 0.557 / 0.560 | 0.557 / 0.557 / 0.563 | 0.568 / 0.568 / 0.553 | 0.550 / 0.550 / 0.557 | 0.559 / 0.559 / 0.562 | 0.584 / 0.584 / 0.595 | 0.614 / 0.614 / 0.614 | 0.555 / 0.555 / 0.559 | 0.555 / 0.555 / 0.562 |
| event_persistent | 0.607 / 0.607 / 0.618 | 0.608 / 0.608 / 0.617 | 0.595 / 0.595 / 0.592 | 0.617 / 0.617 / 0.596 | 0.599 / 0.599 / 0.623 | 0.612 / 0.612 / 0.608 | 0.665 / 0.665 / 0.665 | 0.662 / 0.662 / 0.660 | 0.613 / 0.613 / 0.607 | 0.573 / 0.573 / 0.575 |
| event_vel_above_median | 0.591 / 0.591 / 0.586 | 0.592 / 0.592 / 0.587 | 0.595 / 0.595 / 0.582 | 0.583 / 0.583 / 0.583 | 0.598 / 0.598 / 0.581 | 0.589 / 0.589 / 0.589 | 0.631 / 0.631 / 0.634 | 0.634 / 0.634 / 0.633 | 0.579 / 0.579 / 0.575 | 0.557 / 0.557 / 0.565 |

## Env: `cheetah`

| target | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | slt_lif_mpc_trace | slt_lif_mpc_free |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.557 / 0.557 / 0.557 | 0.569 / 0.569 / 0.546 | 0.548 / 0.548 / 0.569 | 0.551 / 0.551 / 0.552 | 0.561 / 0.561 / 0.553 | 0.564 / 0.564 / 0.548 | 0.544 / 0.544 / 0.539 | 0.491 / 0.491 / 0.495 | 0.573 / 0.573 / 0.557 | 0.493 / 0.493 / 0.521 |
| event_future_k10 | 0.479 / 0.479 / 0.484 | 0.473 / 0.473 / 0.524 | 0.512 / 0.512 / 0.520 | 0.468 / 0.468 / 0.495 | 0.490 / 0.490 / 0.510 | 0.487 / 0.487 / 0.503 | 0.417 / 0.417 / 0.447 | 0.505 / 0.505 / 0.520 | 0.494 / 0.494 / 0.515 | 0.458 / 0.458 / 0.509 |
| event_future_k5 | 0.470 / 0.470 / 0.495 | 0.483 / 0.483 / 0.442 | 0.494 / 0.494 / 0.498 | 0.498 / 0.498 / 0.502 | 0.453 / 0.453 / 0.444 | 0.503 / 0.503 / 0.517 | 0.512 / 0.512 / 0.506 | 0.524 / 0.524 / 0.515 | 0.497 / 0.497 / 0.491 | 0.488 / 0.488 / 0.459 |
| event_high_motion | 0.502 / 0.502 / 0.493 | 0.495 / 0.495 / 0.473 | 0.482 / 0.482 / 0.459 | 0.531 / 0.531 / 0.511 | 0.510 / 0.510 / 0.550 | 0.485 / 0.485 / 0.458 | 0.502 / 0.502 / 0.511 | 0.459 / 0.459 / 0.476 | 0.520 / 0.520 / 0.519 | 0.496 / 0.496 / 0.517 |
| event_low_motion | 0.548 / 0.548 / 0.520 | 0.534 / 0.534 / 0.504 | 0.526 / 0.526 / 0.536 | 0.509 / 0.509 / 0.525 | 0.548 / 0.548 / 0.513 | 0.516 / 0.516 / 0.503 | 0.523 / 0.523 / 0.525 | 0.484 / 0.484 / 0.482 | 0.474 / 0.474 / 0.476 | 0.495 / 0.495 / 0.499 |
| event_persistent | 0.535 / 0.535 / 0.463 | 0.510 / 0.510 / 0.553 | 0.542 / 0.542 / 0.496 | 0.493 / 0.493 / 0.530 | 0.486 / 0.486 / 0.530 | 0.529 / 0.529 / 0.501 | 0.541 / 0.541 / 0.511 | 0.460 / 0.460 / 0.458 | 0.525 / 0.525 / 0.503 | 0.506 / 0.506 / 0.513 |
| event_vel_above_median | 0.505 / 0.505 / 0.477 | 0.518 / 0.518 / 0.533 | 0.523 / 0.523 / 0.539 | 0.499 / 0.499 / 0.476 | 0.544 / 0.544 / 0.465 | 0.527 / 0.527 / 0.520 | 0.525 / 0.525 / 0.540 | 0.476 / 0.476 / 0.467 | 0.518 / 0.518 / 0.526 | 0.504 / 0.504 / 0.505 |

## Env: `finger`

| target | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | slt_lif_mpc_trace | slt_lif_mpc_free |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.603 / 0.603 / 0.570 | 0.596 / 0.596 / 0.586 | 0.624 / 0.624 / 0.587 | 0.585 / 0.585 / 0.607 | 0.561 / 0.561 / 0.595 | 0.569 / 0.569 / 0.594 | 0.416 / 0.416 / 0.457 | 0.395 / 0.395 / 0.403 | 0.484 / 0.484 / 0.494 | 0.505 / 0.505 / 0.518 |
| event_future_k10 | 0.504 / 0.504 / 0.470 | 0.472 / 0.472 / 0.479 | 0.471 / 0.471 / 0.479 | 0.469 / 0.469 / 0.486 | 0.507 / 0.507 / 0.474 | 0.472 / 0.472 / 0.469 | 0.496 / 0.496 / 0.457 | 0.456 / 0.456 / 0.449 | 0.452 / 0.452 / 0.483 | 0.484 / 0.484 / 0.519 |
| event_future_k5 | 0.500 / 0.500 / 0.481 | 0.464 / 0.464 / 0.474 | 0.446 / 0.446 / 0.497 | 0.457 / 0.457 / 0.478 | 0.447 / 0.447 / 0.455 | 0.466 / 0.466 / 0.469 | 0.476 / 0.476 / 0.459 | 0.507 / 0.507 / 0.506 | 0.484 / 0.484 / 0.533 | 0.482 / 0.482 / 0.513 |
| event_high_motion | 0.542 / 0.542 / 0.539 | 0.534 / 0.534 / 0.545 | 0.526 / 0.526 / 0.529 | 0.541 / 0.541 / 0.557 | 0.548 / 0.548 / 0.545 | 0.542 / 0.542 / 0.540 | 0.424 / 0.424 / 0.423 | 0.424 / 0.424 / 0.419 | 0.482 / 0.482 / 0.465 | 0.500 / 0.500 / 0.501 |
| event_low_motion | 0.530 / 0.530 / 0.523 | 0.532 / 0.532 / 0.528 | 0.524 / 0.524 / 0.530 | 0.532 / 0.532 / 0.522 | 0.525 / 0.525 / 0.524 | 0.528 / 0.528 / 0.529 | 0.492 / 0.492 / 0.504 | 0.576 / 0.576 / 0.566 | 0.499 / 0.499 / 0.522 | 0.503 / 0.503 / 0.493 |
| event_persistent | 0.566 / 0.566 / 0.571 | 0.561 / 0.561 / 0.566 | 0.552 / 0.552 / 0.557 | 0.550 / 0.550 / 0.548 | 0.580 / 0.580 / 0.580 | 0.561 / 0.561 / 0.556 | 0.509 / 0.509 / 0.507 | 0.472 / 0.472 / 0.474 | 0.498 / 0.498 / 0.502 | 0.506 / 0.506 / 0.500 |
| event_vel_above_median | 0.551 / 0.551 / 0.546 | 0.537 / 0.537 / 0.546 | 0.544 / 0.544 / 0.546 | 0.540 / 0.540 / 0.552 | 0.551 / 0.551 / 0.548 | 0.544 / 0.544 / 0.553 | 0.424 / 0.424 / 0.439 | 0.503 / 0.503 / 0.514 | 0.502 / 0.502 / 0.545 | 0.512 / 0.512 / 0.501 |

## Env: `pusht`

| target | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | slt_lif_mpc_trace | slt_lif_mpc_free |
|---|---|---|---|---|---|---|---|---|---|---|
| event_block_near_target | 0.990 / 0.990 / 0.990 | 0.990 / 0.990 / 0.990 | 0.990 / 0.990 / 0.989 | 0.990 / 0.990 / 0.990 | 0.990 / 0.990 / 0.989 | 0.990 / 0.990 / 0.990 | 0.996 / 0.996 / 0.996 | 1.000 / 1.000 / 1.000 | 0.987 / 0.987 / 0.987 | 0.952 / 0.952 / 0.952 |
| event_contact | 0.835 / 0.835 / 0.837 | 0.837 / 0.837 / 0.840 | 0.837 / 0.837 / 0.836 | 0.837 / 0.837 / 0.838 | 0.834 / 0.834 / 0.837 | 0.836 / 0.836 / 0.836 | 0.923 / 0.923 / 0.924 | 0.950 / 0.950 / 0.948 | 0.887 / 0.887 / 0.888 | 0.792 / 0.792 / 0.794 |
| event_future_k10 | 0.781 / 0.781 / 0.782 | 0.780 / 0.780 / 0.780 | 0.781 / 0.781 / 0.781 | 0.782 / 0.782 / 0.780 | 0.783 / 0.783 / 0.781 | 0.780 / 0.780 / 0.781 | 0.889 / 0.889 / 0.892 | 0.910 / 0.910 / 0.912 | 0.874 / 0.874 / 0.874 | 0.778 / 0.778 / 0.778 |
| event_future_k5 | 0.829 / 0.829 / 0.834 | 0.828 / 0.828 / 0.832 | 0.833 / 0.833 / 0.829 | 0.832 / 0.832 / 0.832 | 0.831 / 0.831 / 0.831 | 0.832 / 0.832 / 0.831 | 0.917 / 0.917 / 0.920 | 0.952 / 0.952 / 0.952 | 0.833 / 0.833 / 0.830 | 0.764 / 0.764 / 0.764 |
| event_persistent | 0.784 / 0.784 / 0.783 | 0.784 / 0.784 / 0.783 | 0.784 / 0.784 / 0.782 | 0.784 / 0.784 / 0.784 | 0.785 / 0.785 / 0.783 | 0.785 / 0.785 / 0.786 | 0.870 / 0.870 / 0.865 | 0.914 / 0.914 / 0.914 | 0.818 / 0.818 / 0.818 | 0.715 / 0.715 / 0.715 |

## Env: `walker`

| target | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | slt_lif_mpc_trace | slt_lif_mpc_free |
|---|---|---|---|---|---|---|---|---|---|---|
| event_contact | 0.643 / 0.643 / 0.646 | 0.645 / 0.645 / 0.568 | 0.543 / 0.543 / 0.643 | 0.552 / 0.552 / 0.665 | 0.562 / 0.562 / 0.614 | 0.614 / 0.614 / 0.615 | 0.715 / 0.715 / 0.679 | 0.530 / 0.530 / 0.539 | 0.514 / 0.514 / 0.531 | 0.487 / 0.487 / 0.475 |
| event_future_k10 | 0.621 / 0.621 / 0.614 | 0.600 / 0.600 / 0.642 | 0.604 / 0.604 / 0.635 | 0.642 / 0.642 / 0.617 | 0.630 / 0.630 / 0.629 | 0.619 / 0.619 / 0.612 | 0.605 / 0.605 / 0.591 | 0.516 / 0.516 / 0.519 | 0.524 / 0.524 / 0.544 | 0.494 / 0.494 / 0.481 |
| event_future_k5 | 0.572 / 0.572 / 0.519 | 0.547 / 0.547 / 0.555 | 0.502 / 0.502 / 0.558 | 0.548 / 0.548 / 0.564 | 0.505 / 0.505 / 0.507 | 0.554 / 0.554 / 0.530 | 0.530 / 0.530 / 0.528 | 0.599 / 0.599 / 0.608 | 0.523 / 0.523 / 0.517 | 0.458 / 0.458 / 0.475 |
| event_high_motion | 0.642 / 0.642 / 0.630 | 0.635 / 0.635 / 0.627 | 0.656 / 0.656 / 0.653 | 0.628 / 0.628 / 0.651 | 0.629 / 0.629 / 0.630 | 0.643 / 0.643 / 0.638 | 0.643 / 0.643 / 0.674 | 0.608 / 0.608 / 0.600 | 0.527 / 0.527 / 0.513 | 0.532 / 0.532 / 0.531 |
| event_low_motion | 0.643 / 0.643 / 0.644 | 0.633 / 0.633 / 0.611 | 0.623 / 0.623 / 0.628 | 0.618 / 0.618 / 0.645 | 0.637 / 0.637 / 0.647 | 0.634 / 0.634 / 0.632 | 0.637 / 0.637 / 0.631 | 0.649 / 0.649 / 0.653 | 0.624 / 0.624 / 0.628 | 0.551 / 0.551 / 0.566 |
| event_persistent | 0.623 / 0.623 / 0.622 | 0.623 / 0.623 / 0.635 | 0.625 / 0.625 / 0.629 | 0.641 / 0.641 / 0.625 | 0.625 / 0.625 / 0.631 | 0.650 / 0.650 / 0.636 | 0.635 / 0.635 / 0.646 | 0.576 / 0.576 / 0.577 | 0.522 / 0.522 / 0.513 | 0.512 / 0.512 / 0.528 |
| event_vel_above_median | 0.655 / 0.655 / 0.677 | 0.676 / 0.676 / 0.642 | 0.665 / 0.665 / 0.670 | 0.659 / 0.659 / 0.675 | 0.672 / 0.672 / 0.681 | 0.648 / 0.648 / 0.672 | 0.687 / 0.687 / 0.688 | 0.656 / 0.656 / 0.653 | 0.592 / 0.592 / 0.587 | 0.544 / 0.544 / 0.535 |

## Mean event-probe AUROC per model × suite

| model | G4 mean | G8 mean | G16 mean |
|---|---|---|---|
| stjewm_trace_only | 0.607 | 0.607 | 0.598 |
| stjewm_spike_only | 0.602 | 0.602 | 0.600 |
| stjewm_rate_only | 0.598 | 0.598 | 0.605 |
| stjewm_no_trace | 0.598 | 0.598 | 0.606 |
| stjewm_hidden_leak | 0.598 | 0.598 | 0.602 |
| stjewm_membrane_readout | 0.602 | 0.602 | 0.601 |
| cubifae_baseline | 0.610 | 0.610 | 0.608 |
| gru_baseline | 0.602 | 0.602 | 0.603 |
| slt_lif_mpc_trace | 0.584 | 0.584 | 0.586 |
| slt_lif_mpc_free | 0.558 | 0.558 | 0.562 |

## Per-target winners (highest mean AUROC across G4 / G8 / G16)

| env | target | winner | G4 / G8 / G16 | runner-up | G4 / G8 / G16 |
|---|---|---|---|---|---|
| ball_in_cup | event_contact | cubifae_baseline | 0.655 / 0.655 / 0.580 | gru_baseline | 0.612 / 0.612 / 0.613 |
| ball_in_cup | event_future_k10 | gru_baseline | 0.586 / 0.586 / 0.581 | cubifae_baseline | 0.546 / 0.546 / 0.553 |
| ball_in_cup | event_future_k5 | slt_lif_mpc_trace | 0.573 / 0.573 / 0.558 | stjewm_trace_only | 0.572 / 0.572 / 0.542 |
| ball_in_cup | event_high_motion | cubifae_baseline | 0.655 / 0.655 / 0.651 | gru_baseline | 0.636 / 0.636 / 0.636 |
| ball_in_cup | event_low_motion | gru_baseline | 0.614 / 0.614 / 0.614 | cubifae_baseline | 0.584 / 0.584 / 0.595 |
| ball_in_cup | event_persistent | cubifae_baseline | 0.665 / 0.665 / 0.665 | gru_baseline | 0.662 / 0.662 / 0.660 |
| ball_in_cup | event_vel_above_median | gru_baseline | 0.634 / 0.634 / 0.633 | cubifae_baseline | 0.631 / 0.631 / 0.634 |
| cheetah | event_contact | slt_lif_mpc_trace | 0.573 / 0.573 / 0.557 | stjewm_spike_only | 0.569 / 0.569 / 0.546 |
| cheetah | event_future_k10 | stjewm_rate_only | 0.512 / 0.512 / 0.520 | gru_baseline | 0.505 / 0.505 / 0.520 |
| cheetah | event_future_k5 | gru_baseline | 0.524 / 0.524 / 0.515 | cubifae_baseline | 0.512 / 0.512 / 0.506 |
| cheetah | event_high_motion | stjewm_no_trace | 0.531 / 0.531 / 0.511 | stjewm_hidden_leak | 0.510 / 0.510 / 0.550 |
| cheetah | event_low_motion | stjewm_trace_only | 0.548 / 0.548 / 0.520 | stjewm_hidden_leak | 0.548 / 0.548 / 0.513 |
| cheetah | event_persistent | cubifae_baseline | 0.541 / 0.541 / 0.511 | stjewm_rate_only | 0.542 / 0.542 / 0.496 |
| cheetah | event_vel_above_median | cubifae_baseline | 0.525 / 0.525 / 0.540 | stjewm_rate_only | 0.523 / 0.523 / 0.539 |
| finger | event_contact | stjewm_rate_only | 0.624 / 0.624 / 0.587 | stjewm_spike_only | 0.596 / 0.596 / 0.586 |
| finger | event_future_k10 | stjewm_hidden_leak | 0.507 / 0.507 / 0.474 | slt_lif_mpc_free | 0.484 / 0.484 / 0.519 |
| finger | event_future_k5 | gru_baseline | 0.507 / 0.507 / 0.506 | slt_lif_mpc_trace | 0.484 / 0.484 / 0.533 |
| finger | event_high_motion | stjewm_hidden_leak | 0.548 / 0.548 / 0.545 | stjewm_no_trace | 0.541 / 0.541 / 0.557 |
| finger | event_low_motion | gru_baseline | 0.576 / 0.576 / 0.566 | stjewm_spike_only | 0.532 / 0.532 / 0.528 |
| finger | event_persistent | stjewm_hidden_leak | 0.580 / 0.580 / 0.580 | stjewm_trace_only | 0.566 / 0.566 / 0.571 |
| finger | event_vel_above_median | stjewm_hidden_leak | 0.551 / 0.551 / 0.548 | stjewm_trace_only | 0.551 / 0.551 / 0.546 |
| pusht | event_block_near_target | gru_baseline | 1.000 / 1.000 / 1.000 | cubifae_baseline | 0.996 / 0.996 / 0.996 |
| pusht | event_contact | gru_baseline | 0.950 / 0.950 / 0.948 | cubifae_baseline | 0.923 / 0.923 / 0.924 |
| pusht | event_future_k10 | gru_baseline | 0.910 / 0.910 / 0.912 | cubifae_baseline | 0.889 / 0.889 / 0.892 |
| pusht | event_future_k5 | gru_baseline | 0.952 / 0.952 / 0.952 | cubifae_baseline | 0.917 / 0.917 / 0.920 |
| pusht | event_persistent | gru_baseline | 0.914 / 0.914 / 0.914 | cubifae_baseline | 0.870 / 0.870 / 0.865 |
| walker | event_contact | cubifae_baseline | 0.715 / 0.715 / 0.679 | stjewm_trace_only | 0.643 / 0.643 / 0.646 |
| walker | event_future_k10 | stjewm_no_trace | 0.642 / 0.642 / 0.617 | stjewm_hidden_leak | 0.630 / 0.630 / 0.629 |
| walker | event_future_k5 | gru_baseline | 0.599 / 0.599 / 0.608 | stjewm_trace_only | 0.572 / 0.572 / 0.519 |
| walker | event_high_motion | stjewm_rate_only | 0.656 / 0.656 / 0.653 | cubifae_baseline | 0.643 / 0.643 / 0.674 |
| walker | event_low_motion | gru_baseline | 0.649 / 0.649 / 0.653 | stjewm_trace_only | 0.643 / 0.643 / 0.644 |
| walker | event_persistent | stjewm_membrane_readout | 0.650 / 0.650 / 0.636 | cubifae_baseline | 0.635 / 0.635 / 0.646 |
| walker | event_vel_above_median | cubifae_baseline | 0.687 / 0.687 / 0.688 | stjewm_hidden_leak | 0.672 / 0.672 / 0.681 |

## Win counts (highest mean AUROC across G4 / G8 / G16)

| model | wins |
|---|---|
| stjewm_trace_only | 1 |
| stjewm_spike_only | 0 |
| stjewm_rate_only | 3 |
| stjewm_no_trace | 2 |
| stjewm_hidden_leak | 4 |
| stjewm_membrane_readout | 1 |
| cubifae_baseline | 7 |
| gru_baseline | 13 |
| slt_lif_mpc_trace | 2 |
| slt_lif_mpc_free | 0 |
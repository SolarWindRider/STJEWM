# Frozen-encoder sample efficiency (v0.7.7 utility experiment 3)

**Hypothesis**: a calibrated latent should be usable by a tiny linear policy even with little data. A collapse / noise / over-reactive latent should need more data.

## mean_cos_dist_terminal per (model × env × data fraction)

Lower is better. The collapse latent (MLP) gives ~0.0 at all fractions because the policy can't move in a constant latent space.

### env = cheetah

| model | 0.010 data | 0.050 data | 0.100 data | 0.250 data | 1.000 data |
|---|---|---|---|---|---|
| stjewm_trace_only | 0.0046 | 0.0060 | 0.0062 | 0.0104 | 0.0061 |
| stjewm_spike_only | 0.0046 | 0.0060 | 0.0063 | 0.0103 | 0.0060 |
| stjewm_rate_only | 0.0046 | 0.0060 | 0.0063 | 0.0104 | 0.0060 |
| stjewm_no_trace | 0.0045 | 0.0060 | 0.0063 | 0.0104 | 0.0061 |
| stjewm_hidden_leak | 0.0045 | 0.0060 | 0.0063 | 0.0104 | 0.0061 |
| stjewm_membrane_readout | 0.0044 | 0.0059 | 0.0063 | 0.0103 | 0.0061 |
| slt_lif_mpc_trace | 0.0039 | 0.0051 | 0.0055 | 0.0089 | 0.0050 |
| slt_lif_mpc_free | 0.0039 | 0.0051 | 0.0055 | 0.0090 | 0.0051 |
| gru_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
| mlp_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### env = walker

| model | 0.010 data | 0.050 data | 0.100 data | 0.250 data | 1.000 data |
|---|---|---|---|---|---|
| stjewm_trace_only | 0.0875 | 0.0672 | 0.0653 | 0.0840 | 0.0692 |
| stjewm_spike_only | 0.0861 | 0.0696 | 0.0634 | 0.0861 | 0.0719 |
| stjewm_rate_only | 0.0796 | 0.0691 | 0.0642 | 0.0838 | 0.0726 |
| stjewm_no_trace | 0.0865 | 0.0689 | 0.0650 | 0.0820 | 0.0697 |
| stjewm_hidden_leak | 0.0851 | 0.0706 | 0.0640 | 0.0833 | 0.0717 |
| stjewm_membrane_readout | 0.0833 | 0.0673 | 0.0644 | 0.0851 | 0.0770 |
| slt_lif_mpc_trace | 0.0787 | 0.0699 | 0.0657 | 0.0795 | 0.0766 |
| slt_lif_mpc_free | 0.0781 | 0.0708 | 0.0665 | 0.0766 | 0.0817 |
| gru_baseline | 0.0010 | 0.0009 | 0.0008 | 0.0008 | 0.0011 |
| mlp_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### env = reacher

| model | 0.010 data | 0.050 data | 0.100 data | 0.250 data | 1.000 data |
|---|---|---|---|---|---|
| stjewm_trace_only | 0.0530 | 0.0661 | 0.0553 | 0.0437 | 0.0641 |
| stjewm_spike_only | 0.0552 | 0.0666 | 0.0553 | 0.0426 | 0.0641 |
| stjewm_rate_only | 0.0534 | 0.0666 | 0.0553 | 0.0437 | 0.0628 |
| stjewm_no_trace | 0.0532 | 0.0667 | 0.0553 | 0.0438 | 0.0644 |
| stjewm_hidden_leak | 0.0564 | 0.0662 | 0.0552 | 0.0432 | 0.0647 |
| stjewm_membrane_readout | 0.0525 | 0.0671 | 0.0554 | 0.0436 | 0.0647 |
| slt_lif_mpc_trace | 0.0780 | 0.0812 | 0.0734 | 0.0565 | 0.0717 |
| slt_lif_mpc_free | 0.0793 | 0.0814 | 0.0734 | 0.0568 | 0.0806 |
| gru_baseline | 0.0005 | 0.0005 | 0.0005 | 0.0004 | 0.0005 |
| mlp_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### env = finger

| model | 0.010 data | 0.050 data | 0.100 data | 0.250 data | 1.000 data |
|---|---|---|---|---|---|
| stjewm_trace_only | 0.0774 | 0.0661 | 0.0574 | 0.0687 | 0.0601 |
| stjewm_spike_only | 0.0738 | 0.0665 | 0.0568 | 0.0687 | 0.0582 |
| stjewm_rate_only | 0.0787 | 0.0660 | 0.0565 | 0.0696 | 0.0574 |
| stjewm_no_trace | 0.0785 | 0.0659 | 0.0573 | 0.0687 | 0.0553 |
| stjewm_hidden_leak | 0.0733 | 0.0664 | 0.0571 | 0.0681 | 0.0583 |
| stjewm_membrane_readout | 0.0697 | 0.0663 | 0.0567 | 0.0691 | 0.0557 |
| slt_lif_mpc_trace | 0.0787 | 0.0737 | 0.0620 | 0.0740 | 0.0599 |
| slt_lif_mpc_free | 0.0778 | 0.0732 | 0.0622 | 0.0736 | 0.0628 |
| gru_baseline | 0.0006 | 0.0006 | 0.0004 | 0.0006 | 0.0004 |
| mlp_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

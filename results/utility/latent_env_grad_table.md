# Latent-environment gradient correlation (v0.7.7 utility experiment 2)

**Hypothesis**: a calibrated latent whose geometry is meaningful should make the gradient of `1 - cos(z_t, z_goal)` w.r.t. action align (in cosine similarity) with the gradient of env reward w.r.t. the same action. Collapse / noise / over-reactive should decorrelate.

## mean_abs_corr (Pearson cosine) per (model × env)

| model | cheetah | walker | reacher | finger |
|---|---|---|---|---|
| stjewm_trace_only | 0.461 | 0.797 | 0.812 | 0.765 |
| stjewm_spike_only | 0.416 | 0.577 | 0.755 | 0.696 |
| stjewm_rate_only | nan | nan | nan | nan |
| stjewm_no_trace | 0.300 | 0.604 | 0.336 | 0.590 |
| stjewm_hidden_leak | 0.306 | 0.607 | 0.328 | 0.589 |
| stjewm_membrane_readout | nan | nan | nan | nan |
| cubifae_baseline | nan | nan | nan | nan |
| slt_lif_mpc_trace | 0.702 | 0.538 | 0.575 | 0.871 |
| slt_lif_mpc_free | 0.116 | 0.295 | 0.631 | 0.785 |
| gru_baseline | 0.314 | 0.311 | 0.083 | 0.467 |
| mlp_baseline | 0.092 | 0.290 | 0.373 | 0.639 |

## mean_corr (signed)

| model | cheetah | walker | reacher | finger |
|---|---|---|---|---|
| stjewm_trace_only | -0.461 | -0.797 | 0.393 | -0.127 |
| stjewm_spike_only | 0.416 | -0.577 | 0.442 | -0.087 |
| stjewm_rate_only | nan | nan | nan | nan |
| stjewm_no_trace | -0.300 | -0.604 | 0.048 | -0.143 |
| stjewm_hidden_leak | -0.306 | -0.607 | 0.053 | -0.142 |
| stjewm_membrane_readout | nan | nan | nan | nan |
| cubifae_baseline | nan | nan | nan | nan |
| slt_lif_mpc_trace | 0.702 | 0.538 | -0.535 | -0.179 |
| slt_lif_mpc_free | 0.028 | 0.295 | -0.243 | 0.065 |
| gru_baseline | -0.314 | -0.311 | 0.073 | -0.165 |
| mlp_baseline | 0.007 | -0.245 | -0.372 | -0.152 |
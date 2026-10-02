# Latent-environment gradient correlation (v0.7.7 utility experiment 2)

**Hypothesis**: a calibrated latent whose geometry is meaningful should make the gradient of `1 - cos(z_t, z_goal)` w.r.t. action align (in cosine similarity) with the gradient of env reward w.r.t. the same action. Collapse / noise / over-reactive should decorrelate.

## mean_abs_corr (Pearson cosine) per (model × env)

| model | cheetah | walker | reacher | finger |
|---|---|---|---|---|
| stjewm_trace_only | nan | nan | nan | nan |
| stjewm_spike_only | nan | nan | nan | nan |
| stjewm_rate_only | nan | nan | nan | nan |
| stjewm_no_trace | nan | nan | nan | nan |
| stjewm_hidden_leak | nan | nan | nan | nan |
| stjewm_membrane_readout | nan | nan | nan | nan |
| alif_timecell_baseline | nan | nan | nan | nan |
| stacked_lif_trace | nan | nan | nan | nan |
| stacked_lif_free | nan | nan | nan | nan |
| gru_baseline | nan | nan | nan | nan |
| mlp_baseline | nan | nan | nan | nan |

## mean_corr (signed)

| model | cheetah | walker | reacher | finger |
|---|---|---|---|---|
| stjewm_trace_only | nan | nan | nan | nan |
| stjewm_spike_only | nan | nan | nan | nan |
| stjewm_rate_only | nan | nan | nan | nan |
| stjewm_no_trace | nan | nan | nan | nan |
| stjewm_hidden_leak | nan | nan | nan | nan |
| stjewm_membrane_readout | nan | nan | nan | nan |
| alif_timecell_baseline | nan | nan | nan | nan |
| stacked_lif_trace | nan | nan | nan | nan |
| stacked_lif_free | nan | nan | nan | nan |
| gru_baseline | nan | nan | nan | nan |
| mlp_baseline | nan | nan | nan | nan |
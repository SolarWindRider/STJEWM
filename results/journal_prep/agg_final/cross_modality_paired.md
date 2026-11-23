# Audited paired state/pixel CEM summary

Both sides use CEM, but their goals, initialization, horizon and replanning protocols differ.
This is descriptive, not a controlled modality-only experiment. Every mean uses the same paired split/environment cells.
Cell dispersion is not a seed confidence interval; there is one training seed per modality.

| state model | pixel model | paired cells | state env-SR | pixel env-SR | state cosine | pixel cosine |
|---|---|---|---|---|---|---|
| alif_timecell_baseline | alif_timecell_baseline | 85 | 0.231 | 0.108 | 0.009 | 0.004 |
| gru_baseline | gru_baseline | 85 | 0.160 | 0.141 | 0.083 | 0.049 |
| lewm_baseline_v2 | lewm_baseline | 85 | 0.169 | 0.106 | 0.200 | 0.041 |
| lif_transformer_baseline | lif_transformer_baseline | 85 | 0.339 | 0.120 | 0.000 | 0.000 |
| mlp_baseline | mlp_baseline | 85 | 0.268 | 0.205 | 0.067 | 0.000 |
| stacked_lif_free | stacked_lif_free | 85 | 0.273 | 0.176 | 0.031 | 0.000 |
| stacked_lif_trace | stacked_lif_trace | 85 | 0.311 | 0.219 | 0.016 | 0.000 |
| stjewm_hidden_leak | stjewm_hidden_leak | 85 | 0.191 | 0.106 | 0.085 | 0.035 |
| stjewm_membrane_readout | stjewm_membrane_readout | 85 | 0.216 | 0.120 | 0.087 | 0.038 |
| stjewm_no_trace | stjewm_no_trace | 85 | 0.216 | 0.118 | 0.087 | 0.043 |
| stjewm_rate_only | stjewm_rate_only | 85 | 0.294 | 0.165 | 0.051 | 0.029 |
| stjewm_spike_only | stjewm_spike_only | 85 | 0.275 | 0.118 | 0.209 | 0.208 |
| stjewm_trace_only | stjewm | 85 | 0.336 | 0.205 | 0.112 | 0.130 |

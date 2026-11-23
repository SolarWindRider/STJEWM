# Audited G5 three-seed comparison

All three training seeds and all planned environments are paired before averaging.
Environment means are averaged across the three splits within each seed; uncertainty is computed over the three seed means.

| model | cosine mean | seed std | 95% Student-t interval | latent SR@0.05 | env-SR |
|---|---|---|---|---|---|
| stjewm_trace_only | 0.118 | 0.012 | [0.087, 0.149] | 0.328 | 0.217 |
| stjewm_spike_only | 0.213 | 0.016 | [0.172, 0.254] | 0.105 | 0.170 |
| stjewm_rate_only | 0.047 | 0.007 | [0.030, 0.065] | 0.729 | 0.200 |
| stjewm_no_trace | 0.074 | 0.013 | [0.041, 0.107] | 0.719 | 0.132 |
| stjewm_hidden_leak | 0.079 | 0.012 | [0.050, 0.109] | 0.695 | 0.120 |
| stjewm_membrane_readout | 0.074 | 0.013 | [0.041, 0.107] | 0.719 | 0.132 |
| alif_timecell_baseline | 0.028 | 0.021 | [-0.025, 0.081] | 0.920 | 0.142 |
| gru_baseline | 0.079 | 0.008 | [0.058, 0.100] | 0.582 | 0.108 |
| lewm_baseline_v2 | 0.196 | 0.018 | [0.150, 0.241] | 0.240 | 0.109 |
| stacked_lif_trace | 0.017 | 0.009 | [-0.006, 0.040] | 0.957 | 0.233 |
| stacked_lif_free | 0.030 | 0.002 | [0.024, 0.036] | 0.861 | 0.156 |
| mlp_baseline | 0.061 | 0.016 | [0.021, 0.101] | 0.919 | 0.205 |
| lif_transformer_baseline | 0.001 | 0.002 | [-0.003, 0.005] | 0.996 | 0.224 |

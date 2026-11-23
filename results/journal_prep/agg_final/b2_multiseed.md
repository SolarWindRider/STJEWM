# Audited B2 five-model three-seed comparison

All three training seeds and all planned environments are paired before averaging.
Environment means are averaged across the three splits within each seed; uncertainty is computed over the three seed means.

| model | cosine mean | seed std | 95% Student-t interval | latent SR@0.05 | env-SR |
|---|---|---|---|---|---|
| stjewm_trace_only | 0.118 | 0.012 | [0.087, 0.149] | 0.328 | 0.217 |
| stjewm_spike_only | 0.213 | 0.016 | [0.172, 0.254] | 0.105 | 0.170 |
| stacked_lif_trace | 0.017 | 0.009 | [-0.006, 0.040] | 0.957 | 0.233 |
| lewm_baseline_v2 | 0.196 | 0.018 | [0.150, 0.241] | 0.240 | 0.109 |
| mlp_baseline | 0.061 | 0.016 | [0.021, 0.101] | 0.919 | 0.205 |

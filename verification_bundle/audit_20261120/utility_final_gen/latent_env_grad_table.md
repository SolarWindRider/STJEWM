# Audited latent-environment gradient correlation

Both gradients are evaluated at the same restored native state with a strict-load audited checkpoint.
The physical objective is negative native goal-state L2 distance, not environment reward.
Zero-gradient windows are reported as undefined counts, never as zero correlation.

| model | cheetah: mean absolute / signed / undefined | walker: mean absolute / signed / undefined | reacher: mean absolute / signed / undefined | finger: mean absolute / signed / undefined |
|---|---|---|---|---|
| stjewm_trace_only | 0.315 / -0.042 / 0 | 0.462 / -0.326 / 0 | 0.669 / 0.104 / 0 | 0.747 / -0.274 / 0 |
| stjewm_spike_only | 0.337 / -0.054 / 0 | 0.345 / -0.092 / 0 | 0.502 / -0.191 / 0 | 0.724 / -0.163 / 0 |
| stjewm_rate_only | 0.290 / -0.069 / 0 | 0.458 / 0.013 / 0 | 0.807 / -0.001 / 0 | 0.604 / 0.166 / 0 |
| stjewm_no_trace | 0.347 / -0.148 / 0 | 0.452 / -0.406 / 0 | 0.626 / -0.593 / 0 | 0.782 / -0.761 / 0 |
| stjewm_hidden_leak | 0.344 / -0.057 / 0 | 0.430 / -0.427 / 0 | 0.670 / -0.659 / 0 | 0.807 / -0.801 / 0 |
| stjewm_membrane_readout | 0.347 / -0.148 / 0 | 0.452 / -0.406 / 0 | 0.626 / -0.593 / 0 | 0.782 / -0.761 / 0 |
| alif_timecell_baseline | 0.435 / -0.089 / 0 | 0.368 / -0.071 / 0 | 0.761 / -0.009 / 0 | 0.754 / 0.030 / 0 |
| stacked_lif_trace | 0.351 / 0.046 / 0 | 0.434 / -0.042 / 0 | 0.778 / -0.064 / 0 | 0.705 / -0.048 / 0 |
| stacked_lif_free | 0.346 / -0.015 / 0 | 0.557 / -0.056 / 0 | 0.768 / 0.140 / 0 | 0.516 / 0.146 / 0 |
| lewm_baseline_v2 | 0.644 / -0.590 / 0 | 0.483 / -0.476 / 0 | 0.709 / -0.699 / 0 | 0.787 / -0.766 / 0 |
| gru_baseline | 0.345 / -0.100 / 0 | 0.250 / -0.156 / 0 | 0.736 / -0.148 / 0 | 0.722 / 0.150 / 0 |
| mlp_baseline | 0.381 / 0.027 / 7 | 0.141 / -0.098 / 168 | 0.950 / -0.796 / 172 | 0.577 / -0.174 / 55 |

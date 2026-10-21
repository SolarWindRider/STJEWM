# S2: Latent gain / Jacobian stability

32 contexts x 32 directions per (model, env); eps = 0.01x episode latent std. Cells: mean / max over pairs. Verdict: divergent iff mean gain(20) > 1.

**Divergent (pooled gain(20) > 1):** stjewm_spike_only, stjewm_membrane_readout, stjewm_hidden_leak, stacked_lif_trace

## cartpole_2d

### Latent FD gain gain(h) = ||dz_hat_h||/eps (mean / max)

| model | h=1 | h=5 | h=10 | h=20 | verdict | frac gain20>1 |
|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.002 / 0.00503 | 0.002 / 0.00493 | 0.002 / 0.00453 | 0.002 / 0.00434 | contractive | 0.00 |
| stjewm_spike_only | 1.455 / 117 | 22.805 / 413 | 82.578 / 608 | 219.741 / 1.22e+03 | divergent | 0.79 |
| stjewm_membrane_readout | 2.515 / 41.2 | 33.364 / 417 | 83.043 / 543 | 202.290 / 1e+03 | divergent | 1.00 |
| stjewm_hidden_leak | 2.305 / 41.7 | 16.901 / 83.4 | 47.058 / 245 | 121.313 / 831 | divergent | 1.00 |
| stacked_lif_trace | 0.505 / 15.5 | 0.888 / 18.1 | 3.759 / 22.5 | 9.819 / 30.1 | divergent | 0.65 |
| lewm_baseline_v2 | 0.272 / 0.505 | 0.129 / 0.721 | 0.154 / 1.76 | 0.216 / 4.31 | contractive | 0.04 |
| gru_baseline | 0.064 / 0.229 | 0.039 / 0.619 | 0.046 / 1.99 | 0.025 / 0.274 | contractive | 0.00 |
| mlp_baseline | 0.000 / 0.000193 | 0.000 / 0.000197 | 0.000 / 0.000196 | 0.000 / 0.000193 | contractive | 0.00 |

### Obs-side gain ||dz_1||/||dobs|| (mean / max)

| model | gain_obs |
|---|---|
| stjewm_trace_only | 0.000 / 0.0308 |
| stjewm_spike_only | 3.406 / 54 |
| stjewm_membrane_readout | 14.679 / 139 |
| stjewm_hidden_leak | 17.221 / 235 |
| stacked_lif_trace | 0.048 / 5.04 |
| lewm_baseline_v2 | 11.070 / 74.7 |
| gru_baseline | 9.384 / 147 |
| mlp_baseline | 0.000 / 6.97e-06 |

### One-step Jacobian top singular value (autograd power iteration)

| model | top sv | device |
|---|---|---|
| stjewm_trace_only | skipped | cuda: RuntimeError: In order to use an autograd.Function with functorch transfor |
| stjewm_spike_only | skipped | cuda: RuntimeError: In order to use an autograd.Function with functorch transfor |
| stjewm_membrane_readout | skipped | cuda: RuntimeError: In order to use an autograd.Function with functorch transfor |
| stjewm_hidden_leak | skipped | cuda: RuntimeError: In order to use an autograd.Function with functorch transfor |
| stacked_lif_trace | skipped | cuda: RuntimeError: In order to use an autograd.Function with functorch transfor |
| lewm_baseline_v2 | 1.614 | cuda:0 |
| gru_baseline | 0.962 | cpu |
| mlp_baseline | 0.000 | cuda:0 |

## cheetah

### Latent FD gain gain(h) = ||dz_hat_h||/eps (mean / max)

| model | h=1 | h=5 | h=10 | h=20 | verdict | frac gain20>1 |
|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.003 / 0.0061 | 0.003 / 0.00558 | 0.003 / 0.00553 | 0.003 / 0.0111 | contractive | 0.00 |
| stjewm_spike_only | 0.294 / 0.492 | 18.719 / 893 | 101.624 / 1.55e+03 | 451.636 / 2.32e+03 | divergent | 0.40 |
| stjewm_membrane_readout | 1.770 / 19.8 | 28.246 / 175 | 99.886 / 436 | 295.171 / 1.19e+03 | divergent | 1.00 |
| stjewm_hidden_leak | 2.218 / 73.2 | 22.228 / 150 | 71.439 / 391 | 224.603 / 1.08e+03 | divergent | 1.00 |
| stacked_lif_trace | 0.177 / 11.3 | 1.856 / 21 | 3.708 / 23.4 | 11.359 / 27.7 | divergent | 0.72 |
| lewm_baseline_v2 | 0.307 / 0.478 | 0.196 / 0.874 | 0.219 / 1.6 | 0.321 / 9 | contractive | 0.06 |
| gru_baseline | 0.109 / 0.474 | 0.122 / 2.19 | 0.698 / 56.4 | 0.184 / 17.5 | contractive | 0.03 |
| mlp_baseline | 0.000 / 0.000197 | 0.000 / 0.000197 | 0.000 / 0.000204 | 0.000 / 0.000197 | contractive | 0.00 |

### Obs-side gain ||dz_1||/||dobs|| (mean / max)

| model | gain_obs |
|---|---|
| stjewm_trace_only | 0.000 / 0.000837 |
| stjewm_spike_only | 6.784 / 50.7 |
| stjewm_membrane_readout | 13.523 / 166 |
| stjewm_hidden_leak | 18.325 / 177 |
| stacked_lif_trace | 0.000 / 0.000319 |
| lewm_baseline_v2 | 26.074 / 79.4 |
| gru_baseline | 62.164 / 472 |
| mlp_baseline | 0.000 / 5.49e-06 |

### One-step Jacobian top singular value (autograd power iteration)

| model | top sv | device |
|---|---|---|
| stjewm_trace_only | skipped | cuda: RuntimeError: In order to use an autograd.Function with functorch transfor |
| stjewm_spike_only | skipped | cuda: RuntimeError: In order to use an autograd.Function with functorch transfor |
| stjewm_membrane_readout | skipped | cuda: RuntimeError: In order to use an autograd.Function with functorch transfor |
| stjewm_hidden_leak | skipped | cuda: RuntimeError: In order to use an autograd.Function with functorch transfor |
| stacked_lif_trace | skipped | cuda: RuntimeError: In order to use an autograd.Function with functorch transfor |
| lewm_baseline_v2 | 1.437 | cuda:0 |
| gru_baseline | 1.927 | cpu |
| mlp_baseline | 0.000 | cuda:0 |

## Pooled (both envs)

| model | gain1 | gain5 | gain10 | gain20 | gain_obs | verdict |
|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.002 / 0.0061 | 0.003 | 0.003 | 0.003 / 0.0111 | 0.000 / 0.0308 | contractive |
| stjewm_spike_only | 0.875 / 117 | 20.762 | 92.101 | 335.688 / 2.32e+03 | 5.095 / 54 | divergent |
| stjewm_membrane_readout | 2.142 / 41.2 | 30.805 | 91.464 | 248.730 / 1.19e+03 | 14.101 / 166 | divergent |
| stjewm_hidden_leak | 2.262 / 73.2 | 19.565 | 59.248 | 172.958 / 1.08e+03 | 17.773 / 235 | divergent |
| stacked_lif_trace | 0.341 / 15.5 | 1.372 | 3.733 | 10.589 / 30.1 | 0.024 / 5.04 | divergent |
| lewm_baseline_v2 | 0.289 / 0.505 | 0.162 | 0.186 | 0.269 / 9 | 18.572 / 79.4 | contractive |
| gru_baseline | 0.087 / 0.474 | 0.081 | 0.372 | 0.104 / 17.5 | 35.774 / 472 | contractive |
| mlp_baseline | 0.000 / 0.000197 | 0.000 | 0.000 | 0.000 / 0.000197 | 0.000 / 6.97e-06 | contractive |

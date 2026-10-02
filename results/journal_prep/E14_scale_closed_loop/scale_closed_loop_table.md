# E14 — closed-loop env-SR on the scale axis (G4/G8/G16, in-union core)

Protocol: exact E1 recipe (CEM 300/30, horizon 25, budget 50, goal=t+25, 5 eps, pad 128/act 56).
Envs restricted to the common in-union core (cartpole_2d, pendulum_2d, cheetah, pusht) so cross-scale
differences are not confounded with out-of-union coverage. cos = mean_cos_dist (lower is better).

## Pooled per scale x model (4 envs)

| scale | model | env-SR | cos | phys | LeWM-SR@0.05 |
|---|---|---:|---:|---:|---:|
| G4 | stjewm_trace_only | 0.35 | 0.158 | 111.358 | 0.30 |
| G4 | stjewm_spike_only | 0.15 | 0.330 | 356.958 | 0.00 |
| G4 | stjewm_rate_only | 0.35 | 0.013 | 145.323 | 0.95 |
| G4 | stjewm_no_trace | 0.30 | 0.122 | 185.703 | 0.60 |
| G4 | stjewm_hidden_leak | 0.30 | 0.075 | 202.816 | 0.75 |
| G4 | stjewm_membrane_readout | 0.30 | 0.122 | 185.703 | 0.60 |
| G4 | alif_timecell_baseline | 0.25 | 0.083 | 180.605 | 0.80 |
| G4 | stacked_lif_trace | 0.45 | 0.000 | 119.433 | 1.00 |
| G4 | stacked_lif_free | 0.30 | 0.000 | 156.616 | 1.00 |
| G4 | gru_baseline | 0.25 | 0.135 | 311.152 | 0.65 |
| G4 | lewm_baseline_v2 | 0.10 | 0.336 | 445.999 | 0.10 |
| G4 | mlp_baseline | 0.35 | 0.000 | 165.147 | 1.00 |
| G8 | stjewm_trace_only | 0.35 | 0.178 | 115.188 | 0.05 |
| G8 | stjewm_spike_only | 0.20 | 0.288 | 121.823 | 0.05 |
| G8 | stjewm_rate_only | 0.40 | 0.047 | 205.086 | 0.60 |
| G8 | stjewm_no_trace | 0.20 | 0.135 | 161.361 | 0.50 |
| G8 | stjewm_hidden_leak | 0.15 | 0.232 | 182.781 | 0.40 |
| G8 | stjewm_membrane_readout | 0.20 | 0.135 | 161.361 | 0.50 |
| G8 | alif_timecell_baseline | 0.40 | 0.289 | 289.590 | 0.50 |
| G8 | stacked_lif_trace | 0.45 | 0.000 | 119.433 | 1.00 |
| G8 | stacked_lif_free | 0.30 | 0.001 | 125.699 | 1.00 |
| G8 | gru_baseline | 0.05 | 0.191 | 420.809 | 0.35 |
| G8 | lewm_baseline_v2 | 0.15 | 0.308 | 460.382 | 0.15 |
| G8 | mlp_baseline | 0.25 | 0.000 | 152.358 | 1.00 |
| G16 | stjewm_trace_only | 0.30 | 0.101 | 119.084 | 0.25 |
| G16 | stjewm_spike_only | 0.20 | 0.313 | 238.912 | 0.05 |
| G16 | stjewm_rate_only | 0.25 | 0.177 | 164.430 | 0.40 |
| G16 | stjewm_no_trace | 0.20 | 0.330 | 278.761 | 0.25 |
| G16 | stjewm_hidden_leak | 0.20 | 0.224 | 257.670 | 0.35 |
| G16 | stjewm_membrane_readout | 0.20 | 0.330 | 278.761 | 0.25 |
| G16 | alif_timecell_baseline | 0.15 | 0.010 | 385.691 | 0.95 |
| G16 | stacked_lif_trace | 0.35 | 0.000 | 146.523 | 1.00 |
| G16 | stacked_lif_free | 0.25 | 0.135 | 381.523 | 0.25 |
| G16 | gru_baseline | 0.05 | 0.156 | 365.968 | 0.25 |
| G16 | lewm_baseline_v2 | 0.15 | 0.325 | 300.699 | 0.10 |
| G16 | mlp_baseline | 0.35 | 0.002 | 119.424 | 1.00 |

## Per-env env-SR (scale x model)

| env | G4 stjewm_trace_only | G4 stjewm_spike_only | G4 stjewm_rate_only | G4 stjewm_no_trace | G4 stjewm_hidden_leak | G4 stjewm_membrane_readout | G4 alif_timecell_baseline | G4 stacked_lif_trace | G4 stacked_lif_free | G4 gru_baseline | G4 lewm_baseline_v2 | G4 mlp_baseline | G8 stjewm_trace_only | G8 stjewm_spike_only | G8 stjewm_rate_only | G8 stjewm_no_trace | G8 stjewm_hidden_leak | G8 stjewm_membrane_readout | G8 alif_timecell_baseline | G8 stacked_lif_trace | G8 stacked_lif_free | G8 gru_baseline | G8 lewm_baseline_v2 | G8 mlp_baseline | G16 stjewm_trace_only | G16 stjewm_spike_only | G16 stjewm_rate_only | G16 stjewm_no_trace | G16 stjewm_hidden_leak | G16 stjewm_membrane_readout | G16 alif_timecell_baseline | G16 stacked_lif_trace | G16 stacked_lif_free | G16 gru_baseline | G16 lewm_baseline_v2 | G16 mlp_baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cartpole_2d | 1.00 | 0.20 | 1.00 | 0.80 | 1.00 | 0.80 | 0.80 | 1.00 | 1.00 | 0.80 | 0.20 | 1.00 | 1.00 | 0.60 | 1.00 | 0.60 | 0.40 | 0.60 | 1.00 | 1.00 | 1.00 | 0.20 | 0.20 | 0.80 | 1.00 | 0.60 | 0.80 | 0.40 | 0.40 | 0.40 | 0.40 | 1.00 | 0.80 | 0.20 | 0.40 | 1.00 |
| pendulum_2d | 0.20 | 0.40 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.00 | 0.40 | 0.20 | 0.20 | 0.20 | 0.20 | 0.40 | 0.40 | 0.40 | 0.20 | 0.20 | 0.20 | 0.00 | 0.20 | 0.20 |
| cheetah | 0.20 | 0.00 | 0.20 | 0.20 | 0.00 | 0.20 | 0.00 | 0.40 | 0.00 | 0.00 | 0.00 | 0.20 | 0.20 | 0.00 | 0.40 | 0.00 | 0.00 | 0.00 | 0.40 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| pusht | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.20 |

# Event Boundary Alignment (Pearson ρ between obs-event and latent-event)

Pearson correlation between obs first-difference (event strength) and latent first-difference.
High ρ means the latent preserves obs-level event timing. Aggregated across seeds.

| model | ball_in_cup | cartpole_2d | cheetah | finger | pendulum_2d | walker | AVG |
|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.982 | 1.000 | 0.999 | 0.996 | 0.998 | 0.986 | **0.993** |
| stjewm_spike_only | 0.974 | 1.000 | 0.998 | 0.996 | 0.998 | 0.986 | **0.992** |
| stjewm_rate_only | 0.984 | 1.000 | 0.999 | 0.995 | 0.998 | 0.994 | **0.995** |
| stjewm_no_trace | 0.973 | 0.998 | 0.999 | 0.996 | 0.998 | 0.999 | **0.994** |
| stjewm_hidden_leak | 0.968 | 1.000 | 0.998 | 0.995 | 0.998 | 0.996 | **0.993** |
| stjewm_membrane_readout | 0.981 | 1.000 | 0.999 | 0.996 | 0.998 | 0.996 | **0.995** |
| cubifae_baseline | - | - | - | - | - | - | **-** |
| gru_baseline | -0.087 | -0.037 | 0.197 | 0.277 | 0.032 | 0.050 | **0.072** |
| lewm_baseline_v2 | -0.047 | -0.170 | 0.862 | 0.741 | 0.922 | 0.025 | **0.389** |
| slt_lif_mpc_trace | - | - | - | - | - | - | **-** |
| slt_lif_mpc_free | - | - | - | - | - | - | **-** |
| mlp_baseline | 0.278 | -0.038 | -0.023 | 0.084 | 0.067 | -0.055 | **0.052** |

# Latent-goal MPC horizon sweep (v0.7.7 utility experiment 1)

**CEM config**: n_samples=100, n_elites=10, n_iters=10, n_episodes=5

## mean_cos_dist_terminal per (model × env × horizon)

Lower is better. A collapse latent gives ~1e-7; a calibrated latent gives ~0.05; over-reactive gives >0.10 and grows with H.

| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |
|---|---|---|---|---|---|---|
| stjewm_trace_only | cheetah | 0.0178 | 0.0150 | 0.0107 | 0.0115 | 0.0110 |
| stjewm_trace_only | walker | 0.0874 | 0.0959 | 0.0955 | 0.0949 | 0.0953 |
| stjewm_trace_only | reacher | 0.0712 | 0.0907 | 0.0970 | 0.0962 | 0.0936 |
| stjewm_trace_only | finger | 0.0399 | 0.0343 | 0.0346 | 0.0337 | 0.0387 |
| stjewm_spike_only | cheetah | 0.0050 | 0.0070 | 0.0050 | 0.0036 | 0.0054 |
| stjewm_spike_only | walker | 0.0927 | 0.0723 | 0.0763 | 0.0769 | 0.0769 |
| stjewm_spike_only | reacher | 0.0274 | 0.0245 | 0.0229 | 0.0236 | 0.0213 |
| stjewm_spike_only | finger | 0.0883 | 0.0795 | 0.0789 | 0.0803 | 0.0840 |
| stjewm_rate_only | cheetah | 0.0105 | 0.0118 | 0.0124 | 0.0123 | 0.0123 |
| stjewm_rate_only | walker | 0.0391 | 0.0225 | 0.0222 | 0.0225 | 0.0230 |
| stjewm_rate_only | reacher | 0.0199 | 0.0195 | 0.0209 | 0.0215 | 0.0262 |
| stjewm_rate_only | finger | 0.0340 | 0.0329 | 0.0423 | 0.0466 | 0.0404 |
| stjewm_no_trace | cheetah | 0.0283 | 0.0268 | 0.0243 | 0.0252 | 0.0176 |
| stjewm_no_trace | walker | 0.0911 | 0.0870 | 0.0884 | 0.0892 | 0.0901 |
| stjewm_no_trace | reacher | 0.2706 | 0.2800 | 0.2851 | 0.2884 | 0.2895 |
| stjewm_no_trace | finger | 0.1090 | 0.1090 | 0.1090 | 0.1095 | 0.1093 |
| stjewm_hidden_leak | cheetah | 0.0286 | 0.0297 | 0.0200 | 0.0106 | 0.0058 |
| stjewm_hidden_leak | walker | 0.0695 | 0.0711 | 0.0714 | 0.0711 | 0.0711 |
| stjewm_hidden_leak | reacher | 0.2503 | 0.2491 | 0.2536 | 0.2600 | 0.2470 |
| stjewm_hidden_leak | finger | 0.1096 | 0.1066 | 0.1073 | 0.1038 | 0.1018 |
| stjewm_membrane_readout | cheetah | 0.0275 | 0.0196 | 0.0174 | 0.0105 | 0.0134 |
| stjewm_membrane_readout | walker | 0.0922 | 0.0890 | 0.0903 | 0.0905 | 0.0908 |
| stjewm_membrane_readout | reacher | 0.2767 | 0.2829 | 0.2888 | 0.2973 | 0.3022 |
| stjewm_membrane_readout | finger | 0.1076 | 0.1102 | 0.1092 | 0.1061 | 0.1048 |
| slt_lif_mpc_trace | cheetah | 0.0110 | 0.0105 | 0.0068 | 0.0038 | 0.0053 |
| slt_lif_mpc_trace | walker | 0.0556 | 0.0239 | 0.0235 | 0.0230 | 0.0226 |
| slt_lif_mpc_trace | reacher | 0.0178 | 0.0176 | 0.0193 | 0.0235 | 0.0258 |
| slt_lif_mpc_trace | finger | 0.0236 | 0.0277 | 0.0347 | 0.0369 | 0.0328 |
| slt_lif_mpc_free | cheetah | 0.0248 | 0.0136 | 0.0079 | 0.0090 | 0.0079 |
| slt_lif_mpc_free | walker | 0.0307 | 0.0256 | 0.0256 | 0.0256 | 0.0257 |
| slt_lif_mpc_free | reacher | 0.2085 | 0.2231 | 0.2241 | 0.2102 | 0.2063 |
| slt_lif_mpc_free | finger | 0.2245 | 0.2149 | 0.2175 | 0.2098 | 0.2138 |
| gru_baseline | cheetah | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| gru_baseline | walker | 0.0008 | 0.0011 | 0.0012 | 0.0012 | 0.0013 |
| gru_baseline | reacher | 0.0017 | 0.0021 | 0.0022 | 0.0024 | 0.0026 |
| gru_baseline | finger | 0.0007 | 0.0008 | 0.0006 | 0.0006 | 0.0006 |
| mlp_baseline | cheetah | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mlp_baseline | walker | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mlp_baseline | reacher | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mlp_baseline | finger | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## env_success per (model × env × horizon)

Env-native success: |state - goal| < per-env tol. The DMC tol is loose (1.0 for cheetah/walker) so most models get 100% trivially. The cos_dist table is the real signal.

| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |
|---|---|---|---|---|---|---|
| stjewm_trace_only | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_trace_only | walker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_trace_only | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_trace_only | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_spike_only | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_spike_only | walker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_spike_only | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 |
| stjewm_spike_only | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_rate_only | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_rate_only | walker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_rate_only | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_rate_only | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_no_trace | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_no_trace | walker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_no_trace | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_no_trace | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_hidden_leak | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_hidden_leak | walker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_hidden_leak | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_hidden_leak | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_membrane_readout | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_membrane_readout | walker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_membrane_readout | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_membrane_readout | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| slt_lif_mpc_trace | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| slt_lif_mpc_trace | walker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| slt_lif_mpc_trace | reacher | 0.20 | 0.20 | 0.20 | 0.00 | 0.00 |
| slt_lif_mpc_trace | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| slt_lif_mpc_free | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| slt_lif_mpc_free | walker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| slt_lif_mpc_free | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| slt_lif_mpc_free | finger | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| gru_baseline | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| gru_baseline | walker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| gru_baseline | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gru_baseline | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mlp_baseline | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mlp_baseline | walker | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mlp_baseline | reacher | 0.60 | 0.40 | 0.40 | 0.40 | 0.40 |
| mlp_baseline | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
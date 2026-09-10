# Latent-goal MPC horizon sweep (v0.7.7 utility experiment 1)

**CEM config**: n_samples=100, n_elites=10, n_iters=10, n_episodes=3

## mean_cos_dist_terminal per (model × env × horizon)

Lower is better. A collapse latent gives ~1e-7; a calibrated latent gives ~0.05; over-reactive gives >0.10 and grows with H.

| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |
|---|---|---|---|---|---|---|
| stjewm_trace_only | cheetah | 0.0754 | 0.0211 | 0.0208 | 0.0340 | 0.0419 |
| stjewm_trace_only | walker | 0.1691 | 0.1032 | 0.1001 | 0.0963 | 0.0945 |
| stjewm_trace_only | reacher | 0.6496 | 0.6623 | 0.6608 | 0.6363 | 0.5999 |
| stjewm_trace_only | finger | 0.4611 | 0.4448 | 0.4548 | 0.4523 | 0.4444 |
| stjewm_spike_only | cheetah | 0.2047 | 0.0607 | 0.0860 | 0.1067 | 0.0334 |
| stjewm_spike_only | walker | 0.1272 | 0.1566 | 0.1624 | 0.1682 | 0.1755 |
| stjewm_spike_only | reacher | 0.9514 | 0.9545 | 0.9560 | 0.9539 | 0.9567 |
| stjewm_spike_only | finger | 0.3064 | 0.3247 | 0.3213 | 0.3249 | 0.3228 |
| stjewm_rate_only | cheetah | 0.0491 | 0.0336 | 0.0123 | 0.0154 | 0.0071 |
| stjewm_rate_only | walker | 0.0455 | 0.1438 | 0.1498 | 0.1534 | 0.1585 |
| stjewm_rate_only | reacher | 0.8321 | 0.8247 | 0.8244 | 0.8278 | 0.8263 |
| stjewm_rate_only | finger | 0.3009 | 0.3843 | 0.3805 | 0.3716 | 0.3654 |
| stjewm_no_trace | cheetah | 0.2564 | 0.0393 | 0.1286 | 0.1770 | 0.1621 |
| stjewm_no_trace | walker | 0.2809 | 0.1832 | 0.1803 | 0.1783 | 0.1758 |
| stjewm_no_trace | reacher | 0.3382 | 0.3275 | 0.3296 | 0.3276 | 0.3392 |
| stjewm_no_trace | finger | 0.3012 | 0.3152 | 0.3161 | 0.3177 | 0.3171 |
| stjewm_hidden_leak | cheetah | 0.2252 | 0.0732 | 0.1185 | 0.0664 | 0.0311 |
| stjewm_hidden_leak | walker | 0.1943 | 0.3018 | 0.3069 | 0.3159 | 0.3277 |
| stjewm_hidden_leak | reacher | 0.0254 | 0.0278 | 0.0252 | 0.0255 | 0.0241 |
| stjewm_hidden_leak | finger | 0.4372 | 0.3708 | 0.3770 | 0.3815 | 0.3889 |
| stjewm_membrane_readout | cheetah | 0.6371 | 0.4360 | 0.4171 | 0.3978 | 0.4453 |
| stjewm_membrane_readout | walker | 0.0581 | 0.0292 | 0.0280 | 0.0276 | 0.0273 |
| stjewm_membrane_readout | reacher | 0.8547 | 0.8495 | 0.8459 | 0.8423 | 0.8396 |
| stjewm_membrane_readout | finger | 0.3533 | 0.3516 | 0.3519 | 0.3505 | 0.3555 |
| alif_timecell_baseline | cheetah | 0.0435 | 0.0118 | 0.0152 | 0.0201 | 0.0242 |
| alif_timecell_baseline | walker | 0.0166 | 0.0193 | 0.0208 | 0.0225 | 0.0248 |
| alif_timecell_baseline | reacher | 0.1489 | 0.1536 | 0.1469 | 0.1200 | 0.1123 |
| alif_timecell_baseline | finger | 0.4773 | 0.5108 | 0.5163 | 0.5116 | 0.5028 |
| stacked_lif_trace | cheetah | 0.0045 | 0.0021 | 0.0026 | 0.0031 | 0.0042 |
| stacked_lif_trace | walker | 0.0574 | 0.1942 | 0.1970 | 0.1960 | 0.1919 |
| stacked_lif_trace | reacher | 0.1091 | 0.1188 | 0.1216 | 0.1203 | 0.1205 |
| stacked_lif_trace | finger | 0.0447 | 0.0485 | 0.0484 | 0.0563 | 0.0609 |
| stacked_lif_free | cheetah | 0.0694 | 0.1461 | 0.1252 | 0.1554 | 0.2571 |
| stacked_lif_free | walker | 0.0450 | 0.1539 | 0.1456 | 0.1386 | 0.1330 |
| stacked_lif_free | reacher | 0.8801 | 0.8771 | 0.8725 | 0.8708 | 0.8642 |
| stacked_lif_free | finger | 0.3427 | 0.3332 | 0.3265 | 0.3134 | 0.3190 |
| gru_baseline | cheetah | 0.0631 | 0.0088 | 0.0085 | 0.0108 | 0.0136 |
| gru_baseline | walker | 0.0614 | 0.0139 | 0.0138 | 0.0139 | 0.0139 |
| gru_baseline | reacher | 0.2334 | 0.2344 | 0.2344 | 0.2332 | 0.2253 |
| gru_baseline | finger | 0.3795 | 0.3784 | 0.3761 | 0.3778 | 0.3785 |
| mlp_baseline | cheetah | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mlp_baseline | walker | 0.2927 | 0.2927 | 0.2927 | 0.2927 | 0.2927 |
| mlp_baseline | reacher | 0.2845 | 0.2845 | 0.2845 | 0.2845 | 0.2845 |
| mlp_baseline | finger | 0.0415 | 0.0415 | 0.0415 | 0.0415 | 0.0415 |

## env_success per (model × env × horizon)

Env-native success: |state - goal| < per-env tol. The DMC tol is loose (1.0 for cheetah/walker) so most models get 100% trivially. The cos_dist table is the real signal.

| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |
|---|---|---|---|---|---|---|
| stjewm_trace_only | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_trace_only | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_trace_only | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_trace_only | finger | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| stjewm_spike_only | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_spike_only | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_spike_only | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_spike_only | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_rate_only | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_rate_only | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_rate_only | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_rate_only | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_no_trace | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_no_trace | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_no_trace | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_no_trace | finger | 0.67 | 0.67 | 0.67 | 1.00 | 1.00 |
| stjewm_hidden_leak | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_hidden_leak | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_hidden_leak | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_hidden_leak | finger | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| stjewm_membrane_readout | cheetah | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 |
| stjewm_membrane_readout | walker | 0.00 | 0.67 | 0.67 | 0.67 | 0.67 |
| stjewm_membrane_readout | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_membrane_readout | finger | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| alif_timecell_baseline | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| alif_timecell_baseline | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| alif_timecell_baseline | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| alif_timecell_baseline | finger | 0.67 | 0.33 | 0.33 | 0.33 | 0.33 |
| stacked_lif_trace | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stacked_lif_trace | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stacked_lif_trace | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stacked_lif_trace | finger | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| stacked_lif_free | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| stacked_lif_free | walker | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 |
| stacked_lif_free | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stacked_lif_free | finger | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| gru_baseline | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| gru_baseline | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gru_baseline | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gru_baseline | finger | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 |
| mlp_baseline | cheetah | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mlp_baseline | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mlp_baseline | reacher | 0.33 | 0.33 | 0.00 | 0.00 | 0.00 |
| mlp_baseline | finger | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
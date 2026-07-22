# Generalist 5M-Aligned Table

Splits: cross_benchmark_F1, cross_benchmark_F2, generalist_16env, oodc_F1, oodc_F1F2, oodc_F1F3, oodc_F2, oodc_F2F3, oodc_F3
Models: 13 (stjewm_trace_only, stjewm_spike_only, stjewm_rate_only, stjewm_no_trace, stjewm_hidden_leak, stjewm_membrane_readout, cubifae_baseline, gru_baseline, lewm_baseline_v2, slt_lif_mpc_trace, slt_lif_mpc_free, mlp_baseline, spikedreamer_baseline)

## Per-split LeWM-SR (env-SR) per model

| Split | Model | n_envs | LeWM-SR | env-SR | cos_dist |
|---|---|---|---|---|---|
| cross_benchmark_F1 | stjewm_trace_only | 14 | 60.0% | 0.0% | 0.107 |
| cross_benchmark_F1 | stjewm_spike_only | 14 | 55.7% | 0.0% | 0.121 |
| cross_benchmark_F1 | stjewm_rate_only | 14 | 57.1% | 0.0% | 0.117 |
| cross_benchmark_F1 | stjewm_no_trace | 14 | 52.9% | 0.0% | 0.127 |
| cross_benchmark_F1 | stjewm_hidden_leak | 14 | 54.3% | 0.0% | 0.127 |
| cross_benchmark_F1 | stjewm_membrane_readout | 14 | 54.3% | 0.0% | 0.139 |
| cross_benchmark_F1 | cubifae_baseline | 14 | 58.6% | 0.0% | 0.118 |
| cross_benchmark_F1 | gru_baseline | 14 | 91.4% | 0.0% | 0.015 |
| cross_benchmark_F1 | lewm_baseline_v2 | 14 | 34.3% | 0.0% | 0.190 |
| cross_benchmark_F1 | slt_lif_mpc_trace | 14 | 57.1% | 0.0% | 0.111 |
| cross_benchmark_F1 | mlp_baseline | 14 | 100.0% | 0.0% | 0.003 |
| cross_benchmark_F1 | spikedreamer_baseline | 14 | 100.0% | 0.0% | -0.000 |
| cross_benchmark_F2 | cubifae_baseline | 14 | 52.9% | 0.0% | 0.127 |
| cross_benchmark_F2 | gru_baseline | 14 | 87.1% | 0.0% | 0.025 |
| cross_benchmark_F2 | lewm_baseline_v2 | 14 | 25.7% | 0.0% | 0.201 |
| cross_benchmark_F2 | mlp_baseline | 14 | 92.9% | 0.0% | 0.015 |
| cross_benchmark_F2 | spikedreamer_baseline | 14 | 100.0% | 0.0% | 0.000 |
| generalist_16env | cubifae_baseline | 15 | 53.3% | 0.0% | 0.116 |
| generalist_16env | gru_baseline | 15 | 82.7% | 0.0% | 0.038 |
| generalist_16env | lewm_baseline_v2 | 15 | 33.3% | 0.0% | 0.180 |
| generalist_16env | mlp_baseline | 15 | 96.0% | 0.0% | 0.011 |
| oodc_F1 | stjewm_trace_only | 5 | 84.0% | 0.0% | 0.044 |
| oodc_F1 | stjewm_spike_only | 5 | 76.0% | 0.0% | 0.049 |
| oodc_F1 | stjewm_rate_only | 5 | 84.0% | 0.0% | 0.047 |
| oodc_F1 | stjewm_no_trace | 5 | 76.0% | 0.0% | 0.070 |
| oodc_F1 | stjewm_hidden_leak | 5 | 96.0% | 0.0% | 0.037 |
| oodc_F1 | stjewm_membrane_readout | 5 | 80.0% | 0.0% | 0.053 |
| oodc_F1 | cubifae_baseline | 5 | 88.0% | 0.0% | 0.038 |
| oodc_F1 | gru_baseline | 5 | 100.0% | 0.0% | 0.000 |
| oodc_F1 | lewm_baseline_v2 | 5 | 52.0% | 0.0% | 0.145 |
| oodc_F1 | slt_lif_mpc_trace | 5 | 76.0% | 0.0% | 0.050 |
| oodc_F1 | slt_lif_mpc_free | 5 | 76.0% | 0.0% | 0.068 |
| oodc_F1 | mlp_baseline | 5 | 100.0% | 0.0% | 0.000 |
| oodc_F1 | spikedreamer_baseline | 5 | 100.0% | 0.0% | -0.000 |
| oodc_F1F2 | stjewm_trace_only | 10 | 56.0% | 0.0% | 0.092 |
| oodc_F1F2 | stjewm_spike_only | 10 | 62.0% | 0.0% | 0.099 |
| oodc_F1F2 | stjewm_rate_only | 10 | 58.0% | 0.0% | 0.097 |
| oodc_F1F2 | stjewm_no_trace | 10 | 52.0% | 0.0% | 0.110 |
| oodc_F1F2 | stjewm_hidden_leak | 10 | 50.0% | 0.0% | 0.116 |
| oodc_F1F2 | stjewm_membrane_readout | 10 | 50.0% | 0.0% | 0.103 |
| oodc_F1F2 | cubifae_baseline | 10 | 58.0% | 0.0% | 0.092 |
| oodc_F1F2 | gru_baseline | 10 | 100.0% | 0.0% | 0.001 |
| oodc_F1F2 | lewm_baseline_v2 | 10 | 40.0% | 0.0% | 0.165 |
| oodc_F1F2 | slt_lif_mpc_trace | 10 | 58.0% | 0.0% | 0.093 |
| oodc_F1F2 | slt_lif_mpc_free | 10 | 54.0% | 0.0% | 0.090 |
| oodc_F1F2 | mlp_baseline | 10 | 100.0% | 0.0% | 0.000 |
| oodc_F1F2 | spikedreamer_baseline | 10 | 100.0% | 0.0% | 0.000 |
| oodc_F1F3 | stjewm_trace_only | 5 | 76.0% | 0.0% | 0.059 |
| oodc_F1F3 | stjewm_spike_only | 5 | 80.0% | 0.0% | 0.050 |
| oodc_F1F3 | stjewm_rate_only | 5 | 88.0% | 0.0% | 0.054 |
| oodc_F1F3 | stjewm_no_trace | 5 | 92.0% | 0.0% | 0.044 |
| oodc_F1F3 | stjewm_hidden_leak | 5 | 80.0% | 0.0% | 0.053 |
| oodc_F1F3 | stjewm_membrane_readout | 5 | 76.0% | 0.0% | 0.072 |
| oodc_F1F3 | cubifae_baseline | 5 | 80.0% | 0.0% | 0.054 |
| oodc_F1F3 | gru_baseline | 5 | 100.0% | 0.0% | 0.000 |
| oodc_F1F3 | lewm_baseline_v2 | 5 | 48.0% | 0.0% | 0.227 |
| oodc_F1F3 | slt_lif_mpc_trace | 5 | 68.0% | 0.0% | 0.060 |
| oodc_F1F3 | slt_lif_mpc_free | 5 | 80.0% | 0.0% | 0.064 |
| oodc_F1F3 | mlp_baseline | 5 | 100.0% | 0.0% | 0.000 |
| oodc_F1F3 | spikedreamer_baseline | 5 | 100.0% | 0.0% | 0.000 |
| oodc_F2 | stjewm_trace_only | 5 | 36.0% | 0.0% | 0.136 |
| oodc_F2 | stjewm_spike_only | 5 | 52.0% | 0.0% | 0.119 |
| oodc_F2 | stjewm_rate_only | 5 | 48.0% | 0.0% | 0.113 |
| oodc_F2 | stjewm_no_trace | 5 | 52.0% | 0.0% | 0.111 |
| oodc_F2 | stjewm_hidden_leak | 5 | 40.0% | 0.0% | 0.139 |
| oodc_F2 | stjewm_membrane_readout | 5 | 32.0% | 0.0% | 0.167 |
| oodc_F2 | cubifae_baseline | 5 | 40.0% | 0.0% | 0.129 |
| oodc_F2 | gru_baseline | 5 | 100.0% | 0.0% | 0.002 |
| oodc_F2 | lewm_baseline_v2 | 5 | 12.0% | 0.0% | 0.244 |
| oodc_F2 | slt_lif_mpc_trace | 5 | 44.0% | 0.0% | 0.114 |
| oodc_F2 | slt_lif_mpc_free | 5 | 44.0% | 0.0% | 0.142 |
| oodc_F2 | mlp_baseline | 5 | 100.0% | 0.0% | 0.000 |
| oodc_F2 | spikedreamer_baseline | 5 | 100.0% | 0.0% | -0.000 |
| oodc_F2F3 | stjewm_trace_only | 6 | 36.7% | 0.0% | 0.115 |
| oodc_F2F3 | stjewm_spike_only | 6 | 46.7% | 0.0% | 0.112 |
| oodc_F2F3 | stjewm_rate_only | 6 | 46.7% | 0.0% | 0.111 |
| oodc_F2F3 | stjewm_no_trace | 6 | 43.3% | 0.0% | 0.131 |
| oodc_F2F3 | stjewm_hidden_leak | 6 | 53.3% | 0.0% | 0.130 |
| oodc_F2F3 | stjewm_membrane_readout | 6 | 56.7% | 0.0% | 0.118 |
| oodc_F2F3 | cubifae_baseline | 6 | 46.7% | 0.0% | 0.105 |
| oodc_F2F3 | gru_baseline | 6 | 100.0% | 0.0% | 0.002 |
| oodc_F2F3 | lewm_baseline_v2 | 6 | 33.3% | 0.0% | 0.158 |
| oodc_F2F3 | slt_lif_mpc_trace | 6 | 53.3% | 0.0% | 0.099 |
| oodc_F2F3 | slt_lif_mpc_free | 6 | 60.0% | 0.0% | 0.094 |
| oodc_F2F3 | mlp_baseline | 6 | 100.0% | 0.0% | 0.000 |
| oodc_F2F3 | spikedreamer_baseline | 6 | 100.0% | 0.0% | 0.000 |
| oodc_F3 | stjewm_trace_only | 1 | 100.0% | 0.0% | 0.008 |
| oodc_F3 | stjewm_spike_only | 1 | 100.0% | 0.0% | 0.010 |
| oodc_F3 | stjewm_rate_only | 1 | 100.0% | 0.0% | 0.008 |
| oodc_F3 | stjewm_no_trace | 1 | 100.0% | 0.0% | 0.011 |
| oodc_F3 | stjewm_hidden_leak | 1 | 100.0% | 0.0% | 0.009 |
| oodc_F3 | stjewm_membrane_readout | 1 | 100.0% | 0.0% | 0.016 |
| oodc_F3 | cubifae_baseline | 1 | 100.0% | 0.0% | 0.004 |
| oodc_F3 | gru_baseline | 1 | 100.0% | 0.0% | 0.000 |
| oodc_F3 | lewm_baseline_v2 | 1 | 40.0% | 0.0% | 0.087 |
| oodc_F3 | slt_lif_mpc_trace | 1 | 100.0% | 0.0% | 0.007 |
| oodc_F3 | slt_lif_mpc_free | 1 | 100.0% | 0.0% | 0.008 |
| oodc_F3 | mlp_baseline | 1 | 100.0% | 0.0% | 0.000 |
| oodc_F3 | spikedreamer_baseline | 1 | 100.0% | 0.0% | 0.000 |

## Per-env LeWM-SR per model (all splits pooled)

| Env | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | lewm_baseline_v2 | slt_lif_mpc_trace | slt_lif_mpc_free | mlp_baseline | spikedreamer_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ball_in_cup | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| cartpole_2d | 65.0% | 70.0% | 65.0% | 80.0% | 65.0% | 60.0% | 73.3% | 100.0% | 30.0% | 60.0% | 60.0% | 100.0% | 100.0% |
| cheetah | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 75.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| dog | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| finger | 30.0% | 45.0% | 45.0% | 45.0% | 50.0% | 30.0% | 43.3% | 100.0% | 6.7% | 25.0% | 26.7% | 100.0% | 100.0% |
| fish | 20.0% | 0.0% | 0.0% | 0.0% | 20.0% | 0.0% | 20.0% | 26.7% | 13.3% | 20.0% | — | 100.0% | 100.0% |
| hopper | 65.0% | 80.0% | 75.0% | 55.0% | 75.0% | 70.0% | 63.3% | 100.0% | 43.3% | 70.0% | 93.3% | 100.0% | 100.0% |
| humanoid | 10.0% | 0.0% | 0.0% | 15.0% | 10.0% | 5.0% | 6.7% | 100.0% | 3.3% | 0.0% | 6.7% | 100.0% | 100.0% |
| pendulum_2d | 95.0% | 90.0% | 100.0% | 85.0% | 100.0% | 80.0% | 83.3% | 100.0% | 30.0% | 80.0% | 93.3% | 100.0% | 100.0% |
| pusht | — | — | — | — | — | — | 40.0% | 0.0% | 20.0% | — | — | 20.0% | 100.0% |
| quadruped | 65.0% | 65.0% | 65.0% | 65.0% | 55.0% | 60.0% | 63.3% | 100.0% | 50.0% | 65.0% | 60.0% | 100.0% | 100.0% |
| reacher | 60.0% | 40.0% | 60.0% | 40.0% | 20.0% | 40.0% | 46.7% | 100.0% | 13.3% | 80.0% | — | 100.0% | 100.0% |
| stacker | 60.0% | 40.0% | 40.0% | 60.0% | 40.0% | 60.0% | 53.3% | 100.0% | 0.0% | 40.0% | — | 100.0% | 100.0% |
| tworoom | 100.0% | 80.0% | 80.0% | 60.0% | 80.0% | 80.0% | 80.0% | 30.0% | 80.0% | 80.0% | — | 100.0% | 100.0% |
| walker | 35.0% | 65.0% | 70.0% | 45.0% | 45.0% | 60.0% | 60.0% | 100.0% | 13.3% | 80.0% | 60.0% | 100.0% | 100.0% |

## Probes (event-AUROC)

Total probes: 207 (skipped=75, OK=132)

| Target | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | lewm_baseline_v2 | slt_lif_mpc_trace | slt_lif_mpc_free | mlp_baseline | spikedreamer_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| contact | 0.478 | 0.532 | 0.505 | 0.453 | 0.519 | 0.504 | 0.533 | 0.660 | — | — | — | — | — |
| entered | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | — | — | — | — | — |
| k10 | 0.536 | 0.514 | 0.459 | 0.486 | 0.545 | 0.505 | 0.432 | 0.591 | — | — | — | — | — |
| k5 | 0.507 | 0.497 | 0.505 | 0.476 | 0.520 | 0.440 | 0.478 | 0.489 | — | — | — | — | — |
| motion | 0.419 | 0.485 | 0.453 | 0.459 | 0.401 | 0.430 | 0.546 | 0.535 | — | — | — | — | — |
| target | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | — | — | — | — | — |

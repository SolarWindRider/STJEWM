# Generalist 5M-Aligned Table

Splits: cross_benchmark_F1, cross_benchmark_F2, cross_benchmark_F3, generalist_16env, oodc_F1, oodc_F1F2, oodc_F1F3, oodc_F2, oodc_F2F3, oodc_F3
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
| cross_benchmark_F2 | stjewm_trace_only | 14 | 57.1% | 0.0% | 0.121 |
| cross_benchmark_F2 | stjewm_spike_only | 14 | 50.0% | 0.0% | 0.128 |
| cross_benchmark_F2 | stjewm_rate_only | 14 | 58.6% | 0.0% | 0.112 |
| cross_benchmark_F2 | stjewm_no_trace | 14 | 50.0% | 0.0% | 0.130 |
| cross_benchmark_F2 | stjewm_hidden_leak | 14 | 50.0% | 0.0% | 0.139 |
| cross_benchmark_F2 | stjewm_membrane_readout | 14 | 48.6% | 0.0% | 0.151 |
| cross_benchmark_F2 | cubifae_baseline | 14 | 52.9% | 0.0% | 0.127 |
| cross_benchmark_F2 | gru_baseline | 14 | 87.1% | 0.0% | 0.025 |
| cross_benchmark_F2 | lewm_baseline_v2 | 14 | 25.7% | 0.0% | 0.201 |
| cross_benchmark_F2 | mlp_baseline | 14 | 92.9% | 0.0% | 0.015 |
| cross_benchmark_F2 | spikedreamer_baseline | 14 | 100.0% | 0.0% | 0.000 |
| cross_benchmark_F3 | stjewm_trace_only | 14 | 55.7% | 0.0% | 0.112 |
| cross_benchmark_F3 | stjewm_spike_only | 14 | 55.7% | 0.0% | 0.116 |
| cross_benchmark_F3 | stjewm_rate_only | 14 | 55.7% | 0.0% | 0.112 |
| cross_benchmark_F3 | stjewm_no_trace | 11 | 54.5% | 0.0% | 0.131 |
| cross_benchmark_F3 | stjewm_hidden_leak | 6 | 83.3% | 0.0% | 0.051 |
| cross_benchmark_F3 | cubifae_baseline | 14 | 57.1% | 0.0% | 0.108 |
| cross_benchmark_F3 | gru_baseline | 14 | 81.4% | 0.0% | 0.045 |
| cross_benchmark_F3 | lewm_baseline_v2 | 14 | 35.7% | 0.0% | 0.168 |
| cross_benchmark_F3 | mlp_baseline | 14 | 94.3% | 0.0% | 0.014 |
| cross_benchmark_F3 | spikedreamer_baseline | 14 | 100.0% | 0.0% | 0.000 |
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
| cartpole_2d | 66.7% | 70.0% | 66.7% | 83.3% | 66.7% | 60.0% | 71.4% | 100.0% | 28.6% | 60.0% | 60.0% | 100.0% | 100.0% |
| cheetah | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 77.8% | 100.0% | 100.0% | 100.0% | 100.0% |
| dog | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| finger | 36.7% | 43.3% | 56.7% | 43.3% | 46.7% | 32.0% | 45.7% | 100.0% | 8.6% | 25.0% | 26.7% | 100.0% | 100.0% |
| fish | 13.3% | 0.0% | 6.7% | 13.3% | 10.0% | 10.0% | 20.0% | 20.0% | 15.0% | 20.0% | — | 100.0% | 100.0% |
| hopper | 73.3% | 76.7% | 73.3% | 56.7% | 76.0% | 64.0% | 65.7% | 100.0% | 40.0% | 70.0% | 93.3% | 100.0% | 100.0% |
| humanoid | 10.0% | 0.0% | 0.0% | 10.0% | 8.0% | 4.0% | 5.7% | 100.0% | 2.9% | 0.0% | 6.7% | 100.0% | 100.0% |
| pendulum_2d | 96.7% | 83.3% | 100.0% | 90.0% | 100.0% | 84.0% | 85.7% | 100.0% | 34.3% | 80.0% | 93.3% | 100.0% | 100.0% |
| pusht | 20.0% | 40.0% | 10.0% | 0.0% | 0.0% | 0.0% | 40.0% | 0.0% | 13.3% | — | — | 26.7% | 100.0% |
| quadruped | 63.3% | 66.7% | 63.3% | 63.3% | 56.0% | 60.0% | 65.7% | 100.0% | 51.4% | 65.0% | 60.0% | 100.0% | 100.0% |
| reacher | 70.0% | 50.0% | 70.0% | 50.0% | 40.0% | 50.0% | 46.7% | 100.0% | 13.3% | 80.0% | — | 100.0% | 100.0% |
| stacker | 53.3% | 40.0% | 46.7% | 50.0% | 40.0% | 60.0% | 50.0% | 100.0% | 5.0% | 40.0% | — | 100.0% | 100.0% |
| tworoom | 70.0% | 80.0% | 80.0% | 60.0% | 80.0% | 80.0% | 80.0% | 33.3% | 80.0% | 80.0% | — | 93.3% | 100.0% |
| walker | 46.7% | 66.7% | 70.0% | 40.0% | 53.3% | 56.0% | 57.1% | 100.0% | 11.4% | 80.0% | 60.0% | 100.0% | 100.0% |

## Probes (event-AUROC)

Total probes: 520 (skipped=200, OK=320)

| Target | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | lewm_baseline_v2 | slt_lif_mpc_trace | slt_lif_mpc_free | mlp_baseline | spikedreamer_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| contact | 0.490 | 0.517 | 0.514 | 0.488 | 0.523 | 0.520 | 0.504 | 0.563 | — | — | — | — | — |
| entered | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | — | — | — | — | — |
| k10 | 0.534 | 0.526 | 0.477 | 0.520 | 0.528 | 0.507 | 0.510 | 0.492 | — | — | — | — | — |
| k5 | 0.528 | 0.501 | 0.521 | 0.495 | 0.530 | 0.488 | 0.497 | 0.532 | — | — | — | — | — |
| motion | 0.502 | 0.509 | 0.505 | 0.497 | 0.498 | 0.504 | 0.537 | 0.561 | — | — | — | — | — |
| target | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | — | — | — | — | — |

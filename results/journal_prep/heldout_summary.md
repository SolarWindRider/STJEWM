# Held-out eval summary (2026-09-01, 546 cells)

Protocol: CEM 300x30x10, H=5, budget=50, 5 eps x 1 seed; ckpts = existing 5M-aligned (no retrain).

| split | envs | model | n | mean_cos_dist |
|---|---|---|---|---|
| oodc_F1 | | stjewm_trace_only | 8 | 0.0980 |
| oodc_F1 | | stjewm_spike_only | 8 | 0.1084 |
| oodc_F1 | | stjewm_rate_only | 8 | 0.1072 |
| oodc_F1 | | stjewm_no_trace | 8 | 0.1240 |
| oodc_F1 | | stjewm_hidden_leak | 8 | 0.1349 |
| oodc_F1 | | stjewm_membrane_readout | 8 | 0.1142 |
| oodc_F1 | | alif_timecell_baseline | 8 | 0.1134 |
| oodc_F1 | | lif_transformer_baseline | 8 | 0.0000 |
| oodc_F1 | | stacked_lif_trace | 8 | 0.1012 |
| oodc_F1 | | stacked_lif_free | 8 | 0.0948 |
| oodc_F1 | | lewm_baseline_v2 | 8 | 0.1508 |
| oodc_F1 | | gru_baseline | 8 | 0.0065 |
| oodc_F1 | | mlp_baseline | 8 | 0.0002 |
| oodc_F2 | | stjewm_trace_only | 7 | 0.0478 |
| oodc_F2 | | stjewm_spike_only | 7 | 0.0362 |
| oodc_F2 | | stjewm_rate_only | 7 | 0.0374 |
| oodc_F2 | | stjewm_no_trace | 7 | 0.0397 |
| oodc_F2 | | stjewm_hidden_leak | 7 | 0.0490 |
| oodc_F2 | | stjewm_membrane_readout | 7 | 0.0469 |
| oodc_F2 | | alif_timecell_baseline | 7 | 0.0314 |
| oodc_F2 | | lif_transformer_baseline | 7 | -0.0000 |
| oodc_F2 | | stacked_lif_trace | 7 | 0.0357 |
| oodc_F2 | | stacked_lif_free | 7 | 0.0299 |
| oodc_F2 | | lewm_baseline_v2 | 7 | 0.1260 |
| oodc_F2 | | gru_baseline | 7 | 0.0042 |
| oodc_F2 | | mlp_baseline | 7 | 0.0002 |
| oodc_F3 | | stjewm_trace_only | 11 | 0.0976 |
| oodc_F3 | | stjewm_spike_only | 11 | 0.0905 |
| oodc_F3 | | stjewm_rate_only | 11 | 0.0982 |
| oodc_F3 | | stjewm_no_trace | 11 | 0.1052 |
| oodc_F3 | | stjewm_hidden_leak | 11 | 0.1036 |
| oodc_F3 | | stjewm_membrane_readout | 11 | 0.0990 |
| oodc_F3 | | alif_timecell_baseline | 11 | 0.1085 |
| oodc_F3 | | lif_transformer_baseline | 11 | 0.0000 |
| oodc_F3 | | stacked_lif_trace | 11 | 0.1026 |
| oodc_F3 | | stacked_lif_free | 11 | 0.0935 |
| oodc_F3 | | lewm_baseline_v2 | 11 | 0.1710 |
| oodc_F3 | | gru_baseline | 11 | 0.0016 |
| oodc_F3 | | mlp_baseline | 11 | 0.0000 |
| oodc_F1F2 | | stjewm_trace_only | 2 | 0.0069 |
| oodc_F1F2 | | stjewm_spike_only | 2 | 0.0113 |
| oodc_F1F2 | | stjewm_rate_only | 2 | 0.0110 |
| oodc_F1F2 | | stjewm_no_trace | 2 | 0.0103 |
| oodc_F1F2 | | stjewm_hidden_leak | 2 | 0.0147 |
| oodc_F1F2 | | stjewm_membrane_readout | 2 | 0.0139 |
| oodc_F1F2 | | alif_timecell_baseline | 2 | 0.0089 |
| oodc_F1F2 | | lif_transformer_baseline | 2 | 0.0000 |
| oodc_F1F2 | | stacked_lif_trace | 2 | 0.0095 |
| oodc_F1F2 | | stacked_lif_free | 2 | 0.0114 |
| oodc_F1F2 | | lewm_baseline_v2 | 2 | 0.0227 |
| oodc_F1F2 | | gru_baseline | 2 | 0.0185 |
| oodc_F1F2 | | mlp_baseline | 2 | 0.0006 |
| oodc_F1F3 | | stjewm_trace_only | 6 | 0.1365 |
| oodc_F1F3 | | stjewm_spike_only | 6 | 0.1268 |
| oodc_F1F3 | | stjewm_rate_only | 6 | 0.1381 |
| oodc_F1F3 | | stjewm_no_trace | 6 | 0.1583 |
| oodc_F1F3 | | stjewm_hidden_leak | 6 | 0.1296 |
| oodc_F1F3 | | stjewm_membrane_readout | 6 | 0.1662 |
| oodc_F1F3 | | alif_timecell_baseline | 6 | 0.1341 |
| oodc_F1F3 | | lif_transformer_baseline | 6 | 0.0000 |
| oodc_F1F3 | | stacked_lif_trace | 6 | 0.1249 |
| oodc_F1F3 | | stacked_lif_free | 6 | 0.1477 |
| oodc_F1F3 | | lewm_baseline_v2 | 6 | 0.1842 |
| oodc_F1F3 | | gru_baseline | 6 | 0.0021 |
| oodc_F1F3 | | mlp_baseline | 6 | 0.0000 |
| oodc_F2F3 | | stjewm_trace_only | 5 | 0.0431 |
| oodc_F2F3 | | stjewm_spike_only | 5 | 0.0406 |
| oodc_F2F3 | | stjewm_rate_only | 5 | 0.0615 |
| oodc_F2F3 | | stjewm_no_trace | 5 | 0.0366 |
| oodc_F2F3 | | stjewm_hidden_leak | 5 | 0.0495 |
| oodc_F2F3 | | stjewm_membrane_readout | 5 | 0.0480 |
| oodc_F2F3 | | alif_timecell_baseline | 5 | 0.0385 |
| oodc_F2F3 | | lif_transformer_baseline | 5 | -0.0000 |
| oodc_F2F3 | | stacked_lif_trace | 5 | 0.0588 |
| oodc_F2F3 | | stacked_lif_free | 5 | 0.0601 |
| oodc_F2F3 | | lewm_baseline_v2 | 5 | 0.1632 |
| oodc_F2F3 | | gru_baseline | 5 | 0.0003 |
| oodc_F2F3 | | mlp_baseline | 5 | 0.0000 |
| cross_benchmark_F1 | | stjewm_trace_only | 1 | 0.1484 |
| cross_benchmark_F1 | | stjewm_spike_only | 1 | 0.2654 |
| cross_benchmark_F1 | | stjewm_rate_only | 1 | 0.2414 |
| cross_benchmark_F1 | | stjewm_no_trace | 1 | 0.1147 |
| cross_benchmark_F1 | | stjewm_hidden_leak | 1 | 0.0863 |
| cross_benchmark_F1 | | stjewm_membrane_readout | 1 | 0.2550 |
| cross_benchmark_F1 | | alif_timecell_baseline | 1 | 0.1002 |
| cross_benchmark_F1 | | lif_transformer_baseline | 1 | 0.0000 |
| cross_benchmark_F1 | | stacked_lif_trace | 1 | 0.1011 |
| cross_benchmark_F1 | | stacked_lif_free | 1 | 0.1174 |
| cross_benchmark_F1 | | lewm_baseline_v2 | 1 | 0.0998 |
| cross_benchmark_F1 | | gru_baseline | 1 | 0.2093 |
| cross_benchmark_F1 | | mlp_baseline | 1 | 0.1361 |
| cross_benchmark_F2 | | stjewm_trace_only | 1 | 0.0468 |
| cross_benchmark_F2 | | stjewm_spike_only | 1 | 0.1034 |
| cross_benchmark_F2 | | stjewm_rate_only | 1 | 0.0833 |
| cross_benchmark_F2 | | stjewm_no_trace | 1 | 0.0611 |
| cross_benchmark_F2 | | stjewm_hidden_leak | 1 | 0.0616 |
| cross_benchmark_F2 | | stjewm_membrane_readout | 1 | 0.0684 |
| cross_benchmark_F2 | | alif_timecell_baseline | 1 | 0.0413 |
| cross_benchmark_F2 | | lif_transformer_baseline | 1 | 0.0000 |
| cross_benchmark_F2 | | stacked_lif_trace | 1 | 0.0742 |
| cross_benchmark_F2 | | stacked_lif_free | 1 | 0.0582 |
| cross_benchmark_F2 | | lewm_baseline_v2 | 1 | 0.0686 |
| cross_benchmark_F2 | | gru_baseline | 1 | 0.1083 |
| cross_benchmark_F2 | | mlp_baseline | 1 | 0.0450 |
| cross_benchmark_F3 | | stjewm_trace_only | 1 | 0.1375 |
| cross_benchmark_F3 | | stjewm_spike_only | 1 | 0.1207 |
| cross_benchmark_F3 | | stjewm_rate_only | 1 | 0.1951 |
| cross_benchmark_F3 | | stjewm_no_trace | 1 | 0.1111 |
| cross_benchmark_F3 | | stjewm_hidden_leak | 1 | 0.1358 |
| cross_benchmark_F3 | | stjewm_membrane_readout | 1 | 0.1008 |
| cross_benchmark_F3 | | alif_timecell_baseline | 1 | 0.0977 |
| cross_benchmark_F3 | | lif_transformer_baseline | 1 | 0.0000 |
| cross_benchmark_F3 | | stacked_lif_trace | 1 | 0.0944 |
| cross_benchmark_F3 | | stacked_lif_free | 1 | 0.0805 |
| cross_benchmark_F3 | | lewm_baseline_v2 | 1 | 0.2483 |
| cross_benchmark_F3 | | gru_baseline | 1 | 0.0016 |
| cross_benchmark_F3 | | mlp_baseline | 1 | 0.0000 |

## Per-split calibrated-cluster band (STJEWM 6 readouts)

- oodc_F1: STJEWM range 0.098–0.135 (n envs 1)
- oodc_F2: STJEWM range 0.036–0.049 (n envs 1)
- oodc_F3: STJEWM range 0.091–0.105 (n envs 1)
- oodc_F1F2: STJEWM range 0.007–0.015 (n envs 1)
- oodc_F1F3: STJEWM range 0.127–0.166 (n envs 1)
- oodc_F2F3: STJEWM range 0.037–0.062 (n envs 1)
- cross_benchmark_F1: STJEWM range 0.086–0.265 (n envs 1)
- cross_benchmark_F2: STJEWM range 0.047–0.103 (n envs 1)
- cross_benchmark_F3: STJEWM range 0.101–0.195 (n envs 1)

## Regime verdict
- calibrated: STJEWM 6 readouts 0.085–0.096, ALIF 0.085, SL-trace 0.084, SL-free 0.082
- collapse: LIF-Transformer 0.000, MLP 0.004, GRU 0.011
- over-react: LeWM-v2 0.151

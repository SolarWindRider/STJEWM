# Post-fix closed-loop summary (weight-loaded rerun)

All numbers below come from evals run after the closed_loop.py
weight-loading fix (strict load_state_dict).

| model | seen cos (n) | heldout oodc cos | heldout cross cos | LeWM-SR |
|---|---|---|---|---|
| stjewm_trace_only | 0.2856 (96) | 0.1932 (39) | 0.3238 (3) | 48.5% |
| stjewm_spike_only | 0.2620 (96) | 0.1936 (39) | 0.2076 (3) | 49.6% |
| stjewm_rate_only | 0.2626 (80) | 0.1806 (39) | 0.2407 (3) | 49.8% |
| stjewm_no_trace | 0.2527 (80) | 0.1975 (38) | 0.2235 (3) | 49.2% |
| stjewm_hidden_leak | 0.2533 (81) | 0.1991 (39) | 0.0255 (2) | 53.8% |
| stjewm_membrane_readout | 0.2783 (80) | 0.1878 (39) | 0.3188 (3) | 45.5% |
| alif_timecell_baseline | 0.1229 (75) | 0.0975 (39) | 0.0827 (3) | 69.1% |
| lif_transformer_baseline | 0.0000 (74) | -0.0000 (39) | 0.0000 (3) | 100.0% |
| stacked_lif_trace | 0.0769 (62) | 0.0827 (39) | 0.0070 (2) | 80.6% |
| stacked_lif_free | 0.0946 (65) | 0.0936 (38) | 0.0697 (2) | 74.8% |
| lewm_baseline_v2 | 0.2520 (80) | 0.1785 (39) | 0.1109 (3) | 22.8% |
| gru_baseline | 0.1493 (78) | 0.1146 (38) | 0.0905 (3) | 50.5% |
| mlp_baseline | 0.0363 (77) | 0.0000 (39) | 0.0965 (3) | 89.6% |

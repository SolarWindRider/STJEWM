# Post-fix closed-loop summary (weight-loaded rerun)

All numbers below come from evals run after the closed_loop.py
weight-loading fix (strict load_state_dict).

| model | seen cos (n) | heldout oodc cos | heldout cross cos | LeWM-SR |
|---|---|---|---|---|
| stjewm_trace_only | 0.2856 (96) | 0.1932 (39) | 0.3238 (3) | 48.5% |
| stjewm_spike_only | 0.2620 (96) | 0.1936 (39) | 0.2076 (3) | 49.6% |
| stjewm_rate_only | 0.2770 (96) | 0.1806 (39) | 0.2407 (3) | 47.3% |
| stjewm_no_trace | 0.2615 (96) | 0.1975 (38) | 0.2235 (3) | 49.0% |
| stjewm_hidden_leak | 0.2533 (81) | 0.1991 (39) | 0.0255 (2) | 53.8% |
| stjewm_membrane_readout | 0.2805 (96) | 0.1878 (39) | 0.3188 (3) | 45.6% |
| alif_timecell_baseline | 0.1380 (96) | 0.0975 (39) | 0.0827 (3) | 66.0% |
| lif_transformer_baseline | 0.0000 (96) | -0.0000 (39) | 0.0000 (3) | 100.0% |
| stacked_lif_trace | 0.0724 (81) | 0.0827 (39) | 0.0070 (2) | 81.0% |
| stacked_lif_free | 0.0973 (81) | 0.0936 (38) | 0.0697 (2) | 73.6% |
| lewm_baseline_v2 | 0.2197 (96) | 0.1785 (39) | 0.1109 (3) | 32.9% |
| gru_baseline | 0.1486 (96) | 0.1146 (38) | 0.0905 (3) | 50.2% |
| mlp_baseline | 0.0298 (96) | 0.0000 (39) | 0.0965 (3) | 91.7% |

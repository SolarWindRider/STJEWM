# Post-fix closed-loop summary (weight-loaded rerun)

All numbers below come from evals run after the closed_loop.py
weight-loading fix (strict load_state_dict).

| model | seen cos (n) | heldout oodc cos | heldout cross cos | LeWM-SR |
|---|---|---|---|---|
| stjewm_trace_only | 0.1013 (99) | 0.0801 (39) | 0.1100 (3) | 56.2% |
| stjewm_spike_only | 0.1130 (99) | 0.0844 (39) | 0.1573 (3) | 54.1% |
| stjewm_rate_only | 0.1024 (99) | 0.0800 (39) | 0.1244 (3) | 51.9% |
| stjewm_no_trace | 0.1286 (99) | 0.0960 (39) | 0.2131 (3) | 51.3% |
| stjewm_hidden_leak | 0.1277 (99) | 0.0937 (39) | 0.2232 (3) | 50.7% |
| stjewm_membrane_readout | 0.1286 (99) | 0.0960 (39) | 0.2131 (3) | 51.3% |
| alif_timecell_baseline | 0.1067 (99) | 0.0823 (39) | 0.1445 (3) | 54.3% |
| lif_transformer_baseline | 0.0000 (99) | 0.0000 (39) | 0.0000 (3) | 100.0% |
| stacked_lif_trace | 0.1427 (99) | 0.1201 (39) | 0.1326 (3) | 46.3% |
| stacked_lif_free | 0.1543 (99) | 0.1302 (39) | 0.1545 (3) | 45.5% |
| lewm_baseline_v2 | 0.1792 (99) | 0.1608 (39) | 0.1416 (3) | 33.9% |
| gru_baseline | 0.0416 (99) | 0.0194 (39) | 0.1112 (3) | 91.1% |
| mlp_baseline | 0.0000 (99) | 0.0000 (39) | 0.0000 (3) | 100.0% |

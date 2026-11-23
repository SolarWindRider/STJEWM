# Audited diagnostic scaling: G4 / G8 / G16

Primary measurement: forward.emb. All seven planned environments are paired across 36 checkpoints.
These are diagnostic means, not re-evaluated env-SR. No pre-audit regime classifications are carried forward.
Undefined ratios/correlations are excluded only from their metric mean, with the denominator shown explicitly.
One training seed: no seed confidence interval is estimable.

| suite | model | divergence | responsiveness | response defined/expected | event_rho | rho defined/expected |
|---|---|---|---|---|---|---|
| generalist_G4 | alif_timecell_baseline | 0.03430 | 5.743 | 7/7 | 0.305 | 7/7 |
| generalist_G4 | gru_baseline | 0.33354 | 168.459 | 7/7 | 0.173 | 7/7 |
| generalist_G4 | lewm_baseline_v2 | 0.37785 | 113.544 | 7/7 | 0.419 | 7/7 |
| generalist_G4 | mlp_baseline | 0.00000 | 0.000 | 7/7 | 0.122 | 7/7 |
| generalist_G4 | stacked_lif_free | 0.13082 | 820.940 | 7/7 | 0.018 | 7/7 |
| generalist_G4 | stacked_lif_trace | 0.00000 | 0.000 | 7/7 | undefined | 0/7 |
| generalist_G4 | stjewm_hidden_leak | 0.32613 | 119.456 | 7/7 | 0.196 | 7/7 |
| generalist_G4 | stjewm_membrane_readout | 0.32614 | 119.705 | 7/7 | 0.149 | 7/7 |
| generalist_G4 | stjewm_no_trace | 0.32614 | 119.705 | 7/7 | 0.149 | 7/7 |
| generalist_G4 | stjewm_rate_only | 0.00440 | 25.783 | 7/7 | 0.068 | 7/7 |
| generalist_G4 | stjewm_spike_only | 0.03018 | 90.827 | 7/7 | 0.106 | 7/7 |
| generalist_G4 | stjewm_trace_only | 0.00025 | 0.766 | 7/7 | 0.098 | 7/7 |
| generalist_G8 | alif_timecell_baseline | 0.05296 | 9.900 | 7/7 | 0.389 | 7/7 |
| generalist_G8 | gru_baseline | 0.52687 | 352.155 | 7/7 | 0.204 | 7/7 |
| generalist_G8 | lewm_baseline_v2 | 0.39755 | 54.925 | 7/7 | 0.526 | 7/7 |
| generalist_G8 | mlp_baseline | 0.00000 | 0.000 | 7/7 | 0.177 | 7/7 |
| generalist_G8 | stacked_lif_free | 0.09910 | 574.840 | 7/7 | -0.014 | 7/7 |
| generalist_G8 | stacked_lif_trace | 0.00000 | 0.000 | 7/7 | undefined | 0/7 |
| generalist_G8 | stjewm_hidden_leak | 0.32043 | 149.805 | 7/7 | 0.271 | 7/7 |
| generalist_G8 | stjewm_membrane_readout | 0.32202 | 80.918 | 7/7 | 0.214 | 7/7 |
| generalist_G8 | stjewm_no_trace | 0.32202 | 80.918 | 7/7 | 0.214 | 7/7 |
| generalist_G8 | stjewm_rate_only | 0.00919 | 31.980 | 7/7 | 0.207 | 7/7 |
| generalist_G8 | stjewm_spike_only | 0.02059 | 60.602 | 7/7 | 0.113 | 7/7 |
| generalist_G8 | stjewm_trace_only | 0.00018 | 0.566 | 7/7 | -0.032 | 7/7 |
| generalist_G16 | alif_timecell_baseline | 0.49780 | 1937.889 | 7/7 | 0.018 | 7/7 |
| generalist_G16 | gru_baseline | 0.71928 | 411.842 | 7/7 | 0.227 | 7/7 |
| generalist_G16 | lewm_baseline_v2 | 0.35711 | 68.722 | 7/7 | 0.512 | 7/7 |
| generalist_G16 | mlp_baseline | 144018598.09602 | 5934076050.073 | 7/7 | 0.161 | 7/7 |
| generalist_G16 | stacked_lif_free | 0.93668 | 319.229 | 7/7 | 0.385 | 7/7 |
| generalist_G16 | stacked_lif_trace | 0.02198 | 73.238 | 7/7 | 0.039 | 7/7 |
| generalist_G16 | stjewm_hidden_leak | 0.23277 | 15.361 | 7/7 | 0.722 | 7/7 |
| generalist_G16 | stjewm_membrane_readout | 0.23682 | 15.972 | 7/7 | 0.712 | 7/7 |
| generalist_G16 | stjewm_no_trace | 0.23682 | 15.972 | 7/7 | 0.712 | 7/7 |
| generalist_G16 | stjewm_rate_only | 0.06439 | 112.679 | 7/7 | -0.021 | 7/7 |
| generalist_G16 | stjewm_spike_only | 0.02467 | 68.938 | 7/7 | 0.147 | 7/7 |
| generalist_G16 | stjewm_trace_only | 0.00015 | 0.062 | 7/7 | 0.088 | 7/7 |

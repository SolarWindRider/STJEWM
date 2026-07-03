# Stress Suite — All Methods × All Tasks

Regenerated from results/<env>/<model>/eval.json (v0.7.2 — all 48 cells freshly re-evaluated).
STJEWM modes averaged over 3 seeds.

## Env-SR (%), the honest capability metric

| Env | stjewm_trace_only | stjewm_hidden_leak | stjewm_spike_only | stjewm_no_trace | stjewm_membrane_readout | stjewm_rate_only | cubifae_baseline | spikedreamer_baseline | slt_lif_mpc_trace | slt_lif_mpc_free | lewm_baseline_v2 | gru_baseline | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pusht_ood | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | n/a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| tworoom_long | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | n/a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cartpole_flicker | 0.0 | 2.0 | 0.0 | 0.0 | 2.0 | n/a | 2.0 | 0.0 | 0.0 | 6.0 | 2.0 | 68.0 | 30.0 |
| cheetah_velhidden | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | n/a | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| **AVG** | **25.0** | **25.5** | **25.0** | **25.0** | **25.5** | n/a | **25.5** | **25.0** | **25.0** | **26.5** | **25.5** | **42.0** | **32.5** |

## LeWM-SR (cos_dist < 0.1, %)

| Env | stjewm_trace_only | stjewm_hidden_leak | stjewm_spike_only | stjewm_no_trace | stjewm_membrane_readout | stjewm_rate_only | cubifae_baseline | spikedreamer_baseline | slt_lif_mpc_trace | slt_lif_mpc_free | lewm_baseline_v2 | gru_baseline | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pusht_ood | 64.0 | 14.0 | 30.0 | 16.0 | 12.0 | n/a | 16.0 | 0.0 | 0.0 | 32.0 | 22.0 | 0.0 | 82.0 |
| tworoom_long | 88.0 | 96.0 | 86.0 | 80.0 | 88.0 | n/a | 76.0 | 0.0 | 78.0 | 96.0 | 80.0 | 12.0 | 100.0 |
| cartpole_flicker | 16.0 | 26.0 | 18.0 | 24.0 | 18.0 | n/a | 20.0 | 0.0 | 16.0 | 44.0 | 30.0 | 92.0 | 100.0 |
| cheetah_velhidden | 98.0 | 82.0 | 96.0 | 90.0 | 80.0 | n/a | 98.0 | 0.0 | 96.0 | 94.0 | 94.0 | 100.0 | 100.0 |
| **AVG** | **66.5** | **54.5** | **57.5** | **52.5** | **49.5** | n/a | **52.5** | **0.0** | **47.5** | **66.5** | **56.5** | **51.0** | **95.5** |

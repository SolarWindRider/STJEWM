# Generalist vs Specialist — comparison table

Spec: `configs/generalist_4env_2k.json`

Models: stjewm_trace_only, stjewm_spike_only, stjewm_rate_only, stjewm_no_trace, stjewm_hidden_leak, stjewm_membrane_readout, cubifae_baseline, gru_baseline, lewm_baseline_v2, slt_lif_mpc_trace, slt_lif_mpc_free, mlp_baseline

Cells: env-native success rate (%) and LeWM-SR (cos_dist<0.1) (%)
'-' = eval JSON not yet produced

| env | stjewm_trace_only env-SR | stjewm_trace_only LeWM-SR | stjewm_spike_only env-SR | stjewm_spike_only LeWM-SR | stjewm_rate_only env-SR | stjewm_rate_only LeWM-SR | stjewm_no_trace env-SR | stjewm_no_trace LeWM-SR | stjewm_hidden_leak env-SR | stjewm_hidden_leak LeWM-SR | stjewm_membrane_readout env-SR | stjewm_membrane_readout LeWM-SR | cubifae_baseline env-SR | cubifae_baseline LeWM-SR | gru_baseline env-SR | gru_baseline LeWM-SR | lewm_baseline_v2 env-SR | lewm_baseline_v2 LeWM-SR | slt_lif_mpc_trace env-SR | slt_lif_mpc_trace LeWM-SR | slt_lif_mpc_free env-SR | slt_lif_mpc_free LeWM-SR | mlp_baseline env-SR | mlp_baseline LeWM-SR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cartpole_2d | 100.0 | 80.0 | - | - | - | - | - | - | 100.0 | 80.0 | - | - | - | - | 100.0 | 100.0 | 100.0 | 20.0 | - | - | - | - | - | - |
| pendulum_2d | 0.0 | 100.0 | - | - | - | - | - | - | 0.0 | 100.0 | - | - | - | - | 0.0 | 100.0 | 0.0 | 0.0 | - | - | - | - | - | - |
| cheetah | 100.0 | 100.0 | - | - | - | - | - | - | 100.0 | 100.0 | - | - | - | - | 100.0 | 100.0 | 100.0 | 80.0 | - | - | - | - | - | - |
| pusht | 0.0 | 20.0 | - | - | - | - | - | - | 0.0 | 0.0 | - | - | - | - | 0.0 | 0.0 | 0.0 | 0.0 | - | - | - | - | - | - |

### env-SR AVG per model

| model | all 20 | std only | stress only |
|---|---|---|---|
| stjewm_trace_only | 50.0 | 50.0 | 0.0 |
| stjewm_spike_only | 0.0 | 0.0 | 0.0 |
| stjewm_rate_only | 0.0 | 0.0 | 0.0 |
| stjewm_no_trace | 0.0 | 0.0 | 0.0 |
| stjewm_hidden_leak | 50.0 | 50.0 | 0.0 |
| stjewm_membrane_readout | 0.0 | 0.0 | 0.0 |
| cubifae_baseline | 0.0 | 0.0 | 0.0 |
| gru_baseline | 50.0 | 50.0 | 0.0 |
| lewm_baseline_v2 | 50.0 | 50.0 | 0.0 |
| slt_lif_mpc_trace | 0.0 | 0.0 | 0.0 |
| slt_lif_mpc_free | 0.0 | 0.0 | 0.0 |
| mlp_baseline | 0.0 | 0.0 | 0.0 |
### LeWM-SR AVG per model

| model | all 20 | std only | stress only |
|---|---|---|---|
| stjewm_trace_only | 75.0 | 75.0 | 0.0 |
| stjewm_spike_only | 0.0 | 0.0 | 0.0 |
| stjewm_rate_only | 0.0 | 0.0 | 0.0 |
| stjewm_no_trace | 0.0 | 0.0 | 0.0 |
| stjewm_hidden_leak | 70.0 | 70.0 | 0.0 |
| stjewm_membrane_readout | 0.0 | 0.0 | 0.0 |
| cubifae_baseline | 0.0 | 0.0 | 0.0 |
| gru_baseline | 75.0 | 75.0 | 0.0 |
| lewm_baseline_v2 | 25.0 | 25.0 | 0.0 |
| slt_lif_mpc_trace | 0.0 | 0.0 | 0.0 |
| slt_lif_mpc_free | 0.0 | 0.0 | 0.0 |
| mlp_baseline | 0.0 | 0.0 | 0.0 |

# Generalist vs Specialist — comparison table

Spec: `configs/generalist_4env_2k.json`

Models: stjewm_trace_only, stjewm_hidden_leak, lewm_baseline_v2, gru_baseline

Cells: env-native success rate (%) and LeWM-SR (cos_dist<0.1) (%)
'-' = eval JSON not yet produced

| env | stjewm_trace_only env-SR | stjewm_trace_only LeWM-SR | stjewm_hidden_leak env-SR | stjewm_hidden_leak LeWM-SR | lewm_baseline_v2 env-SR | lewm_baseline_v2 LeWM-SR | gru_baseline env-SR | gru_baseline LeWM-SR |
|---|---|---|---|---|---|---|---|---|
| cartpole_2d | 100.0 | 80.0 | 100.0 | 80.0 | 100.0 | 20.0 | 100.0 | 100.0 |
| pendulum_2d | 0.0 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| cheetah | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 80.0 | 100.0 | 100.0 |
| pusht | 0.0 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### env-SR AVG per model

| model | all 20 | std only | stress only |
|---|---|---|---|
| stjewm_trace_only | 50.0 | 50.0 | 0.0 |
| stjewm_hidden_leak | 50.0 | 50.0 | 0.0 |
| lewm_baseline_v2 | 50.0 | 50.0 | 0.0 |
| gru_baseline | 50.0 | 50.0 | 0.0 |
### LeWM-SR AVG per model

| model | all 20 | std only | stress only |
|---|---|---|---|
| stjewm_trace_only | 75.0 | 75.0 | 0.0 |
| stjewm_hidden_leak | 70.0 | 70.0 | 0.0 |
| lewm_baseline_v2 | 25.0 | 25.0 | 0.0 |
| gru_baseline | 75.0 | 75.0 | 0.0 |

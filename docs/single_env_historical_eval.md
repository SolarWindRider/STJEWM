# Single-env historical eval (archived 2026-08-13)

Early v0.7.x single-env Stacked-LIF experiments (results/<env>/<model>/eval.json)
were deleted in the legacy cleanup. The underlying checkpoints were irrecoverable
(training commands not recorded). Only these eval numbers survived and are kept
for the record; none feed the 5M audit tables (FULL_METRIC_MATRIX / MAIN_TABLE_*).

- `cheetah_velhidden/alif_timecell_baseline`: success_rate=0.98
- `cheetah_velhidden/gru_baseline`: success_rate=1.0
- `cheetah_velhidden/lewm_baseline_v2`: success_rate=0.94
- `cheetah_velhidden/lif_transformer_baseline`: success_rate=1.0
- `cheetah_velhidden/mlp_baseline`: success_rate=1.0
- `cheetah_velhidden/stacked_lif_free`: success_rate=0.94
- `cheetah_velhidden/stacked_lif_trace`: success_rate=0.96
- `cheetah_velhidden/stjewm_hidden_leak`: success_rate=0.8200000000000001
- `cheetah_velhidden/stjewm_membrane_readout`: success_rate=0.8
- `cheetah_velhidden/stjewm_no_trace`: success_rate=0.8999999999999999
- `cheetah_velhidden/stjewm_spike_only`: success_rate=0.96
- `cheetah_velhidden/stjewm_trace_only`: success_rate=0.98
- `cartpole_2d/gru_baseline`: success_rate=0.9199999999999999
- `cartpole_2d/lewm_baseline_no_goal`: success_rate=0.8600000000000001
- `cartpole_2d/lewm_baseline_v2`: success_rate=0.8600000000000001
- `cartpole_2d/lif_transformer_baseline`: success_rate=1.0
- `cartpole_2d/mlp_baseline`: success_rate=1.0
- `cartpole_2d/stacked_lif_free`: success_rate=0.8
- `cartpole_2d/stacked_lif_trace`: success_rate=1.0
- `cartpole_2d/stjewm_nogoal`: success_rate=0.8600000000000001
- `cartpole_2d/stjewm_rate_only`: success_rate=0.76
- `cartpole_2d/stjewm_v2`: success_rate=0.8600000000000001
- `cartpole_flicker/alif_timecell_baseline`: success_rate=0.2
- `cartpole_flicker/gru_baseline`: success_rate=0.9199999999999999
- `cartpole_flicker/lewm_baseline_v2`: success_rate=0.30000000000000004
- `cartpole_flicker/lif_transformer_baseline`: success_rate=1.0
- `cartpole_flicker/mlp_baseline`: success_rate=1.0
- `cartpole_flicker/stacked_lif_free`: success_rate=0.44
- `cartpole_flicker/stacked_lif_trace`: success_rate=0.16
- `cartpole_flicker/stjewm_hidden_leak`: success_rate=0.26
- `cartpole_flicker/stjewm_membrane_readout`: success_rate=0.18
- `cartpole_flicker/stjewm_no_trace`: success_rate=0.24000000000000002
- `cartpole_flicker/stjewm_rate_only`: success_rate=0.6000000000000001
- `cartpole_flicker/stjewm_spike_only`: success_rate=0.18
- `cartpole_flicker/stjewm_trace_only`: success_rate=0.16

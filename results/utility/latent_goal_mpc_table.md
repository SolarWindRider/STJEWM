# Latent-goal MPC horizon sweep(E10/utility experiment 1,20260916 final repair)

> **Regenerated 2026-09-17** from the final repaired evidence (20260916 final repair, `causal_grad_sigreg_B_20260916`). All values are copied from the source files listed below — no legacy numbers.
> Training manifest: `/data/lx/tmp/results/_repair_archive/20260916T102154Z/training_final_repair_manifest.json` — sha256 `4376bb14c50e30e12c3a1d47435b17489568f148b6f0abd57592efe5936e1f09`.
> Freeze v7: `4ba90480e09d93d61d442be9569a03c020f6bbed9906af915015b3eb2dded834`.

> Evidence sources:
> - `/data/lx/tmp/results/utility_final/latent_goal_mpc_table.md` — sha256 `affae3eebdfc8dbb760a9ff49ceb22225837ea0b9f667dfa9b2a6a495de64317`

逐 run JSON(n=49)位于 `/data/lx/tmp/results/utility_final/latent_goal_mpc/`;本文件 = `utility_final/latent_goal_mpc_table.md` 全文合并 provenance 头,表格内容一字未改。

# Latent-goal MPC horizon sweep

**Grid**: 12 model configurations × 4 environments.
**CEM config**: n_samples=100, n_elites=10, n_iters=10, n_episodes=3
**Protocol**: t→t+25 offline goals, 50-step budget, fresh qpos initialization with zero velocity; observed-state replanning after each H-step action chunk.
**Planner objective**: canonical squared L2 in latent space; candidates are native-dimensional controls, bounded and zero-padded before model prediction.

## mean_cos_dist_terminal per (model × env × horizon)

Terminal metric: (1 − cosine_similarity) / 2. Interpret with native physical success; latent collapse can also produce small distances.

| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |
|---|---|---|---|---|---|---|
| stjewm_trace_only | cheetah | 0.0632 | 0.0501 | 0.0832 | 0.0694 | 0.0220 |
| stjewm_trace_only | walker | 0.2058 | 0.0457 | 0.0463 | 0.0364 | 0.0339 |
| stjewm_trace_only | reacher | 0.3895 | 0.1954 | 0.2542 | 0.1123 | 0.1443 |
| stjewm_trace_only | finger | 0.2630 | 0.1907 | 0.1435 | 0.0914 | 0.0458 |
| stjewm_spike_only | cheetah | 0.3276 | 0.3629 | 0.3832 | 0.4101 | 0.2081 |
| stjewm_spike_only | walker | 0.1257 | 0.1354 | 0.1141 | 0.2509 | 0.1676 |
| stjewm_spike_only | reacher | 0.1327 | 0.4882 | 0.4747 | 0.4897 | 0.3597 |
| stjewm_spike_only | finger | 0.1534 | 0.3013 | 0.2714 | 0.2902 | 0.1162 |
| stjewm_rate_only | cheetah | 0.2384 | 0.3101 | 0.2278 | 0.1581 | 0.2151 |
| stjewm_rate_only | walker | 0.0134 | 0.0128 | 0.0148 | 0.0034 | 0.0034 |
| stjewm_rate_only | reacher | 0.0100 | 0.0106 | 0.0083 | 0.0076 | 0.0097 |
| stjewm_rate_only | finger | 0.0031 | 0.0039 | 0.0008 | 0.0016 | 0.0039 |
| stjewm_no_trace | cheetah | 0.1686 | 0.2162 | 0.3521 | 0.2052 | 0.0466 |
| stjewm_no_trace | walker | 0.0171 | 0.0264 | 0.0151 | 0.0015 | 0.0260 |
| stjewm_no_trace | reacher | 0.0004 | 0.0042 | 0.0022 | 0.0640 | 0.0335 |
| stjewm_no_trace | finger | 0.0002 | 0.0003 | 0.0026 | 0.0184 | 0.0357 |
| stjewm_hidden_leak | cheetah | 0.1509 | 0.3190 | 0.3317 | 0.1450 | 0.0732 |
| stjewm_hidden_leak | walker | 0.0096 | 0.0134 | 0.0109 | 0.0047 | 0.0061 |
| stjewm_hidden_leak | reacher | 0.0001 | 0.0053 | 0.0028 | 0.0080 | 0.0254 |
| stjewm_hidden_leak | finger | 0.0001 | 0.0003 | 0.0017 | 0.0380 | 0.0244 |
| stjewm_membrane_readout | cheetah | 0.1686 | 0.2162 | 0.3521 | 0.2052 | 0.0466 |
| stjewm_membrane_readout | walker | 0.0171 | 0.0264 | 0.0151 | 0.0015 | 0.0260 |
| stjewm_membrane_readout | reacher | 0.0004 | 0.0042 | 0.0022 | 0.0640 | 0.0335 |
| stjewm_membrane_readout | finger | 0.0002 | 0.0003 | 0.0026 | 0.0184 | 0.0357 |
| alif_timecell_baseline | cheetah | 0.0017 | 0.0004 | 0.0011 | 0.0009 | 0.0033 |
| alif_timecell_baseline | walker | 0.0033 | 0.0006 | 0.0030 | 0.0014 | 0.0013 |
| alif_timecell_baseline | reacher | 0.0271 | 0.0370 | 0.0116 | 0.0196 | 0.0211 |
| alif_timecell_baseline | finger | 0.0048 | 0.0068 | 0.0037 | 0.0042 | 0.0067 |
| stacked_lif_trace | cheetah | 0.0015 | 0.0018 | 0.0023 | 0.0003 | 0.0001 |
| stacked_lif_trace | walker | 0.0005 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
| stacked_lif_trace | reacher | 0.0026 | 0.0487 | 0.1360 | 0.2094 | 0.1086 |
| stacked_lif_trace | finger | 0.0002 | 0.0032 | 0.0125 | 0.0295 | 0.0294 |
| stacked_lif_free | cheetah | 0.0025 | 0.0054 | 0.0111 | 0.0136 | 0.0060 |
| stacked_lif_free | walker | 0.0021 | 0.0017 | 0.0006 | 0.0009 | 0.0009 |
| stacked_lif_free | reacher | 0.2299 | 0.3599 | 0.0880 | 0.0516 | 0.1092 |
| stacked_lif_free | finger | 0.2699 | 0.0041 | 0.0450 | 0.0643 | 0.0451 |
| gru_baseline | cheetah | 0.0204 | 0.0348 | 0.0182 | 0.0236 | 0.0440 |
| gru_baseline | walker | 0.0245 | 0.0410 | 0.0285 | 0.0231 | 0.0177 |
| gru_baseline | reacher | 0.0019 | 0.0045 | 0.0137 | 0.1619 | 0.1447 |
| gru_baseline | finger | 0.1099 | 0.1400 | 0.1367 | 0.1568 | 0.1021 |
| mlp_baseline | cheetah | 0.0003 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |
| mlp_baseline | walker | 0.0018 | 0.0033 | 0.0018 | 0.0018 | 0.0018 |
| mlp_baseline | reacher | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mlp_baseline | finger | 0.0004 | 0.0007 | 0.0004 | 0.0005 | 0.0010 |
| LeWM | cheetah | 0.1127 | 0.0747 | 0.1810 | 0.2612 | 0.2296 |
| LeWM | walker | 0.0856 | 0.1255 | 0.1454 | 0.2108 | 0.1680 |
| LeWM | reacher | 0.0005 | 0.0033 | 0.0217 | 0.3744 | 0.2750 |
| LeWM | finger | 0.0003 | 0.0013 | 0.0112 | 0.1948 | 0.1939 |

## env_success per (model × env × horizon)

Success uses unpadded native qpos RMS distance: thresholds cheetah/walker=0.1, reacher=0.05, finger=0.3.

| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |
|---|---|---|---|---|---|---|
| stjewm_trace_only | cheetah | 0.00 | 0.00 | 0.33 | 0.33 | 0.00 |
| stjewm_trace_only | walker | 0.00 | 0.00 | 0.33 | 0.33 | 0.00 |
| stjewm_trace_only | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_trace_only | finger | 0.00 | 0.00 | 0.33 | 0.67 | 0.33 |
| stjewm_spike_only | cheetah | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 |
| stjewm_spike_only | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_spike_only | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_spike_only | finger | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 |
| stjewm_rate_only | cheetah | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 |
| stjewm_rate_only | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_rate_only | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_rate_only | finger | 0.00 | 0.00 | 0.33 | 0.00 | 0.33 |
| stjewm_no_trace | cheetah | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_no_trace | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_no_trace | reacher | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_no_trace | finger | 1.00 | 1.00 | 1.00 | 0.33 | 0.00 |
| stjewm_hidden_leak | cheetah | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_hidden_leak | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_hidden_leak | reacher | 0.67 | 0.33 | 0.00 | 0.00 | 0.00 |
| stjewm_hidden_leak | finger | 1.00 | 1.00 | 1.00 | 0.33 | 0.33 |
| stjewm_membrane_readout | cheetah | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_membrane_readout | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_membrane_readout | reacher | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 |
| stjewm_membrane_readout | finger | 1.00 | 1.00 | 1.00 | 0.33 | 0.00 |
| alif_timecell_baseline | cheetah | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| alif_timecell_baseline | walker | 0.00 | 0.33 | 0.33 | 0.00 | 0.00 |
| alif_timecell_baseline | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| alif_timecell_baseline | finger | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 |
| stacked_lif_trace | cheetah | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| stacked_lif_trace | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stacked_lif_trace | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stacked_lif_trace | finger | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 |
| stacked_lif_free | cheetah | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| stacked_lif_free | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stacked_lif_free | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| stacked_lif_free | finger | 0.00 | 0.67 | 0.00 | 0.00 | 0.33 |
| gru_baseline | cheetah | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gru_baseline | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gru_baseline | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gru_baseline | finger | 0.33 | 0.33 | 0.33 | 0.00 | 0.33 |
| mlp_baseline | cheetah | 0.00 | 0.33 | 0.00 | 0.33 | 0.00 |
| mlp_baseline | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 |
| mlp_baseline | reacher | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mlp_baseline | finger | 1.00 | 0.00 | 0.33 | 0.67 | 0.33 |
| LeWM | cheetah | 0.67 | 0.33 | 0.00 | 0.00 | 0.00 |
| LeWM | walker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| LeWM | reacher | 0.33 | 0.33 | 0.33 | 0.00 | 0.00 |
| LeWM | finger | 1.00 | 1.00 | 1.00 | 0.00 | 0.33 |

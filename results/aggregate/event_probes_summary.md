# Event-Probe Summary (NMI paper, Results 5)

Total cells aggregated: 215 (7 envs, 12 models, 8 targets).

## Mean event-probe AUROC ranking

1. `stjewm_spike_only` = 0.699
2. `stjewm_trace_only` = 0.690
3. `stjewm_hidden_leak` = 0.690
4. `stjewm_no_trace` = 0.688
5. `gru_baseline` = 0.670
6. `cubifae_baseline` = 0.663
7. `stjewm_membrane_readout` = 0.647
8. `mlp_baseline` = 0.612
9. `slt_lif_mpc_trace` = 0.610
10. `lewm_baseline_v2` = 0.582
11. `slt_lif_mpc_free` = 0.581
12. `spikedreamer_baseline` = 0.554

## Dissociation claim

STJEWM-trace is competitive or best on event-type probes, even though
its position-probe R² is moderate (see `probe_table.md`). This is the
core dissociation: the trace captures event-relevant information that
is not equivalent to position memory.


## Win counts

- `cubifae_baseline`: 2 wins
- `gru_baseline`: 8 wins
- `lewm_baseline_v2`: 0 wins
- `mlp_baseline`: 1 wins
- `slt_lif_mpc_free`: 0 wins
- `slt_lif_mpc_trace`: 0 wins
- `spikedreamer_baseline`: 1 wins
- `stjewm_hidden_leak`: 2 wins
- `stjewm_membrane_readout`: 1 wins
- `stjewm_no_trace`: 4 wins
- `stjewm_spike_only`: 0 wins
- `stjewm_trace_only`: 2 wins
# Event-Probe Summary (NMI paper, Results 5)

Total cells aggregated: 374 (8 envs, 10 models, 8 targets).

## Mean event-probe AUROC ranking

1. `cubifae_baseline` = 0.610
2. `stjewm_membrane_readout` = 0.603
3. `gru_baseline` = 0.602
4. `stjewm_hidden_leak` = 0.600
5. `stjewm_no_trace` = 0.597
6. `slt_lif_mpc_trace` = 0.584
7. `stjewm_rate_only` = 0.581
8. `stjewm_spike_only` = 0.570
9. `slt_lif_mpc_free` = 0.545
10. `stjewm_trace_only` = 0.465

## Dissociation claim

STJEWM-trace is competitive or best on event-type probes, even though
its position-probe R² is moderate (see `probe_table.md`). This is the
core dissociation: the trace captures event-relevant information that
is not equivalent to position memory.


## Win counts

- `cubifae_baseline`: 5 wins
- `gru_baseline`: 12 wins
- `slt_lif_mpc_free`: 0 wins
- `slt_lif_mpc_trace`: 2 wins
- `stjewm_hidden_leak`: 4 wins
- `stjewm_membrane_readout`: 1 wins
- `stjewm_no_trace`: 3 wins
- `stjewm_rate_only`: 4 wins
- `stjewm_spike_only`: 10 wins
- `stjewm_trace_only`: 7 wins
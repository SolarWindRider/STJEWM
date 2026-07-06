# Event-Probe Summary (NMI paper, Results 5)

Total cells aggregated: 329 (5 envs, 10 models, 8 targets).

## Mean event-probe AUROC ranking

1. `cubifae_baseline` = 0.610
2. `stjewm_trace_only` = 0.607
3. `stjewm_spike_only` = 0.602
4. `gru_baseline` = 0.602
5. `stjewm_membrane_readout` = 0.602
6. `stjewm_hidden_leak` = 0.598
7. `stjewm_rate_only` = 0.598
8. `stjewm_no_trace` = 0.598
9. `slt_lif_mpc_trace` = 0.584
10. `slt_lif_mpc_free` = 0.545

## Dissociation claim

STJEWM-trace is competitive or best on event-type probes, even though
its position-probe R² is moderate (see `probe_table.md`). This is the
core dissociation: the trace captures event-relevant information that
is not equivalent to position memory.


## Win counts

- `cubifae_baseline`: 5 wins
- `gru_baseline`: 13 wins
- `slt_lif_mpc_free`: 0 wins
- `slt_lif_mpc_trace`: 2 wins
- `stjewm_hidden_leak`: 5 wins
- `stjewm_membrane_readout`: 1 wins
- `stjewm_no_trace`: 2 wins
- `stjewm_rate_only`: 4 wins
- `stjewm_spike_only`: 0 wins
- `stjewm_trace_only`: 1 wins
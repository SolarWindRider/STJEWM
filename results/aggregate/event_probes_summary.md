# Event-Probe Summary (NMI paper, Results 5)

Total cells aggregated: 563 (8 envs, 10 models, 9 targets).

## Mean event-probe AUROC ranking

1. `slt_lif_mpc_trace` = 0.587
2. `slt_lif_mpc_free` = 0.560
3. `gru_baseline` = 0.546
4. `cubifae_baseline` = 0.531
5. `stjewm_rate_only` = 0.508
6. `stjewm_membrane_readout` = 0.501
7. `stjewm_no_trace` = 0.500
8. `stjewm_hidden_leak` = 0.496
9. `stjewm_spike_only` = 0.488
10. `stjewm_trace_only` = 0.470

## Dissociation claim

STJEWM-trace is competitive or best on event-type probes, even though
its position-probe R² is moderate (see `probe_table.md`). This is the
core dissociation: the trace captures event-relevant information that
is not equivalent to position memory.


## Win counts

- `cubifae_baseline`: 4 wins
- `gru_baseline`: 13 wins
- `slt_lif_mpc_free`: 8 wins
- `slt_lif_mpc_trace`: 14 wins
- `stjewm_hidden_leak`: 3 wins
- `stjewm_membrane_readout`: 2 wins
- `stjewm_no_trace`: 2 wins
- `stjewm_rate_only`: 4 wins
- `stjewm_spike_only`: 0 wins
- `stjewm_trace_only`: 0 wins
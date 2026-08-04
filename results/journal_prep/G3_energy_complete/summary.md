# G3 Complete Energy / Sparsity Coverage

State-modality measurements on `cross_benchmark_F1`, seed 0. **G3 new** rows were measured in this run; **P11 old** rows are copied unchanged from `P11_energy/measurements.json`.

| Model | Provenance | Dense MFLOPs/step | Effective MFLOPs/step | Sparsity | Trainable params | Effective / dense | vs GRU dense | vs MLP dense | vs LeWM dense |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stjewm_rate_only | G3 new | 5.158 | 0.478 | 93.338% | 2,697,600 | 0.0928× | 0.0467× | 0.0479× | 0.0490× |
| stjewm_no_trace | G3 new | 5.158 | 0.465 | 93.598% | 2,697,600 | 0.0902× | 0.0454× | 0.0466× | 0.0476× |
| stjewm_hidden_leak | G3 new | 5.232 | 0.477 | 93.468% | 2,697,600 | 0.0911× | 0.0465× | 0.0477× | 0.0488× |
| stjewm_membrane_readout | G3 new | 5.158 | 0.481 | 93.294% | 2,697,600 | 0.0932× | 0.0469× | 0.0481× | 0.0492× |
| cubifae_baseline | G3 new | 9.963 | 9.686 | 100.000% | 4,984,986 | 0.9722× | 0.9458× | 0.9702× | 0.9914× |
| slt_lif_mpc_trace | G3 new | 10.182 | 2.125 | 99.118% | 5,110,560 | 0.2087× | 0.2075× | 0.2129× | 0.2175× |
| slt_lif_mpc_free | G3 new | 10.066 | 1.940 | 99.199% | 5,051,520 | 0.1927× | 0.1894× | 0.1943× | 0.1985× |
| spikedreamer_baseline | G3 new | 10.070 | 9.573 | 99.812% | 5,119,488 | 0.9507× | 0.9347× | 0.9588× | 0.9798× |
| stjewm_trace_only | P11 old | 5.232 | 0.483 | 93.338% | 2,697,600 | 0.0924× | 0.0472× | 0.0484× | 0.0495× |
| stjewm_spike_only | P11 old | 5.158 | 0.465 | 93.598% | 2,697,600 | 0.0902× | 0.0454× | 0.0466× | 0.0476× |
| gru_baseline | P11 old | 10.241 | 10.241 | 0.000% | 5,131,840 | 1.0000× | 1.0000× | 1.0258× | 1.0482× |
| mlp_baseline | P11 old | 9.984 | 9.984 | 0.000% | 5,000,704 | 1.0000× | 0.9749× | 1.0000× | 1.0219× |
| lewm_baseline_v2 | P11 old | 9.770 | 9.770 | 0.000% | 4,970,016 | 1.0000× | 0.9540× | 0.9786× | 1.0000× |

## Interpretation

- Sparsity is measured as `1 − nonzero/total` across every returned `spike_layers` tensor over 2 random batches × 2 samples, sequence length 3.
- Effective FLOPs discount only event-driven SNN linears by measured active fraction. Dense input/action projections stay dense. In hybrid CubiFAE and SpikeDreamer, time-cell/Transformer/fusion computation stays dense.
- Dense non-SNN models are **GRU, MLP, and LeWM-v2**; they expose no spike tensor, so sparsity is defined as **0%** and effective equals dense.
- SNN/hybrid models are all six STJEWM variants, CubiFAE, SLT-trace, SLT-free, and SpikeDreamer; all have measured sparsity **>0%**. CubiFAE produced zero spikes on these forwards (100% measured sparsity), but its dense time-cell convolution and fusion dominate effective FLOPs.
- The `vs GRU/MLP/LeWM dense` columns are the requested STJEWM-family effective-vs-dense comparator ratios and are also shown for other models for completeness.

## Counting scope

- Linear layers count `2 × in × out` FLOPs per step. Biases, normalization, nonlinearities, membrane/trace elementwise updates, softmax, and tensor additions are consistently excluded.
- Transformer attention interactions count QKᵀ and AV and are amortized per token at checkpoint history length. CubiFAE Conv1d counts multiply-adds for one output step.
- Shared frozen pixel ViT is irrelevant here because this is the required state-only coverage.

Machine-readable complete results: `measurements.json` (13 successful rows, with ledger breakdowns and per-layer sparsity).

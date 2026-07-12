# Training-data-budget scaling (was v0.7.7 'data-budget compression')

Per-env training-data-budget scaling (0.5x / 1.0x / 2.0x of BASE_PER_ENV=10K
windows per env) for the 12 G16 generalist models. The 1.0x cell re-uses the
existing G16 ckpt; 0.5x and 2.0x are freshly trained at the new budget.

NOTE on terminology: this is *training-data-budget scaling*, not model
compression / latent-dim reduction / dataset distillation. Only the
per-env `max_windows` changes; model class and capacity are unchanged.
| model | frac | env-SR (avg ± std) | div (avg) | resp (avg) | ρ (avg) |
|-------|------|--------------------|-----------|------------|---------|
| stjewm_trace_only | 0.5 | 0.944 ± 0.136 | 0.0120 | 0.204 | 0.993 |
| stjewm_trace_only | 1.0 | 0.833 ± 0.408 | 0.0117 | 0.206 | 0.993 |
| stjewm_trace_only | 2.0 | 0.944 ± 0.136 | 0.0107 | 0.199 | 0.994 |
| stjewm_spike_only | 0.5 | 0.944 ± 0.136 | 0.0115 | 0.200 | 0.995 |
| stjewm_spike_only | 1.0 | 0.889 ± 0.272 | 0.0110 | 0.210 | 0.995 |
| stjewm_spike_only | 2.0 | 0.833 ± 0.408 | 0.0110 | 0.202 | 0.987 |
| mlp_baseline | 0.5 | 0.833 ± 0.408 | 0.0002 | 0.636 | -0.060 |
| mlp_baseline | 1.0 | 0.833 ± 0.408 | 0.0002 | 0.548 | -0.070 |
| mlp_baseline | 2.0 | 0.833 ± 0.408 | 0.0002 | 0.474 | -0.085 |

Cells with frac=1.0 reuse existing G16 outputs. Cells with frac≠1.0
train fresh ckpts with per-env max_windows = round(BASE_PER_ENV × frac).

## Robustness narrative

Lower absolute drift across the frac axis = more robust.


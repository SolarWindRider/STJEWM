# P1-1 Energy / Efficiency Measurement

This report is generated from `measurements.json`; all numeric entries below are rendered from that file after the checkpoint forwards completed.

## Scope and reproducibility

- Checkpoint split: `cross_benchmark_F1`; seed directory: `seed_0`.
- Random forward protocol: `2` batches × `2` samples, sequence length from each checkpoint's `history_size`, seed `20260802`.
- Device: `cpu`; Python/PyTorch execution used the repository environment.
- State inputs are `(B,T,128)` and actions are `(B,T,56)`. Pixel inputs are `(B,T,3,84,84)` and actions are `(B,T,56)`.
- A Linear with shape `(din,dout)` contributes `2×din×dout` FLOPs per token. Biases, activations, LayerNorm, membrane/trace elementwise updates, softmax, and tensor adds are excluded consistently.
- Counted always-dense path: state/pixel projection and action encoder. Counted dynamic path: STJEWM MultiCompartment cell linears, post-cell MLPs, gated-trace gate, and mode-specific readout; GRU gates/output; MLP FFN; or LeWM AdaLN/QKV/output/MLP/attention interactions/output projection.
- Pixel-mode frozen ViT backbone is excluded from every row because it is shared across the comparison. Its trainable projection is included. Thus pixel numbers are predictor-side FLOPs, not end-to-end camera encoding FLOPs.
- STJEWM sparsity is measured as `1 − nonzero/total` over every layer's returned binary soma `spike_layers` tensor on the random forwards. The prescribed effective estimate is `always_dense + active_fraction × dynamic_SNN/readout`; dense baselines have no spike tensor and receive sparsity `0` / active fraction `1`.

## Per-model FLOP table

`Params` is trainable parameters; `total` additionally includes frozen parameters (notably the excluded pixel ViT). FLOPs are per input token/step, with transformer attention amortized over the reported sequence length.

| Modality | Model | Status | T | Trainable params | Total params | Dense input/action MFLOPs | Dense dynamic MFLOPs | Dense total MFLOPs/step | Sparsity | Effective dynamic MFLOPs | Effective MFLOPs/step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| state | stjewm_rate_only | ok | 3 | 2,697,600 | 8,198,976 | 0.144 | 5.014 | 5.158 | 93.338% | 0.334 | 0.478 |
| state | stjewm_no_trace | ok | 3 | 2,697,600 | 8,198,976 | 0.144 | 5.014 | 5.158 | 93.598% | 0.321 | 0.465 |
| state | stjewm_hidden_leak | ok | 3 | 2,697,600 | 8,198,976 | 0.144 | 5.087 | 5.232 | 93.468% | 0.332 | 0.477 |
| state | stjewm_membrane_readout | ok | 3 | 2,697,600 | 8,198,976 | 0.144 | 5.014 | 5.158 | 93.294% | 0.336 | 0.481 |
| state | alif_timecell_baseline | ok | 3 | 4,984,986 | 4,984,986 | 0.207 | 9.756 | 9.963 | 100.000% | 9.479 | 9.686 |
| state | stacked_lif_trace | ok | 3 | 5,110,560 | 5,110,560 | 2.054 | 8.129 | 10.182 | 99.118% | 0.072 | 2.125 |
| state | stacked_lif_free | ok | 3 | 5,051,520 | 5,051,520 | 1.874 | 8.192 | 10.066 | 99.199% | 0.066 | 1.940 |
| state | lif_transformer_baseline | ok | 3 | 5,119,488 | 5,119,488 | 0.106 | 9.964 | 10.070 | 99.812% | 9.467 | 9.573 |

## Explicit STJEWM effective-vs-dense ratios

Each ratio is `STJEWM effective FLOPs/step ÷ comparator dense FLOPs/step` within the same modality. Values below 1.0 indicate a lower estimated predictor-side cost under the event discount.

| Modality | STJEWM variant | vs GRU dense | vs MLP dense | vs LeWM-v2 dense |
|---|---|---:|---:|---:|
| state | stjewm_trace_only | — | — | — |
| state | stjewm_spike_only | — | — | — |
| pixel | stjewm_trace_only | — | — | — |
| pixel | stjewm_spike_only | — | — | — |

## Measured spike activity

| Modality | Model | Spike elements | Nonzero spikes | Active fraction | Sparsity source | Per-layer sparsity |
|---|---|---:|---:|---:|---|---|
| state | stjewm_rate_only | 4,608 | 307 | 6.662% | measured from all STJEWM soma spike_layers on random forwards | L0=98.220%, L1=88.455% |
| state | stjewm_no_trace | 4,608 | 295 | 6.402% | measured from all STJEWM soma spike_layers on random forwards | L0=98.438%, L1=88.759% |
| state | stjewm_hidden_leak | 4,608 | 301 | 6.532% | measured from all STJEWM soma spike_layers on random forwards | L0=98.958%, L1=87.977% |
| state | stjewm_membrane_readout | 4,608 | 309 | 6.706% | measured from all STJEWM soma spike_layers on random forwards | L0=98.220%, L1=88.368% |
| state | alif_timecell_baseline | 4,464 | 0 | 0.000% | measured from all STJEWM soma spike_layers on random forwards | L0=100.000%, L1=100.000% |
| state | stacked_lif_trace | 64,512 | 569 | 0.882% | measured from all STJEWM soma spike_layers on random forwards | L0=99.020%, L1=98.884%, L2=99.182%, L3=99.132%, L4=99.219%, L5=99.219%, L6=99.107%, L7=99.182% |
| state | stacked_lif_free | 61,440 | 492 | 0.801% | measured from all STJEWM soma spike_layers on random forwards | L0=99.102%, L1=99.102%, L2=99.362%, L3=99.141%, L4=99.180%, L5=99.232%, L6=99.284%, L7=99.193% |
| state | lif_transformer_baseline | 6,912 | 13 | 0.188% | measured from all STJEWM soma spike_layers on random forwards | L0=99.624%, L1=100.000% |

## Missing or failed inputs

- None; all requested state and pixel checkpoints were present and loaded.

## Component interpretation

The comparison is deliberately about the learned world-model predictor after observation projection. The shared frozen pixel ViT is reported in `total_params` for transparency but excluded from FLOPs; including the identical ViT once on both sides would add a common constant and not change the relative predictor ranking. The SNN discount is an analytical event-driven estimate, not a hardware benchmark: it discounts the counted STJEWM stack/readout matmuls by measured active soma fraction while leaving input/action projections dense.

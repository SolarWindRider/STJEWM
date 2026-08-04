# Full Metric Matrix — Record & Gap List (v0.7.17)

> Canonical main table: `results/journal_prep/FULL_METRIC_MATRIX.md` (pushed `e14c72b`).
> This file records the table and the exact gaps to fill with new experiments.

## The canonical table (13 models × 16 metrics)

| Model | cos↓ | LeWM@.05 | envSR | event-ρ | AUROC-c | AUROC-k5 | AUROC-mot | effFLOPs | dense | spar% | train. | 3-seed cos | pos R² | fut R² | goal R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STJEWM-trace | 0.105 | 0.373 | 0 | 0.9987 | 0.517 | 0.540 | 0.477 | 0.483 | 5.23 | 93.3 | 2.70 | 0.119±0.002 | -0.017 | -0.024 | -0.086 |
| STJEWM-spike | 0.108 | 0.373 | 0 | 0.9988 | 0.517 | 0.542 | 0.491 | 0.465 | 5.16 | 93.6 | 2.70 | 0.114±0.010 | -0.066 | -0.035 | -0.053 |
| STJEWM-rate | 0.103 | 0.416 | 0 | 0.9988 | 0.525 | 0.532 | 0.496 | — | — | — | — | — | — | — | — |
| STJEWM-no-trace | 0.119 | 0.364 | 0 | — | 0.516 | 0.526 | 0.485 | — | — | — | — | — | — | — | — |
| STJEWM-leak | 0.119 | 0.398 | 0 | — | 0.524 | 0.565 | 0.477 | — | — | — | — | — | — | — | — |
| STJEWM-membrane | 0.124 | 0.375 | 0 | 0.9987 | 0.526 | 0.504 | 0.490 | — | — | — | — | — | — | — | — |
| CuBiFAE | 0.105 | 0.422 | 0 | — | 0.507 | 0.518 | 0.497 | — | — | — | — | — | — | — | — |
| SLT-trace | 0.091 | 0.422 | 0 | 0.9996 | — | — | — | — | — | — | — | 0.115±0.004 | — | — | — |
| SLT-free | 0.105 | 0.386 | 0 | — | — | — | — | — | — | — | — | — | — | — | — |
| LeWM-v2 | 0.183 | 0.225 | 0 | 0.7515 | — | — | — | 9.770 | 9.77 | 0 | 4.97 | 0.194±0.011 | 0.605 | 0.396 | 0.168 |
| GRU | 0.020 | 0.894 | 0 | -0.111 | 0.595 | 0.560 | 0.545 | 10.241 | 10.24 | 0 | 5.13 | — | — | — | — |
| MLP | 0.007 | 0.948 | 0 | -0.022 | — | — | — | 9.984 | 9.98 | 0 | 5.00 | 0.004±0.000 | -0.043 | -0.023 | -0.042 |
| SpikeDreamer | 0.000 | 1.000 | 0 | — | — | — | — | — | — | — | — | — | — | — | — |

## GAPS to fill (experiment → missing models)

### G1. event-ρ (B1Fix covered 8 models; missing 5 + GRU/MLP env coverage)
- Missing models: **stjewm_no_trace, stjewm_hidden_leak, cubifae_baseline, slt_lif_mpc_free, spikedreamer_baseline**
- GRU/MLP only have cheetah (2 cells); extend to all 4 envs (cheetah, ball_in_cup, pendulum_2d, finger) × 2 splits
- Total: 5 models × 4 envs × 2 splits = 40 cells + 4 GRU/MLP cells = 44 cells

### G2. event-AUROC (5m probes covered 8 models; missing 5)
- Missing: **slt_lif_mpc_trace, slt_lif_mpc_free, lewm_baseline_v2, mlp_baseline, spikedreamer_baseline**
- Use `probe_all.sh` pattern (probe.py --probe-target, 6 targets × envs present in cross_benchmark_F1 split)
- Note: lewm/mlp AUROC exists at 3/5-epoch (P13) but not at 1-epoch — fill the 1-epoch cells

### G3. effFLOPs/dense/sparsity (P11 covered 5 models; missing 8)
- Missing: **stjewm_rate_only, stjewm_no_trace, stjewm_hidden_leak, stjewm_membrane_readout, cubifae_baseline, slt_lif_mpc_trace, slt_lif_mpc_free, spikedreamer_baseline**
- Reuse `measure_energy.py` (state modality)

### G4. probe R² (B3 covered 4 models; missing 9)
- Missing: **stjewm_rate_only, stjewm_no_trace, stjewm_hidden_leak, stjewm_membrane_readout, cubifae_baseline, slt_lif_mpc_trace, slt_lif_mpc_free, gru_baseline, spikedreamer_baseline**
- cross_benchmark_F1 split, targets: position, velocity, future_k, goal_direction, contact

### G5. 3-seed cos (B2 covered 5 models; missing 8)
- Missing: **stjewm_rate_only, stjewm_no_trace, stjewm_hidden_leak, stjewm_membrane_readout, cubifae_baseline, slt_lif_mpc_free, gru_baseline, spikedreamer_baseline**
- HEAVIEST: train seeds 1-2 (state, 5M protocol), 3 splits (cross_benchmark_F1, oodc_F2, generalist_16env) = 8 × 3 × 2 = 48 ckpts + eval

## After all gaps filled: regenerate FULL_METRIC_MATRIX.md with zero '—' in the covered columns.

## Experiment → script reference
| Experiment | Script | Ckpt source |
|---|---|---|
| G1 event-ρ | `code/scripts/event_align.py --env --model --ckpt --out --pad-obs-to 128 --action-dim-eval 56` | `results/5m/<split>/<model>/seed_0/final.pt` |
| G2 AUROC | `python -m code.scripts.probe --env --model --ckpt --probe-target --pad-obs-to 128 --action-dim-eval 56 --max-windows 200` | same |
| G3 FLOPs | `python code/scripts/generalist_v0_7_5_5m/measure_energy.py` (read its CLI first) | same |
| G4 probe R² | `code/scripts/probe.py` (B3-fixed) | same |
| G5 3-seed | `B2_multiseed_launcher.sh` pattern (train seed 1/2) | train new → `results/5m_seed{1,2}/` |

## Model → build name map (for training/loading)
- STJEWM readouts: `--model stjewm --readout-mode {trace_only,spike_only,rate_only,no_trace,hidden_leak,membrane_readout}`
- Baselines: `--model {cubifae_baseline, s lt_lif_mpc_trace, slt_lif_mpc_free, lewm_baseline, gru_baseline, mlp_baseline, spikedreamer_baseline}`
- N_LAYERS: stjewm=4, cubifae=2, slt=8, lewm=3, gru=2, mlp=12, spikedreamer=3

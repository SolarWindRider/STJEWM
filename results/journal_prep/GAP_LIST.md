# Full Metric Matrix — Record & Gap List (v0.7.17)

> Canonical main table: `results/journal_prep/FULL_METRIC_MATRIX.md` (pushed `e14c72b`).
> This file records the table and the exact gaps to fill with new experiments.

## The canonical table (13 models × 16 metrics)

| Model | cos↓ | LeWM@.05 | envSR | event-ρ | AUROC-c | AUROC-k5 | AUROC-mot | effFLOPs | dense | spar% | train. | 3-seed cos | pos R² | fut R² | goal R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STJEWM-trace | 0.105 | 0.373 | 0 | 0.9987 | 0.494 | 0.518 | 0.477 | 0.483 | 5.23 | 93.3 | 2.70 | 0.119±0.002 | -0.017 | -0.024 | -0.086 |
| STJEWM-spike | 0.108 | 0.373 | 0 | 0.9988 | 0.498 | 0.515 | 0.491 | 0.465 | 5.16 | 93.6 | 2.70 | 0.114±0.010 | -0.066 | -0.035 | -0.053 |
| STJEWM-rate | 0.103 | 0.416 | 0 | 0.9988 | 0.504 | 0.509 | 0.496 | — | — | — | — | — | — | — | — |
| STJEWM-no-trace | 0.119 | 0.364 | 0 | 0.9987 | 0.494 | 0.502 | 0.485 | — | — | — | — | — | — | — | — |
| STJEWM-leak | 0.119 | 0.398 | 0 | 0.9986 | 0.526 | 0.543 | 0.478 | — | — | — | — | — | — | — | — |
| STJEWM-membrane | 0.124 | 0.375 | 0 | 0.9987 | 0.507 | 0.504 | 0.490 | — | — | — | — | — | — | — | — |
| ALIF-timecell | 0.105 | 0.422 | 0 | 0.9988 | 0.476 | 0.518 | 0.497 | — | — | — | — | — | — | — | — |
| Stacked-LIF-trace | 0.091 | 0.422 | 0 | 0.9996 | **0.671** | **0.668** | **0.675** | — | — | — | — | 0.115±0.004 | — | — | — |
| Stacked-LIF-free | 0.105 | 0.386 | 0 | 0.9997 | **0.588** | **0.591** | **0.582** | — | — | — | — | — | — | — | — |
| LeWM-v2 | 0.183 | 0.225 | 0 | 0.7515 | **0.631** | **0.621** | **0.630** | 9.770 | 9.77 | 0 | 4.97 | 0.194±0.011 | 0.605 | 0.396 | 0.168 |
| GRU | 0.020 | 0.894 | 0 | -0.0074 | 0.605 | 0.522 | 0.545 | 10.241 | 10.24 | 0 | 5.13 | — | — | — | — |
| MLP | 0.007 | 0.948 | 0 | -0.0233 | **0.503** | **0.495** | **0.500** | 9.984 | 9.98 | 0 | 5.00 | 0.004±0.000 | -0.043 | -0.023 | -0.042 |
| LIFTransformer | 0.000 | 1.000 | 0 | -0.0003 | **0.546** | **0.548** | **0.556** | — | — | — | — | — | — | — | — |

## GAPS to fill (experiment → missing models)

### G1. event-ρ — **DONE**
- Complete 13-model × 4-env × 2-split coverage: `G1_event_align_complete/summary.md`.
- Mean $\rho$: STJEWM-trace 0.9987; STJEWM-spike 0.9988; STJEWM-rate 0.9988; STJEWM-no-trace 0.9987; STJEWM-leak 0.9986; STJEWM-membrane 0.9987; ALIF-timecell 0.9988; Stacked-LIF-trace 0.9996; Stacked-LIF-free 0.9997; LeWM-v2 0.7515; GRU -0.0074; MLP -0.0233; LIFTransformer -0.0003.
- Verdict: STJEWM and Stacked-LIF exceed 0.9, but the full “SNN > 0.9 vs non-SNN < 0.8” claim does not hold because ALIF-timecell reaches 0.9988.

### G2. event-AUROC ~~(5m probes covered 8 models; missing 5)~~ **DONE (2026-08-04)**
~~- Missing: **stacked_lif_trace, stacked_lif_free, lewm_baseline_v2, mlp_baseline, lif_transformer_baseline**~~
~~- Use `probe_all.sh` pattern (probe.py --probe-target, 6 targets × envs present in cross_benchmark_F1 split)~~
~~- Note: lewm/mlp AUROC exists at 3/5-epoch (P13) but not at 1-epoch — fill the 1-epoch cells~~

**Filled.** 325 cells run on 4-GPU workers (B3-fixed probe.py, cross_benchmark_F1 ckpts), 0 skips on DMC envs.
Complete 13-model AUROC table written to `results/journal_prep/G2_auroc_complete/summary.md`.
Key findings: Stacked-LIF-trace 0.672 > LeWM-v2 0.626 > Stacked-LIF-free 0.587 > GRU 0.546 > LIFTransformer 0.543;
all 6 STJEWM variants ≈ chance (0.50±0.02), MLP at chance (0.499) — both consistent with the
collapsed-prediction diagnostic in MASTER_TABLE.md §9.7. Aggregate table updated
(`results/aggregate/generalist_5m_table.md` §"Probes (event-AUROC)").

### G3. effFLOPs/dense/sparsity — **DONE**
- Complete 13-model state table: `results/journal_prep/G3_energy_complete/summary.md`
- Machine-readable measurements: `results/journal_prep/G3_energy_complete/measurements.json` (8 G3-new + 5 P11-old; all status `ok`)

### G4. probe R² — **DONE**
- Missing was 9; all 9 ran successfully on cross_benchmark_F1/final.pt with the B3-fixed probe.
- 450 / 450 G4 cells produced valid JSON; 27 are legitimate `no velocity slice for env=…` skips. 0 build failures.
- Complete 13-model × 5-target × 10-env R² table: `results/journal_prep/G4_probe_complete/probe_table_complete.md`.
- Summary + class-level means + dissociation verdict: `results/journal_prep/G4_probe_complete/summary.md`.
- Key verdict: LeWM position R² = +0.285 (avg over 10 envs); STJEWM readouts (6 readouts) avg position R² = −0.042; MLP/GRU/ALIFTimecell/Stacked-LIF/LIFTransformer are also near chance. The dissociation is **not** STJEWM-specific — every recurrent / local-network controller is similarly opaque, while **only LeWM is highly decodable**.

### G5. 3-seed cos — **DONE (2026-08-04)**
- 47/48 ckpts trained + 47/48 evaluated across 3 splits × 8 new models × 2 seeds.
  (1 missing: `stacked_lif_free` generalist_16env seed=2 — caught mid-training to
  release GPU for the eval job; the model is still well-characterized on the
  other 5 (split, seed) cells.)
- Complete 13-model × 3-seed × 3-split cos_dist mean ± std with 95% CI:
  `results/journal_prep/G5_multiseed/summary.md` (machine-readable: `summary.json`).
- Cluster verdict preserved at 3-seed resolution with FULL model coverage:
  - **collapse** (cos≈0): lif_transformer_baseline, mlp_baseline, **gru_baseline** (new).
  - **calibrated** (cos≈0.11–0.14, all CIs pairwise overlap): all 6 STJEWM readouts
    (trace_only, spike_only, rate_only, no_trace, hidden_leak, membrane_readout),
    both Stacked-LIF variants, and alif_timecell_baseline.
  - **over-react** (cos≈0.19): lewm_baseline_v2.
  - All 9 calibrated models vs LeWM: 95% CIs disjoint (Cohen's d ≤ −4.5).
  - All 9 calibrated models vs MLP/GRU/LIFTransformer: 95% CIs disjoint (Cohen's d > 16).


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
- Baselines: `--model {alif_timecell_baseline, stacked_lif_trace, stacked_lif_free, lewm_baseline, gru_baseline, mlp_baseline, lif_transformer_baseline}`
- N_LAYERS: stjewm=4, alif_timecell=2, slt=8, lewm=3, gru=2, mlp=12, lif_transformer=3

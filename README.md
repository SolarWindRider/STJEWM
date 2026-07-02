# ST-JEWM: Spike-Trace Joint-Embedding World Model

> **Can the event history of a spiking dynamical system itself become a
> world-model predictive state, when the downstream predictor and planner
> are forbidden from reading the continuous membrane potential?**

A **pure-SNN** reconstruction-free world model whose predictive latent
is read out from a **post-spike trace** rather than a continuous recurrent
hidden state. The trace is bounded in [0,1] per dim, content-aware
(forget gate `alpha = sigma(W[r_{t-1}, s_t, c_t])`), and event-driven.

This repository contains the code, evaluations, and paper for ST-JEWM.
The full PDF is at `paper/paper.pdf`. Source: `paper/paper.md` and
`paper/paper.tex`. Current version: v0.5 (2026-07-02).

## Headline results (v0.5, post-NMI-refactor)

We ran 3 new experiments after the v0.4 paper, expecting to strengthen
the trace-vs-membrane story. The results are *more nuanced* than
expected: two of the three are **negative** and one is **positive**.

1. **Stress-difficulty sweep (3 levels per env, 50 cells)**: the
   v0.4 claim "membrane_readout collapses to 0% AVG on stress" was
   driven by pusht_ood + tworoom_long hitting 0% env-native at every
   difficulty for *every* model. On the LeWM-SR metric, the membrane
   variant is within 0-10pp of trace at every difficulty. **The
   protocol is not motivated by a measurable stress disadvantage of
   membrane readout**; it is motivated by the abstraction and the
   event-alignment correlation.
2. **Event-window causal ablation (6 cells)**: zeroing the trace at
   event-aligned env steps does *not* reduce env-SR more than zeroing
   at matched non-event or random steps. **The strong causal claim
   "the trace specifically carries event info the planner uses" is not
   supported.** The weak event-correlation claim (Sec 4.3.1) is.
3. **Event-type linear probes (144 cells, 7 envs x 8 models x 3
   targets)**: the cleanest new positive result. **STJEWM readouts
   and a 7.3M continuous-RNN GRU both reach mean AUROC ~0.68 on
   event-type targets; the LeWM Transformer reaches 0.58 and the
   stateless MLP 0.61.** The event-alignment property is shared by
   SN training and continuous-RNN training, not trace-specific. The
   Transformer and stateless MLP are the two that lack it.

The **revised claim**: the membrane-forbidden protocol is a
*constructive* constraint that forces the planner to read an
event-correlated state without requiring the planner to *use* the
event-window component specifically. The post-spike trace is a
competitive, bounded, event-correlated predictive state under that
protocol. The relevant design choice for event-aligned predictive
state is the **recurrent dynamics** (SN or continuous RNN), not
the trace specifically.

| Model | env-SR (16 env) | event-probe AUROC (144 cells) |
|---|---|---|
| STJEWM-rate        | 85.7% | n/a (rate-only ablation) |
| LeWM Transformer   | 85.4% (5-ep) | 0.582 |
| GRU 7.3M (continuous RNN) | 83.7% | 0.670 |
| STJEWM-trace       | 83.9% | **0.690** |
| STJEWM-spike       | 82.3% | 0.654 |
| STJEWM-no-trace    | 81.7% | 0.644 |
| STJEWM-leak        | 79.7% | 0.690 |
| MLP 1.3M (stateless) | 80.9% | 0.612 |
| STJEWM-membrane   | 80.4% | 0.647 |

The standard suite is **saturated**; the event-probe suite is the
new benchmark.

## Repository layout

```
.
├── README.md                        # this file
├── LICENSE                          # MIT (code)
├── CITATION.cff                     # citation metadata
├── CONTRIBUTING.md                  # contributor guide
├── code/
│   ├── stjewm.py                   # the model (ReadoutMode enum, 6 branches)
│   ├── lewm_transformer_baseline.py # 5.07M Transformer baseline
│   ├── gru_baseline.py             # 7.3M continuous-RNN baseline
│   ├── mlp_baseline.py              # 1.3M no-history baseline
│   ├── sigreg.py                    # spike-train regulariser
│   ├── snn_cell.py                  # MultiCompartment SNN cell
│   ├── theory/                      # theoretical writeups
│   ├── core/
│   │   ├── cem.py                  # CEM planner (LeWM App. B + F.1)
│   │   ├── encode.py                # encode_obs / encode_history
│   │   ├── envs/                   # env class registry (22 envs)
│   │   └── data/loaders.py          # dataset loaders (h5 / npz)
│   ├── train/train.py               # single trainer (all 9 model dirs)
│   ├── eval/closed_loop.py          # CEM planner + env-native SR
│   ├── scripts/
│   │   ├── probe.py                # linear probe on frozen encoder (event-type + position)
│   │   ├── event_window_ablation.py  # causal ablation (Sec 4.5.1)
│   │   ├── aggregate_event_probes.py # event-probe aggregation
│   │   ├── aggregate_stress_sweep.py # stress-sweep aggregation
│   │   ├── aggregate_event_window_ablation.py
│   │   ├── run_event_probes.sh      # full 144-cell event-probe sweep
│   │   ├── run_stress_sweep.sh      # 50-cell stress-sweep
│   │   ├── run_event_window_ablation.sh
│   │   ├── eval_stress_baselines.sh, eval_stress_suite.sh, ...
│   │   └── ~30 more analysis scripts
├── data/
│   └── delayed_t_maze_30k.npz       # working-memory dataset
├── results/                         # per-env ckpts + aggregate tables
│   └── aggregate/                   # final summary tables (HEADLINE FILES)
│       ├── summary_5way.md           # 5-condition LeWM-SR (16 envs)
│       ├── env_sr_table.md           # env-SR (the honest metric)
│       ├── stress_full_table.md      # 4-task stress
│       ├── stress_sweep_table.md     # 3-level difficulty sweep (Sec 4.5.2)
│       ├── event_window_ablation_table.md  # causal ablation (Sec 4.5.1)
│       ├── event_probes_table.md     # 144-cell event-probe (Sec 4.6)
│       ├── event_probes_summary.md   # event-probe win summary
│       ├── probe_table.md            # legacy position probe
│       ├── event_align_table.md      # Pearson corr with event boundaries
│       ├── flops_table.md            # efficiency
│       ├── lewm_sr_vs_env_sr.md      # MLP latent-collapse analysis
│       └── dt_modes/, eval_*/        # per-cell JSONs (gitignored)
├── paper/
│   ├── paper.pdf                    # v0.5 compiled PDF
│   ├── paper.md                     # v0.5 markdown source
│   ├── paper.tex                    # v0.5 LaTeX source
│   ├── v0_references.md             # bibliography
│   └── figs/                        # 8 figures (architecture, 5-way, stress, ...)
├── docs/
│   ├── ARCHITECTURE.md              # model architecture writeup
│   ├── HONEST_RESULTS.md            # v0.4 reframe
│   ├── LEWM_SR_ARTIFACT.md         # MLP latent-collapse analysis
│   ├── SATURATION_ANALYSIS.md       # why standard suite is saturated
│   ├── GOAL_LOSS_FIX.md             # with-goal vs no-goal fix
│   ├── TWOROOM_BUGFIX.md            # env reset bug fix
│   ├── GIF_PAIRS.md                 # GIF comparison protocol
│   ├── BENCHMARKS_REPORT.md         # 1-week sprint benchmarks
│   ├── FINAL_RESULTS_REPORT.md       # final 1-week results
│   ├── FRESH_RUN_REPORT.md          # fresh-run validation
│   └── report/refs/                 # upstream references (LeWM paper PDF)
└── logs/                            # training/eval logs (gitignored)
```

## Reproducing

```bash
# Probe sweep (144 cells, ~1.5h on CPU; AUROC per env x target x model)
bash code/scripts/run_event_probes.sh
python -m code.scripts.aggregate_event_probes  # writes event_probes_table.md

# Stress-difficulty sweep (50 cells, ~1h on CPU; per-difficulty env-SR + LeWM-SR)
bash code/scripts/run_stress_sweep.sh
python -m code.scripts.aggregate_stress_sweep

# Event-window causal ablation (6 cells, ~10min)
bash code/scripts/run_event_window_ablation.sh
python -m code.scripts.aggregate_event_window_ablation

# Standard 5-way + 9-way + stress (already done; rerun only if needed)
bash code/scripts/eval_v1_readout.sh trace_only
bash code/scripts/eval_v2_5way.sh
bash code/scripts/eval_stress_suite.sh
python -m code.scripts.make_5way_metrics
```

## Status (v0.5, 2026-07-02)

| Component | Status | Output |
|---|---|---|
| 9-condition standard suite (16 envs, env-SR) | done | `results/aggregate/env_sr_table.md` |
| 4-task unsaturated stress suite (env-SR) | done | `results/aggregate/stress_full_table.md` |
| **Stress-difficulty sweep (3 levels, 50 cells)** | done | `results/aggregate/stress_sweep_table.md` |
| **Event-window causal ablation (6 cells)** | done | `results/aggregate/event_window_ablation_table.md` |
| **Event-type linear probes (144 cells, 7 envs x 8 models x 3 targets)** | done | `results/aggregate/event_probes_table.md` |
| Event-boundary alignment (6 DMC, d=3.36) | done | `results/aggregate/event_align_table.md` |
| Position / velocity / goal probe (legacy) | done | `results/aggregate/probe_table.md` |
| FLOPs / efficiency (4 models) | done | `results/aggregate/flops_table.md` |
| Trace necessity (64 ablation evals) | done | `results/trace_necessity/SUMMARY.md` |
| LeWM-SR artifact analysis | done | `results/aggregate/lewm_sr_vs_env_sr.md` |
| Paper v0.5 PDF | done | `paper/paper.pdf` |

## Pre-push checklist (GitHub)

- [x] Add `LICENSE` (MIT for code) — done
- [x] Add `CONTRIBUTING.md` and `CITATION.cff` — done
- [x] Make `paper/paper.md` the canonical source — done
- [ ] GitHub Actions: `tectonic` rebuild PDF on push
- [ ] GitHub Actions: `make_5way_metrics.py` regression check
- [x] Push to GitHub via PAT (SSH key not configured) — done
- [x] Tag v0.5 release — done

## License

Code: MIT. Paper text + figures: CC-BY-4.0. Data (LeWM suite, PushT, etc.):
inherits from upstream LeWM / dmc_control / OGBench licenses.

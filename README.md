# ST-JEWM: Spike-Trace Joint-Embedding World Model

> **Can the event history of a spiking dynamical system itself become a
> world-model predictive state, when the downstream predictor and planner
> are forbidden from reading the continuous membrane potential?**

A **pure-SNN** reconstruction-free world model whose predictive latent
is read out from a **post-spike trace** rather than a continuous recurrent
hidden state. The trace is bounded in [0,1] per dim, content-aware
(forget gate `alpha = sigma(W[r_{t-1}, s_t, c_t])`), and event-driven.

This repository contains the code, evaluations, and paper for ST-JEWM.
The full PDF is at `paper/paper.pdf` (v0.4, 2026-07-02). Source:
`paper/paper.md` and `paper/paper.tex`.

## Headline result (v0.4, honest)

The membrane-forbidden protocol is **necessary**, not arbitrary: the
`membrane_readout` ablation collapses to **0% env-SR** on the 4-task
unsaturated stress suite, while all other STJEWM readouts stay at
~40%. See `docs/HONEST_RESULTS.md` for the full re-analysis and
`docs/LEWM_SR_ARTIFACT.md` for why the LeWM-SR metric is gaming-able.

| Model | env-SR (16 env) | env-SR (stress) |
|---|---|---|
| STJEWM-rate        | 85.7% | 40.4% |
| LeWM Transformer   | 85.4% (5-ep) | n/a |
| **STJEWM-trace**   | **83.9%** | **40.4%** |
| GRU (7.3M)         | 83.7% | 42.0% |
| STJEWM-spike       | 82.3% | 40.0% |
| STJEWM-no-trace    | 81.7% | 40.0% |
| MLP (1.3M)         | 80.9% | 32.5% |
| STJEWM-membrane   | 80.4% | **0.0%** |
| STJEWM-leak        | 79.7% | 40.8% |

The standard suite is **saturated** (all models within 6pp). The stress
suite is where the membrane-forbidden protocol shows its value.

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
│   │   ├── envs/                   # env class registry
│   │   │   ├── base.py              # BaseEnv, EnvSpec
│   │   │   ├── dmc_env.py           # DMCStateEnv (mujoco)
│   │   │   ├── gym_envs.py           # CartPole, Pendulum
│   │   │   ├── reacher_env.py
│   │   │   ├── swm_envs.py          # PushT, TwoRoom
│   │   │   └── delayed_t_maze.py    # working-memory probe
│   │   └── viz/                     # trajectory GIFs
│   ├── data/loaders.py              # dataset loaders (h5 / npz)
│   ├── train/train.py               # single trainer (all 9 models)
│   ├── eval/closed_loop.py          # CEM planner + env-native SR
│   └── scripts/                     # 50+ pipeline scripts
├── data/
│   └── delayed_t_maze_30k.npz       # working-memory dataset (synthetic)
├── results/                         # per-env ckpts + aggregate tables
│   ├── ball_in_cup/...              # 16 standard env dirs (gitignored)
│   ├── cartpole_flicker/...         # 4 stress env dirs (gitignored)
│   └── aggregate/                   # final summary tables
│       ├── summary_5way.md          # 5-condition LeWM-SR
│       ├── env_sr_table.md           # env-SR (the honest metric)
│       ├── stress_full_table.md      # 4-task stress
│       ├── probe_table.md           # linear probe R²
│       ├── event_align_table.md      # event boundary alignment
│       ├── flops_table.md           # efficiency
│       ├── dt_summary.md            # delayed T-Maze
│       ├── lewm_sr_vs_env_sr.md      # the MLP artifact analysis
│       ├── summary_4way.md          # 4-condition legacy
│       ├── dt_modes/                # 5 DT eval JSONs
│       ├── eval_v1_readout/         # 71 STJEWM-v1 evals
│       ├── eval_v2_5way/            # 64 9-way evals
│       ├── stress_logs/              # STJEWM stress evals
│       ├── stress_baselines/         # 8 GRU/MLP stress evals
│       ├── lewm_no_trace_eval/       # 4 legacy lewm evals
│       ├── eval_logs/                # 17 .log files (gitignored)
│       ├── gifs/                    # 43 MB trajectory gifs (gitignored)
│       └── gif_inventory.json       # GIF manifest
├── paper/
│   ├── paper.pdf                    # v0.4 compiled PDF (510KB)
│   ├── paper.md                     # v0.4 markdown source
│   ├── paper.tex                    # v0.4 LaTeX source
│   ├── v0_references.md             # bibliography
│   └── figs/                        # 8 figures (architecture, 5-way, stress, ...)
├── docs/
│   ├── ARCHITECTURE.md              # model architecture writeup
│   ├── HONEST_RESULTS.md            # v0.4 reframing
│   ├── LEWM_SR_ARTIFACT.md         # the MLP latent-collapse analysis
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

The repo is designed to be reproducible end-to-end. Each model ckpt is
trained with the same trainer (`code/train/train.py`) on the same
hyper-parameters. The eval pipeline is `code/scripts/eval_v1_readout.sh`
(standard 5-way) + `code/scripts/eval_v2_5way.sh` (9-way) +
`code/scripts/eval_stress_baselines.sh` (4-task stress).

```bash
# Train (all 9 model ckpts over 16 envs)
bash code/scripts/retrain_with_readout_modes.sh   # 5 STJEWM modes
bash code/scripts/baseline_train_seq.sh          # GRU + MLP

# Eval
bash code/scripts/eval_v1_readout.sh trace_only   # one mode, all envs
bash code/scripts/eval_v2_5way.sh                 # all models, all envs
bash code/scripts/eval_stress_baselines.sh        # stress suite

# Aggregate
python -m code.scripts.make_5way_metrics         # builds summary_5way.md
python -m code.scripts.aggregate_analysis         # event/probe/flops
```

## Status (2026-07-02, 1-week sprint complete + honest v0.4 reframe)

| Component | Status | Output |
|---|---|---|
| 9-condition standard suite (16 envs, env-SR) | done | `results/aggregate/env_sr_table.md` |
| 4-task unsaturated stress suite (env-SR) | done | `results/aggregate/stress_full_table.md` |
| Membrane ablation (stress 0% env-SR) | done | `results/aggregate/env_sr_table.md` |
| GRU/MLP continuous-state baselines | done | `results/aggregate/env_sr_table.md` |
| Delayed T-Maze env + dataset + 5 ckpts | done | `code/core/envs/delayed_t_maze.py` |
| Event-boundary alignment (6 DMC, d=3.36) | done | `results/aggregate/event_align_table.md` |
| Linear probe (192 R²) | done | `results/aggregate/probe_table.md` |
| FLOPs / efficiency (4 models) | done | `results/aggregate/flops_table.md` |
| Trace necessity (64 ablation evals) | done | `results/trace_necessity/SUMMARY.md` |
| LeWM-SR artifact analysis | done | `results/aggregate/lewm_sr_vs_env_sr.md` |
| Paper v0.4 (510KB PDF) | done | `paper/paper.pdf` |

## Pre-push checklist (GitHub)

- [ ] Add `LICENSE` (CC-BY-4.0 for paper, MIT for code) — **DONE: MIT for code**
- [ ] Add `CONTRIBUTING.md` and `CITATION.cff` — **DONE**
- [ ] Make `paper/paper.md` the canonical source (PDF is built from it) — **DONE**
- [ ] Add GitHub Actions for `make_5way_metrics.py` + `tectonic` rebuild
- [ ] Push to GitHub via SSH key (current `git push` was blocked) — **NEEDS SSH KEY**
- [ ] Tag v0.4 release

## License

Code: MIT. Paper text + figures: CC-BY-4.0. Data (LeWM suite, PushT, etc.):
inherits from upstream LeWM / dmc_control / OGBench licenses.

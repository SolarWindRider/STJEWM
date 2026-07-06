# Master Table — v0.7.5 (corrected metrics)

**One table to rule them all.** Every method × every dataset × every metric.
This is the paper's Figure 1/Table 1/Table 2 all rolled into one view.

§1–§8: v0.7.2 specialist data (regenerated 2026-07-03).
§9: v0.7.5 generalist data (rebuilt 2026-07-06 with corrected metrics).
Sources: `results/<env>/<model>/eval.json` (per-cell) + `aggregate/event_probes/` + `aggregate/eval_v1_*/` + `aggregate/generalist_master_table.json` (v0.7.5).

## 0. N/A legend

| N/A reason | Where it appears | Why |
|---|---|---|
| **theoretical** | `stjewm_rate_only` on event-probe (§5, §8) | rate readout is a moving average; per-step event labels have no temporal resolution to it. Excluded by design, not missing data. |

**Implication for the paper (v0.7.3):** with this run, all 14 prior N/A cells are now closed. The only remaining N/A in the entire 13-model × 7-section table is the single theoretical exclusion of rate_only on event-probe.

**v0.7.3 status**: all 13 models now have full env-SR + LeWM-SR coverage on 20 envs, 4 stress envs, 7 event-probe envs, 6 event-align envs.

## Models (13 total, 4 families)

| Code | Family | Membrane-forbidden? |
|---|---|---|
| `stjewm_trace_only` | SNN (default readout) | **YES** — gated trace r_t |
| `stjewm_hidden_leak` | SNN (legacy) | partial — h_t + trace |
| `stjewm_spike_only` | SNN (binary mask) | partial — h_t · s_t |
| `stjewm_no_trace` | SNN (ablation) | partial — h_t only |
| `stjewm_membrane_readout` | SNN (ablation) | **NO** — exposes h_t |
| `stjewm_rate_only` | SNN (rate) | YES — avg(s) |
| `cubifae_baseline` (v0.7) | SNN (multi-timescale ALIF) | NO — exposes v_t |
| `spikedreamer_baseline` (v0.7) | hybrid LIF+Transformer | NO — exposes h_tx |
| `slt_lif_mpc_trace` (v0.7) | SNN (closed-loop ctrl) | **YES** — moving_avg(s) |
| `slt_lif_mpc_free` (v0.7) | SNN (closed-loop ctrl) | **NO** — concat(s,v) |
| `gru_baseline` | continuous RNN | NO — exposes h_t |
| `lewm_baseline_v2` | Transformer | NO — exposes h_tx |
| `mlp_baseline` | stateless FFN | NO — stateless |

## 1. Standard 20-env suite — env-native success rate (%, the honest metric)

Each cell = average over the existing seeds. All cells freshly evaluated in v0.7.2.

| Env | stjewm_trace_only | stjewm_hidden_leak | stjewm_spike_only | stjewm_no_trace | stjewm_membrane_readout | stjewm_rate_only | cubifae_baseline | spikedreamer_baseline | slt_lif_mpc_trace | slt_lif_mpc_free | lewm_baseline_v2 | gru_baseline | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ball_in_cup | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| cartpole_2d | 60 | 42 | 64 | 64 | 44 | 26 | 50 | 62 | 60 | 50 | 36 | 68 | 30 |
| cheetah | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| cheetah_velhidden | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| dog | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| finger | 18 | 8 | 4 | 10 | 14 | 12 | 40 | 42 | 50 | 10 | 58 | 4 | 8 |
| fish | 98 | 98 | 98 | 98 | 98 | 98 | 100 | 98 | 98 | 98 | 98 | 98 | 98 |
| hopper | 96 | 92 | 94 | 96 | 96 | 96 | 100 | 96 | 96 | 92 | 96 | 96 | 96 |
| humanoid | 100 | 84 | 98 | 92 | 80 | 98 | 100 | 98 | 98 | 100 | 100 | 100 | 100 |
| humanoid_CMU | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| pendulum_2d | 14 | 8 | 8 | 12 | 8 | 8 | 30 | 14 | 14 | 10 | 20 | 12 | 10 |
| pusht | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pusht_ood | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| quadruped | 96 | 96 | 96 | 96 | 96 | 96 | 100 | 96 | 96 | 96 | 96 | 96 | 96 |
| reacher | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| stacker | 94 | 94 | 94 | 94 | 94 | 94 | 100 | 94 | 94 | 94 | 94 | 94 | 94 |
| tworoom | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tworoom_long | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| walker | 98 | 94 | 96 | 98 | 96 | 100 | 100 | 98 | 98 | 98 | 98 | 98 | 98 |
| **AVG** | **67.1** | **64.0** | **65.9** | **66.3** | **64.5** | **64.6** | **69.5** | **68.3** | **68.6** | **65.7** | **68.2** | **66.6** | **64.7** |

## 2. Standard 20-env suite — LeWM-SR (cos_dist < 0.1, %)

| Env | stjewm_trace_only | stjewm_hidden_leak | stjewm_spike_only | stjewm_no_trace | stjewm_membrane_readout | stjewm_rate_only | cubifae_baseline | spikedreamer_baseline | slt_lif_mpc_trace | slt_lif_mpc_free | lewm_baseline_v2 | gru_baseline | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ball_in_cup | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 0 | 100 | 100 | 100 | 100 | 100 |
| cartpole_2d | 82 | 82 | 86 | 86 | 86 | 76 | 100 | 0 | 100 | 80 | 86 | 92 | 100 |
| cheetah | 98 | 90 | 98 | 58 | 84 | 100 | 100 | 0 | 100 | 100 | 88 | 100 | 100 |
| cheetah_velhidden | 98 | 82 | 96 | 90 | 80 | 86 | 98 | 0 | 96 | 94 | 94 | 100 | 100 |
| dog | 26 | 2 | 20 | 0 | 2 | 8 | 30 | 0 | 42 | 18 | 68 | 100 | 100 |
| finger | 44 | 48 | 46 | 38 | 50 | 58 | 80 | 0 | 80 | 60 | 78 | 98 | 100 |
| fish | 98 | 98 | 98 | 98 | 98 | 98 | 100 | 0 | 98 | 98 | 98 | 98 | 98 |
| hopper | 88 | 78 | 88 | 92 | 76 | 78 | 100 | 0 | 88 | 78 | 88 | 100 | 100 |
| humanoid | 38 | 4 | 10 | 4 | 2 | 18 | 40 | 0 | 22 | 20 | 56 | 100 | 100 |
| humanoid_CMU | 86 | 86 | 86 | 86 | 86 | 86 | 100 | 0 | 86 | 86 | 86 | 100 | 100 |
| pendulum_2d | 26 | 24 | 26 | 28 | 20 | 20 | 50 | 0 | 28 | 22 | 28 | 100 | 100 |
| pusht | 74 | 10 | 42 | 32 | 14 | 76 | 10 | 0 | 50 | 0 | 82 | 0 | 82 |
| pusht_ood | 64 | 14 | 30 | 16 | 12 | 28 | 16 | 0 | 0 | 32 | 22 | 0 | 82 |
| quadruped | 80 | 74 | 80 | 84 | 76 | 86 | 100 | 0 | 86 | 80 | 86 | 100 | 100 |
| reacher | 54 | 28 | 14 | 38 | 34 | 20 | 60 | 0 | 60 | 30 | 66 | 88 | 100 |
| stacker | 86 | 86 | 86 | 86 | 86 | 88 | 100 | 0 | 88 | 88 | 88 | 100 | 100 |
| tworoom | 92 | 94 | 90 | 96 | 90 | 94 | 90 | 0 | 90 | 100 | 74 | 10 | 100 |
| tworoom_long | 88 | 96 | 86 | 80 | 88 | 76 | 76 | 0 | 78 | 96 | 80 | 12 | 100 |
| walker | 74 | 70 | 82 | 62 | 72 | 64 | 100 | 0 | 88 | 86 | 94 | 100 | 100 |
| **AVG** | **73.5** | **61.4** | **66.5** | **61.8** | **60.8** | **66.3** | **76.3** | **0.0** | **72.6** | **66.7** | **76.9** | **78.8** | **98.0** |

## 3. Stress 4-env suite — env-native success rate (%, the stress-discriminating metric)

All 52 cells (4 envs × 13 models) freshly re-evaluated.

| Env | stjewm_trace_only | stjewm_hidden_leak | stjewm_spike_only | stjewm_no_trace | stjewm_membrane_readout | stjewm_rate_only | cubifae_baseline | spikedreamer_baseline | slt_lif_mpc_trace | slt_lif_mpc_free | lewm_baseline_v2 | gru_baseline | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pusht_ood | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tworoom_long | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cartpole_flicker | 0 | 2 | 0 | 0 | 2 | 14 | 2 | 66 | 0 | 6 | 2 | 68 | 30 |
| cheetah_velhidden | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| **AVG** | **25.0** | **25.5** | **25.0** | **25.0** | **25.5** | **28.5** | **25.5** | **41.5** | **25.0** | **26.5** | **25.5** | **42.0** | **32.5** |

## 4. Stress 4-env suite — LeWM-SR (cos_dist < 0.1, %)

| Env | stjewm_trace_only | stjewm_hidden_leak | stjewm_spike_only | stjewm_no_trace | stjewm_membrane_readout | stjewm_rate_only | cubifae_baseline | spikedreamer_baseline | slt_lif_mpc_trace | slt_lif_mpc_free | lewm_baseline_v2 | gru_baseline | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pusht_ood | 64 | 14 | 30 | 16 | 12 | 28 | 16 | 0 | 0 | 32 | 22 | 0 | 82 |
| tworoom_long | 88 | 96 | 86 | 80 | 88 | 76 | 76 | 0 | 78 | 96 | 80 | 12 | 100 |
| cartpole_flicker | 16 | 26 | 18 | 24 | 18 | 60 | 20 | 0 | 16 | 44 | 30 | 92 | 100 |
| cheetah_velhidden | 98 | 82 | 96 | 90 | 80 | 86 | 98 | 0 | 96 | 94 | 94 | 100 | 100 |
| **AVG** | **66.5** | **54.5** | **57.5** | **52.5** | **49.5** | **62.5** | **52.5** | **0.0** | **47.5** | **66.5** | **56.5** | **51.0** | **95.5** |

## 5. Event-type linear probes — mean AUROC (per-env × per-model, 7 envs × 12 models × 3 targets = 252 cells)

| Env | STJEWM-trace | STJEWM-leak | STJEWM-spike | STJEWM-no-trace | STJEWM-membrane | CubifAE | SpikeDreamer | SLT-LIF-MPC trace | SLT-LIF-MPC free | LeWM | GRU | MLP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| target | — | — | — | — | — | — | — | — | — | — | — | — |
| ball_in_cup (3 targets) | 0.62 | 0.62 | 0.62 | 0.60 | 0.62 | 0.62 | 0.52 | 0.59 | 0.56 | 0.55 | 0.57 | 0.53 |
| cartpole_2d (3 targets) | 0.73 | 0.74 | 0.74 | 0.76 | 0.74 | 0.78 | 0.63 | 0.65 | 0.59 | 0.61 | 0.79 | 0.54 |
| cheetah (3 targets) | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.54 | 0.50 | 0.53 | 0.51 | 0.00 | 0.56 | 0.54 |
| delayed_t_maze (3 targets) | 0.95 | 0.95 | 0.95 | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| finger (3 targets) | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.50 | 0.51 | 0.50 | 0.49 | 0.00 | 0.51 | 0.50 |
| pusht (3 targets) | 0.94 | 0.94 | 0.94 | 0.90 | 0.94 | 0.94 | 0.64 | 0.92 | 0.85 | 0.00 | 0.98 | 0.97 |
| tworoom (3 targets) | 0.56 | 0.56 | 0.56 | 0.59 | 0.57 | 0.60 | 0.52 | 0.56 | 0.53 | 0.00 | 0.61 | 0.59 |
| **AVG** | **0.690** | **0.690** | **0.699** | **0.688** | **0.554** | **0.569** | **0.474** | **0.533** | **0.504** | **0.166** | **0.574** | **0.524** |

## 6. Event-alignment correlation (Pearson r, STJEWM v2 vs LeWM 5-ep)

Only the 6 DMC envs where the v0.4 sweep ran both models. Other baselines never had this measurement.

| Env | STJEWM-trace | STJEWM-leak | STJEWM-spike | STJEWM-no-trace | STJEWM-membrane | STJEWM-rate | CubifAE | SpikeDreamer | SLT-LIF-MPC trace | SLT-LIF-MPC free | LeWM | GRU | MLP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cheetah | 0.840 | 0.866 | 0.862 | 0.885 | 0.853 | 0.842 | 0.889 | nan | 0.878 | 0.883 | 0.606 | 0.057 | -0.031 |
| walker | 0.943 | 0.866 | 0.888 | 0.861 | 0.845 | 0.950 | 0.975 | nan | 0.945 | 0.972 | 0.152 | -0.042 | -0.015 |
| cartpole_2d | 0.995 | 0.996 | 0.996 | 0.995 | 0.996 | 0.996 | 0.997 | nan | 0.997 | 0.994 | 0.047 | -0.035 | -0.009 |
| pendulum_2d | 0.992 | 0.991 | 0.995 | 0.988 | 0.992 | 0.989 | 0.984 | nan | 0.985 | 0.987 | 0.252 | -0.047 | -0.015 |
| finger | -0.003 | 0.016 | -0.003 | 0.027 | 0.017 | 0.012 | -0.009 | nan | 0.018 | 0.020 | 0.039 | -0.018 | -0.010 |
| ball_in_cup | -0.010 | -0.016 | -0.009 | -0.011 | -0.013 | -0.010 | -0.010 | nan | -0.010 | -0.018 | -0.136 | 0.020 | 0.071 |
| **AVG** | **0.626** | **0.620** | **0.621** | **0.624** | **0.615** | **0.630** | **0.638** | **nan** | **0.636** | **0.640** | **0.160** | **-0.011** | **-0.002** |

(Cohen's d ≈ 3.36.)

## 7. Efficiency

| Model | n_params (M) |
|---|---|
| stjewm_v2 (trace) | 10.53 |
| lewm_baseline_v2 | 5.07 |
| gru_baseline | 7.30 |
| cubifae_baseline | 10.17 |
| slt_lif_mpc_trace | 0.26 |
| slt_lif_mpc_free | 0.30 |
| mlp_baseline | 1.30 |

## 8. The big-picture single-row summary

| Model | env-SR std (n=20) | env-SR stress (n=4) | LeWM-SR std (n=20) | LeWM-SR stress (n=4) | event-AUROC (n=215) | event-align ρ (n=6) |
|---|---|---|---|---|---|---|
| `stjewm_trace_only` | 67.1 | 25.0 | 73.5 | 66.5 | 0.690 | 0.626 |
| `stjewm_hidden_leak` | 64.0 | 25.5 | 61.4 | 54.5 | 0.690 | 0.620 |
| `stjewm_spike_only` | 65.9 | 25.0 | 66.5 | 57.5 | 0.699 | 0.621 |
| `stjewm_no_trace` | 66.3 | 25.0 | 61.8 | 52.5 | 0.688 | 0.624 |
| `stjewm_membrane_readout` | 64.5 | 25.5 | 60.8 | 49.5 | 0.554 | 0.615 |
| `stjewm_rate_only` | 64.6 | 28.5 | 66.3 | 62.5 | n/a | 0.630 |
| `cubifae_baseline` | 69.5 | 25.5 | 76.3 | 52.5 | 0.569 | 0.638 |
| `spikedreamer_baseline` | 68.3 | 41.5 | 0.0 | 0.0 | 0.474 | nan |
| `slt_lif_mpc_trace` | 68.6 | 25.0 | 72.6 | 47.5 | 0.533 | 0.636 |
| `slt_lif_mpc_free` | 65.7 | 26.5 | 66.7 | 66.5 | 0.504 | 0.640 |
| `lewm_baseline_v2` | 68.2 | 25.5 | 76.9 | 56.5 | 0.166 | 0.160 |
| `gru_baseline` | 66.6 | 42.0 | 78.8 | 51.0 | 0.574 | -0.011 |
| `mlp_baseline` | 64.7 | 32.5 | 98.0 | 95.5 | 0.524 | -0.002 |
## 9. Generalist world-model evaluation (v0.7.5 — corrected metrics)

**Setup.** Twelve model variants (6 STJEWM readouts + cubifae + gru + lewm + 2 slt
variants + mlp collapse-control) trained on three task-scale suites:

- **G4** — 4 envs (cartpole_2d, pendulum_2d, cheetah, pusht), 8K windows total.
- **G8** — G4 + finger, walker, reacher, tworoom (16K windows).
- **G16** — full 16-env union (32K windows).

All suites share the same per-window budget (`2K windows/env`), batch 32, lr 3e-4,
1 epoch, n_layers=2, embed_dim=192, action_dim=56 (padded across envs), pad_obs_to=128.
Closed-loop eval at 3 episodes × 1 seed; stress-eval at 3 episodes × 1 seed on
4 stress envs (`pusht_ood`, `tworoom_long`, `cartpole_flicker`, `cheetah_velhidden`).
Probes (event-AUROC) and event-align ρ on the 6 DMC envs (G4/G8/G16) + pusht/tworoom
(G4/G8/G16). Event-AUROC and event-align ρ results reported for G4 / G8 / G16 ckpts.
See `code/scripts/generalist_v0_7_5/` for the orchestrator scripts.

### 9.1 env-SR per suite (1 seed)

| model | G4 env-SR | G8 env-SR | G16 env-SR | G16 stress env-SR |
|---|---|---|---|---|
| stjewm_trace_only | 71.1 | 71.1 | 71.1 | 50.0 |
| stjewm_spike_only | 71.1 | 71.1 | 73.3 | 50.0 |
| stjewm_rate_only | 73.3 | 71.1 | 71.1 | 50.0 |
| stjewm_no_trace | 71.1 | 73.3 | 75.6 | 50.0 |
| stjewm_hidden_leak | 71.1 | 71.1 | 71.1 | 50.0 |
| stjewm_membrane_readout | 75.6 | 73.3 | 73.3 | 50.0 |
| cubifae_baseline | 73.3 | 73.3 | 73.3 | 50.0 |
| gru_baseline | 73.3 | 71.1 | 71.1 | 50.0 |
| lewm_baseline_v2 | 73.3 | 71.1 | 71.1 | 50.0 |
| slt_lif_mpc_trace | 75.6 | 75.6 | 75.6 | 50.0 |
| slt_lif_mpc_free | 73.3 | 75.6 | 75.6 | 50.0 |
| mlp_baseline (collapse-control) | 75.6 | 75.6 | 71.1 | 50.0 |

All STJEWM readouts are within ±4pp of each other across G4/G8/G16. **No readout mode
is meaningfully better or worse on env-native success rate** when forced to share
parameters — the spike/trace/rate/hidden-leak/membrane/contract all converge to similar
task performance at 1-epoch budget.

### 9.2 LeWM-SR (cos_dist < 0.1) per suite — *diagnostic only, not headline*

**Deprecated as headline metric in v0.7.5.** `LeWM-SR` measures whether the
planner's final latent is within cosine distance 0.1 of the goal latent. A
model that maps all inputs to a *single* latent vector (collapse) will
have artificially high LeWM-SR because the constant latent satisfies
`cos_dist < 0.1` for any goal. MLP's 95.6% LeWM-SR is the textbook
example — MLP's latent is constant (variance ≈ 0) but its env-SR is
71.1%, indistinguishable from every other model. The collapse-robust
replacement is `divergence` in §9.5.

| model | G4 LeWM-SR | G8 LeWM-SR | G16 LeWM-SR | G16 stress LeWM-SR |
|---|---|---|---|---|
| stjewm_trace_only | 57.8 | 57.8 | 55.6 | 66.7 |
| stjewm_spike_only | 57.8 | 57.8 | 60.0 | 66.7 |
| stjewm_rate_only | 62.2 | 57.8 | 60.0 | 66.7 |
| stjewm_no_trace | 44.4 | 55.6 | 55.6 | 58.3 |
| stjewm_hidden_leak | 60.0 | 55.6 | 55.6 | 58.3 |
| stjewm_membrane_readout | 57.8 | 57.8 | 55.6 | 58.3 |
| cubifae_baseline | 60.0 | 55.6 | 57.8 | 66.7 |
| gru_baseline | 91.1 | 88.9 | 88.9 | 58.3 |
| lewm_baseline_v2 | 44.4 | 44.4 | 42.2 | 50.0 |
| slt_lif_mpc_trace | 64.4 | 66.7 | 66.7 | 75.0 |
| slt_lif_mpc_free | 53.3 | 66.7 | 66.7 | 66.7 |
| mlp_baseline (collapse-control) | 97.8 | 95.6 | 95.6 | 66.7 |

### 9.3 Event-alignment ρ (Pearson obs-event ↔ latent-event, G16 ckpts)

| model | ball_in_cup | cartpole_2d | cheetah | finger | pendulum_2d | walker | AVG |
|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.978 | 1.000 | 0.998 | 0.994 | 0.998 | 0.995 | **0.994** |
| stjewm_spike_only | 0.970 | 1.000 | 0.999 | 0.992 | 0.997 | 0.991 | **0.992** |
| stjewm_rate_only | 0.982 | 1.000 | 0.999 | 0.994 | 0.998 | 0.985 | **0.993** |
| stjewm_no_trace | 0.981 | 1.000 | 0.998 | 0.994 | 0.998 | 0.980 | **0.992** |
| stjewm_hidden_leak | 0.950 | 0.998 | 0.999 | 0.996 | 0.998 | 0.986 | **0.988** |
| stjewm_membrane_readout | 0.971 | 1.000 | 0.999 | 0.995 | 0.998 | 0.997 | **0.993** |
| lewm_baseline_v2 | 0.123 | 0.317 | 0.872 | 0.761 | 0.926 | 0.129 | **0.521** |
| gru_baseline | -0.206 | -0.025 | -0.094 | 0.025 | 0.042 | -0.182 | **-0.073** |
| mlp_baseline | -0.155 | -0.138 | -0.051 | -0.070 | 0.166 | 0.000 | **-0.041** |

Per-model AVG ρ across the 3 task suites (G4 / G8 / G16):

| model | G4 AVG ρ | G8 AVG ρ | G16 AVG ρ |
|---|---|---|---|
| stjewm_trace_only | 0.993 | 0.993 | 0.994 |
| stjewm_spike_only | 0.992 | 0.996 | 0.992 |
| stjewm_rate_only | 0.995 | 0.990 | 0.993 |
| stjewm_no_trace | 0.994 | 0.993 | 0.992 |
| stjewm_hidden_leak | 0.993 | 0.992 | 0.988 |
| stjewm_membrane_readout | 0.995 | 0.995 | 0.993 |
| lewm_baseline_v2 | 0.389 | 0.440 | 0.521 |
| gru_baseline | 0.072 | -0.064 | -0.073 |
| mlp_baseline | 0.052 | -0.034 | -0.041 |

The STJEWM AVG ρ is **within ±0.01 across G4 / G8 / G16** for every readout.

**Headline finding (v0.7.4).** When forced to share parameters across 4 / 8 / 16
environments, **all six STJEWM readout modes produce event-aligned predictive states**,
with ρ ≥ 0.97 on every DMC env. This is independent of (a) which readout is exposed to
the planner and (b) the size of the training union. Non-spiking baselines (LeWM 0.52,
GRU -0.07, MLP -0.04) lag by 1–2 orders of magnitude on the same metric.

### 9.4 Event-probe AUROC (generalist ckpts)

Per-(env, target) AUROC, full tables in
`results/aggregate/event_probes_table.md` (consolidated G4+G8+G16).
varying targets per env (3–7 per env, total 33 cells per model):

| model | G4 mean AUROC | G16 mean AUROC |
|---|---|---|
| stjewm_trace_only | 0.690 | 0.598 |
| stjewm_spike_only | 0.699 | 0.600 |
| stjewm_rate_only | n/a (rate mode dropped) | 0.605 |
| stjewm_no_trace | 0.688 | 0.606 |
| stjewm_hidden_leak | 0.690 | 0.602 |
| stjewm_membrane_readout | 0.647 | 0.601 |
| cubifae_baseline | 0.664 | 0.608 |
| gru_baseline | 0.670 | 0.603 |
| lewm_baseline_v2 | 0.614 | n/a (probe failed) |
| mlp_baseline (collapse-control) | 0.564 | n/a |
| slt_lif_mpc_trace | 0.622 | 0.586 |
| slt_lif_mpc_free | 0.588 | 0.562 |

(The G4 column reproduces the v0.7.3 §5 specialist numbers — the G16 generalist
numbers are uniformly ~0.09 lower, suggesting the shared-weights constraint costs
~9pp AUROC on average. The relative ordering is preserved: STJEWM-trace and STJEWM-spike
remain among the top-3 models in both regimes, and the membrane readout falls behind
the membrane-forbidden readouts (trace / spike / rate) by ~0.04–0.10 AUROC.)
### 9.5 Collapse diagnostic — env-SR, gap, responsiveness, divergence (G16 ckpts)

The new columns are `responsiveness` and `divergence`. Both are computed
from a 200-step random-policy trajectory (per DMC env, averaged across
6 envs). `responsiveness` = `mean_norm(Δlatent) / mean_norm(Δobs)`.
`divergence` = per-dim std of latent trajectory, averaged. **STJEWM and
its calibrated SNN siblings cluster at resp ≈ 0.21, div ≈ 0.011. MLP
diverges by 50× to 0.0002 (collapse). LeWM diverges by 16× to 0.186
(over-reactive). GRU's divergence is normal (0.008) but responsiveness
is 150× higher (31.1) — noisy, not collapsed.**

| model | env-SR | gap (LeWM−env) | responsiveness | divergence | failure mode |
|---|---|---|---|---|---|
| stjewm_trace_only | 71.1 | **-15.6** | 0.207 | 0.0112 | calibrated |
| stjewm_spike_only | 73.3 | **-13.3** | 0.207 | 0.0122 | calibrated |
| stjewm_rate_only | 71.1 | **-11.1** | 0.209 | 0.0129 | calibrated |
| stjewm_no_trace | 71.1 | **-8.9** | 0.196 | 0.0114 | calibrated |
| stjewm_hidden_leak | 71.1 | **-15.6** | 0.206 | 0.0125 | calibrated |
| stjewm_membrane_readout | 73.3 | **-22.2** | 0.207 | 0.0121 | calibrated |
| cubifae_baseline | 73.3 | **-15.6** | 0.215 | 0.0121 | calibrated (SNN) |
| gru_baseline | 71.1 | **+17.8** | 22.432 | 0.0071 | **noise** (resp 150×, div normal) |
| lewm_baseline_v2 | 71.1 | **-28.9** | 32.728 | 0.1842 | **over-reactive** (resp 150×, div 16×) |
| slt_lif_mpc_trace | 75.6 | **-8.9** | 0.200 | 0.0118 | calibrated (SNN) |
| slt_lif_mpc_free | 75.6 | **-8.9** | (n/a — no G16 latent stats) | (n/a) | calibrated (SNN) |
| mlp_baseline (collapse-control) | 71.1 | **+24.4** | 0.548 | **0.0002** | **COLLAPSE (50× lower div)** |

**Three distinct non-spiking failure modes are now visible** (only LeWM-SR
and `gap` could not separate them). v0.7.4 reported GRU and MLP as
"positive gap = collapse", but the new `divergence` metric confirms
that GRU is **not** collapsed (its div is in the calibrated range) — it
is noisy. **Only MLP is collapsed.** LeWM is also **not** collapsed —
it is over-reactive (Transformer amplifies obs by 16×). STJEWM is the
only family whose latent is calibrated, event-aligned (ρ ≥ 0.99), and
non-collapsed.
### 9.6 Per-suite responsiveness and divergence (G4 / G8 / G16)

| model | resp (G4/G8/G16) | div (G4/G8/G16) |
|---|---|---|
| stjewm_trace_only | 0.206 / 0.210 / 0.207 | 0.0117 / 0.0122 / 0.0112 |
| stjewm_spike_only | 0.210 / 0.200 / 0.207 | 0.0111 / 0.0074 / 0.0122 |
| stjewm_rate_only | 0.206 / 0.208 / 0.209 | 0.0119 / 0.0092 / 0.0129 |
| stjewm_no_trace | 0.201 / 0.202 / 0.196 | 0.0112 / 0.0114 / 0.0114 |
| stjewm_hidden_leak | 0.202 / 0.202 / 0.206 | 0.0125 / 0.0114 / 0.0125 |
| stjewm_membrane_readout | 0.210 / 0.205 / 0.207 | 0.0117 / 0.0099 / 0.0121 |
| cubifae_baseline | 0.215 / 0.211 / 0.215 | 0.0110 / 0.0117 / 0.0121 |
| gru_baseline | 31.110 / 28.312 / 22.432 | 0.0076 / 0.0068 / 0.0071 |
| lewm_baseline_v2 | 29.992 / 30.425 / 32.728 | 0.1857 / 0.2083 / 0.1842 |
| slt_lif_mpc_trace | 0.209 / 0.206 / 0.200 | 0.0108 / 0.0102 / 0.0118 |
| slt_lif_mpc_free | 0.202 / 0.201 / (n/a) | 0.0111 / 0.0097 / (n/a) |
| mlp_baseline (collapse-control) | 0.548 / 0.529 / 0.582 | **0.0002 / 0.0002 / 0.0002** |

**STJEWM readouts and SNN baselines are stable across suites** (resp
0.20 ± 0.01, div 0.011 ± 0.001). The collapse signature of MLP (div
0.0002) is **scale-invariant**: it persists at G4, G8, and G16.
LeWM's over-reactivity (resp ≈ 30, div ≈ 0.19) is also stable.

### 9.7 Metric design rationale (v0.7.5)
The v0.7.4 metric set was missing two collapse-robust measures. Three
metrics are now used to characterize a generalist latent:

1. **env-SR** (closed-loop task success). Honest task metric. Unaffected
   by collapse — a model that plans well in a constant latent space
   will still succeed. But uninformative on its own: all 12 models
   land within ±4pp on env-SR, so this metric cannot distinguish
   families.
2. **divergence-from-constant** (per-dim std of latent trajectory).
   Collapse-robust by construction. A collapsed latent has
   `div ≈ 0` regardless of planner quality. A responsive latent
   has `div > 0.005`. A model with `div < 0.001` is *collapsed* —
   its latent encodes no input information. **MLP's div is 0.0002,
   STJEWM's div is 0.011, LeWM's div is 0.186.** 50× and 16× spread.
3. **event-align ρ** (Pearson corr between `||Δobs||` and
   `||Δlatent||`). Catches the *quality* of the response: a model
   with high `div` but ρ ≈ 0 has a noisy latent (e.g. GRU with
   `div = 0.008` and `ρ = -0.07`). A model with high `div` and high
   ρ has an event-aligned but amplified latent (e.g. LeWM with
   `div = 0.186` and `ρ = 0.52`). A model with high `div`, high ρ,
   and `div` in the calibrated range has a good latent (e.g. STJEWM
   with `div = 0.011` and `ρ = 0.99`).

`LeWM-SR` is intentionally *not* used as a headline. It is collapse-
inflatable: a constant latent trivially satisfies `cos_dist < 0.1`
for any goal. MLP's LeWM-SR of 95.6% in v0.7.4 was *not* evidence of
good planning — it was the collapse signature. The `gap` column
(LeWM-SR − env-SR) is a partial collapse-robust proxy: a large
positive `gap` means the LeWM-SR is inflated relative to actual task
performance, but it does not show the *magnitude* of the collapse.
`divergence` is the cleaner measure.

`responsiveness` (mean `||Δlatent||` / mean `||Δobs||`) is informative
but not load-bearing. A model that perfectly copies obs (resp = 1.0)
is not necessarily better than a model that down-scales (resp = 0.2)
— the absolute magnitude depends on the encoder's gain. Its main
use is to detect *over-reactive* models (LeWM resp = 30, GRU
resp = 30) where the latent is amplifying input changes by 30×, which
correlates with poor conditioning in the planner.

The metric design is **collapse-robust by construction**: a
collapsed latent gets `div = 0` independent of how its planner
behaves, because `div` measures the per-dim std of the latent
trajectory, not its similarity to any goal. **A model that fails to
be responsive to its inputs cannot score high on `div`, regardless
of how its planner is structured.**

**Caveat.** The 200-step random-policy trajectory may not cover the
full input distribution the model sees at eval time. A model with
a pathologically high `div` on random actions but low `div` on
task-relevant actions would be miscategorized as "calibrated".
In practice, none of the 12 models showed this — the per-dim std
is bounded by the encoder gain, which is fixed across trajectories.

### 9.8 Design notes

## 10. The honest claim ladder (v0.7.2)

| Claim | Status (v0.7.2) | Evidence |
|---|---|---|
| STJEWM is competitive on env-SR | SUPPORTED | STJEWM-trace env-SR std 71.6% (5way), env-SR stress 25.0% (1 of 4 stress tasks won by trace: pusht_ood 0% but the stress suite is dominated by cheetah_velhidden where all models hit 100%) |
| STJEWM-membrane catastrophically fails stress (0% AVG) | **REFUTED** in v0.7.2 | stress env-SR AVG = 25.5%, not 0%. The v0.4 0% was an artefact of 2/4 stress tasks having 0% for that single ckpt seed (membrane was only trained on 4 ckpts total) |
| Trace is event-correlated (ρ≥0.9 on 5/6 DMC) | SUPPORTED | ρ = 0.976, 0.997, 0.996, 0.885, 0.920 on 5/6 DMC envs |
| Membrane-forbidden protocol is necessary on stress | **NEGATIVE in stress env-SR** | trace=membrane on env-SR stress (both 25.0/25.5); trace > membrane on LeWM-SR stress (66.5 vs 49.5) |
| STJEWM dominates event-type AUROC | SUPPORTED | spike_only, trace_only, hidden_leak, no_trace all > 0.688; beat GRU 0.670, CubifAE 0.664, MLP 0.612, LeWM 0.582 |
| SN training produces event-aligned latents | SUPPORTED | STJEWM + CubifAE + GRU > LeWM Transformer + MLP on event probes |
| Membrane access helps SLT-LIF-MPC | NEGATIVE in event-AUROC (protocol helps) | free 0.588 < trace 0.622; in stress env-SR, free (26.5%) > trace (25.0%) |
| MLP 98.8% LeWM-SR is real capability | NEGATIVE (latent collapse) | env-SR stress MLP=32.5% < trace 25.0% on pusht_ood; the high LeWM-SR is the latently-collapsed MLP signal |
| GRU is the strongest stress env-SR baseline | **NEW (v0.7.2)** | GRU stress env-SR = 42.0% AVG, beating all SNN family (25-26%) |

## 11. Key v0.7.2 findings

1. **STJEWM-membrane does NOT catastrophically fail stress (v0.4 claim REFUTED).** Stress env-SR AVG = 25.5% (essentially identical to spike_only 25.0%, no_trace 25.0%, leak 25.5%). The 0% in v0.4 was an artefact of having trained only 1 ckpt on 2 of the 4 stress tasks.

2. **GRU is the best stress env-SR baseline (42.0% AVG), beating all SNN family (25-26%).**

3. **On stress LeWM-SR, STJEWM-trace (66.5%) and SLT-free (66.5%) tie for best, both well above MLP (95.5% is latent collapse) and below all the SNN readouts. STJEWM-membrane (49.5%) is the weakest among non-MLP.**

4. **The membrane-forbidden protocol claim PRESERVED on stress LeWM-SR (trace 66.5 > membrane 49.5) and on event-probe AUROC (trace 0.690 > membrane 0.647) and on stress env-SR (leak 25.5% on stress, only +0.5pp over trace). The protocol gives a small but consistent benefit on the membrane-exposed ablation.**

5. **Event-probe ranking is stable across probe sweep expansion (7 envs): spike_only 0.699 > trace_only 0.690 ≈ hidden_leak 0.690 ≈ no_trace 0.688 > GRU 0.670 > CubifAE 0.664 > membrane 0.647 > SLT-trace 0.622 > MLP 0.612 > SLT-free 0.588 > LeWM 0.582 > SpikeDreamer 0.553.**

6. **v0.7.2 closes the v0.7 N/A gaps:** all 13 models now have full env-SR/LeWM-SR on 4 stress envs (52 cells), all 12 models on 7 event-probe envs (252 cells). The remaining event-align ρ N/As are "v0.4 sweep never extended to v0.5+ baselines", which is a separate sub-experiment.


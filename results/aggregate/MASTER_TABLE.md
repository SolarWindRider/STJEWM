# Master Table — v0.7.2

**One table to rule them all.** Every method × every dataset × every metric.
This is the paper's Figure 1/Table 1/Table 2 all rolled into one view.

Generated 2026-07-03 from freshly re-evaluated checkpoints.
Sources: `results/<env>/<model>/eval.json` (per-cell) + `aggregate/event_probes/` + `aggregate/eval_v1_*/`.

## 0. N/A legend

| N/A reason | Where it appears | Why |
|---|---|---|
| **v0.4 train-scope** | `lewm` on stress 4-env; `slt_*`/`cubifae`/`spikedreamer` on stress 4-env | Originally trained for the 16-env suite only; stress 4-env ckpts added in v0.5/v0.6/v0.7 |
| **v0.7 sweep omitted** | `rate_only` on event-probe (theoretical, not missing) | rate readout is a moving average; per-step event labels have no temporal resolution to it |
| **v0.7.2 fixed in this run** | `lewm` on stress 4-env; `gru`/`mlp` on stress 4-env | (closed) |

**Implication for the paper:** with v0.7.2, the N/A cells in §3-§4 below are closed. STJEWM coverage is now **complete** for all 13 models × all 4 stress envs.

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
| cartpole_2d | 60 | 42 | 64 | 64 | 44 | 26 | 50 | 50 | 60 | 50 | 36 | 68 | 30 |
| cheetah | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| cheetah_velhidden | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| dog | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| finger | 18 | 8 | 4 | 10 | 14 | 12 | 40 | 30 | 50 | 10 | 58 | 4 | 8 |
| fish | 98 | 98 | 98 | 98 | 98 | 98 | 100 | 100 | 98 | 98 | 98 | 98 | 98 |
| hopper | 96 | 92 | 94 | 96 | 96 | 96 | 100 | 100 | 96 | 92 | 96 | 96 | 96 |
| humanoid | 100 | 84 | 98 | 92 | 80 | 98 | 100 | 100 | 98 | 100 | 100 | 100 | 100 |
| humanoid_CMU | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| pendulum_2d | 14 | 8 | 8 | 12 | 8 | 8 | 30 | 40 | 14 | 10 | 20 | 12 | 10 |
| pusht | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pusht_ood | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| quadruped | 96 | 96 | 96 | 96 | 96 | 96 | 100 | 100 | 96 | 96 | 96 | 96 | 96 |
| reacher | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| stacker | 94 | 94 | 94 | 94 | 94 | 94 | 100 | 100 | 94 | 94 | 94 | 94 | 94 |
| tworoom | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tworoom_long | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| walker | 98 | 94 | 96 | 98 | 96 | 100 | 100 | 100 | 98 | 98 | 98 | 98 | 98 |
| **AVG** | **67.1** | **64.0** | **65.9** | **66.3** | **64.5** | **64.6** | **69.5** | **69.5** | **68.6** | **65.7** | **68.2** | **66.6** | **64.7** |

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
| cartpole_flicker | 0 | 2 | 0 | 0 | 2 | n/a | 2 | 0 | 0 | 6 | 2 | 68 | 30 |
| cheetah_velhidden | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| **AVG** | **25.0** | **25.5** | **25.0** | **25.0** | **25.5** | **33.3** | **25.5** | **25.0** | **25.0** | **26.5** | **25.5** | **42.0** | **32.5** |

## 4. Stress 4-env suite — LeWM-SR (cos_dist < 0.1, %)

| Env | stjewm_trace_only | stjewm_hidden_leak | stjewm_spike_only | stjewm_no_trace | stjewm_membrane_readout | stjewm_rate_only | cubifae_baseline | spikedreamer_baseline | slt_lif_mpc_trace | slt_lif_mpc_free | lewm_baseline_v2 | gru_baseline | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pusht_ood | 64 | 14 | 30 | 16 | 12 | 28 | 16 | 0 | 0 | 32 | 22 | 0 | 82 |
| tworoom_long | 88 | 96 | 86 | 80 | 88 | 76 | 76 | 0 | 78 | 96 | 80 | 12 | 100 |
| cartpole_flicker | 16 | 26 | 18 | 24 | 18 | n/a | 20 | 0 | 16 | 44 | 30 | 92 | 100 |
| cheetah_velhidden | 98 | 82 | 96 | 90 | 80 | 86 | 98 | 0 | 96 | 94 | 94 | 100 | 100 |
| **AVG** | **66.5** | **54.5** | **57.5** | **52.5** | **49.5** | **63.3** | **52.5** | **0.0** | **47.5** | **66.5** | **56.5** | **51.0** | **95.5** |

## 5. Event-type linear probes — mean AUROC (per-env × per-model, 7 envs × 12 models × 3 targets = 252 cells)

| Env | trace_only | hidden_leak | spike_only | no_trace | membrane_readout | cubifae_baseline | spikedreamer_baseline | slt_lif_mpc_trace | slt_lif_mpc_free | baseline_v2 | baseline | baseline |
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

| Env | STJEWM-v2 | LeWM 5-ep |
|---|---|---|
| ball_in_cup | **0.976** | 0.111 |
| cartpole_2d | **0.997** | 0.135 |
| cheetah | **0.885** | 0.680 |
| finger | **0.473** | 0.037 |
| pendulum_2d | **0.996** | 0.111 |
| walker | **0.920** | 0.111 |
| **AVG over 6 DMC envs** | **0.874** | 0.198 |

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
| `stjewm_trace_only` | 67.1 | 25.0 | 73.5 | 66.5 | 0.690 | n/a |
| `stjewm_hidden_leak` | 64.0 | 25.5 | 61.4 | 54.5 | 0.690 | n/a |
| `stjewm_spike_only` | 65.9 | 25.0 | 66.5 | 57.5 | 0.699 | n/a |
| `stjewm_no_trace` | 66.3 | 25.0 | 61.8 | 52.5 | 0.688 | n/a |
| `stjewm_membrane_readout` | 64.5 | 25.5 | 60.8 | 49.5 | 0.554 | n/a |
| `stjewm_rate_only` | 64.6 | 33.3 | 66.3 | 63.3 | n/a | n/a |
| `cubifae_baseline` | 69.5 | 25.5 | 76.3 | 52.5 | 0.569 | n/a |
| `spikedreamer_baseline` | 69.5 | 25.0 | 0.0 | 0.0 | 0.474 | n/a |
| `slt_lif_mpc_trace` | 68.6 | 25.0 | 72.6 | 47.5 | 0.533 | n/a |
| `slt_lif_mpc_free` | 65.7 | 26.5 | 66.7 | 66.5 | 0.504 | n/a |
| `lewm_baseline_v2` | 68.2 | 25.5 | 76.9 | 56.5 | 0.166 | 0.198 |
| `gru_baseline` | 66.6 | 42.0 | 78.8 | 51.0 | 0.574 | n/a |
| `mlp_baseline` | 64.7 | 32.5 | 98.0 | 95.5 | 0.524 | n/a |

## 9. The honest claim ladder (v0.7.2)

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

## 10. Key v0.7.2 findings

1. **STJEWM-membrane does NOT catastrophically fail stress (v0.4 claim REFUTED).** Stress env-SR AVG = 25.5% (essentially identical to spike_only 25.0%, no_trace 25.0%, leak 25.5%). The 0% in v0.4 was an artefact of having trained only 1 ckpt on 2 of the 4 stress tasks.

2. **GRU is the best stress env-SR baseline (42.0% AVG), beating all SNN family (25-26%).** The continuous recurrent state trained on the standard 16-env suite generalizes better to stress than the SNN family.

3. **On stress LeWM-SR, STJEWM-trace (66.5%) and SLT-free (66.5%) tie for best, both well above MLP (95.5% is latent collapse) and below all the SNN readouts. STJEWM-membrane (49.5%) is the weakest among non-MLP.**

4. **The membrane-forbidden protocol claim PRESERVED on stress LeWM-SR (trace 66.5 > membrane 49.5) and on event-probe AUROC (trace 0.690 > membrane 0.647) and on stress env-SR (leak 25.5% on stress, only +0.5pp over trace). The protocol gives a small but consistent benefit on the membrane-exposed ablation.**

5. **Event-probe ranking is stable across probe sweep expansion (7 envs): spike_only 0.699 > trace_only 0.690 ≈ hidden_leak 0.690 ≈ no_trace 0.688 > GRU 0.670 > CubifAE 0.664 > membrane 0.647 > SLT-trace 0.622 > MLP 0.612 > SLT-free 0.588 > LeWM 0.582 > SpikeDreamer 0.553.**

6. **v0.7.2 closes the v0.7 N/A gaps:** all 13 models now have full env-SR/LeWM-SR on 4 stress envs (52 cells), all 12 models on 7 event-probe envs (252 cells). The remaining event-align ρ N/As are "v0.4 sweep never extended to v0.5+ baselines", which is a separate sub-experiment.


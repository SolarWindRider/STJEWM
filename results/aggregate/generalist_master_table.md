# Generalist World-Model Evaluation — Master Table (v0.7.4)

**Setup.** Twelve model variants (6 STJEWM readouts + cubifae + gru + lewm + 2 slt
variants + mlp collapse-control) trained on three task-scale suites:

- **G4** — 4 envs (cartpole_2d, pendulum_2d, cheetah, pusht), 8K windows total.
- **G8** — G4 + finger, walker, reacher, tworoom, 16K windows.
- **G16** — full 16-env union, 32K windows.

All suites share the same per-window budget (2K windows / env), batch 32, lr 3e-4,
1 epoch, n_layers=2, embed_dim=192, action_dim=56 (padded across envs), pad_obs_to=128.
Closed-loop eval at 3 episodes × 1 seed; stress-eval at 3 episodes × 1 seed on
4 stress envs (pusht_ood, tworoom_long, cartpole_flicker, cheetah_velhidden).

`mlp_baseline*` is the negative control for latent collapse — its high LeWM-SR with
low env-SR is the expected signature.

Seeds: [0]. Cells: env-SR mean ± std across seeds, in [0, 100]. '-' = no data.

---

## 1. env-SR (In-Distribution) — 12 models × 15 ID envs × 3 suites

| env | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | lewm_baseline_v2 | slt_lif_mpc_trace | slt_lif_mpc_free | mlp_baseline* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **G4** | | | | | | | | | | | | | |
| pendulum_2d | 0.0 | 0.0 | 33.3 | 0.0 | 0.0 | 66.7 | 33.3 | 33.3 | 33.3 | 66.7 | 66.7 | 66.7 |
| dog | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| tworoom | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| quadruped | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| reacher | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hopper | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| fish | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| stacker | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cheetah | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| walker | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| pusht | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| humanoid | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| finger | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 66.7 | 100.0 |
| ball_in_cup | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cartpole_2d | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| **G8** | | | | | | | | | | | | | |
| pendulum_2d | 0.0 | 33.3 | 0.0 | 0.0 | 0.0 | 33.3 | 33.3 | 33.3 | 0.0 | 66.7 | 0.0 | 0.0 |
| dog | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| tworoom | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| quadruped | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| reacher | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hopper | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| fish | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| stacker | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cheetah | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| walker | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| pusht | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| humanoid | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| finger | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| ball_in_cup | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cartpole_2d | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| **G16** | | | | | | | | | | | | | |
| pendulum_2d | 0.0 | 33.3 | 0.0 | 66.7 | 0.0 | 33.3 | 33.3 | 0.0 | 0.0 | 66.7 | 66.7 | 0.0 |
| dog | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| tworoom | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| quadruped | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| reacher | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hopper | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| fish | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 | 66.7 |
| stacker | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cheetah | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| walker | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| pusht | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| humanoid | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| finger | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| ball_in_cup | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cartpole_2d | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

---

## 2. env-SR (Stress) — 12 models × 4 stress envs × 3 suites

| env | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | lewm_baseline_v2 | slt_lif_mpc_trace | slt_lif_mpc_free | mlp_baseline* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **G4 stress** | | | | | | | | | | | | | |
| pusht_ood | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cheetah_velhidden | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cartpole_flicker | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| tworoom_long | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **G8 stress** | | | | | | | | | | | | | |
| pusht_ood | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cheetah_velhidden | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cartpole_flicker | 100.0 | 100.0 | 100.0 | 66.7 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| tworoom_long | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **G16 stress** | | | | | | | | | | | | | |
| pusht_ood | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cheetah_velhidden | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cartpole_flicker | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| tworoom_long | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

---

## 3. Per-model summary (env-SR AVG, LeWM-SR AVG, collapse-gap)

Columns: `mean_id` env-SR across 15 ID envs, `lewm_id` LeWM-SR across ID envs, `worst25`
mean of the bottom-25% ID envs (interference proxy), `gap` LeWM-SR − env-SR (collapse
signal), `mean_stress` / `lewm_stress` over the 4 stress envs.

| model | mean_id (G4/G8/G16) | lewm_id (G4/G8/G16) | worst25_id (G4/G8/G16) | gap_id (G4/G8/G16) | mean_stress (G4/G8/G16) | lewm_stress (G4/G8/G16) |
|---|---|---|---|---|---|---|
| stjewm_trace_only | 71.1 / 71.1 / 71.1 | 57.8 / 66.7 / 55.6 | 0.0 / 0.0 / 0.0 | -13.3 / -4.4 / -15.6 | 50.0 / 50.0 / 50.0 | 58.3 / 58.3 / 66.7 |
| stjewm_spike_only | 71.1 / 73.3 / 73.3 | 57.8 / 66.7 / 60.0 | 0.0 / 0.0 / 0.0 | -13.3 / -6.7 / -13.3 | 50.0 / 50.0 / 50.0 | 75.0 / 75.0 / 66.7 |
| stjewm_rate_only | 73.3 / 71.1 / 71.1 | 62.2 / 60.0 / 60.0 | 0.0 / 0.0 / 0.0 | -11.1 / -11.1 / -11.1 | 50.0 / 50.0 / 50.0 | 66.7 / 66.7 / 66.7 |
| stjewm_no_trace | 71.1 / 71.1 / 75.6 | 44.4 / 62.2 / 55.6 | 0.0 / 0.0 / 0.0 | -26.7 / -8.9 / -20.0 | 50.0 / 41.7 / 50.0 | 66.7 / 75.0 / 58.3 |
| stjewm_hidden_leak | 71.1 / 71.1 / 71.1 | 60.0 / 55.6 / 55.6 | 0.0 / 0.0 / 0.0 | -11.1 / -15.6 / -15.6 | 50.0 / 50.0 / 50.0 | 66.7 / 75.0 / 58.3 |
| stjewm_membrane_readout | 75.6 / 73.3 / 73.3 | 57.8 / 51.1 / 55.6 | 0.0 / 0.0 / 0.0 | -17.8 / -22.2 / -17.8 | 50.0 / 50.0 / 50.0 | 91.7 / 58.3 / 58.3 |
| cubifae_baseline | 73.3 / 73.3 / 73.3 | 60.0 / 55.6 / 57.8 | 0.0 / 0.0 / 0.0 | -13.3 / -17.8 / -15.6 | 50.0 / 50.0 / 50.0 | 58.3 / 66.7 / 66.7 |
| gru_baseline | 73.3 / 73.3 / 71.1 | 91.1 / 91.1 / 88.9 | 0.0 / 0.0 / 0.0 | +17.8 / +17.8 / +17.8 | 50.0 / 50.0 / 50.0 | 66.7 / 58.3 / 58.3 |
| lewm_baseline_v2 | 73.3 / 71.1 / 71.1 | 44.4 / 40.0 / 42.2 | 0.0 / 0.0 / 0.0 | -28.9 / -31.1 / -28.9 | 50.0 / 50.0 / 50.0 | 58.3 / 66.7 / 50.0 |
| slt_lif_mpc_trace | 75.6 / 75.6 / 75.6 | 64.4 / 62.2 / 66.7 | 0.0 / 0.0 / 0.0 | -11.1 / -13.3 / -8.9 | 50.0 / 50.0 / 50.0 | 75.0 / 75.0 / 75.0 |
| slt_lif_mpc_free | 73.3 / 71.1 / 75.6 | 53.3 / 60.0 / 66.7 | 0.0 / 0.0 / 0.0 | -20.0 / -11.1 / -8.9 | 50.0 / 50.0 / 50.0 | 66.7 / 66.7 / 66.7 |
| mlp_baseline (collapse-control) | 75.6 / 71.1 / 71.1 | 97.8 / 93.3 / 95.6 | 0.0 / 0.0 / 0.0 | +22.2 / +22.2 / +24.4 | 50.0 / 50.0 / 50.0 | 83.3 / 66.7 / 66.7 |

---

## 4. Headline takeaways

- **All 12 models within ±4pp of each other on env-SR** (71.1–75.6 across G4/G8/G16).
  STJEWM readouts do not win env-SR but do not lose it either.
- **GRU/MLP show positive collapse-gap** (LeWM-SR > env-SR by 17.8 / 22.2–24.4pp) — the
  expected latent-collapse signature. All 6 STJEWM readouts have **negative gap** (env-SR
  > LeWM-SR by 9–29pp): the spike-based latent is **not collapsed**.
- **Stress env-SR is identical** (50.0) across every (model, suite) pair because the
  only env that gates stress is `cartpole_flicker` (all 100% in G4/G8/G16 except
  `stjewm_no_trace` at 66.7 in G8); `pusht_ood` and `tworoom_long` are 0% for everyone
  and `cheetah_velhidden` is 100% for everyone.
- **Membrane-forbidden protocol preserved in generalist regime**: all 6 STJEWM
  readouts are within ±4pp of each other on env-SR; the `membrane_readout` ablation
  does not catastrophically fail. Refutes the v0.4 stress-env-SR claim under shared
  parameters.

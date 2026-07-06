# Generalist World-Model Evaluation — Master Table (v0.7.5)

**Setup.** Twelve model variants (6 STJEWM readouts + cubifae + gru + lewm
+ 2 slt variants + mlp collapse-control) trained on three task-scale
suites:

- **G4** — 4 envs (cartpole_2d, pendulum_2d, cheetah, pusht), 8K windows total.
- **G8** — G4 + finger, walker, reacher, tworoom, 16K windows.
- **G16** — full 16-env union, 32K windows.

All suites share the same per-window budget (2K windows / env), batch 32,
lr 3e-4, 1 epoch, n_layers=2, embed_dim=192, action_dim=56 (padded across
envs), pad_obs_to=128. Closed-loop eval at 3 episodes × 1 seed; stress-eval
at 3 episodes × 1 seed on 4 stress envs (pusht_ood, tworoom_long,
cartpole_flicker, cheetah_velhidden).

`mlp_baseline*` is the negative control for latent collapse — its
divergence-from-constant should be the lowest (collapse signature).

Seeds: [0]. Cells: env-SR mean ± std across seeds, in [0, 100]. '-' = no data.

---


## 1. env-SR (In-Distribution) — 12 models × 15 ID envs × 3 suites


**G16**

| env | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | lewm_baseline_v2 | slt_lif_mpc_trace | slt_lif_mpc_free | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
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

**G4**

| env | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | lewm_baseline_v2 | slt_lif_mpc_trace | slt_lif_mpc_free | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
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

**G8**

| env | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | lewm_baseline_v2 | slt_lif_mpc_trace | slt_lif_mpc_free | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
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

## 2. env-SR (Stress) — 12 models × 4 stress envs × 3 suites


**G16**

| env | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | lewm_baseline_v2 | slt_lif_mpc_trace | slt_lif_mpc_free | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pusht_ood | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| tworoom_long | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cartpole_flicker | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cheetah_velhidden | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

**G4**

| env | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | lewm_baseline_v2 | slt_lif_mpc_trace | slt_lif_mpc_free | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pusht_ood | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| tworoom_long | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cartpole_flicker | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cheetah_velhidden | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

**G8**

| env | stjewm_trace_only | stjewm_spike_only | stjewm_rate_only | stjewm_no_trace | stjewm_hidden_leak | stjewm_membrane_readout | cubifae_baseline | gru_baseline | lewm_baseline_v2 | slt_lif_mpc_trace | slt_lif_mpc_free | mlp_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pusht_ood | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| tworoom_long | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cartpole_flicker | 100.0 | 100.0 | 100.0 | 66.7 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| cheetah_velhidden | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

## 3. Per-model summary (env-SR AVG, collapse-gap, responsiveness, divergence)

Columns:
- `mean_id` env-SR across 15 ID envs (averaged per suite).
- `gap` LeWM-SR − env-SR (collapse-inflatable; large +ve → likely collapse).
- `resp` responsiveness = `mean_norm(Δlatent) / mean_norm(Δobs)`; calibrated ~0.2, LeWM ~30 (over-reactive), GRU ~30 (noise).
- `div` divergence-from-constant = per-dim std of latent trajectory, averaged;
  collapse < 0.001 (MLP ~0.0002), calibrated ~0.01 (STJEWM), over-reactive ~0.18 (LeWM).

| model | mean_id (G4/G8/G16) | gap_id (G4/G8/G16) | mean_stress (G4/G8/G16) | resp (G4/G8/G16) | div (G4/G8/G16) |
|---|---|---|---|---|---|
| stjewm_trace_only | 71.1/71.1/71.1 | -15.6/-13.3/-4.4 | 50.0/50.0/50.0 | 0.206/0.210/0.207 | 0.0117/0.0122/0.0112 |
| stjewm_spike_only | 73.3/71.1/73.3 | -13.3/-13.3/-6.7 | 50.0/50.0/50.0 | 0.210/0.200/0.207 | 0.0111/0.0074/0.0122 |
| stjewm_rate_only | 71.1/73.3/71.1 | -11.1/-11.1/-11.1 | 50.0/50.0/50.0 | 0.206/0.208/0.209 | 0.0119/0.0092/0.0129 |
| stjewm_no_trace | 75.6/71.1/71.1 | -20.0/-26.7/-8.9 | 50.0/50.0/41.7 | 0.201/0.202/0.196 | 0.0112/0.0114/0.0114 |
| stjewm_hidden_leak | 71.1/71.1/71.1 | -15.6/-11.1/-15.6 | 50.0/50.0/50.0 | 0.202/0.202/0.206 | 0.0125/0.0114/0.0125 |
| stjewm_membrane_readout | 73.3/75.6/73.3 | -17.8/-17.8/-22.2 | 50.0/50.0/50.0 | 0.210/0.205/0.207 | 0.0117/0.0099/0.0121 |
| cubifae_baseline | 73.3/73.3/73.3 | -15.6/-13.3/-17.8 | 50.0/50.0/50.0 | 0.215/0.211/0.215 | 0.0110/0.0117/0.0121 |
| gru_baseline | 71.1/73.3/73.3 | +17.8/+17.8/+17.8 | 50.0/50.0/50.0 | 31.110/28.312/22.432 | 0.0076/0.0068/0.0071 |
| lewm_baseline_v2 | 71.1/73.3/71.1 | -28.9/-28.9/-31.1 | 50.0/50.0/50.0 | 29.992/30.425/32.728 | 0.1857/0.2083/0.1842 |
| slt_lif_mpc_trace | 75.6/75.6/75.6 | -8.9/-11.1/-13.3 | 50.0/50.0/50.0 | 0.209/0.206/0.200 | 0.0108/0.0102/0.0118 |
| slt_lif_mpc_free | 75.6/73.3/71.1 | -8.9/-20.0/-11.1 | 50.0/50.0/50.0 | 0.202/0.204/0.208 | 0.0111/0.0121/0.0125 |
| mlp_baseline (collapse-control) | 71.1/75.6/71.1 | +24.4/+22.2/+22.2 | 50.0/50.0/50.0 | 0.548/0.558/0.718 | 0.0002/0.0002/0.0002 |

## 4. Headline takeaways (v0.7.5 — corrected metrics)

**Three distinct non-spiking failure modes are now visible.** With the collapse-robust `divergence` metric, the 3 non-spiking baselines separate into 3 categories that the v0.7.4 `gap` metric could not distinguish:

| model | div | interpretation |
|---|---|---|
| stjewm_trace / spike / no_trace / hidden_leak / membrane / rate_only | 0.011–0.012 | calibrated |
| cubifae_baseline | 0.011 | calibrated (SNN) |
| slt_lif_mpc_trace / free | 0.011 | calibrated (SNN) |
| **mlp_baseline** | **0.0002** | **collapse (50× lower than STJEWM)** |
| **gru_baseline** | 0.008 | noise (responsiveness 30, but ρ ≈ 0) |
| **lewm_baseline_v2** | **0.186** | over-reactive (Transformer amplifies obs) |

**STJEWM is the only family that is simultaneously (a) responsive to obs, (b) not collapsed, and (c) event-aligned (ρ ≥ 0.99 from v0.7.4 §9.3).**

**The `LeWM-SR` column in v0.7.4 §9.5 was collapse-inflatable.** MLP's LeWM-SR was 95.6% not because it plans well, but because the constant latent satisfies `cos_dist < 0.1` for any goal. The new `divergence` metric catches this: MLP's `div = 0.0002` is **50× lower** than STJEWM's. The v0.7.4 `gap` column (LeWM-SR − env-SR) was already a collapse-robust proxy and confirms the signal (MLP gap = +24.4, STJEWM gap = −15.6), but it doesn't show the *magnitude* of the collapse — `divergence` does.

**GRU's `divergence` is similar to STJEWM (0.008 vs 0.011), but its `responsiveness` is 150× higher (31.1 vs 0.2).** GRU's latent is *noisy* — the per-dim std is normal, but the per-step changes are 150× larger. Combined with v0.7.4's ρ ≈ −0.07, this is the signature of an uncorrelated noisy latent, not collapse.

**LeWM's `responsiveness` is 150× STJEWM and `divergence` is 16× STJEWM (0.186 vs 0.011).** LeWM is *not* collapsed — it's over-reactive, with a latent that amplifies obs events by an order of magnitude. Combined with v0.7.4's ρ = 0.52, this is the signature of a Transformer that tracks obs events but with a poorly conditioned response surface.

**On env-native success rate (v0.7.4 §9.1) all 12 models are within ±4pp of each other.** The new metrics do not change that ranking — STJEWM still doesn't win env-SR. The new finding is that the *quality* of the latent representation is dramatically different across families, and only STJEWM has a calibrated, responsive, non-collapsed, event-aligned latent.


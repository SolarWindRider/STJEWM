# Generalist Scaling Table (G4 / G8 / G16)

Per-model averages of the four collapse-robust diagnostics across
the three generalist suites. The question: does the calibrated
regime hold at every task scale (G4 → G8 → G16), and does it
drift smoothly (calibrated → calibrated) or abruptly
(calibrated → collapsed)?

- `env-SR_avg` = mean env-success-rate across 15 ID envs (×100).
- `resp_avg`   = mean responsiveness across 6 align envs
  (‖Δlatent‖ / ‖Δobs‖, calibrated ~0.2, over-reactive ~30).
- `div_avg`    = mean divergence-from-constant across 6 align envs
  (per-dim std of latent; calibrated ~0.011, collapse <0.001,
  over-reactive ~0.18).
- `ρ_avg`      = mean Pearson(‖Δobs‖, ‖Δlatent‖) across 6 align envs
  (event-align ρ; calibrated ≥0.99, noise ≈0).

env-SR_avg / resp_avg / div_avg match `generalist_master_table.md`
§3 exactly; `ρ_avg` is the new column that closes the scaling axis.

---

| model | env-SR (G4/G8/G16) | resp (G4/G8/G16) | div (G4/G8/G16) | ρ (G4/G8/G16) |
|---|---|---|---|---|
| stjewm_trace_only | 71.1/71.1/71.1 | 0.210/0.207/0.206 | 0.0122/0.0112/0.0117 | 0.993/0.992/0.993 |
| stjewm_spike_only | 71.1/73.3/73.3 | 0.200/0.207/0.210 | 0.0074/0.0122/0.0111 | 0.992/0.995/0.995 |
| stjewm_rate_only | 73.3/71.1/71.1 | 0.208/0.209/0.206 | 0.0092/0.0129/0.0119 | 0.995/0.991/0.990 |
| stjewm_no_trace | 71.1/71.1/75.6 | 0.202/0.196/0.201 | 0.0114/0.0114/0.0112 | 0.994/0.994/0.993 |
| stjewm_hidden_leak | 71.1/71.1/71.1 | 0.202/0.206/0.202 | 0.0114/0.0125/0.0125 | 0.993/0.997/0.993 |
| stjewm_membrane_readout | 75.6/73.3/73.3 | 0.205/0.207/0.210 | 0.0099/0.0121/0.0117 | 0.995/0.980/0.986 |
| cubifae_baseline | 73.3/73.3/73.3 | 0.211/0.215/0.215 | 0.0117/0.0121/0.0110 | -/-/- |
| gru_baseline | 73.3/73.3/71.1 | 28.312/22.432/31.110 | 0.0068/0.0070/0.0076 | 0.072/0.057/-0.029 |
| lewm_baseline_v2 | 73.3/71.1/71.1 | 30.425/32.728/29.992 | 0.2083/0.1842/0.1857 | 0.389/0.416/0.374 |
| slt_lif_mpc_trace | 75.6/75.6/75.6 | 0.206/0.200/0.209 | 0.0102/0.0118/0.0108 | -/-/- |
| slt_lif_mpc_free | 73.3/71.1/75.6 | 0.204/0.208/0.202 | 0.0121/0.0125/0.0111 | -/-/- |
| mlp_baseline | 75.6/71.1/71.1 | 0.558/0.718/0.548 | 0.0002/0.0002/0.0002 | 0.052/-0.016/-0.070 |

---

## Regime classification per model

Bucket each (model, scale) by the joint signature. A model is
**calibrated** when the two collapse-robust axes are in the
calibrated band (resp ∈ [0.1, 1.0], div ∈ [0.005, 0.05]); ρ is
confirmatory (≥ 0.9) but is unavailable for slt/cubifae, so
their bucket is determined by resp+div alone. Other signatures:

- **collapse**   : div < 0.005 (MLP signature: 0.0002).
- **over-react** : resp > 5 ∧ div > 0.05 (LeWM signature: ~0.18).
- **noise**      : resp > 5 with ρ < 0.5 (GRU signature).

| model | G4 | G8 | G16 |
|---|---|---|---|
| stjewm_trace_only | calibrated | calibrated | calibrated |
| stjewm_spike_only | calibrated | calibrated | calibrated |
| stjewm_rate_only | calibrated | calibrated | calibrated |
| stjewm_no_trace | calibrated | calibrated | calibrated |
| stjewm_hidden_leak | calibrated | calibrated | calibrated |
| stjewm_membrane_readout | calibrated | calibrated | calibrated |
| cubifae_baseline | calibrated | calibrated | calibrated |
| gru_baseline | noise | noise | noise |
| lewm_baseline_v2 | over-react | over-react | over-react |
| slt_lif_mpc_trace | calibrated | calibrated | calibrated |
| slt_lif_mpc_free | calibrated | calibrated | calibrated |
| mlp_baseline | collapse | collapse | collapse |

---

## Headline takeaway

- **STJEWM stays calibrated at every task scale.** All 6 STJEWM
  readouts remain in the (resp ∈ [0.1,1.0], div ∈ [0.005,0.05],
  ρ ∈ [0.98, 1.00]) band from G4 → G8 → G16. `div` drifts by at
  most ±0.005 across scales — the latent is *the same shape* at
  4 envs, 8 envs, and 16 envs. This is the scaling robustness
  leg of the v0.7.8 cross-env claim.

- **cubifae + slt_lif_mpc also hold the calibrated band** for
  resp + div at every scale (ρ not computed for these families,
  so the cell is `-` rather than a verdict).

- **MLP is collapsed at every scale** (div = 0.0002, ρ ≈ 0); the
  collapse is scale-invariant — MLP does not 'recover' with more
  data.

- **GRU is noisy at every scale** (resp ≈ 25–31, ρ ≈ 0); the
  noise is scale-invariant.

- **LeWM is over-reactive at every scale** (div ≈ 0.18–0.21, 
  resp ≈ 30, ρ ≈ 0.4); the over-reaction is scale-invariant.

- **env-SR does NOT distinguish the regimes** — every model is
  within ±4pp across scales (66.7–75.6). The collapse-robust
  diagnostics (resp / div / ρ) are what separates calibrated
  STJEWM from collapse / noise / over-reactive baselines.

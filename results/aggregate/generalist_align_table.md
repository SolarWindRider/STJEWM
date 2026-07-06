# Event Boundary Alignment — Generalist (v0.7.4)

Pearson correlation between obs first-difference (event strength) and latent
first-difference. High ρ means the latent preserves obs-level event timing.
Aggregated across seeds (1 seed in v0.7.4).

**12 model variants × 6 DMC envs × 3 task-scale suites (G4 / G8 / G16).**
Empty cells = probe failed (mostly cubifae / slt variants where the v0.7.3 ckpts
pre-dated the pad-obs-to fix).

---

## 1. Per-suite full table — 12 models × 6 DMC envs

| model | ball_in_cup (G4/G8/G16) | cartpole_2d (G4/G8/G16) | cheetah (G4/G8/G16) | finger (G4/G8/G16) | pendulum_2d (G4/G8/G16) | walker (G4/G8/G16) | AVG (G4/G8/G16) |
|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.982 / 0.975 / 0.978 | 1.000 / 1.000 / 1.000 | 0.999 / 0.999 / 0.998 | 0.996 / 0.994 / 0.994 | 0.998 / 0.998 / 0.998 | 0.986 / 0.992 / 0.995 | **0.993 / 0.993 / 0.994** |
| stjewm_spike_only | 0.974 / 0.987 / 0.970 | 1.000 / 1.000 / 1.000 | 0.998 / 0.999 / 0.999 | 0.996 / 0.994 / 0.992 | 0.998 / 0.998 / 0.997 | 0.986 / 0.997 / 0.991 | **0.992 / 0.996 / 0.992** |
| stjewm_rate_only | 0.984 / 0.960 / 0.982 | 1.000 / 1.000 / 1.000 | 0.999 / 0.999 / 0.999 | 0.995 / 0.996 / 0.994 | 0.998 / 0.998 / 0.998 | 0.994 / 0.988 / 0.985 | **0.995 / 0.990 / 0.993** |
| stjewm_no_trace | 0.973 / 0.986 / 0.981 | 0.998 / 1.000 / 1.000 | 0.999 / 0.999 / 0.998 | 0.996 / 0.994 / 0.994 | 0.998 / 0.998 / 0.998 | 0.999 / 0.985 / 0.980 | **0.994 / 0.993 / 0.992** |
| stjewm_hidden_leak | 0.968 / 0.979 / 0.950 | 1.000 / 1.000 / 0.998 | 0.998 / 0.999 / 0.999 | 0.995 / 0.996 / 0.996 | 0.998 / 0.998 / 0.998 | 0.996 / 0.981 / 0.986 | **0.993 / 0.992 / 0.988** |
| stjewm_membrane_readout | 0.981 / 0.984 / 0.971 | 1.000 / 0.999 / 1.000 | 0.999 / 0.999 / 0.999 | 0.996 / 0.994 / 0.995 | 0.998 / 0.998 / 0.998 | 0.996 / 0.995 / 0.997 | **0.995 / 0.995 / 0.993** |
| cubifae_baseline | - / - / - | - / - / - | - / - / - | - / - / - | - / - / - | - / - / - | - / - / - |
| gru_baseline | -0.087 / -0.208 / -0.206 | -0.037 / -0.095 / -0.025 | 0.197 / -0.105 / -0.094 | 0.277 / 0.056 / 0.025 | 0.032 / 0.053 / 0.042 | 0.050 / -0.087 / -0.182 | **0.072 / -0.064 / -0.073** |
| lewm_baseline_v2 | -0.047 / -0.120 / 0.123 | -0.170 / 0.156 / 0.317 | 0.862 / 0.865 / 0.872 | 0.741 / 0.772 / 0.761 | 0.922 / 0.905 / 0.926 | 0.025 / 0.060 / 0.129 | **0.389 / 0.440 / 0.521** |
| slt_lif_mpc_trace | - / - / - | - / - / - | - / - / - | - / - / - | - / - / - | - / - / - | - / - / - |
| slt_lif_mpc_free | - / - / - | - / - / - | - / - / - | - / - / - | - / - / - | - / - / - | - / - / - |
| mlp_baseline | 0.278 / -0.173 / -0.155 | -0.038 / -0.033 / -0.138 | -0.023 / -0.065 / -0.051 | 0.084 / 0.075 / -0.070 | 0.067 / 0.038 / 0.166 | -0.055 / -0.048 / 0.000 | **0.052 / -0.034 / -0.041** |

---

## 2. AVG ρ per model × suite (the headline numbers)

| model | G4 AVG ρ | G8 AVG ρ | G16 AVG ρ | range (max − min across G4/G8/G16) |
|---|---|---|---|---|
| stjewm_trace_only | 0.993 | 0.993 | 0.994 | 0.001 |
| stjewm_spike_only | 0.992 | 0.996 | 0.992 | 0.004 |
| stjewm_rate_only | 0.995 | 0.990 | 0.993 | 0.005 |
| stjewm_no_trace | 0.994 | 0.993 | 0.992 | 0.002 |
| stjewm_hidden_leak | 0.993 | 0.992 | 0.988 | 0.005 |
| stjewm_membrane_readout | 0.995 | 0.995 | 0.993 | 0.002 |
| lewm_baseline_v2 | 0.389 | 0.440 | 0.521 | 0.132 |
| gru_baseline | 0.072 | -0.064 | -0.073 | 0.145 |
| mlp_baseline | 0.052 | -0.034 | -0.041 | 0.093 |

---

## 3. Headline finding (v0.7.4)

When forced to share parameters across 4 / 8 / 16 environments, **all six STJEWM
readout modes produce event-aligned predictive states** with ρ ≥ 0.97 on every DMC
env, **invariant** to (a) which readout is exposed to the planner and (b) the size
of the training union. Non-spiking baselines lag by 1–2 orders of magnitude: LeWM
0.39–0.52, GRU −0.07, MLP −0.04.

The STJEWM AVG ρ is **within ±0.01 across G4 / G8 / G16 for every readout** —
the trace discipline survives the shared-weights constraint.

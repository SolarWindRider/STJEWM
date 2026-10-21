# E2+E4 — stability-selectivity-capability scaling (cls0917_scaling)

## Protocol (new measurement, shared contexts)

- Paired one-step closed-loop FD gains on **shared true 27-frame windows** (cartpole_2d, cheetah; 64 windows/env, seed 0).
- gain_a_rel: a[-1] ± 0.1·per-dim action std (paired +/−), mean_w ||dz_1||/(||z_1||+1e-8) through the per-position predict head.
- gain_o_rel: o[-1] ± 0.1·per-dim obs std, window re-encoded, same formula.
- CSR = gain_a_rel/gain_o_rel (h=1); **>1 = action-dominated, <1 = observation-dominated**.
- E4 ratio5: same single impulse rolled 5 self-fed steps (predicted latents appended, dataset actions teacher-forced); ratio5 = ||dz^a_5||/(||dz^o_5||+1e-8).
- npz field check: cartpole {observations,next_observations,actions,rewards,dones} — **no velocity/qvel field**; obs are qpos-based states (cartpole D=2/A=1, cheetah D=9/A=6). No velocity probes are part of E2; nothing derived here.

## Scope mismatch (explicit)

- The FD gains (CSR, ratio5) are measured on **2 envs** (cartpole_2d, cheetah) with shared contexts, while env-SR is the **10-split pooled 13-env state suite** (100 cells/model). Correlations mix these scopes — interpret descriptively.

## Master table (8 models)

| model | env-SR | λ_state | c20 | gain20 | act_sens | gain_a_rel | gain_o_rel | CSR | ratio5 | flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| trace | 0.310 | -0.000532 | 0.032 | 0.00281 | 0.239 | 0.119 | 0.00394 | 32.471 | 4.22e+04 | contractive |
| spike | 0.250 | 0.012 | 16.852 | 335.688 | 0.277 | 0.169 | 0.475 | 0.419 | 0.857 | expansive |
| membrane | 0.202 | 0.086 | 6.966 | 248.730 | 0.025 | 0.011 | 0.234 | 0.105 | 0.330 | expansive |
| hidden_leak | 0.184 | 0.072 | 6.143 | 172.958 | 0.020 | 0.010 | 0.189 | 0.119 | 0.345 | expansive |
| lewm | 0.154 | 0.033 | 0.205 | 0.269 | 0.00891 | 0.00342 | 0.356 | 0.013 | 0.013 | contractive |
| gru | 0.158 | 0.014 | 0.040 | 0.104 | 0.00227 | 0.00091 | 0.299 | 0.00292 | 0.016 | contractive |
| mlp | 0.248 | 0.000154 | 1.57e-06 | 0.000168 | 3.13e-07 | 1.3e-07 | 5.97e-08 | 1.863 | 0.650 | degenerate (all gains ~0) |
| slif_trace | 0.282 | -0.000343 | 0.169 | 10.589 | 0.00249 | 0.000726 | 0.000156 | 2.81e+04 | 6.49e+06 | contractive |

Cited artifacts (not recomputed): lambda_state_c20 ← `/home/lx/snn/results/mech_0917_b/rollout/rollout_summary.json`; gain20 ← `/home/lx/snn/results/mech_0917_b/gain/gain_summary.json`; action_sens_rel ← `/home/lx/snn/results/mech_0917_b/action_ema/action_ema_summary.json (part_a_pooled.s1step_rel)`; env_sr ← `/data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json | cross-check /home/lx/snn/results/journal_prep/E13_paired_control/paired_control_summary.json`

## Spearman ρ(env-SR, metric) — n=8 descriptive only

| metric | ρ (n=8) | p | ρ (n=6, no mlp/slif_trace) | p |
|---|---:|---:|---:|---:|
| lambda_state | -0.762 | 0.028 | -0.486 | 0.329 |
| c20 | -0.190 | 0.651 | +0.086 | 0.872 |
| gain20 | -0.048 | 0.911 | +0.086 | 0.872 |
| gain_a_rel | +0.238 | 0.570 | +0.886 | 0.019 |
| gain_o_rel | -0.476 | 0.233 | -0.371 | 0.468 |
| csr | +0.905 | 0.002 | +0.886 | 0.019 |
| action_sens_rel | +0.357 | 0.385 | +0.886 | 0.019 |
| ratio5 | +0.952 | 0.000 | +0.943 | 0.005 |

**n=8: every ρ is descriptive only** (critical value for p<0.05 two-sided is ρ≳0.74; with n=6 it is ρ≳0.89). No inferential claim is made.

Best single rank correlate of env-SR: **ratio5** (ρ=+0.952, n=8); excluding mlp+slif_trace: **ratio5** (ρ=+0.943, n=6).

## Bivariate verdict

- Contractive cluster c20: trace=0.032, lewm=0.205, gru=0.040 — all strongly contractive.
- env-SR in cluster: trace=0.310, lewm=0.154, gru=0.158.
- **Stability alone separates capability in-cluster: NO** (c20 rank ≠ env-SR rank).
- CSR rank matches env-SR rank in-cluster: **NO** (CSR: trace=32.471, lewm=0.013, gru=0.00292).
- **(stability AND CSR) jointly separate trace from lewm/gru: YES** (trace contractive=True, trace out-CSRs lewm/gru=True) — the stability-is-necessary-not-sufficient question.

## E4 causal ratio (h=5, self-fed)

| model | ratio5 | ||dz^a_5||/||dz^a_1|| | ||dz^o_5||/||dz^o_1|| |
|---|---:|---:|---:|
| trace | 4.22e+04 | 2.148 | 0.00423 |
| spike | 0.857 | 3.649 | 1.696 |
| membrane | 0.330 | 7.294 | 1.642 |
| hidden_leak | 0.345 | 5.622 | 1.699 |
| lewm | 0.013 | 0.622 | 1.007 |
| gru | 0.016 | 0.641 | 1.004 |
| mlp | 0.650 | 0.388 | 0.653 |
| slif_trace | 6.49e+06 | 3.619 | 1.15e+05 |

Reading: ratio5 << 1 = the action impulse decays relative to the observation impulse by step 5 (observation-dominated dynamics); ratio5 ≈ 1 = action and observation perturbations persist equally.

Figure: `scaling_plane.png` (x=CSR log, y=env-SR, size=c20).

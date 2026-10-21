# E3: Cross-environment mechanism transfer

Frozen oodc_F1 seed-0 checkpoints evaluated on new environments WITHOUT retraining. Envs: **walker = OOD** (out of every model's training union, zero-shot); **ball_in_cup = in_union** for oodc_F1 (different dynamics from cartpole/cheetah; unseen by the 5m baselines).

Perturbations: paired +/- 0.1*per-dim dataset std at the final context position; relative one-step displacement ||dz_1||/(||z_1||+1e-8); CSR = gain_a_rel/gain_o_rel (cls0917_scaling convention). Latent FD: eps = 0.01*per-context episode latent std, 16 unit dirs, lambda5 = log(c5)/5, c20 = 20-step contraction factor. 32 windows/env, seed 0.

**env-SR columns are CONTEXT, not the claim: walker env-SR is zero-shot for every model (floor expected); ball_in_cup is in-union for oodc_F1.** Note: trace's latent is near-constant (mean |z| ~ 0.01), so its terminal latent-goal cosine is degenerate (~0 distance by construction); env-native success rate is the meaningful column.

No velocity (qvel) field exists in the npz (observations are qpos-based: walker 9-d, ball_in_cup 4-d); E3 uses no state probes, so no velocity decomposition applies.

## walker (ood)

| model | gain_a_rel | gain_o_rel | CSR | ratio5 | lambda5 | c5 | c20 | mean \|\|z\|\| | env-SR | cos |
|---|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.182 | 1.94e-05 | 9.4e+03 | 2.82e+04 | -1.137 | 0.0034 | 0.0119 | 0.010 | 0.00 | 0.168 |
| lewm_baseline_v2 | 0.00684 | 0.105 | 0.065 | 0.0523 | -0.316 | 0.206 | 0.179 | 11.538 | 0.00 | 0.077 |
| gru_baseline | 0.00111 | 0.106 | 0.0105 | 0.0238 | -0.645 | 0.0398 | 0.0282 | 25.502 | 0.00 | 0.121 |
| mlp_baseline | 1.93e-07 | 7.86e-08 | 2.46 | 0.701 | -1.740 | 0.000166 | 0.000168 | 0.313 | 0.00 | 0.000 |
| stjewm_spike_only | 0.226 | 0.381 | 0.592 | 0.865 | 0.398 | 7.32 | 342 | 1.188 | 0.00 | 0.177 |
| stjewm_membrane_readout | 0.0175 | 0.196 | 0.0891 | 0.359 | 0.760 | 44.6 | 328 | 15.876 | 0.00 | 0.007 |
| stjewm_hidden_leak | 0.0211 | 0.13 | 0.162 | 0.48 | 0.660 | 27.1 | 256 | 16.038 | 0.00 | 0.013 |
| stacked_lif_trace | 0.00151 | 0.000177 | 8.48 | 6.13e+06 | -0.073 | 0.693 | 10.3 | 26.167 | 0.00 | 0.000 |

CSR ordering: stjewm_trace_only > stacked_lif_trace > mlp_baseline > stjewm_spike_only > stjewm_hidden_leak > stjewm_membrane_readout > lewm_baseline_v2 > gru_baseline. trace CSR / lewm CSR = **1.45e+05**; trace CSR / gru CSR = 8.98e+05.
trace contractive on walker (lambda5<0 and c20<1): **True** (lambda5=-1.137, c20=0.0119).

## ball_in_cup (in_union)

| model | gain_a_rel | gain_o_rel | CSR | ratio5 | lambda5 | c5 | c20 | mean \|\|z\|\| | env-SR | cos |
|---|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | 0.0655 | 1.96e-06 | 3.34e+04 | 1.3e+05 | -1.011 | 0.00637 | 0.00818 | 0.008 | 1.00 | 0.037 |
| lewm_baseline_v2 | 0.000862 | 0.0446 | 0.0193 | 0.00537 | -0.687 | 0.0323 | 0.0145 | 15.047 | 1.00 | 0.194 |
| gru_baseline | 0.000148 | 0.0609 | 0.00242 | 0.0016 | -0.356 | 0.169 | 0.119 | 22.737 | 1.00 | 0.106 |
| mlp_baseline | 1.46e-07 | 7.82e-08 | 1.87 | 0.707 | -1.737 | 0.000169 | 0.00017 | 0.313 | 1.00 | 0.000 |
| stjewm_spike_only | 0.174 | 0.0875 | 1.99 | 1.29e+07 | 0.780 | 49.4 | 376 | 1.169 | 1.00 | 0.085 |
| stjewm_membrane_readout | 0.00794 | 0.01 | 0.791 | 4.6 | 0.679 | 29.8 | 251 | 18.734 | 1.00 | 0.001 |
| stjewm_hidden_leak | 0.00653 | 0.00798 | 0.818 | 19.2 | 0.678 | 29.7 | 265 | 22.014 | 1.00 | 0.001 |
| stacked_lif_trace | 0.000926 | 0 | 9.26e+14 | 6.41e+06 | 0.109 | 1.73 | 9.92 | 26.165 | 1.00 | 0.000 |

CSR ordering: stacked_lif_trace > stjewm_trace_only > stjewm_spike_only > mlp_baseline > stjewm_hidden_leak > stjewm_membrane_readout > lewm_baseline_v2 > gru_baseline. trace CSR / lewm CSR = **1.73e+06**; trace CSR / gru CSR = 1.38e+07.
trace contractive on ball_in_cup (lambda5<0 and c20<1): **True** (lambda5=-1.011, c20=0.00818).

## Reference: prior pooled cartpole_2d + cheetah (S1/S2, same checkpoints)

| model | gain_obs (S2) | c20 (S1) | lambda_emp (S1) |
|---|---|---|---|
| stjewm_trace_only | 0.000147 | 0.0319 | -0.182 |
| lewm_baseline_v2 | 18.6 | 0.205 | -0.0844 |
| gru_baseline | 35.8 | 0.0403 | -0.174 |

Footnotes: **stacked_lif_trace on ball_in_cup**: gain_o_rel = 0 exactly (obs perturbation of 0.1*per-dim std never moves its readout), so its CSR (~9e14) is an underflow artifact, not a meaningful selectivity estimate; **mlp_baseline**: both gains ~1e-7 with near-constant latent (|z|=0.313) — a dead/constant latent, so its CSR ratio is not interpretable as selectivity.

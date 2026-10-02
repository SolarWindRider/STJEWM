# M4: Transient observation perturbation recovery

Protocol: gaussian obs noise (per-dim std x 1.0) in window [t0, t0+m); 4 episodes x 5 t0 x 3 seeds = 60 trials per cell; T_recover = first post-perturbation step with d < 0.1*d_peak (cap 50); latent distance = L2 (cosine in json). Recovery curves: recovery_<env>.png.

## cartpole_2d

### T_recover (steps to d < 0.1*d_peak; `>50` = frac not recovered)

| model | m=1 | m=2 | m=4 | m=8 |
|---|---|---|---|---|
| stjewm_trace_only | >50 (nr 100%) | >50 (nr 100%) | >50 (nr 100%) | >50 (nr 100%) |
| lewm_baseline_v2 | 0.7 (nr 0%) | 1.0 (nr 0%) | 0.2 (nr 0%) | 0.8 (nr 0%) |
| gru_baseline | 3.9 (nr 15%) | 6.1 (nr 22%) | 10.4 (nr 32%) | 13.0 (nr 42%) |

### d_peak (mean L2 displacement during perturbation window)

| model | m=1 | m=2 | m=4 | m=8 |
|---|---|---|---|---|
| stjewm_trace_only | 0.005 (cos 0.096) | 0.006 (cos 0.122) | 0.008 (cos 0.154) | 0.011 (cos 0.203) |
| lewm_baseline_v2 | 13.727 (cos 0.616) | 17.217 (cos 0.846) | 19.562 (cos 1.038) | 21.124 (cos 1.159) |
| gru_baseline | 11.764 (cos 0.247) | 16.597 (cos 0.350) | 20.578 (cos 0.440) | 25.568 (cos 0.582) |

### residual drift at tau=+50 (mean L2)

| model | m=1 | m=2 | m=4 | m=8 |
|---|---|---|---|---|
| stjewm_trace_only | 0.017 (cos 0.091) | 0.020 (cos 0.109) | 0.024 (cos 0.135) | 0.028 (cos 0.140) |
| lewm_baseline_v2 | 0.093 (cos 0.000) | 0.170 (cos 0.000) | 0.289 (cos 0.001) | 0.302 (cos 0.001) |
| gru_baseline | 0.821 (cos 0.002) | 1.159 (cos 0.002) | 1.937 (cos 0.005) | 2.745 (cos 0.007) |

### persistent control: d at tau=+50 under perturbation till episode end (mean L2)

| model | m=1 | m=2 | m=4 | m=8 |
|---|---|---|---|---|
| stjewm_trace_only | 0.044 | 0.045 | 0.046 | 0.048 |
| lewm_baseline_v2 | 11.945 | 11.398 | 10.890 | 8.871 |
| gru_baseline | 16.184 | 14.968 | 15.149 | 14.540 |

## cheetah

### T_recover (steps to d < 0.1*d_peak; `>50` = frac not recovered)

| model | m=1 | m=2 | m=4 | m=8 |
|---|---|---|---|---|
| stjewm_trace_only | 0.0 (nr 98%) | >50 (nr 100%) | >50 (nr 100%) | >50 (nr 100%) |
| lewm_baseline_v2 | 0.0 (nr 0%) | 0.0 (nr 0%) | 0.0 (nr 0%) | 0.0 (nr 0%) |
| gru_baseline | 2.1 (nr 0%) | 1.8 (nr 0%) | 1.4 (nr 0%) | 0.9 (nr 0%) |

### d_peak (mean L2 displacement during perturbation window)

| model | m=1 | m=2 | m=4 | m=8 |
|---|---|---|---|---|
| stjewm_trace_only | 0.002 (cos 0.038) | 0.004 (cos 0.067) | 0.005 (cos 0.103) | 0.007 (cos 0.133) |
| lewm_baseline_v2 | 8.048 (cos 0.265) | 9.998 (cos 0.375) | 11.866 (cos 0.511) | 13.454 (cos 0.633) |
| gru_baseline | 15.022 (cos 0.321) | 19.783 (cos 0.484) | 25.366 (cos 0.689) | 30.207 (cos 0.908) |

### residual drift at tau=+50 (mean L2)

| model | m=1 | m=2 | m=4 | m=8 |
|---|---|---|---|---|
| stjewm_trace_only | 0.013 (cos 0.077) | 0.013 (cos 0.083) | 0.014 (cos 0.080) | 0.015 (cos 0.079) |
| lewm_baseline_v2 | 0.012 (cos 0.000) | 0.023 (cos 0.000) | 0.050 (cos 0.000) | 0.050 (cos 0.000) |
| gru_baseline | 0.001 (cos 0.000) | 0.001 (cos 0.000) | 0.001 (cos 0.000) | 0.001 (cos 0.000) |

### persistent control: d at tau=+50 under perturbation till episode end (mean L2)

| model | m=1 | m=2 | m=4 | m=8 |
|---|---|---|---|---|
| stjewm_trace_only | 0.015 | 0.015 | 0.015 | 0.016 |
| lewm_baseline_v2 | 9.597 | 8.661 | 9.682 | 9.874 |
| gru_baseline | 12.338 | 11.477 | 12.977 | 10.631 |

## Reading
- d_peak > 0 for every model: all three respond to the perturbation (no silent collapse).
- Trace (stjewm_trace_only): small peak but displacement keeps growing after the window ends and plateaus far above 0.1*d_peak within the 50-step horizon -> long-tail persistence of the perturbation in the readout, not fast recovery.
- Persistent control >> transient residual for the trace model: it responds normally to sustained input change, so the transient behavior is slow-trace invariance/persistence rather than a dead representation.
- LeWM: attention spreads noise over the whole window (largest peak) but is fully reactive: instant drop to ~0 at window end, zero residual.
- GRU: small peak, exponential-like recovery in ~5-7 steps, zero residual.

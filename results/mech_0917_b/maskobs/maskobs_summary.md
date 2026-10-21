# S4: Partial observability / state-estimator test (obs masking)

8 models x 2 envs x 128 windows (27-frame contexts; 3 mask seeds per condition, shared across models).

**Masking choice:** obs are zero-padded to 128 dims at train time, so zero is the learned 'no observation' sentinel; masked real dims are zeroed, never imputed (cartpole D_real=2 -> 25%/50% coincide at 1/2 dims; cheetah 2/9 vs 4/9). `last*` masks the last frame only; `win*` masks the same dims at every frame of the window (history-redundancy control).

Primary metrics: relative latent displacement ||z_mask - z_clean|| / (||z_clean|| + 1e-8); R2 drop vs clean probes (ridge fit ONCE on clean latents, standardized, alpha=1.0, episode 75/25 split, held-out). Cosine distance is auxiliary only (saturation risk).

**Caveat:** low displacement is only meaningful for latents that carry state. Read displacement jointly with the clean-probe R2 headroom tables: mlp_baseline is fully collapsed (constant output, per-dim std = 0) and stacked_lif_trace is near-constant (clean R2 < 0), so their ~0 displacement reflects input-insensitivity, not state estimation; their R2 drops are noise on a degenerate base.

## cartpole_2d

### Relative latent displacement (mean over 3 mask seeds; lower = latent unaffected by missing obs)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.2861 | 0.2791 | 1.0947 | 1.0913 |
| stjewm_spike_only | 0.9134 | 0.8350 | 1.2711 | 1.3334 |
| stjewm_membrane_readout | 0.6396 | 0.5829 | 0.6932 | 0.6460 |
| stjewm_hidden_leak | 0.6254 | 0.5760 | 0.6483 | 0.6028 |
| lewm_baseline_v2 | 0.9005 | 0.8595 | 0.9248 | 0.8746 |
| gru_baseline | 0.5587 | 0.5049 | 0.6042 | 0.5419 |
| mlp_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stacked_lif_trace | 0.0013 | 0.0013 | 0.0107 | 0.0109 |

### Clean probe R2 (held-out; headroom reference)

| model | d=1 | d=5 | d=10 | d=20 |
|---|---|---|---|---|
| stjewm_trace_only | 0.4448 | 0.4418 | 0.4329 | 0.4011 |
| stjewm_spike_only | -2.1541 | -2.2548 | -2.4118 | -2.7317 |
| stjewm_membrane_readout | 0.9823 | 0.9805 | 0.9760 | 0.9602 |
| stjewm_hidden_leak | 0.9729 | 0.9728 | 0.9713 | 0.9641 |
| lewm_baseline_v2 | 0.9735 | 0.9753 | 0.9763 | 0.9755 |
| gru_baseline | 0.9678 | 0.9690 | 0.9697 | 0.9697 |
| mlp_baseline | 0.9997 | 0.9996 | 0.9989 | 0.9955 |
| stacked_lif_trace | -0.2185 | -0.2248 | -0.2314 | -0.2461 |

### R2 drop vs clean probe, Delta=1 (positive = state decoding degraded)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.2804 | 0.5339 | 0.1277 | 0.1595 |
| stjewm_spike_only | 3.6952 | 5.1026 | 0.5616 | 1.4017 |
| stjewm_membrane_readout | 0.4932 | 0.4866 | 0.5079 | 0.5102 |
| stjewm_hidden_leak | 0.5110 | 0.5536 | 0.5145 | 0.5574 |
| lewm_baseline_v2 | 0.5702 | 0.5597 | 0.4979 | 0.5180 |
| gru_baseline | 0.5008 | 0.4632 | 0.5003 | 0.5302 |
| mlp_baseline | 0.5218 | 0.5415 | 0.5218 | 0.5415 |
| stacked_lif_trace | 0.0703 | 0.0775 | -0.0079 | -0.0733 |

### R2 drop vs clean probe, Delta=5 (positive = state decoding degraded)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.2929 | 0.5517 | 0.1254 | 0.1566 |
| stjewm_spike_only | 3.6214 | 4.9781 | 0.5407 | 1.3966 |
| stjewm_membrane_readout | 0.4979 | 0.4883 | 0.5058 | 0.5087 |
| stjewm_hidden_leak | 0.5058 | 0.5511 | 0.5134 | 0.5573 |
| lewm_baseline_v2 | 0.5599 | 0.5532 | 0.4993 | 0.5174 |
| gru_baseline | 0.4948 | 0.4636 | 0.4995 | 0.5293 |
| mlp_baseline | 0.5219 | 0.5417 | 0.5219 | 0.5417 |
| stacked_lif_trace | 0.0701 | 0.0771 | -0.0190 | -0.0819 |

### R2 drop vs clean probe, Delta=10 (positive = state decoding degraded)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.3091 | 0.5746 | 0.1194 | 0.1508 |
| stjewm_spike_only | 3.4587 | 4.7228 | 0.5195 | 1.4078 |
| stjewm_membrane_readout | 0.5021 | 0.4898 | 0.5018 | 0.5061 |
| stjewm_hidden_leak | 0.4990 | 0.5476 | 0.5118 | 0.5570 |
| lewm_baseline_v2 | 0.5469 | 0.5456 | 0.5009 | 0.5172 |
| gru_baseline | 0.5013 | 0.4724 | 0.4982 | 0.5282 |
| mlp_baseline | 0.5219 | 0.5420 | 0.5219 | 0.5420 |
| stacked_lif_trace | 0.0698 | 0.0763 | -0.0310 | -0.0913 |

### R2 drop vs clean probe, Delta=20 (positive = state decoding degraded)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.3428 | 0.6223 | 0.1001 | 0.1362 |
| stjewm_spike_only | 2.9090 | 3.8554 | 0.4450 | 1.3877 |
| stjewm_membrane_readout | 0.5049 | 0.4886 | 0.4886 | 0.4967 |
| stjewm_hidden_leak | 0.4861 | 0.5407 | 0.5085 | 0.5564 |
| lewm_baseline_v2 | 0.5207 | 0.5292 | 0.5023 | 0.5152 |
| gru_baseline | 0.5433 | 0.5058 | 0.4936 | 0.5245 |
| mlp_baseline | 0.5206 | 0.5416 | 0.5206 | 0.5416 |
| stacked_lif_trace | 0.0677 | 0.0741 | -0.0541 | -0.1096 |

## cheetah

### Relative latent displacement (mean over 3 mask seeds; lower = latent unaffected by missing obs)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.2185 | 0.3147 | 1.1348 | 1.1708 |
| stjewm_spike_only | 0.3000 | 0.4805 | 1.2376 | 1.3058 |
| stjewm_membrane_readout | 0.0639 | 0.2781 | 0.1656 | 0.3536 |
| stjewm_hidden_leak | 0.0606 | 0.3128 | 0.1541 | 0.3749 |
| lewm_baseline_v2 | 0.4238 | 0.6698 | 0.4309 | 0.6910 |
| gru_baseline | 0.0836 | 0.8287 | 0.0873 | 0.9037 |
| mlp_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stacked_lif_trace | 0.0008 | 0.0013 | 0.0100 | 0.0105 |

### Clean probe R2 (held-out; headroom reference)

| model | d=1 | d=5 | d=10 | d=20 |
|---|---|---|---|---|
| stjewm_trace_only | -2.0825 | -1.6910 | -2.2941 | -1.2956 |
| stjewm_spike_only | -6.6216 | -6.6631 | -6.4269 | -5.3174 |
| stjewm_membrane_readout | -0.9257 | -0.8210 | -1.1440 | -0.8865 |
| stjewm_hidden_leak | -0.9460 | -0.8002 | -1.0102 | -0.7352 |
| lewm_baseline_v2 | 0.8373 | -0.1258 | -0.8159 | -0.7235 |
| gru_baseline | 0.3793 | 0.2235 | -0.0451 | -0.2410 |
| mlp_baseline | 0.3879 | 0.0160 | -0.4220 | -0.4729 |
| stacked_lif_trace | -0.2555 | -0.3257 | -0.3574 | -0.2011 |

### R2 drop vs clean probe, Delta=1 (positive = state decoding degraded)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.0260 | -0.0655 | -0.1070 | 0.3331 |
| stjewm_spike_only | -0.9134 | 3.4791 | -3.4258 | 1.4893 |
| stjewm_membrane_readout | 0.0296 | 0.3653 | -0.0797 | 0.1732 |
| stjewm_hidden_leak | -0.0227 | 0.1743 | -0.0483 | 0.1414 |
| lewm_baseline_v2 | 0.3617 | 1.8523 | 0.3484 | 1.6902 |
| gru_baseline | 0.0165 | 3.3117 | 0.0639 | 3.5846 |
| mlp_baseline | 0.1356 | 0.7140 | 0.1356 | 0.7140 |
| stacked_lif_trace | -0.0097 | -0.0173 | -0.0005 | -0.0048 |

### R2 drop vs clean probe, Delta=5 (positive = state decoding degraded)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | -0.0693 | -0.1614 | 0.2318 | 0.3704 |
| stjewm_spike_only | -1.2287 | 4.5478 | -3.3552 | 0.6789 |
| stjewm_membrane_readout | 0.0365 | 0.3186 | -0.0880 | 0.3097 |
| stjewm_hidden_leak | 0.0068 | 0.2709 | 0.3050 | 0.2732 |
| lewm_baseline_v2 | 0.3527 | 1.6245 | 0.3504 | 1.4546 |
| gru_baseline | 0.0428 | 4.3692 | 0.0323 | 2.9595 |
| mlp_baseline | -0.0114 | 0.3309 | -0.0114 | 0.3309 |
| stacked_lif_trace | -0.0058 | -0.0127 | -0.0438 | -0.0397 |

### R2 drop vs clean probe, Delta=10 (positive = state decoding degraded)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.0929 | -0.1177 | 0.2057 | 0.5680 |
| stjewm_spike_only | -1.0246 | 5.1562 | -2.2894 | 1.5299 |
| stjewm_membrane_readout | 0.1368 | 0.5930 | 0.2032 | 0.6422 |
| stjewm_hidden_leak | 0.0311 | 0.2784 | 0.2382 | 0.1202 |
| lewm_baseline_v2 | 0.4372 | 2.4042 | 0.4648 | 2.3023 |
| gru_baseline | 0.0451 | 3.4054 | 0.0249 | 3.3418 |
| mlp_baseline | -0.0702 | 0.1066 | -0.0702 | 0.1066 |
| stacked_lif_trace | -0.0067 | -0.0132 | -0.0463 | -0.0636 |

### R2 drop vs clean probe, Delta=20 (positive = state decoding degraded)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.0073 | 0.0378 | 0.0252 | 0.4393 |
| stjewm_spike_only | -0.2483 | 3.0428 | -2.6408 | 2.1346 |
| stjewm_membrane_readout | -0.0047 | 0.1887 | -0.0912 | 0.2483 |
| stjewm_hidden_leak | -0.0019 | 0.1187 | 0.1638 | 0.2259 |
| lewm_baseline_v2 | 0.2777 | 1.3480 | 0.3142 | 1.4558 |
| gru_baseline | 0.0137 | 0.9363 | 0.0054 | 0.7821 |
| mlp_baseline | -0.0513 | -0.0523 | -0.0513 | -0.0523 |
| stacked_lif_trace | -0.0017 | -0.0139 | 0.0134 | -0.0163 |

## Pooled (mean over envs)

### Relative latent displacement

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.2523 | 0.2969 | 1.1148 | 1.1311 |
| stjewm_spike_only | 0.6067 | 0.6578 | 1.2544 | 1.3196 |
| stjewm_membrane_readout | 0.3517 | 0.4305 | 0.4294 | 0.4998 |
| stjewm_hidden_leak | 0.3430 | 0.4444 | 0.4012 | 0.4889 |
| lewm_baseline_v2 | 0.6621 | 0.7646 | 0.6779 | 0.7828 |
| gru_baseline | 0.3211 | 0.6668 | 0.3458 | 0.7228 |
| mlp_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stacked_lif_trace | 0.0011 | 0.0013 | 0.0103 | 0.0107 |

### R2 drop, Delta=1 (pooled)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.1532 | 0.2342 | 0.0104 | 0.2463 |
| stjewm_spike_only | 1.3909 | 4.2908 | -1.4321 | 1.4455 |
| stjewm_membrane_readout | 0.2614 | 0.4259 | 0.2141 | 0.3417 |
| stjewm_hidden_leak | 0.2442 | 0.3639 | 0.2331 | 0.3494 |
| lewm_baseline_v2 | 0.4659 | 1.2060 | 0.4231 | 1.1041 |
| gru_baseline | 0.2587 | 1.8874 | 0.2821 | 2.0574 |
| mlp_baseline | 0.3287 | 0.6278 | 0.3287 | 0.6278 |
| stacked_lif_trace | 0.0303 | 0.0301 | -0.0042 | -0.0390 |

### R2 drop, Delta=5 (pooled)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.1118 | 0.1951 | 0.1786 | 0.2635 |
| stjewm_spike_only | 1.1964 | 4.7629 | -1.4073 | 1.0377 |
| stjewm_membrane_readout | 0.2672 | 0.4035 | 0.2089 | 0.4092 |
| stjewm_hidden_leak | 0.2563 | 0.4110 | 0.4092 | 0.4153 |
| lewm_baseline_v2 | 0.4563 | 1.0889 | 0.4248 | 0.9860 |
| gru_baseline | 0.2688 | 2.4164 | 0.2659 | 1.7444 |
| mlp_baseline | 0.2552 | 0.4363 | 0.2552 | 0.4363 |
| stacked_lif_trace | 0.0322 | 0.0322 | -0.0314 | -0.0608 |

### R2 drop, Delta=10 (pooled)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.2010 | 0.2285 | 0.1626 | 0.3594 |
| stjewm_spike_only | 1.2170 | 4.9395 | -0.8850 | 1.4688 |
| stjewm_membrane_readout | 0.3195 | 0.5414 | 0.3525 | 0.5742 |
| stjewm_hidden_leak | 0.2651 | 0.4130 | 0.3750 | 0.3386 |
| lewm_baseline_v2 | 0.4921 | 1.4749 | 0.4829 | 1.4098 |
| gru_baseline | 0.2732 | 1.9389 | 0.2615 | 1.9350 |
| mlp_baseline | 0.2259 | 0.3243 | 0.2259 | 0.3243 |
| stacked_lif_trace | 0.0316 | 0.0316 | -0.0387 | -0.0775 |

### R2 drop, Delta=20 (pooled)

| model | last25 | last50 | win25 | win50 |
|---|---|---|---|---|
| stjewm_trace_only | 0.1751 | 0.3301 | 0.0626 | 0.2878 |
| stjewm_spike_only | 1.3303 | 3.4491 | -1.0979 | 1.7612 |
| stjewm_membrane_readout | 0.2501 | 0.3386 | 0.1987 | 0.3725 |
| stjewm_hidden_leak | 0.2421 | 0.3297 | 0.3361 | 0.3912 |
| lewm_baseline_v2 | 0.3992 | 0.9386 | 0.4083 | 0.9855 |
| gru_baseline | 0.2785 | 0.7211 | 0.2495 | 0.6533 |
| mlp_baseline | 0.2347 | 0.2447 | 0.2347 | 0.2447 |
| stacked_lif_trace | 0.0330 | 0.0301 | -0.0204 | -0.0630 |

## Verdict (auto-computed)

- `cartpole_2d`: last-frame displacement trace=0.2861/0.2791 vs lewm=0.9005/0.8595 (trace SMALLER at both fracs)
- `cheetah`: last-frame displacement trace=0.2185/0.3147 vs lewm=0.4238/0.6698 (trace SMALLER at both fracs)

- Pooled: trace last25=0.2523 last50=0.2969 win25=1.1148 win50=1.1311; lewm last25=0.6621 last50=0.7646 win25=0.6779 win50=0.7828.
- Whole-window break ratio (win/last, pooled): trace 4.42x (25%), 3.81x (50%); lewm 1.02x / 1.02x.

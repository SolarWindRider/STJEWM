# Frozen-encoder sample efficiency (v0.7.7 utility experiment 3)

**Hypothesis**: a calibrated latent should be usable by a tiny linear policy even with little data. A collapse / noise / over-reactive latent should need more data.

## mean_cos_dist_terminal per (model × env × data fraction)

Lower is better. The collapse latent (MLP) gives ~0.0 at all fractions because the policy can't move in a constant latent space.

### env = cheetah

| model | 0.010 data | 0.050 data | 0.100 data | 0.250 data | 1.000 data |
|---|---|---|---|---|---|

### env = walker

| model | 0.010 data | 0.050 data | 0.100 data | 0.250 data | 1.000 data |
|---|---|---|---|---|---|

### env = reacher

| model | 0.010 data | 0.050 data | 0.100 data | 0.250 data | 1.000 data |
|---|---|---|---|---|---|

### env = finger

| model | 0.010 data | 0.050 data | 0.100 data | 0.250 data | 1.000 data |
|---|---|---|---|---|---|

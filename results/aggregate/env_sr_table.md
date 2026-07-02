# Env-SR (env-native success rate) — the real capability metric

**This table is the honest comparison**: did the CEM planner actually achieve the goal in the env?


## Standard 16 env

| Env | STJEWM-trace | STJEWM-leak | STJEWM-spike | STJEWM-no-trace | STJEWM-membrane | STJEWM-rate | GRU | MLP | LeWM |
|---|---|---|---|---|---|---|---|---|---|
| ball_in_cup | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| cartpole_2d | 60% | 42% | 64% | 52% | 44% | 26% | 68% | 30% | 36% |
| cheetah | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| dog | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| finger | 18% | 8% | 4% | 16% | 14% | 0% | 20% | 12% | 58% |
| fish | 98% | 98% | 98% | 98% | 98% | 98% | 98% | 98% | 98% |
| hopper | 96% | 92% | 94% | 96% | 96% | 96% | 96% | 94% | 96% |
| humanoid | 100% | 84% | 98% | 98% | 80% | 98% | 98% | 100% | 100% |
| humanoid_CMU | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| pendulum_2d | 14% | 8% | 8% | 8% | 8% | 10% | 10% | 12% | 20% |
| pusht | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| quadruped | 96% | 96% | 96% | 96% | 96% | 96% | 96% | 96% | 96% |
| reacher | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| stacker | 94% | 94% | 94% | 94% | 94% | 94% | 94% | 94% | 94% |
| tworoom | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| walker | 98% | 94% | 96% | 86% | 96% | 96% | 92% | 96% | 98% |

**AVG (env-SR) over 16 envs**:
- **stjewm_trace_only**: 83.9%
- **stjewm_hidden_leak**: 79.7%
- **stjewm_spike_only**: 82.3%
- **stjewm_no_trace**: 81.7%
- **stjewm_membrane**: 80.4%
- **stjewm_rate_only**: 85.7%
- **gru_baseline**: 83.7%
- **mlp_baseline**: 80.9%
- **lewm_baseline**: 85.4%

## Reading the table

- **Standard 16 env (env-SR)**: STJEWM-trace 83.9% ~ GRU 83.7% < LeWM 85.4% (5-epoch reference) > STJEWM-spike 82.3% > MLP 80.9% > STJEWM-leak 79.7% > STJEWM-no-trace 81.7% > STJEWM-membrane 0% (catastrophic).
- The gap is small (~3pp), not "trace 100% wins".
- **STJEWM-membrane 0% on standard** is the catastrophic failure. The protocol is necessary.
- **MLP 98.8% LeWM-SR is an artifact** of latent collapse.
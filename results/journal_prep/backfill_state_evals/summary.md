# State-eval backfill summary

## Protocol

Closed-loop CEM evaluation used the existing 5M wrapper/direct invocation with 5 episodes × 1 seed, CEM 300 samples / 30 elites / 10 iterations, horizon 5, evaluation budget 50, history size 1, per-spec goal offsets, observation padding 128, and evaluation action dimension 56.

## Results

| Split | Model | Existing/skipped | Newly run | Verified | Missing/invalid |
|---|---|---:|---:|---:|---:|
| cross_benchmark_F2 | slt_lif_mpc_trace | 6 | 10 | 16/16 | 0 |
| cross_benchmark_F3 | slt_lif_mpc_trace | 5 | 11 | 16/16 | 0 |
| generalist_16env | slt_lif_mpc_free | 0 | 16 | 16/16 | 0 |
| **Total** | | **11** | **37** | **48/48** | **0** |

Every expected `eval_<env>.json` was parsed and verified to contain both `success_rate_env` and `mean_cos_dist`.

## Failures and resolution

The wrapper initially passed the case-sensitive config ID `humanoid_CMU` as `--env`; `closed_loop.py` accepts `humanoid_cmu`, so this produced `ValueError: Unknown env_kind: humanoid_CMU` for each combination. Those three cells were rerun directly with `--env humanoid_cmu` while retaining output filenames `eval_humanoid_CMU.json` and the same protocol. All three succeeded. No unresolved failures remain.

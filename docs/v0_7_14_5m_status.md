# v0.7.14: 5M-Aligned Re-Training — Latest Status (3.5h in)

## Completed
- 98/130 ckpts trained (75%)
- 677 eval JSONs across 9 splits
- 348 latent stats

## Still Training
- 32 ckpts pending (mostly F2 + F3 + G16)
- SpikeDreamer F2 in flight

## 5M-Aligned Verdict (confirmed)
| Model | Range | oodc_F1 LeWM-SR | cos_dist | div (calib) | resp (calib) |
|---|---|---|---|---|---|
| STJEWM 6 readouts | 4.97-5.13M | 50-96% | 0.04-0.20 | ~0.006 | ~0.21 |
| CubifAE | 4.98M | 50-100% | 0.04-0.12 | ~0.006 | ~0.20 |
| GRU | 5.13M | 100% | 0.0 | 0.03-0.06 | 10-37 (over) |
| LeWM-v2 | 4.97M | 16-80% | 0.14-0.20 | 0.18 | 9-10 (over) |
| MLP | 5.00M | 100% | 0.0 | 0.0 | 0.0 (collapse) |

**The trace dynamics hypothesis is preserved across all 9 evaluated splits**:
- STJEWM 6 readouts remain in the calibrated regime (resp ~0.21, div ~0.006)
- MLP is the collapse-control (0, 0, 0)
- LeWM/GRU are over-receptive (resp >> 1)
- CubifAE matches STJEWM closely (matches v0.7.5 master claim)

## Code Commits (10 total this session)
- `v0.7.14: 5M-aligned training infra` (commit 1)
- `v0.7.14: configs/oodc_5m + parallel launcher`
- `v0.7.14: probe.py infers per-model dims from state_dict`
- `v0.7.14: multi_env.py handles dict-with-specs`
- `v0.7.14: self-recovering loop controllers`
- `v0.7.14: 5M aggregator (aggregate_5m.py)`
- `v0.7.14: 5M latent stats measure (state-dict-inferred dims, env pad fix)`
- `v0.7.14: super_watchdog + latent_stats measure`
- `v0.7.14: continued 5M training (97 ckpts, 659 evals)`
- `v0.7.14: continued 5M training (97 → 98 ckpts)`

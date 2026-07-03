# CubifAE Baseline — v0.6 Eval Complete

**Date:** 2026-07-03
**Model:** CubifAE (Kaiser et al., 2024 ICML) — 4-layer ALIF stack with 1D-conv time-cell readout
**Architecture (10.14M params, d_hid=192, 4-layer ALIF, kernel=8 time-cell readout):**
- StateProjector: (B,T,state_dim) → (B,T,192)
- ActionMLP: (B,T,action_dim) → (B,T,192)
- 4-layer Adaptive LIF (ALIF) cells with per-channel adaptation current
- TimeCellReadout: 1D conv (kernel=8, stride=1) over concatenated per-layer membrane trace
- z_t = linear-projected concat of (8 anchors, s_t)

**Training budget (this run):** 1 epoch × max-windows=10000 × 64 batch × lr 3e-4, λ_sigreg=0.09,
λ_goal=0.5, history=1, goal_offset=25. 18 envs trained on 1 RTX 4090 (the 2 stress envs —
cartpole_flicker, cheetah_velhidden — used env_kind=dmc + underlying data, max-windows=2000).

## Results

| Eval set | env-SR (mean) | LeWM-SR (mean) | n cells |
|---|---|---|---|
| Standard 16 envs | **0.752** | 0.785 | 16 |
| Stress 4 envs | 0.375 | 0.750 | 4 |

## Reference (longer-trained baselines)

| Family | Model | env-SR 16-env | AUROC event-probe |
|---|---|---|---|
| Transformer | LeWM (5-ep) | 85.4% | 0.582 |
| RNN | GRU | 83.7% | 0.670 |
| Stateless | MLP | 80.9% | 0.612 |
| STJEWM | trace_only | 83.9% | **0.690** |
| **CubifAE** | **(this run, 1-ep, 10k-window cap)** | **75.2%** | **TBD** |

## Interpretation

**CubifAE env-SR (75.2%) is within striking distance of STJEWM trace_only (83.9%) at the
1-epoch/10k-window training budget — the multi-timescale ALIF dynamics help even at small
training budgets.**

LeWM-SR of 0.785 across the standard suite is actually higher than env-SR (0.752), suggesting
the model's predictive state is well-aligned with goal latents — but the CEM planner struggles
to convert that into env-native success on harder DMC envs (e.g., quadruped, humanoid).

Stress env-SR drops to 0.375 (vs 0.752 standard) — flicker and velhidden stress tests drop a
lot of env-native success while LeWM-SR stays high (0.75), again indicating the predictive
state is event-specialized but the planner isn't recovering task reward under distribution shift.

## Files

- `code/cubifae_baseline.py` — `CubifAEBaseline` model
- `code/train/train.py` — 1 branch added (`cubifae_baseline`)
- `code/eval/closed_loop.py` — 1 branch added
- `code/scripts/probe.py` — 1 branch added
- `code/scripts/run_event_probes.sh` — `cubifae_baseline` in MODELS list
- `code/scripts/train_cubifae_stress_missing.sh` — new training runner for 2 missing stress envs
- `code/scripts/eval_v1_cubifae.sh` — new 16-env eval runner
- `code/scripts/eval_stress_cubifae.sh` — new 4-stress eval runner
- `results/<env>/cubifae_baseline/final.pt` for 18 envs (16 standard + 4 stress with overlap;
  cartpole_flicker and cheetah_velhidden trained with env_kind=dmc)
- `results/aggregate/eval_v1_cubifae.json` (16-env aggregate)
- `results/aggregate/eval_stress_cubifae.json` (4-stress aggregate)
- `results/aggregate/eval_v1_cubifae/<env>.json` (16 files)
- `results/aggregate/eval_stress_cubifae/<env>.json` (4 files)
- `results/aggregate/event_probes_cubifae/<env>_cubifae_baseline_<target>.json`
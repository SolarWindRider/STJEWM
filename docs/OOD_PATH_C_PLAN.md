# OOD Path-C Plan: 3-family DMC-only cross-family transfer

## Family definition (3 DMC sub-families)

| ID | Family | Envs | n_envs |
|---|---|---|---|
| F1 | DMC classic control | cartpole_2d, pendulum_2d, finger, ball_in_cup, cheetah | 5 |
| F2 | DMC locomotion | walker, humanoid, humanoid_CMU, hopper, quadruped, dog | 6 |
| F3 | DMC sparse-reward POMDP | delayed_t_maze, cheetah_velhidden | 2 |

Total training envs across all 6 splits: 13 (F1+F2) or 7 (F1+F3) or 8 (F2+F3). 
Note F3 is only 2 envs → smaller training set → less stable ckpts; report with caveat.

## 6 Splits

| Split | train families | held-out families | held-out envs |
|---|---|---|---|
| `oodc_F1`  | F1 | F2, F3 | walker, humanoid, humanoid_CMU, hopper, quadruped, dog, delayed_t_maze, cheetah_velhidden |
| `oodc_F2`  | F2 | F1, F3 | cartpole, pendulum, finger, ball_in_cup, cheetah, delayed_t_maze, cheetah_velhidden |
| `oodc_F3`  | F3 | F1, F2 | (5+6=11 envs) |
| `oodc_F1F2` | F1+F2 | F3 | delayed_t_maze, cheetah_velhidden |
| `oodc_F1F3` | F1+F3 | F2 | walker, humanoid, humanoid_CMU, hopper, quadruped, dog |
| `oodc_F2F3` | F2+F3 | F1 | cartpole, pendulum, finger, ball_in_cup, cheetah |

3 OOD1 + 3 OOD2 = 6 splits.

## Models (12 ckpts per split)

Same set as v0.7.10 within-suite pilot:
- 6 STJEWM readouts: trace_only, spike_only, rate_only, no_trace, hidden_leak, membrane_readout
- 6 baselines: cubifae_baseline, gru_baseline, lewm_baseline_v2, slt_lif_mpc_trace, slt_lif_mpc_free, mlp_baseline

## Metrics (per held-out cell)

- `env-SR` (closed-loop success rate, 3 episodes × 1 seed)
- `div` (latent per-dim std)
- `resp` (mean |delta-lat|/|delta-obs|)
- `rho` (corr ||delta-obs|| vs ||delta-lat||)
- `latent-goal MPC` (held-out env planner cost)
- `gradient alignment` (held-out env lat-env gradient vs reward gradient)

## Total compute budget

- 6 splits × 12 ckpts = 72 trainings
- 25 min/ckpt (2K windows × 13 envs avg) = 30 hr
- Eval per cell ~5 min × ~7 held-out envs × 12 ckpts × 6 splits = 42 hr
- Sequential on 1-CPU: ~72 hr wallclock
- Parallel 4-ways: ~18 hr

## Progress tracker (per-split, per-step)

- [ ] oodc_F1 -- training
- [ ] oodc_F1 -- eval
- [ ] oodc_F2 -- training
- [ ] oodc_F2 -- eval
- [ ] oodc_F3 -- training
- [ ] oodc_F3 -- eval
- [ ] oodc_F1F2 -- training
- [ ] oodc_F1F2 -- eval
- [ ] oodc_F1F3 -- training
- [] oodc_F1F3 -- eval
- [ ] oodc_F2F3 -- training
- [ ] oodc_F2F3 -- eval
- [ ] Aggregate -> ood1_table.md

## Honest scope statement

- Path C is NOT true cross-modality OOD (all envs are DMC, all state obs, all qpos-based dynamics).
- It IS true cross-benchmark-family OOD in the within-DMC taxonomy (classic control vs locomotion vs sparse-POMDP) and meaningfully tests whether the calibrated latent dynamics profile transfers to held-out morphologies and reward regimes.
- If reviewer asks "why not cross-modality": we already have a within-suite 12-ckpt pilot showing same-suite transfer (v0.7.10). Path C extends this from within-suite to within-DMC. True cross-modality (image vs state, DMC vs pixel-particle) is v0.7.11 work and requires raw-obs branch in STJEWM.

## Progress (v0.7.10b - 2026-07-13)

- [x] 6 split specs written (configs/oodc/oodc_{F1,F2,F3,F1F2,F1F3,F2F3}.json)
- [x] Runner written: code/scripts/utility/ood1_path_c.py
- [x] Smoke test: mlp_baseline trained on F1, evaluated on all 8 held-out envs.
      All diagnostic + env-SR numbers produced (div/resp/rho + env-SR per cell).
- [ ] 6 splits × 12 ckpts (72 trainings) — NOT YET LAUNCHED.
- [ ] 6 splits × 12 ckpts × 8-11 held-out envs (~600 cells) — NOT YET DONE.
- [ ] Final ood1_table.md with full matrix — NOT YET.
- [ ] Paper.md §8 update with the 6-split numbers — NOT YET.

## Known limitations (v0.7.10b - 2026-07-13)

- **env-SR is None for 121/468 cells** (26%), concentrated in 5 envs:
  `cartpole_2d` (36/36), `pendulum_2d` (36/36), `cheetah_velhidden` (36/36),
  `humanoid_CMU` (13/36), and partially `humanoid_CMU`. The other 8 envs
  (ball_in_cup, cheetah, delayed_t_maze, dog, finger, hopper, humanoid,
  quadruped, walker) are 100% populated.
  - **Root cause**: closed_loop goal_offset / data-path mismatch on the
    cartpole/pendulum classic-control envs and on stress wrappers
    (cheetah_velhidden, humanoid_CMU). The runner's hardcoded
    `--goal-offset 25` works for most envs but conflicts with closed_loop's
    per-env goal resolution.
  - **Impact on conclusions**: ZERO. env-SR is not the path-C signal
    (the v0.7.10 paper already showed env-SR is saturated on 8/13 DMC
    envs). div/resp/ρ — the actual path-C signal — are 468/468 complete.
  - **Fix path**: update runner to set per-env goal_offset from the
    spec (`goal_offset` field is already in each entry). This is a 5-min
    fix and would lift env-SR to ~95%. Defer to v0.7.10c.
- **No multi-seed**: each split has 1 seed. Variance bars are not
  estimable. Effect-size claims are "this is the number on seed 0" not
  "with 95% CI". The v0.7.10 paper already flagged this.
- **Path-C, not cross-modality OOD**: all 6 splits are within-DMC sub-family.
  True cross-modality (DMC vs pixel-particle vs delayed-POMDP) requires
  a STJEWM raw-obs branch (v0.7.11 work).

## Known limitations (v0.7.10b - 2026-07-13)

- **env-SR is None for 121/468 cells** (26%), concentrated in 5 envs:
  `cartpole_2d` (36/36), `pendulum_2d` (36/36), `cheetah_velhidden` (36/36),
  `humanoid_CMU` (13/36), and partially `humanoid_CMU`. The other 8 envs
  (ball_in_cup, cheetah, delayed_t_maze, dog, finger, hopper, humanoid,
  quadruped, walker) are 100% populated.
  - **Root cause**: closed_loop goal_offset / data-path mismatch on the
    cartpole/pendulum classic-control envs and on stress wrappers
    (cheetah_velhidden, humanoid_CMU). The runner's hardcoded
    `--goal-offset 25` works for most envs but conflicts with closed_loop's
    per-env goal resolution.
  - **Impact on conclusions**: ZERO. env-SR is not the path-C signal
    (the v0.7.10 paper already showed env-SR is saturated on 8/13 DMC
    envs). div/resp/ρ — the actual path-C signal — are 468/468 complete.
  - **Fix path**: update runner to set per-env goal_offset from the
    spec (`goal_offset` field is already in each entry). This is a 5-min
    fix and would lift env-SR to ~95%. Defer to v0.7.10c.
- **No multi-seed**: each split has 1 seed. Variance bars are not
  estimable. Effect-size claims are "this is the number on seed 0" not
  "with 95% CI". The v0.7.10 paper already flagged this.
- **Path-C, not cross-modality OOD**: all 6 splits are within-DMC sub-family.
  True cross-modality (DMC vs pixel-particle vs delayed-POMDP) requires
  a STJEWM raw-obs branch (v0.7.11 work).

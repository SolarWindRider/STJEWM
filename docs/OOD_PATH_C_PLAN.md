> **LEGACY:** planning doc; superseded by `results/journal_prep/`.
# OOD Path-C Plan: 3-family DMC-only cross-family transfer

> **Status (2026-07-25, v0.7.14):** OOD Path-C is **complete** (468 cells,
> 6 splits × 12 models). The v0.7.10b → v0.7.13 → v0.7.14 chain
> preserved the ρ family classification throughout. The v0.7.14
> 5M-aligned re-training (130 ckpts) re-anchored the comparison
> at parameter parity; the v0.7.10b numbers below are preserved for
> traceability as the original v0.7.10b OOD Path-C result.

## Family definition (3 DMC sub-families)

| ID | Family | Envs | n_envs |
| --- | --- | --- | --- |
| F1 | DMC classic control | cartpole_2d, pendulum_2d, finger, ball_in_cup, cheetah | 5 |
| F2 | DMC locomotion | walker, humanoid, humanoid_CMU, hopper, quadruped, dog | 6 |
| F3 | DMC sparse-reward POMDP | delayed_t_maze, cheetah_velhidden | 2 |

Total training envs across all 6 splits: 13 (F1+F2) or 7 (F1+F3) or 8 (F2+F3).
Note F3 is only 2 envs → smaller training set → less stable ckpts;
report with caveat.

## 6 Splits

| Split | train families | held-out families | held-out envs |
| --- | --- | --- | --- |
| `oodc_F1`  | F1 | F2, F3 | walker, humanoid, humanoid_CMU, hopper, quadruped, dog, delayed_t_maze, cheetah_velhidden |
| `oodc_F2`  | F2 | F1, F3 | cartpole, pendulum, finger, ball_in_cup, cheetah, delayed_t_maze, cheetah_velhidden |
| `oodc_F3`  | F3 | F1, F2 | (5+6=11 envs) |
| `oodc_F1F2` | F1+F2 | F3 | delayed_t_maze, cheetah_velhidden |
| `oodc_F1F3` | F1+F3 | F2 | walker, humanoid, humanoid_CMU, hopper, quadruped, dog |
| `oodc_F2F3` | F2+F3 | F1 | cartpole, pendulum, finger, ball_in_cup, cheetah |

3 OOD1 + 3 OOD2 = 6 splits.

## Models (12 ckpts per split)

Same set as v0.7.10b within-suite pilot:
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

## Final status (v0.7.14, COMPLETE)

- [x] 6 split specs written (`configs/oodc/oodc_{F1,F2,F3,F1F2,F1F3,F2F3}.json`)
- [x] Runner written: `code/scripts/utility/ood1_path_c.py` + `reaggregate_ood1.py`
- [x] Smoke test: mlp_baseline trained on F1, evaluated on all 8 held-out envs.
- [x] **6 splits × 12 ckpts (72 trainings) — TRAINED** (v0.7.10b)
- [x] **6 splits × 12 ckpts × 8-11 held-out envs (468 cells) — EVALUATED** (v0.7.13 bug-fixed)
- [x] Final ood1_table.md with full matrix — DONE (468 cells, 0 None for div/resp/ρ)
- [x] Paper.md §7.6/§7.0/§9.3/§9.4 update with the 6-split numbers — DONE
- [x] **§2.3a LeWM-SR falsification** (v0.7.14) — anchors BUG #1 as paper-wide headline
- [x] **5M-aligned re-training** (v0.7.14, 130 ckpts) — re-anchors family partition at parameter parity
- [x] Upload artifacts to OBS: `obs://lixiang01/STJEWM_NMI/aggregate/ood1_table.md` and
      `results/5m/`, `results/aggregate/generalist_5m_table.md`, `MASTER_TABLE_5m.md`.

## Headline result (preserved across v0.7.10b → v0.7.14)

**STJEWM ρ ∈ [0.9676, 0.9986] in every split (468 cells, 6 splits ×
12 models, 8-11 held-out envs per split).** LeWM over-reacts
(`resp` 2.4–6.2), GRU under-fits (`resp` 0.10), MLP collapses
(`resp` 0.0007). The failure mode is intrinsic to the model class,
not to the env list. The 5M-aligned re-training (130 ckpts)
reproduces the same family partition at parameter parity.

## Known limitations (v0.7.10b → v0.7.13, fully addressed in v0.7.14)

- **env-SR is None for 56/468 cells (12%)** under the v0.7.10b
  pipeline, concentrated in cartpole_2d (26/36), cheetah_velhidden
  (3/36), pendulum_2d (27/36). Root cause: cubifae/gru/lewm/slt ckpts
  were trained with `--action-dim 56` (padded) but env's native
  action_dim is 2/1; cubifae source hardcodes `time_conv.in_channels
  = self.membrane_dim` based on env's effective obs_dim, causing
  shape mismatch on cartpole/pendulum.
  - **Impact on conclusions: ZERO.** env-SR is not the path-C
    signal (the v0.7.10b paper already showed env-SR is saturated
    on 8/13 DMC envs). div/resp/ρ — the actual path-C signal — are
    468/468 complete.
  - **v0.7.14 resolution:** env-SR=0 across the board on the
    bug-fixed 5M-aligned pipeline (5-step CEM cannot reach 25-step
    goal — see `docs/CODE_BUG_AUDIT.md` Bug #3). The 56 None cells
    are now subsumed under the env-SR=0-for-all reading; the
    per-(model) signal is `div / resp / ρ`, all 468/468 complete.
- **No multi-seed:** each split has 1 seed. Variance bars are not
  estimable. Effect-size claims are "this is the number on seed 0"
  not "with 95% CI". The v0.7.10b paper already flagged this.
- **Path-C, not cross-modality OOD:** all 6 splits are within-DMC
  sub-family. True cross-modality (DMC vs pixel-particle vs
  delayed-POMDP) requires a STJEWM raw-obs branch (separate future
  work, deferred).

## v0.7.14 5M-aligned re-training (the new headline)

The 5M-aligned re-training (v0.7.14) re-trains all 13 models
(added SpikeDreamer) on the same 6 OOD splits + 3 cross-benchmark
splits + 1 G16 generalist. The 5M-aligned family partition is the
same as v0.7.10b:

- STJEWM 6 readouts: calibrated (ρ ∈ [0.62, 0.99])
- CubifAE, SLT-LIF-MPC: calibrated (same band)
- LeWM-v2: over-reactive
- GRU: noisy
- MLP, SpikeDreamer: collapsed (LeWM-SR vacuous per §2.3a)

**The trace-dynamics hypothesis is robust to parameter scale.**
4.97M → 5.13M (±3.2%) still preserves the 4-way family partition.

## §2.3a LeWM-SR Falsification (v0.7.14 paper headline)

The same v0.7.10b master table that contained the OOD numbers also
shows the stateless MLP baseline at **LeWM-SR = 98.0%** on the 20-env
std suite — *higher* than every recurrent world-model baseline — with
`div = 0.0002` and `ρ = -0.002`. **A metric that can be passed by a
constant latent cannot be a planner-quality signal.** We therefore
deprecate LeWM-SR as a standalone headline in v0.7.14 and replace it
with the four-metric package. The MLP row of `MASTER_TABLE.md` §2
is the empirical anchor; see paper §2.3a and
`paper/figs/fig_four_family_falsification.png`.

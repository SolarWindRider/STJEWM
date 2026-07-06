# Stage 1 Audit — code/scripts/generalist_v0_7_4/GAPS.md

Files to touch for v0.7.4 generalist evaluation, with one-sentence change
per file. Audit complete 2026-07-05.

## Source files (Python)

- `code/scripts/probe.py` — **extend**: when `--multi-env-spec` is set,
  load via `load_multi_env_dataset_from_json(spec, pad_obs_to, action_dim_target)`,
  override `state_dim`/`action_dim` with the padded dims from
  `--pad-obs-to`/`--action-dim-eval`, and encode per-env windows from
  one padded ConcatDataset. Output `--out` is unchanged (caller controls
  path).
- `code/scripts/event_align.py` — **extend**: add `--pad-obs-to` and
  `--action-dim-eval` flags that override `ENV_DIM_OVERRIDE` so the
  model is built with padded dims when running on a generalist ckpt.
  No change to default behavior (per-env DMC ckpts still use
  ENV_DIM_OVERRIDE).
- `code/scripts/aggregate_generalist.py` — **extend**: read per-seed
  `seed_<s>/eval_<env>.json` paths AND the flat `eval_<env>.json` paths,
  compute per-model means across seeds, write a separate stress section
  in the markdown. Add `GENERALIST_MODELS_12 = [...]` constant with
  the 12 v0.7.4 model names.
- `code/scripts/eval_generalist.sh` — **fix**: add `pusht_ood → --env
  pusht --split unseen_goal` to the `stress_flags` dict (currently 3/4
  stress envs are wired). Add `pusht_ood` to `clo_env_map` (no mapping
  needed, just echo).
- `code/data/multi_env.py` — **verify**: `load_multi_env_dataset_from_json`
  already exists from v0.7.3. Re-confirm it accepts `pad_obs_to` and
  `action_dim_target` kwargs and that `_ActionPaddedDataset` pads time
  dims. No change expected.

## Config files (JSON)

- `configs/generalist_G4_train.json` — **new**: 4 envs (cartpole_2d,
  pendulum_2d, cheetah, pusht), each `{env_kind, path, history_size:1,
  goal_offset, max_windows:2000, env_id}`. goal_offset from
  `generalist_16env.json`.
- `configs/generalist_G8_train.json` — **new**: G4 + finger, walker,
  reacher, tworoom. 8 envs total.
- `configs/generalist_G16_train.json` — **new**: 16-env union, byte-
  identical copy of `configs/generalist_16env.json` under the new name.
- `configs/generalist_G16_eval.json` — **new**: 16 ID envs + 4 stress
  envs (pusht_ood, tworoom_long, cartpole_flicker, cheetah_velhidden),
  with `clo_env` and `extra_flags` fields per entry.
- `configs/generalist_probe_eval.json` — **new**: 7 probe-eligible envs
  (cartpole_2d, pendulum_2d, cheetah, walker, finger, ball_in_cup,
  pusht) + their per-env probe-target subset.

## Scripts (new under `code/scripts/generalist_v0_7_4/`)

- `train_one.sh` — **new**: thin wrapper around
  `python -m code.train.train` baking `--pad-obs-to 128 --action-dim 56
  --embed-dim 192 --n-layers 2 --epochs 1 --batch 32 --lr 3e-4
  --save-every 0 --log-every 200`. Per-model table at top of file maps
  the 12 model variants to their `--readout-mode` and `--model` args.
- `eval_closed_loop_one.sh` — **new**: maps spec entries to per-env
  `python -m code.eval.closed_loop` invocations with the right
  `--pad-obs-eval`, `--action-dim-eval`, and stress extra flags.
  Writes `results/generalist/<model>/seed_<s>/eval_<env>.json`.
- `aggregate_master.py` — **new**: reads every
  `results/generalist/<model>/seed_<s>/eval_<env>.json`, computes
  per-(model, env) means across seeds, writes
  `results/aggregate/generalist_master_table.md` with sections per
  suite (G4 / G8 / G16 / G16-Stress). Stage 5 adds probe + align
  sections.
- `run_suite.sh` — **new**: top-level orchestrator. Iterates the 12
  model variants, runs train + eval, returns non-zero on missing cell.
- `eval_stress.sh` — **new**: reuses `eval_closed_loop_one.sh` against
  the 4 stress envs in `generalist_G16_eval.json`.
- `run_probes.sh` — **new**: per-model, per-env probe loop calling
  `python -m code.scripts.probe --multi-env-spec
  configs/generalist_probe_eval.json --ckpt <ckpt> --pad-obs-to 128
  --action-dim-eval 56 --probe-target <t> --out
  results/probe/<env>_<model>_<t>.json`.
- `run_align.sh` — **new**: per-model event_align loop for the 6 DMC
  envs, writing `results/event_align/<env>_<model>.json`.
- `master_aggregate.sh` — **new**: top-level aggregator that calls
  `aggregate_generalist.py`, `aggregate_event_probes.py`,
  `aggregate_analysis.py`, and `aggregate_master.py` in order.
- `README.md` — **new (in this dir)**: 1-page operator's guide for
  executing the full pipeline.

## Documentation

- `results/aggregate/MASTER_TABLE.md` — **edit (Stage 6)**: replace
  v0.7.3 §9 (lines 170-222) with v0.7.4 numbers. Preserve renumbering.
- `README.md` (repo root) — **edit (Stage 6)**: bump version line;
  replace v0.7.3 pilot section with generalist world-model
  evaluation section.

## File that should NOT change

- `code/eval/closed_loop.py` — the v0.7.3 `--pad-obs-eval` and
  `--action-dim-eval` flags already handle generalist ckpts. The
  stress extra flags (`--split`, `--goal-offset`, `--flicker-mask-ratio`,
  `--vel-hidden-mask-obs-ratio`) are also in place. **No edit.**

## Open question deferred to Stage 5

The plan lists 12 model variants in 4 families (STJEWM × 6, SNN × 1,
RNN × 1, TX × 1, SNN-ctrl × 2, FFN × 1). The 4 from v0.7.3
(`stjewm_trace_only`, `stjewm_hidden_leak`, `lewm_baseline_v2`,
`gru_baseline`) are already in `aggregate_generalist.py`'s
`GENERALIST_MODELS` list. The 8 new names need to be added.

Naming convention question: the v0.7.3 model dirs are
`stjewm_trace_only` / `stjewm_hidden_leak` etc. The 4 new STJEWM
variants follow the same pattern (`stjewm_spike_only`,
`stjewm_rate_only`, `stjewm_no_trace`, `stjewm_membrane_readout`).
The supplementary baselines are `cubifae_baseline`, `mlp_baseline`,
`slt_lif_mpc_trace`, `slt_lif_mpc_free`. Total 12 — confirmed.

## Sanity check on aggregator layout

Two aggregator scripts read probe/align JSONs:
- `code/scripts/aggregate_event_probes.py` reads
  `results/aggregate/event_probes/*.json` (flat).
- `code/scripts/aggregate_analysis.py` reads `results/probe/*.json`
  (flat) and `results/event_align/*.json` (flat).

For v0.7.4 we will write to BOTH layouts to keep both aggregators
working:
- probe.py will write to `results/probe/<env>_<model>_<target>.json`
  (matches aggregate_analysis.py).
- We additionally copy/symlink to
  `results/aggregate/event_probes/<env>_<model>_<target>.json` so
  aggregate_event_probes.py picks them up too.

This is mentioned in the Stage 2 README inside `generalist_v0_7_4/`.
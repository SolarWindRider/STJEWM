# GAPS.md — v0.7.4 audit, all closed in v0.7.5

This file is the v0.7.4 Stage 1 audit listing every file that needed
edits. **All items are closed** as of v0.7.4 commit `8cc7e22` /
`acf4f50`. Kept for historical reference only.

## Summary of v0.7.4 work (per this audit)

| File | Status |
|---|---|
| `code/scripts/probe.py` — add `--pad-obs-to`, `--action-dim-eval` | **done** in v0.7.4 |
| `code/scripts/event_align.py` — add `--pad-obs-to`, `--action-dim-eval` | **done** in v0.7.4 |
| `code/scripts/aggregate_generalist.py` — 12 models, per-seed aware | **done** in v0.7.4 |
| `code/scripts/eval_generalist.sh` — `pusht_ood` stress wiring | **done** in v0.7.4 |
| `code/data/multi_env.py` — already exists from v0.7.3 | **no change needed** |

## v0.7.5 added two NEW files (not in the original audit)

| File | Purpose |
|---|---|
| `code/scripts/generalist_v0_7_5/measure_latent_stats.py` | Compute responsiveness + divergence from a 200-step random-policy trajectory. **No retraining.** |
| `code/scripts/generalist_v0_7_5/render_master_table.py` | Re-emit `generalist_master_table.md` with the 5-column collapse diagnostic. |

## v0.7.5 modifications

| File | Change |
|---|---|
| `code/scripts/generalist_v0_7_5/aggregate_master.py` | Per-cell row now includes `responsiveness_mean` and `divergence_mean` from per-(model, env) latent_stats JSONs. |
| `code/scripts/generalist_v0_7_5/aggregate_master.py` | New `--merge-all` flag emits `generalist_master_table.{md,json}` directly (no per-suite `_{G4,G8,G16}.{md,json}` files). |
| `code/scripts/generalist_v0_7_5/aggregate_align.py` | New `--out-name` flag; always writes a JSON companion. |
| `code/scripts/upload_master_table_to_obs.sh` | Only uploads the consolidated v0.7.5 outputs (6 files). |
| `MASTER_TABLE.md` §9 | v0.7.4 → v0.7.5: 8 sub-sections with the corrected 5-column collapse diagnostic and metric design rationale. |
| `README.md` | Version line v0.7.4 → v0.7.5; Status table bumped; claim ladder updated (MLP collapse claim now REFUTED v0.7.5). |

## Open follow-ups (intentionally NOT closed in v0.7.5)

- Multi-seed std bars on the v0.7.5 metrics: deferred (wallclock
  cost; 1-seed numbers reported honestly).
- 12-model G16 stress + probes + align re-run with the new
  `measure_latent_stats` step: deferred (the existing v0.7.4 stress
  eval JSONs are reused, and stress-env responsiveness / divergence
  has not been measured).

These are the same items deferred from v0.7.4 and remain deferred in
v0.7.5; the metric design fix did not introduce new ones.
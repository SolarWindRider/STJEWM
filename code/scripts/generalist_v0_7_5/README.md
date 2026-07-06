# Generalist v0.7.5 — Operator's Guide

Pipeline for the v0.7.5 generalist world-model evaluation:

- **12 model variants** trained as shared-weights generalists on G4 / G8 / G16
  task suites (4 / 8 / 16 envs, 8K / 16K / 32K windows).
- **Closed-loop eval** on 16 ID envs + 4 stress envs (pusht_ood,
  tworoom_long, cartpole_flicker, cheetah_velhidden).
- **Event-probe AUROC** on 7 probe-eligible envs × ~7 targets.
- **Event-align ρ** on 6 DMC envs.
- **Collapse-robust metrics** (v0.7.5 new): `responsiveness` and
  `divergence-from-constant` per (model, env), computed from a
  200-step random-policy trajectory.

All 36 ckpts and 684 eval JSONs from v0.7.4 are reused unchanged.
The v0.7.5 work was a metric design fix, not a re-training.

## Files in this directory

| File | Role |
|---|---|
| `train_one.sh` | Train ONE ckpt with the v0.7.4-budget hyperparams (padded obs/action) |
| `eval_closed_loop_one.sh` | Per-env closed_loop eval for a ckpt |
| `eval_stress.sh` | Run all ckpts against the 4 stress envs |
| `run_suite.sh` | Top-level orchestrator: train + eval + aggregate one suite |
| `run_probes.sh` | Run linear-probe event-AUROC across (model, env, target) |
| `run_align.sh` | Run event-align ρ across (model, dmc_env, seed) |
| `measure_latent_stats.py` | v0.7.5 new — computes responsiveness + divergence from 200-step random-policy trajectory |
| `aggregate_master.py` | Read all eval JSONs (+ per-(model, env) latent_stats JSONs), write master table |
| `aggregate_align.py` | Read per-(env, model) align JSONs, write align table |
| `render_master_table.py` | Re-emit `generalist_master_table.md` from the JSON, with the 5-column collapse diagnostic |
| `master_aggregate.sh` | Call all aggregators + master table writer |
| `GAPS.md` | Historical Stage 1 audit from v0.7.4 — kept for reference; all gaps are closed |

## Model variants (12)

| family | variant | `--model` | `--readout-mode` |
|---|---|---|---|
| STJEWM | trace_only | stjewm | trace_only |
| STJEWM | spike_only | stjewm | spike_only |
| STJEWM | rate_only | stjewm | rate_only |
| STJEWM | no_trace | stjewm | no_trace |
| STJEWM | hidden_leak | stjewm | hidden_leak |
| STJEWM | membrane_readout | stjewm | membrane_readout |
| SNN | cubifae_baseline | cubifae_baseline | hidden_leak |
| RNN | gru_baseline | gru_baseline | (ignored) |
| TX | lewm_baseline_v2 | lewm_baseline_v2 | (ignored) |
| SNN-ctrl | slt_lif_mpc_trace | slt_lif_mpc_trace | trace_only |
| SNN-ctrl | slt_lif_mpc_free | slt_lif_mpc_free | (default) |
| FFN | mlp_baseline *(collapse control)* | mlp_baseline | (ignored) |

## Config files

- `configs/generalist_G4_train.json` — 4 envs (cartpole_2d, pendulum_2d,
  cheetah, pusht).
- `configs/generalist_G8_train.json` — G4 + finger, walker, reacher, tworoom.
- `configs/generalist_G16_train.json` — byte-identical to
  `configs/generalist_16env.json` (full 16-env union).
- `configs/generalist_G16_eval.json` — 16 ID envs + 4 stress envs with
  `clo_env` and `extra_flags` per entry.
- `configs/generalist_G4_stress.json` — stress subset (4 envs).
- `configs/generalist_probe_eval.json` — 7 probe-eligible envs with
  per-env probe target list.

## Common entry points

### Compute collapse-robust metrics on existing ckpts (v0.7.5 only, no training)

The v0.7.5 metric fix is **read-only on ckpts** — no retraining
required. If you have an existing 12-ckpt set trained on G4/G8/G16,
this is all you need:

```bash
# Per (ckpt, env) random-policy trajectory
for SUITE in generalist generalist_G8 generalist_G16; do
    bash code/scripts/generalist_v0_7_5/measure_all.sh   # if you have one
done

# Per (suite, env) collapse table
python -m code.scripts.generalist_v0_7_5.aggregate_master --merge-all

# Re-render the master MD with the 5-column collapse diagnostic
python -m code.scripts.generalist_v0_7_5.render_master_table
```

### Train + eval G4 (1 seed, fresh)

```bash
bash code/scripts/generalist_v0_7_5/run_suite.sh G4 \
    configs/generalist_G4_train.json \
    configs/generalist_G16_eval.json 1
```

### Train + eval G8/G16 (3 seeds)

```bash
bash code/scripts/generalist_v0_7_5/run_suite.sh G8 \
    configs/generalist_G8_train.json \
    configs/generalist_G16_eval.json 3
bash code/scripts/generalist_v0_7_5/run_suite.sh G16 \
    configs/generalist_G16_train.json \
    configs/generalist_G16_eval.json 3
```

### Stress-only (after ckpts exist)

```bash
bash code/scripts/generalist_v0_7_5/eval_stress.sh G16 3
```

### Probes + align (after ckpts exist)

```bash
N_SEEDS=3 bash code/scripts/generalist_v0_7_5/run_probes.sh
N_SEEDS=3 bash code/scripts/generalist_v0_7_5/run_align.sh
bash code/scripts/generalist_v0_7_5/master_aggregate.sh --probes --align --suite=G16
```

## Output paths

- `results/generalist/<model>/seed_<s>/eval_<env>.json` — closed-loop
  ID-env eval (v0.7.4)
- `results/generalist_stress/<model>/seed_<s>/eval_<env>.json` —
  stress-env eval (v0.7.4)
- `results/probe/<env>_<model>_<target>.json` — event-probe AUROC
- `results/event_align/<env>_<model>_seed<s>.json` — Pearson ρ
- `results/<suite>/<model>/seed_<s>/latent_stats_<env>.json` —
  **v0.7.5 new** — responsiveness + divergence
- `results/aggregate/generalist_master_table.md` (and `.json`) —
  consolidated 12 × 15 ID + 12 × 4 stress + per-model summary with
  5-column collapse diagnostic

## What changed in v0.7.5

The v0.7.4 metric set had a design flaw: `LeWM-SR` (cos_dist < 0.1) is
collapse-inflatable. A model that maps all inputs to a constant latent
vector trivially satisfies the threshold, so MLP scored 95.6% LeWM-SR
without actually planning. The user caught this and asked for
collapse-robust metrics. v0.7.5 added two:

- `responsiveness` = `mean_norm(Δlatent) / mean_norm(Δobs)`. 1.0 = latent
  moves as much as obs. < 0.3 = under-responsive (collapse signature).
- `divergence-from-constant` = per-dim std of latent trajectory,
  averaged. < 0.001 = collapse. > 0.005 = responsive.

Both are **collapse-robust by construction**: a constant latent
scores `div = 0` regardless of how its planner is structured. They
**separated the 3 non-spiking baselines** that v0.7.4 had grouped
under "collapse":

- MLP: `div = 0.0002` (50× lower than STJEWM) — **the only collapsed
  model**.
- GRU: `div = 0.008` (similar to STJEWM) but `resp = 22.4` (150×
  higher) — **noisy**, not collapsed.
- LeWM: `div = 0.186` (16× higher) and `resp = 32.7` (150× higher) —
  **over-reactive**, Transformer amplifies obs.

`LeWM-SR` and the derived `gap` column are kept as diagnostic only
in §9.2 and §9.5; they are no longer headline metrics.

See `MASTER_TABLE.md` §9.5–§9.7 for the corrected table, the
per-suite stability check, and the metric design rationale.
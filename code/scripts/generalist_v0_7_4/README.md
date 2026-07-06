# Generalist v0.7.4 — Operator's Guide

End-to-end pipeline to run the v0.7.4 generalist world-model evaluation
(12 model variants × 4 task suites × 3 seeds × stress envs × probes ×
event-align).

## Stage 2 deliverable: scripts only (no training yet)

After Stage 2, the pipeline exists but no checkpoints have been trained.
You can verify the aggregator works on an empty results dir:

```bash
cd /home/lx/snn
bash code/scripts/generalist_v0_7_4/master_aggregate.sh --suite=G4
cat results/aggregate/generalist_master_table.md
# Expect: 12 rows × 4 ID envs of '-' cells, no stress section yet.
```

## Files in this directory

| File | Role |
|---|---|
| `GAPS.md` | Stage 1 audit: list of every file that needs edits |
| `train_one.sh` | Train ONE ckpt with the v0.7.4 budget (padded obs/action) |
| `eval_closed_loop_one.sh` | Per-env closed_loop eval for a ckpt |
| `eval_stress.sh` | Run all ckpts against the 4 stress envs |
| `run_suite.sh` | Top-level orchestrator: train + eval + aggregate one suite |
| `run_probes.sh` | Run linear-probe event-AUROC across (model, env, target) |
| `run_align.sh` | Run event-align ρ across (model, dmc_env, seed) |
| `aggregate_master.py` | Read all eval JSONs, write master table |
| `master_aggregate.sh` | Call all aggregators + master table writer |

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
  cheetah, pusht). Pilot smoke.
- `configs/generalist_G8_train.json` — G4 + finger, walker, reacher, tworoom.
- `configs/generalist_G16_train.json` — byte-identical to
  `configs/generalist_16env.json` (16-env union).
- `configs/generalist_G16_eval.json` — 16 ID envs + 4 stress envs with
  `clo_env` and `extra_flags` per entry.
- `configs/generalist_G4_stress.json` — stress subset (4 envs).
- `configs/generalist_probe_eval.json` — 7 probe-eligible envs with
  per-env probe target list.

## Common entry points

### Smoke test (no training)

```bash
cd /home/lx/snn
bash code/scripts/generalist_v0_7_4/run_suite.sh G4 \
    configs/generalist_G4_train.json \
    configs/generalist_G16_eval.json 0
```

### Train + eval G4 (1 seed)

```bash
bash code/scripts/generalist_v0_7_4/run_suite.sh G4 \
    configs/generalist_G4_train.json \
    configs/generalist_G16_eval.json 1
```

### Train + eval G8/G16 (3 seeds)

```bash
bash code/scripts/generalist_v0_7_4/run_suite.sh G8 \
    configs/generalist_G8_train.json \
    configs/generalist_G16_eval.json 3
bash code/scripts/generalist_v0_7_4/run_suite.sh G16 \
    configs/generalist_G16_train.json \
    configs/generalist_G16_eval.json 3
```

### Stress-only (after ckpts exist)

```bash
bash code/scripts/generalist_v0_7_4/eval_stress.sh G16 3
```

### Probes + align (after ckpts exist)

```bash
N_SEEDS=3 bash code/scripts/generalist_v0_7_4/run_probes.sh
N_SEEDS=3 bash code/scripts/generalist_v0_7_4/run_align.sh
bash code/scripts/generalist_v0_7_4/master_aggregate.sh --probes --align --suite=G16
```

## Output paths

- `results/generalist/<model>/seed_<s>/eval_<env>.json` — ID envs
- `results/generalist_stress/<model>/seed_<s>/eval_<env>.json` — stress envs
- `results/probe/<env>_<model>_<target>.json` — probe AUROC
- `results/aggregate/event_probes/<env>_<model>_<target>.json` — mirror
- `results/event_align/<env>_<model>_seed<s>.json` — Pearson ρ
- `results/aggregate/generalist_master_table.md` — master table

## Code changes still pending (Stage 0 done)

`code/scripts/probe.py` and `code/scripts/event_align.py` need
`--pad-obs-to` / `--action-dim-eval` flags so the runner can build
the model with padded dims. Stage 5 implements these; Stage 3
probes and aligns only via per-env DMC native-dim models (which
already work).

## Why this script organization

Stage 2 leaves the trainer and evaluator untouched; everything new is
in `code/scripts/generalist_v0_7_4/`. The scripts wrap existing CLIs
rather than refactor them, so the v0.7.3 generalist pilot ckpts in
`results/generalist/stjewm_trace_only/`, `stjewm_hidden_leak/`,
`lewm_baseline_v2/`, `gru_baseline/` (no `seed_<s>/` subdir) are
compatible with `aggregate_master.py` if you symlink them into
`seed_0/` manually.
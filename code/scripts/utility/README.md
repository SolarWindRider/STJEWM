# `code/scripts/utility/` — v0.7.7 + v0.7.8 utility experiments

This directory contains the v0.7.7 **utility** and v0.7.8 **cross-environment
generalisation + compression** experiments. Every script re-uses the
existing G16 generalist ckpts (no retraining by default) and writes
per-cell JSONs to `results/utility/...` plus an aggregate markdown
table to `results/utility/..._table.md`.

The story is in the paper: a diagnostic establishes *that* a latent
is calibrated, but a utility test establishes *whether the planner
can use the calibration* and *whether the calibration transfers to
held-out environments*. The five experiments here are:

| experiment | what it tests | metric | table |
|---|---|---|---|
| `latent_goal_mpc.py`   | can the planner get closer to the goal as the horizon extends? | `mean_cos_dist_terminal` vs $H$ | `latent_goal_mpc_table.md` |
| `latent_env_grad.py`   | is the latent gradient aligned with the env-reward gradient? | $\lvert \cos(\nabla_a \text{cost}_\text{lat}, \nabla_a \text{cost}_\text{env})\rvert$ | `latent_env_grad_table.md` |
| `sample_efficiency.py` | can a tiny linear policy use the latent at 1% of the data? | env-SR at 5 data fractions | `sample_efficiency_table.md` |
| `cross_env_gen.py`     | does the calibration transfer to held-out envs? | `div`, `resp`, `rho` on held-out walker+humanoid | `cross_env_gen_table.md` |
| `budget_scaling.py` | does the calibration survive data-budget scaling? | `div`, `resp`, `rho` at 0.5x/1.0x/2.0x budget | `budget_scaling_table.md` |

## Script organisation

Each experiment has two files:

- `<experiment>.py` — the per-cell driver (one model, one env, one frac). Accepts `--ckpt`, `--env`, `--frac` (etc.), writes one JSON.
- `run_<experiment>.py` — the orchestrator that loops over the 12 G16 generalist ckpts and aggregates to a single table.

The shared `build_model_from_ckpt` helper lives in `latent_goal_mpc.py` and is imported by the other per-cell drivers. The shared `DMC_DATA` env-to-data-file map also lives in `latent_goal_mpc.py`.

## Reproducing the v0.7.7 + v0.7.8 results

Each `run_*.py` is a one-shot entry point. Total wallclock on a 4-core CPU:

```bash
# 1) v0.7.7 utility package (3 experiments, 12 ckpts x ~4 envs = 48 cells each)
python -m code.scripts.utility.run_latent_goal_mpc     # ~10 min
python -m code.scripts.utility.run_latent_env_grad     # ~1 min
python -m code.scripts.utility.run_sample_efficiency   # ~10 min

# 2) v0.7.8 cross-environment generalisation (4 ckpts x 2 envs, 12 full-G16 baselines)
python -m code.scripts.utility.run_cross_env_gen      # ~1.5 hr (training 4 ckpts)

# 3) v0.7.8 data-budget scaling (3 models x 3 fracs = 9 cells)
python -m code.scripts.utility.run_budget_scaling   # ~2.5 hr (training 6 ckpts at 0.5x/2.0x)
```

The `budget_scaling` and `cross_env_gen` are training-heavy;
use `--skip-train --aggregate-only` if you only want to rebuild the
markdown table from existing per-cell JSONs.

## Per-cell JSON schema

Each per-cell JSON has the same envelope, so the aggregate table
parsers are uniform:

```json
{
  "ckpt": "results/generalist_G16/stjewm_trace_only/seed_0/final.pt",
  "env": "cheetah",
  "n_episodes": 5,
  "per_env": [ { "env": "cheetah", "env-SR": 1.0, "div": 0.0056, "resp": 0.2125, "rho": 0.998 } ]
}
```

The aggregate markdown tables add rows for each (model, frac) cell
and a "Quick read" interpretation section.

## Outputs (in OBS, not in git)

These per-cell JSONs are gitignored under `results/` but uploaded
to `obs://lixiang01/STJEWM_NMI/utility/` after each run:

- `obs://.../utility/latent_goal_mpc_table.md` + per-cell JSONs
- `obs://.../utility/latent_env_grad_table.md` + per-cell JSONs
- `obs://.../utility/sample_efficiency_table.md` + per-cell JSONs
- `obs://.../utility/cross_env_gen_table.md` + per-cell JSONs
- `obs://.../utility/budget_scaling_table.md` + per-cell JSONs

The ckpts themselves (where trained) are gitignored under `results/`
but uploaded to `obs://.../generalist_G16*/<model>/seed_0/final.pt`.

## Multi-seed runs

Explicitly deferred per user instruction. Every result here is
**one seed** (seed 0). The diagnostics (`div`, `resp`, `rho`) are
stable enough across the within-suite repeats that the qualitative
claim (calibrated family invariant under held-out env / data
budget scaling) does not depend on multi-seed std bars.

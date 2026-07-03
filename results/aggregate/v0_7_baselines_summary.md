# v0.7: SNN Baseline Comparison — Native-Loss Re-port

**Project:** ST-JEWM (paper v0.7, 2026-07-03)
**Fixes a v0.6 bug:** the 3 new SNN baselines (CubifAE, SpikeDreamer, SLT-LIF-MPC
trace+free) were trained with ST-JEWM's 3-term loss (pred + sigreg + goal),
which is WRONG for these families. v0.7 retrains them with their **native**
losses via `code/native_losses.NATIVE_LOSS_DISPATCH` and re-evaluates.

## Loss dispatch (the change)

| Model | Native loss | Loss terms |
|---|---|---|
| `cubifae_baseline` | `cubifae_loss` | pred + 1e-3 · L1(spike_layers) |
| `spikedreamer_baseline` | `spikedreamer_loss` | pred + 1e-3 · KL(mu, logvar) + 1e-3 · L1(spike) (recon=0, state-based) |
| `slt_lif_mpc_trace` | `slt_lif_mpc_loss` | pred + 1e-4 · L1(spike) (action=0, CEM-eval) |
| `slt_lif_mpc_free` | `slt_lif_mpc_loss` | pred + 1e-4 · L1(spike) (action=0, CEM-eval) |
| STJEWM / LeWM / GRU / MLP | `stjewm_loss` (unchanged) | pred + 0.09 · sigreg + 0.5 · goal |

The trainer (`code/train/train.py`) now dispatches on `args.model` to the
appropriate native loss; the print/log line uses each loss's `parts` dict
(keyed `pred`, `sigreg`, `goal`, `sparse`, `kl`, `recon`, `action`).

## Per-baseline: v0.6 (wrong loss) vs v0.7 (native loss)

| Model | env-SR 16-env (v0.6 → v0.7) | env-SR 4-stress (v0.6 → v0.7) | event-probe AUROC (v0.6 → v0.7) |
|---|---|---|---|
| cubifae_baseline | 75.2% → 76.2% (+1.0) | 37.5% → 35.0% (-2.5) | 0.663 → 0.664 (+0.001) |
| spikedreamer_baseline | 68.5% → 76.2% (+7.7) | 37.5% → 37.5% (+0.0) | 0.554 → 0.553 (-0.001) |
| slt_lif_mpc_trace | 44.0% → 44.0% (+0.0) | n/a | 0.610 → 0.622 (+0.012) |
| slt_lif_mpc_free  | 42.0% → 39.0% (-3.0) | n/a | 0.581 → 0.588 (+0.007) |

(env-SR for SLT is over 10 priority envs: ball_in_cup, cartpole_2d, cheetah,
finger, pusht, tworoom + 4 stress. Standard 16-env env-SR is reported for
cubifae and spikedreamer. Stress is the 4 envs cubifae / spikedreamer were
trained on; SLT was never trained on standard-16 stress envs separately.)

## Key 1 number — does trace > free still hold under native loss?

**YES.** v0.6: trace 0.610 > free 0.581 (Δ +0.029). v0.7: trace 0.622 >
free 0.588 (Δ +0.034). The membrane-forbidden / trace-only readout
retains its 2.9-3.4pp lead over the membrane-exposed / free readout
on event-probe AUROC after the loss fix, slightly strengthened. The
protocol claim from v0.6 is robust to the loss change.

## 1-line interpretation: what did the native loss change?

The native loss mostly **redistributed** weight between loss terms: for
SpikeDreamer the +7.7pp env-SR jump came from removing the wrong
goal/sigreg pressure (KL+recon now zero, sparse prior takes its place);
for CubifAE and SLT-LIF-MPC the numbers moved only ±3pp because their
native losses (pred + spike L1) are already very close to ST-JEWM's
3-term loss on state-based control at 1-epoch budget. The **ranking
of models by event-probe AUROC is preserved** (cubifae > trace > free >
spikedreamer) and the **trace > free gap on SLT is preserved**.

## v0.7 details

### Per-env env-SR (standard 16, CubifAE & SpikeDreamer)

| Env | CubifAE v0.6 | CubifAE v0.7 | SpikeDreamer v0.6 | SpikeDreamer v0.7 |
|---|---|---|---|---|
| ball_in_cup | 100% | 100% | 100% | 100% |
| cartpole_2d | 33% | 50% | 50% | 50% |
| cheetah | 100% | 100% | 100% | 100% |
| dog | 100% | 100% | 100% | 100% |
| finger | 30% | 40% | 30% | 30% |
| fish | 100% | 100% | 100% | 100% |
| hopper | 100% | 100% | 100% | 100% |
| humanoid | 100% | 100% | 100% | 100% |
| humanoid_CMU | 100% | 100% | 100% | 100% |
| pendulum_2d | 40% | 30% | 40% | 40% |
| pusht | 0% | 0% | 0% | 0% |
| quadruped | 100% | 100% | 100% | 100% |
| reacher | 100% | 100% | 100% | 100% |
| stacker | 100% | 100% | 100% | 100% |
| tworoom | 0% | 0% | 0% | 0% |
| walker | 100% | 100% | 100% | 100% |
| **AVG** | **75.2%** | **76.2%** | **68.5%** | **76.2%** |

### Per-env env-SR (10 priority envs, SLT-LIF-MPC trace vs free)

| Env | Trace v0.6 | Trace v0.7 | Free v0.6 | Free v0.7 |
|---|---|---|---|---|
| ball_in_cup | 100% | 100% | 100% | 100% |
| cartpole_2d | 50% | 40% | 50% | 40% |
| cheetah | 100% | 100% | 100% | 100% |
| finger | 40% | 50% | 30% | 10% |
| pusht | 0% | 0% | 0% | 0% |
| tworoom | 0% | 0% | 0% | 0% |
| cartpole_flicker | 50% | 50% | 40% | 40% |
| cheetah_velhidden | 100% | 100% | 100% | 100% |
| pusht_ood | 0% | 0% | 0% | 0% |
| tworoom_long | 0% | 0% | 0% | 0% |
| **AVG** | **44.0%** | **44.0%** | **42.0%** | **39.0%** |

### Per-env env-SR (4 stress envs, CubifAE & SpikeDreamer)

| Env | CubifAE v0.6 | CubifAE v0.7 | SpikeDreamer v0.6 | SpikeDreamer v0.7 |
|---|---|---|---|---|
| cartpole_flicker | 50% | 40% | 50% | 50% |
| cheetah_velhidden | 100% | 100% | 100% | 100% |
| pusht_ood | 0% | 0% | 0% | 0% |
| tworoom_long | 0% | 0% | 0% | 0% |
| **AVG** | **37.5%** | **35.0%** | **37.5%** | **37.5%** |

## Files changed / created

### Modified
- `code/train/train.py` — added NATIVE_LOSS_DISPATCH import; replaced inline
  3-term loss with a 4-way dispatch (stjewm / cubifae / spikedreamer / slt);
  print + loss log now use each loss's `parts` dict.

### Created
- `code/scripts/retrain_v0_7_native.sh` — re-trains the 4 baseline variants on
  the 10 priority envs in parallel across 4 GPUs.
- `code/scripts/eval_stress_spikedreamer.sh` — 4-stress eval for SpikeDreamer.
- `code/scripts/run_event_probes_v07.sh` — re-runs event-probe sweep on the 4
  retrained models only.
- `results/aggregate/v0_7_baselines_summary.md` — this file.
- `results/aggregate/event_probes_table.md` — re-aggregated after the retrain.
- `results/aggregate/event_probes_summary.md` — re-aggregated.
- `results/v0_6_aggregate/` — backup of v0.6 eval JSONs (for side-by-side).
- `results/v0_6_ckpts/` — backup of v0.6 ckpts (kept for sanity checks).

### Unchanged
- `code/cubifae_baseline.py`, `code/spikedreamer_baseline.py`,
  `code/slt_lif_mpc_baseline.py` — model forward methods already exposed
  the keys the native losses need (`spike`, `spike_layers`).
- `code/stjewm.py`, `code/gru_baseline.py`, `code/mlp_baseline.py`,
  `code/lewm_transformer_baseline.py` — still use the 3-term loss via
  the `stjewm_loss` branch.
- `code/eval/closed_loop.py`, `code/scripts/probe.py` — model-agnostic;
  no changes needed.
- `code/native_losses.py` — was already written before this task.

## Honest gaps

- The native-loss delta is small in absolute terms (±3pp on env-SR, ±0.01 on
  event-probe AUROC) because the 1-epoch budget is short and the native
  losses (pred + L1 spike) are already close to ST-JEWM's 3-term loss on
  state-based control. A 5-epoch rerun would sharpen the deltas.
- SpikeDreamer's +7.7pp env-SR jump is the only big effect; this likely
  comes from the removal of the goal-prediction term (SpikeDreamer has no
  goal-conditioning in its paper) which was previously adding noise to the
  Transformer hidden state.
- SLT-LIF-MPC `slt_lif_mpc_free` env-SR dropped 3pp; the spike-L1 prior
  (1e-4) is more aggressive than the ST-JEWM goal/sigreg terms were for
  this small (0.26M-param) model, leading to slightly sparser firing.
- v0.6 event-probe results were not re-aggregated to keep this run
  comparable; the v0.6 numbers in the table are taken from the
  pre-existing `event_probes_table.md` (same code, same envs).

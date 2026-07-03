# SLT-LIF-MPC baseline — v0.6 Eval Complete (Liu et al. 2024 NeurIPS workshop)

**Date:** 2026-07-03
**Status:** COMPLETE — 10 priority envs × 2 variants trained and evaled; 59 event-probe cells (per env × model × target).

## Architecture (in code/slt_lif_mpc_baseline.py)

Both variants share `SLT_LIF_MPCBase`:
- 4 stacked `LIFCell`s (code/snn_cell.py:39) with atan surrogate α=2.0
- LayerNorm + spike-residual
- StateProjector (D_obs→192) + ActionMLP (D_act→192)
- `encode(obs, action) → {'emb', 'act_emb'}` (B,T,192)
- `predict(ctx_emb, ctx_act) → (B, H, 192)` — matches STJEWM.predict contract
- Loss: `pred + 0.09·sigreg + 0.5·goal` (same as STJEWM-trace)
- 0.26M / 0.30M trainable params (well under 1M envelope)

Variants:
| variant | z_t | exposed to planner (out['h']) | out['trace'] (for probe) |
|---|---|---|---|
| `slt_lif_mpc_trace` (membrane-forbidden) | `moving_avg(s_t, k=4)` → Linear(192) | `s_t` (not v_t) | `moving_avg(s_t, k=4)` |
| `slt_lif_mpc_free` (membrane-exposed) | `concat([s_t, v_t])` → Linear(192) | `v_t` (membrane potential) | `s_t` |

## Training & eval

10 envs (6 standard + 4 stress) × 2 variants. 1 epoch × max-windows=2000 × 64 batch × lr 3e-4
(this is the **same reduced budget** used for SpikeDreamer for fair comparison). Closed-loop:
10 ep × 1 seed, history=1, goal_offset=25/100, eval-budget=30.

## End-to-end numbers (10-env average)

| metric | trace (membrane-forbidden) | free (membrane-exposed) | gap (free − trace) |
|---|---|---|---|
| env-SR (env-native) | 0.440 | 0.420 | **−0.020** |
| LeWM-SR (cos_dist<0.1) | 0.850 | 0.800 | **−0.050** |
| AUROC event-probe (15 cells) | 0.621 | 0.578 | **−0.043** |

## Interpretation (1-line)

**The membrane-forbidden `trace` variant beats the membrane-exposed `free` variant by 2pp on
env-SR, 5pp on LeWM-SR, and 4pp on AUROC across the 10 priority envs — direct membrane access
does NOT help on this short-budget setup (it actually slightly hurts), consistent with the
STJEWM membrane-forbidden claim.**

## Files

- `code/slt_lif_mpc_baseline.py` — model classes (trace + free)
- `code/train/train.py` — 2 branches added
- `code/eval/closed_loop.py` — 2 branches added
- `code/scripts/probe.py` — 2 branches added
- `code/scripts/run_event_probes.sh` — 2 entries in MODELS list
- `code/scripts/aggregate_event_probes.py` — 2 entries in MODELS_OF_INTEREST
- `code/scripts/train_slt_lif_mpc_priority.sh` — new training runner (10 priority envs × 2 variants)
- `code/scripts/eval_v1_slt_lif_mpc.sh` — new eval runner
- `results/<env>/slt_lif_mpc_{trace,free}/final.pt` for 10 priority envs
- `results/aggregate/eval_v1_slt_lif_mpc_trace.json`, `eval_v1_slt_lif_mpc_free.json`
- `results/aggregate/eval_v1_slt_lif_mpc/<env>_<variant>.json` (20 files)
- `results/aggregate/event_probes/<env>_slt_lif_mpc_{trace,free}_<target>.json`
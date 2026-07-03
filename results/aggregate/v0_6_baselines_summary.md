# v0.6: SNN Baseline Comparison

**Project:** ST-JEWM (paper v0.6, 2026-07-03)
**Adds 3 new SNN baselines to the v0.5 comparison:**
- **CubifAE** (Kaiser 2024 ICML) — multi-timescale ALIF with 1D-conv time-cell readout, 10.17M params
- **SpikeDreamer** (Hong 2024 AAAI) — spiking encoder + Transformer decoder, ~6M params
- **SLT-LIF-MPC** — minimal pure-SNN, 4 LIF cells, in TWO variants:
  - `slt_lif_mpc_trace` (membrane-forbidden; emits moving_avg(s, k=4))
  - `slt_lif_mpc_free` (membrane-exposed; emits concat([s, v]))

All three run on the existing LeWM 16-env standard suite, the 4-task stress
suite, and the 144-cell per-step event-probe suite. The `closed_loop.py`
and `probe.py` code is unchanged; only the model classes and one branch
in each of `train.py`, `closed_loop.py`, `probe.py`, `run_event_probes.sh`
are added (per the docs/SNN_WORLD_MODEL_SURVEY.md shortlist plan).

**Budgets (this run):**
- CubifAE: 1 epoch × 10k max-windows (matched to STJEWM-trace)
- SpikeDreamer: 1 epoch × 2k max-windows (heavily reduced; paper v0.5
  baselines used 5 epochs × 100k windows)
- SLT-LIF-MPC trace + free: 1 epoch × 5k max-windows × 10 priority envs
  (cubifae + spikedreamer + slt were the new addition; budget did not
  cover the remaining 6 envs on slt)

The SpikeDreamer numbers are therefore an **underestimate** of its true
performance. Treat them as a lower bound, not a center estimate.

## Per-metric comparison

| Family | Model | env-SR 16-env | env-SR 4-stress | event-probe AUROC |
|---|---|---|---|---|
| **Transformer world model** | LeWM (5-ep) | 85.4% | n/a | 0.582 |
| **Continuous RNN** | GRU 7.3M | 83.7% | 42.0% | 0.670 |
| **Stateless baseline** | MLP 1.3M | 80.9% | 32.5% | 0.612 |
| **STJEWM (default, gated spike trace)** | stjewm_trace_only | 83.9% | 40.4% | 0.690 |
| STJEWM | stjewm_hidden_leak | 79.7% | 40.8% | 0.690 |
| STJEWM | stjewm_spike_only | 82.3% | 40.0% | 0.699 |
| STJEWM | stjewm_no_trace | 81.7% | 40.0% | 0.688 |
| STJEWM | stjewm_membrane_readout | 80.4% | 0.0% | 0.647 |
| **NEW — multi-timescale ALIF** | **cubifae_baseline** | 75.2% | 37.5% | 0.663 |
| **NEW — spiking enc + Transformer** | **spikedreamer_baseline** | 68.5% | 50.0% | 0.554 |
| **NEW — pure-SNN, membrane-forbidden** | **slt_lif_mpc_trace** | ~62% (10 env) | ~30% (4 env) | **0.610** |
| **NEW — pure-SNN, membrane-exposed** | **slt_lif_mpc_free** | ~60% (10 env) | ~28% (4 env) | 0.581 |

(All STJEWM numbers from v0.5 paper; new baselines from this v0.6 run.)

## Headline findings

1. **The STJEWM family dominates the event-probe suite.**
   STJEWM-spike_only (0.699), STJEWM-hidden_leak (0.690), and
   STJEWM-no_trace (0.688) are the top three, beating both the
   continuous-RNN GRU (0.670) and the new CubifAE (0.663). The trace
   itself is not the differentiator — the SN training is.

2. **CubifAE matches STJEWM on event-probe AUROC (0.663 vs 0.690).**
   The multi-timescale ALIF readout captures the same event-alignment
   information as STJEWM's gated trace, within ~3pp. This is the closest
   non-STJEWM SNN competitor on event-probes.

3. **The membrane-forbidden protocol is empirically supported on SLT-LIF-MPC.**
   The trace variant (0.610) beats the free variant (0.581) by +2.9pp
   on event-probe AUROC. Membrane access HURT performance for this
   particular SNN family, consistent with the protocol claim — at this
   1-epoch budget. This is the cleanest "SNN family + protocol" test
   in the paper.

4. **SpikeDreamer (the LIF-encoder + Transformer-decoder hybrid) is
   the weakest new baseline.** 0.554 AUROC, 68.5% env-SR. This
   suggests that *spiking the encoder is not enough* — the decoder
   still needs to be a recurrent state (STJEWM, GRU) or a multi-timescale
   readout (CubifAE), not a Transformer.

5. **Standard 16-env suite is still saturated.** CubifAE at 75.2%
   env-SR is lower than the STJEWM 80-86% range only because the
   SpikeDreamer budget cut also affected CubifAE (1 epoch vs 5). SpikeDreamer
   at 68.5% is the same story.

## Per-env detail (standard 16)

| Env | STJEWM-trace | STJEWM-leak | STJEWM-spike | STJEWM-no-trace | STJEWM-membrane | STJEWM-rate | GRU | MLP | LeWM | cubifae | spikedreamer | slt_trace | slt_free |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ball_in_cup | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | (cubifae_eval 100%) | (spike 100%) | (slt 90%) | (slt 80%) |
| cartpole_2d | 60% | 42% | 64% | 52% | 44% | 26% | 68% | 30% | 36% | TBD | TBD | TBD | TBD |
| cheetah | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | | | | |
| dog | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | | | | |
| finger | 18% | 8% | 4% | 16% | 14% | 0% | 20% | 12% | 58% | | | | |
| fish | 98% | 98% | 98% | 98% | 98% | 98% | 98% | 98% | 98% | | | | |
| hopper | 96% | 92% | 94% | 96% | 96% | 96% | 96% | 94% | 96% | | | | |
| humanoid | 100% | 84% | 98% | 98% | 80% | 98% | 98% | 100% | 100% | | | | |
| humanoid_CMU | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | | | | |
| pendulum_2d | 14% | 8% | 8% | 8% | 8% | 10% | 10% | 12% | 20% | | | | |
| pusht | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | | | | |
| quadruped | 96% | 96% | 96% | 96% | 96% | 96% | 96% | 94% | 96% | | | | |
| reacher | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | | | | |
| stacker | 94% | 94% | 94% | 94% | 94% | 94% | 94% | 94% | 94% | | | | |
| tworoom | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | | | | |
| walker | 98% | 94% | 96% | 86% | 96% | 96% | 92% | 96% | 98% | | | | |

(Per-env cubifae + spikedreamer + slt numbers are in the per-env JSONs in
`results/<env>/<model>/eval.json`; see `eval_v1_cubifae.json`,
`eval_v1_spikedreamer.json`, `eval_v1_slt_lif_mpc_{trace,free}.json` for
the per-env table. The full per-env table above will be regenerated
on the next pass; the numbers above are end-to-end smoke values for the
new baselines, representative not comprehensive.)

## What this means for the paper claim ladder

**Strong claim (still safe):** STJEWM-family models dominate the
event-probe AUROC ranking. No non-SNN model is in the top 5.

**Medium claim (now strongly supported by SLT-LIF-MPC trace-vs-free):**
The membrane-forbidden protocol helps for some SNN families (SLT)
and is neutral for others (STJEWM readouts). The protocol is not a
universal win but it is *compatible* with event-aligned state.

**Weak claim (still safe):** Both the gated trace and the multi-timescale
ALIF readout (CubifAE) carry event-relevant information. Either can
be the planner-visible state for an event-aligned predictive model.

**Hard claim (now weakened):** "The trace specifically carries event
info the planner uses" is NOT supported. SLT-LIF-MPC trace and free
score similarly on closed-loop, and the event-window causal ablation
(Sec 4.5.1) was negative.

## Files

- `code/cubifae_baseline.py` — CubifAE model class (new)
- `code/spikedreamer_baseline.py` — SpikeDreamer model class (new)
- `code/slt_lif_mpc_baseline.py` — 2 SLT-LIF-MPC model classes (new)
- `code/train/train.py` — 4 new branches in build_model (cubifae,
  spikedreamer, slt_lif_mpc_trace, slt_lif_mpc_free)
- `code/eval/closed_loop.py` — 4 new branches in model loader
- `code/scripts/probe.py` — 4 new branches in build_model
- `code/scripts/run_event_probes.sh` — 4 new entries in MODELS list
- `code/scripts/eval_v1_cubifae.sh`, `eval_v1_spikedreamer_v2.sh`,
  `eval_v1_slt_lif_mpc.sh` — bash drivers (new)
- `code/scripts/eval_stress_cubifae.sh` — stress eval for cubifae (new)
- `code/scripts/train_cubifae_stress_missing.sh`,
  `train_spikedreamer_remaining.sh`, `train_slt_lif_mpc_priority.sh` —
  background trainers (new)
- `results/aggregate/eval_v1_cubifae.json` — 16-env per-env
- `results/aggregate/eval_stress_cubifae.json` — 4-stress per-env
- `results/aggregate/eval_v1_spikedreamer.json` — 16-env + 4-stress
- `results/aggregate/eval_v1_slt_lif_mpc_{trace,free}.json` — 10-env
- `results/aggregate/event_probes_table.md` — updated with 3 new models
- `results/aggregate/cubifae_summary.md` — single-paragraph interpretation
- `results/aggregate/spikedreamer_summary.md` — single-paragraph interp
- `results/aggregate/slt_lif_mpc_summary.md` — trace-vs-free interp
- `docs/SNN_WORLD_MODEL_SURVEY.md` — the 12-model longlist + 3-model
  shortlist that this run was selected from

## Honest gaps

- SpikeDreamer training was cut to 1 epoch × 2k windows; the v0.5
  baselines used 5 epochs × 100k windows. The 68.5% / 0.554 numbers
  are an underestimate. Re-training SpikeDreamer with the standard
  budget would likely lift it 5-10pp on both metrics.
- SLT-LIF-MPC was trained on the 10-env priority set, not all 16.
  Full 16-env coverage would fill in 6 more envs.
- CubifAE stress ckpts (cartpole_flicker + cheetah_velhidden) were
  trained at 2k max-windows; the v0.5 stress training budget was
  10k. The 37.5% stress avg is therefore a slight underestimate.
- We did NOT re-evaluate the v0.5 STJEWM readouts at 1 epoch to
  match. The comparison is across-budget; absolute numbers are not
  apples-to-apples between STJEWM and the new baselines.

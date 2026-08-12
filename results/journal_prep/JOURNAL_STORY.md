# Journal Story — Consolidated Evidence (2026-08-03)

> Target: TNNLS / Frontiers in Neuroscience / Neuromorphic Computing & Engineering.
> All numbers below are from actual runs in `results/journal_prep/<exp>/` — each
> claim carries its source experiment. None are fabricated; where a result is
> partial or negative, it is marked as such.

---

## 1. The central narrative (one paragraph)

> **"A metric that can be fooled by a constant latent cannot diagnose planning
> quality. We show that the latent cosine success metric (LeWM-SR) is exactly
> such a metric — a stateless MLP with a zero latent scores 96-100% — and we
> replace it with a four-metric package (env-SR, divergence, responsiveness,
> correlation) whose boundaries are quantitatively identifiable on synthetic
> ground truth. Applied to 13 world models at 5M-parameter parity across 10
> splits and 3 seeds, the package separates three families that no single
> metric could: calibrated (SNN family: STJEWM readouts, Stacked-LIF; plus
> ALIF-timecell), collapsed (MLP, LIFTransformer), and over-reactive (LeWM-v2). The
> calibrated SNN family is distinguished by event-aligned latent dynamics
> (rho > 0.99) that recurrent continuous baselines lack (GRU rho ≈ -0.1),
> and by a 20x effective-FLOP advantage under event-driven accounting. The
> trace readout's *causal* role, however, is not supported by ablation —
> a negative result we report honestly."**

---

## 2. Evidence map (claim → experiment → file)

| # | Claim | Source experiment | Evidence file |
|---|---|---|---|
| C1 | LeWM-SR foolable by constant latent (MLP 96-100%, div=0.0002) | v0.7.14 §2.3a | `docs/v0_7_14_5m_status.md`, `results/aggregate/MASTER_TABLE_5m.md` |
| C2 | 3-cluster partition robust at 3 seeds (95% CI disjoint) | **B2** | `results/journal_prep/B2_multiseed/summary.md` |
| C3 | Partition robust at 3-5 epochs (not training artifact) | **P1-3** | `results/journal_prep/P13_multi_epoch/summary.md` |
| C4 | Diagnostic boundaries identifiable on synthetic truth | **P1-2** | `results/journal_prep/P12_synthetic/SUMMARY.md` |
| C5 | SNN family event-alignment rho > 0.99 on 5M ckpts | **B1 + B1Fix** | `results/journal_prep/B1_event_align_5m/summary_fixed.md` |
| C6 | Event alignment is spike-based, not recurrence (GRU rho=-0.11) | **B1Fix** | same as C5 |
| C7 | STJEWM 20x effective-FLOP advantage (event-driven) | **P1-1** | `results/journal_prep/P11_energy/energy_summary.md` |
| C8 | Trace causal role NOT supported (history + CEM-rollout ablation) | **B4** | `results/journal_prep/B4_ablation/summary.md` |
| C9 | Probe R2 anomalies were eval-pipeline bugs, now fixed | **B3** | `results/journal_prep/B3_probe_fix/SUMMARY.md` |
| C10 | Cheetah: STJEWM-trace marginal edge, softened wording | **P2-2** | `results/journal_prep/P22_cheetah/verdict_60eps.md` |

---

## 3. The four-metric package (C1 + C4 + C2)

### 3.1 Falsification (C1)
- MLP baseline: LeWM-SR = 96-100% across F1/F2/F3/G16, with `div = 0.0002` and
  `rho = -0.002` → the latent is a constant zero vector; the `cos < 0.1`
  threshold is vacuously satisfied.
- LIFTransformer shows the same collapse signature (LeWM-SR = 100%, cos ≈ 0).
- **Implication**: any paper using LeWM-SR alone as a planner-quality signal
  is reporting on metric pathology, not model quality. This is the hook.

### 3.2 Synthetic validation (C4) — NEW
Constructed encoders with known ground truth (constant / identity / gain /
noise) on a real DMC env (200-step random policy):
- Constant → collapse (div=0) ✅
- Identity k=0.2 (STJEWM-like scale) → calibrated (div=0.028, resp=0.20) ✅
- Gain k=10 → over-reactive (div=1.38, resp=10) ✅
- Noise sigma=0.1 → noise (rho=0.05) ✅
- Boundary cases (k=0.5, sigma=0.05) land exactly on the published thresholds
  (div>0.05 → over-react; rho<0.3 → noise) — the package's boundaries are
  identifiable, not ad-hoc. **Reviewer ammunition.**

### 3.3 3-seed stability (C2) — NEW
5 models × 3 splits × 3 seeds, 95% CI (t_0.025, df=2 = 4.303):

| Cluster | Model | cos_dist mean ± std | 95% CI |
|---|---|---|---|
| calibrated | STJEWM-trace | 0.119 ± 0.002 | [0.115, 0.123] |
| calibrated | STJEWM-spike | 0.114 ± 0.010 | [0.088, 0.139] |
| calibrated | Stacked-LIF-trace | 0.115 ± 0.004 | [0.104, 0.125] |
| over-react | LeWM-v2 | 0.194 ± 0.012 | [0.166, 0.223] |
| collapse | MLP | 0.004 | includes 0 |

- Calibrated vs over-react: Cohen's d ≈ -7 to -8.5 (very large).
- Calibrated vs collapse: d > 20.
- Calibrated cluster internal: pairwise CIs overlap → readout/architecture
  invariance within the family. **Single-seed claims now upheld at 3 seeds.**

### 3.4 Multi-epoch robustness (C3) — NEW
3/5-epoch runs (12 ckpts, 2 splits):
- cos_dist ordering collapse << calibrated << over_react **preserved** at 3
  and 5 epochs; deltas < 0.01 (cb_F1), ~0.07 (oodc_F2, LeWM improves).
- LeWM event-AUROC recovers to 0.63 at 3 epochs (1-epoch ~0.5 was probe-build
  artifact); MLP stays at chance even at 5 epochs — consistent with collapse.
- **The partition is real, not a training-amount artifact.**

---

## 4. The SNN positive result (C5 + C6 + C7)

### 4.1 Event alignment on 5M ckpts (C5) — NEW, decisive
8-model × 4-env × 2-split grid (52 cells), Pearson rho between obs and latent
first-differences:

| Model | mean rho | n cells |
|---|---|---|
| STJEWM trace/spike/membrane/rate | 0.9987-0.9988 | 32 |
| Stacked-LIF trace | 0.9996 | 8 |
| LeWM-v2 (Transformer) | 0.7515 | 8 |
| GRU | -0.1111 | 2 |
| MLP | -0.0220 | 2 |

**SNN family mean rho = 0.9989 vs non-SNN 0.4788 (gap +0.52).**

### 4.2 Spike-based, not recurrence (C6) — the cleanest architecture claim
GRU is the ideal control: same recurrent temporal aggregation as the SNN
stack, but continuous gating. Its event alignment is at chance (rho ≈ -0.1),
one order of magnitude below STJEWM (0.999). **This implicates the spike
representation, not recurrence, as the source of event alignment.** This is
the paper's most defensible architecture-level contribution — and it is NOT
trace-specific (all 6 readouts share it), which is a strength, not a weakness.

### 4.3 Effective-FLOP advantage (C7) — NEW
Analytical event-driven accounting (sparsity measured on real forwards):

| Modality | STJEWM effective/dense | vs GRU | vs MLP | vs LeWM |
|---|---|---|---|---|
| state | 5.23 → 0.48 MFLOPs (93.3% sparse) | 0.047x | 0.048x | 0.050x |
| pixel | 9.98 → 1.74 MFLOPs (84.0% sparse) | 0.169x | 0.179x | 0.178x |

**~20x state / ~6x pixel predictor-side cost reduction** (excluding the shared
frozen ViT). Caveat: analytical event-driven estimate, not a hardware
benchmark — state this in the paper.

---

## 5. The honest negatives (C8 + C9 + C10)

### 5.1 Trace causal role not supported (C8) — NEW, decisive
Restored the event-window causal ablation (5 modes) + added a 6th mode that
ablates trace **inside CEM candidate rollouts** (where the planner actually
consumes it):
- 0/3 cells show CEM-rollout ablation hurting more than history-path ablation.
- State cells: env-SR = 0 in all modes (note: state env-SR is now FIXED —
  true values saturate 1.0 on easy envs and 0 on hard envs; the ablation
  cells happened to be hard envs where env-SR is 0 regardless of ablation).
- Pixel cheetah: both history-ablate-all and CEM-rollout drop 10pp — no
  differential trace use.
- **The strong form of the trace-dynamics causal claim is rejected.** The
  correlation (C5) stands; the causal role does not. Report both.

### 5.1a env-SR aggregation bug fix (v0.7.18.1) — reviewer armor
`closed_loop.py` constructed `ClosedLoopResult` without `success_rate_env`/
`std`, so top-level env-SR was always 0.0 despite correct per_seed values.
Fixed; 867 state eval JSONs re-aggregated from per_seed. TRUE state env-SR:
- Easy envs saturate 1.0: ball_in_cup 1.0, cartpole 0.99, cheetah 0.997,
  finger 0.89 (all models succeed — CEM reaches simple posture goals).
- Hard envs 0: dog, humanoid, quadruped, reacher, stacker, tworoom (all
  models fail — 5-step CEM cannot plan long-horizon locomotion).
- Per-model means 0.34-0.38 (low discrimination; discrimination comes from
  cos_dist, unchanged).

### 5.1b FAIR parameter rerun (v0.7.18.4) — strengthens the partition
Original 5m state run trained STJEWM at n_layers=2 (trainable 2.70M) vs ~5M
baselines — parameter-unfair. Retrained all 6 STJEWM readouts × 10 splits at
n_layers=4 (trainable 5.06M, verified) with identical protocol:
- cos_dist delta < 0.004 for every readout (trace 0.105→0.104, spike
  0.108→0.111, rate 0.103→0.103, no_trace 0.119→0.123, leak 0.119→0.120,
  membrane 0.124→0.122); env-SR delta < 0.03.
- Fair cluster partition identical: calibrated = STJEWM 6 + Stacked-LIF 2 + ALIF-timecell
  (0.103-0.123), collapse = GRU/MLP/LIFTransformer, over-react = LeWM 0.183.
- **Implication: STJEWM calibration is parameter-robust (2.70M vs 5.06M
  identical) — the family partition holds under parameter-fair comparison.**
- Pixel trainable clarification: 5.00M total (projector 0.074M + SNN
  predictor 4.93M), NOT 0.07M (that was only the projector).

### 5.2 Probe R2 anomalies were bugs (C9) — NEW, reviewer armor
Root causes: (1) sequential train/val split on episode-ordered data → probe
predicts out-of-distribution on val (R2 = -6.7M); (2) `ss_tot + 1e-9` epsilon
masked near-constant targets (R2 = -1.27M). Fixed with random split (seed
12345) + winsorization + near-constant flagging. Corrected values: finger
-125.967 → -0.052, fish -2188 → -0.231. **The corrected data shows encoders
do not learn strong position representations (R2 in [-0.2, 0.2]) — the
event-vs-position dissociation is a real finding, now on solid ground.**

### 5.3 Cheetah edge: noisy, softened (C10) — NEW
60 episodes/cell (10 splits, paired): STJEWM-trace 0.143 vs Stacked-LIF-trace 0.098,
pooled t = +4.15 (p = 0.0025), Wilcoxon p = 0.0078. BUT: two 30-eps halves
independently non-significant (t = 2.15, 1.12), 4/10 splits flip direction.
**Wording: "marginal advantage on the hardest env with a split-dependent
effect; not claimed as a strong edge."**

---

## 6. How this changes the paper

### Now defensible (NEW strength)
- Metric package with synthetic-validation boundaries (C4) — methods paper core.
- 3-seed CIs (C2) — statistical bottom line.
- SNN event-alignment + GRU control (C5/C6) — architecture claim, family-wide.
- 20x effective FLOPs (C7) — efficiency claim, the SNN selling point.
- Probe-bug fix (C9) — pre-empts the reviewer's cheapest kill.

### Must be softened / removed
- "Trace dynamics advantage" → correlation-only (C8 kills causality).
- "STJEWM beats Stacked-LIF" → indistinguishable within calibrated cluster (C2).
- "STJEWM event-AUROC best" → was probe bug; LeWM recovers at 3 epochs (P1-3).
- Cheetah edge → "marginal, split-dependent" (C10).

### Paper structure suggestion
1. Intro: "single-metric evaluation of latent world models is broken" (C1)
2. Methods: four-metric package + synthetic boundary validation (C4)
3. Results A: 3-cluster partition, 3-seed, multi-epoch robust (C2, C3)
4. Results B: SNN event alignment is spike-based (C5, C6), with efficiency (C7)
5. Results C: the negative — trace causality rejected (C8)
6. Discussion: what the negative means; honest limits

---

## 7. File inventory

```
results/journal_prep/
  B1_event_align_5m/        summary.md (STJEWM 32 cells), summary_fixed.md (8-model)
  B2_multiseed/             summary.md (3-seed CIs, cluster verdict)
  B3_probe_fix/             probe_table_fixed.md (200 cells), SUMMARY.md
  B4_ablation/              summary.md (6-mode ablation, CEM-rollout verdict)
  P11_energy/               energy_summary.md (10 rows), measurements.json
  P12_synthetic/            SUMMARY.md (6 encoders + sweeps)
  P13_multi_epoch/          summary.md (3/5-epoch partition + AUROC)
  P22_cheetah/              verdict_60eps.md (pooled t=4.15, NOISY EDGE)
```

Code changed (committed or pending): `code/scripts/probe.py` (B3),
`code/scripts/event_align.py` (B1Fix), `code/core/cem.py` + `code/eval/closed_loop.py`
(B4, additive hooks), `code/scripts/event_window_ablation.py` (B4, new),
`code/scripts/generalist_v0_7_5_5m/measure_energy.py` (P11, new).

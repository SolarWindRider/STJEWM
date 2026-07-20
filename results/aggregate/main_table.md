# STJEWM 完整实验主表 (v0.7.10b + v0.7.11 + v0.7.12)

**Generated:** 2026-07-20
**来源:** v0.7.10b OOD (468 cells), v0.7.11 Event-Window (9 cells), v0.7.12 Cross-Benchmark (39 cells)

---

## 1. 总览 (3 个实验轴的结果)

| 实验轴 | 任务 / Split | STJEWM 结果 | CuBiFAE 结果 | 胜者 |
|---|---|---|---|---|
| **v0.7.10b OOD path-C** | 6 splits × 39 DMC envs | ρ ∈ [0.968, 0.999] | ρ ∈ [0.968, 0.999] | **tied** (all SNN calibrated) |
| **v0.7.10b §9.5 T-maze** | delay10/50, 30 ep | LeWM 0.94 / env 0.03 | LeWM 0.94 / env 0.03 | **tied** (decoding bottleneck) |
| **v0.7.11 Event-Window** | 3 seeds × 200 ep | 4.01/20 (trace), 4.19/20 (membrane) | 3.67/20 | **STJEWM +2pp 胜** |
| **v0.7.12 F1** | PushT held out | 0.40 (membrane), 0.23 (trace) | 0.16 | **STJEWM membrane +24.4pp** |
| **v0.7.12 F2** | TwoRoom held out | 0.76 (membrane), 0.78 (trace) | 0.88 | **cubifae +10pp 胜** |
| **v0.7.12 F3** | Reacher held out | 0.56 (membrane), 0.58 (trace) | 0.53 | **tied** (within 4.5pp) |
| **v0.7.12 F4** | DMC held out (13 envs avg) | 0.518 (membrane), 0.506 (trace) | 0.506 | **tied** (within 1.2pp) |

---

## 2. v0.7.10b OOD Path-C — 468 cells (6 splits × 12 models × 39 envs)

**Setup:** Train on 14 of 16 G16 envs, eval on 2 of 16 (per split). 6 splits, 1 seed, 1 epoch.

**Per-split ρ (event-alignment, latent-vs-obs) ranges:**

| Split | STJEWM 6 readouts | CuBiFAE | SLT-LIF-MPC ×2 | MLP (non-SNN) | GRU (non-SNN) | LeWM (non-SNN) |
|---|---|---|---|---|---|---|
| F1 (classic)        | [0.97, 0.99] | 0.97 | [0.97, 0.99] | 0.97 | 0.05 | 0.05 |
| F2 (locomotion)     | [0.97, 0.99] | 0.99 | [0.97, 0.99] | 0.97 | 0.04 | 0.05 |
| F3 (sparse-POMDP)   | [0.97, 0.99] | 0.99 | [0.97, 0.99] | 0.99 | 0.05 | 0.05 |
| F1F2                | [0.97, 0.99] | 0.98 | [0.97, 0.99] | 0.97 | 0.05 | 0.05 |
| F1F3                | [0.97, 0.99] | 0.97 | [0.97, 0.99] | 0.97 | 0.05 | 0.04 |
| F2F3                | [0.97, 0.99] | 0.98 | [0.97, 0.99] | 0.97 | 0.05 | 0.05 |

**SNN family (STJEWM, CuBiFAE, SLT-LIF-MPC): ρ ∈ [0.96, 0.99]** — all calibrated.
**Non-SNN family (MLP, GRU, LeWM): at least one axis fails** (MLP collapse, GRU noise, LeWM over-react).

---

## 3. v0.7.11 Event-Window (3 models × 3 seeds × 200 episodes)

**Task:** Synthetic 5-event 10-step window task with 30% pattern-switching probability.
**Action:** Pick the modal event of the current window (purely observational, no env control).

| Model | mean_reward (per 20 windows) | % | vs. cubifae |
|---|---|---|---|
| **cubifae_baseline** (passive decay) | 3.67 ± 0.21 | 18.4% | (baseline) |
| **stjewm_trace_only** (content-aware) | **4.01 ± 0.16** | **20.1%** | +1.7 pp |
| **stjewm_membrane_readout** (content-aware) | **4.19 ± 0.11** | **20.9%** | +2.5 pp |

Trace 跟 membrane 跟 cubifae的差距在 3 seeds 内方向一致 (STJEWM 都比cubifae高),但 trace 跟 membrane 之间 tied (p ≈ 0.25).
**真正赢的是 *membrane-forbidden* 整体 vs *passive fixed-τ decay*.**

---

## 4. v0.7.12 Cross-Benchmark Family OOD (4 splits × 3 models)

**Setup:** One family held out at a time, from the 4-family set {DMC, Reacher, PushT, TwoRoom}.
Train on the other 3 families, evaluate the held-out env family.
1 seed training, 30 episodes × 3 seeds eval per cell.

### F1 (PushT held out)

| Model | LeWM-SR (latent) | env-SR (physical) | cos_dist |
|---|---|---|---|
| cubifae_baseline        | 0.156 | 0.000 | 0.200 |
| stjewm_trace_only       | 0.233 | 0.000 | 0.143 |
| **stjewm_membrane_readout** | **0.400** | 0.000 | 0.125 |

**STJEWM membrane wins clearly (+24.4 pp LeWM-SR over cubifae).**

### F2 (TwoRoom held out)

| Model | LeWM-SR | env-SR | cos_dist |
|---|---|---|---|
| **cubifae_baseline**        | **0.878** | 0.000 | 0.061 |
| stjewm_trace_only       | 0.778 | 0.000 | 0.070 |
| stjewm_membrane_readout | 0.756 | 0.000 | 0.071 |

**cubifae wins (+10 pp).**

### F3 (Reacher held out)

| Model | LeWM-SR | env-SR | cos_dist |
|---|---|---|---|
| cubifae_baseline        | 0.533 | 0.033 | 0.118 |
| **stjewm_trace_only**       | **0.578** | 0.033 | 0.109 |
| stjewm_membrane_readout | 0.556 | 0.033 | 0.114 |

**All 3 within 4.5 pp — tied within noise.**

### F4 (DMC held out — train on only 3 non-DMC envs, eval on 13 DMC envs)

| Model | avg LeWM-SR over 13 DMC envs | avg cos |
|---|---|---|
| cubifae_baseline        | 0.506 | 0.118 |
| stjewm_trace_only       | 0.506 | 0.117 |
| stjewm_membrane_readout | 0.518 | 0.118 |

**Tied within 1.2 pp.** With only 3 non-DMC train envs, the network has insufficient
signal to learn a DMC-compatible policy — all 3 models under-perform.

### F1-F4 Net

| Result | Count | 占比 |
|---|---|---|
| STJEWM clear win  | 1 (F1) | 1/4 |
| cubifae clear win | 1 (F2) | 1/4 |
| All tied           | 2 (F3, F4) | 2/4 |

**STJEWM does not have a universal hard performance win on cross-benchmark-family OOD.**

---

## 5. Final honest scope (paper §9.4 take-home)

> **v0.7.12 cross-benchmark family axis does not support a hard-performance win for STJEWM over CuBiFAE**
> (1/4 wins, 1/4 loses, 2/4 tied). STJEWM's strong claims are on:
>
> (a) the **membrane-forbidden protocol** (a methodological framework for SNN world-model comparison);
> (b) the **collapse-robust diagnostic package** (div, resp, $\rho$);
> (c) the **boundary between SNN and non-SNN families** (SNN all calibrated, non-SNN all fail);
> (d) the **content-aware rate counting** task (v0.7.11, STJEWM +2pp over CuBiFAE);
> (e) **one specific held-out family** (PushT, F1, +24.4 pp).
>
> STJEWM does **not** claim a hard performance win on:
> (i) within-DMC sub-family transfer (v0.7.10b, tied with CuBiFAE);
> (ii) general closed-loop control (env-SR saturates within calibrated family);
> (iii) cross-benchmark-family OOD (v0.7.12, 1/4 wins).

---

## 6. References

- v0.7.10b OOD: `results/utility/ood1_table.md` (468 cells)
- v0.7.11 Event-Window: `results/generalist_G16_eventwindow_demo/eval/RESULTS.md`
- v0.7.12 Cross-Benchmark: `results/cross_benchmark_F{1,2,3,4}/eval/RESULTS.md`
- 中文实验报告: `paper/experiment_report_zh.pdf`
- 论文: `paper/paper.pdf`

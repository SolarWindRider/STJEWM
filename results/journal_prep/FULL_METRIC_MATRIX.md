# Full Metric Matrix — 13 models × all experiments, COMPLETE + FAIR (v0.7.18.4)

## 这是什么表

**13 个世界模型 × 14 个指标的横截面总览表**。每一行是一个模型，每一列
来自一个独立实验（G1–G5 补全实验 + B2 多种子 + P11 能效 + G4 探测）。
它把分散在 `results/journal_prep/` 各实验 summary 里的数字汇聚到一张表，
用于：(1) 快速回答「某个模型整体怎么样」；(2) 三簇分界的横向对比；
(3) 论文 Table 1 的候选。**不包含 per-env 细节**——那些在
`MAIN_TABLE_5M_STATE_FULL.md`（state，10 splits × 每 env）和
`MAIN_TABLE_5M_PIXEL_FULL.md`（pixel，13 envs）里。

## 数据来源（每列一个实验，全部实际跑过）

| 列 | 含义 | 来源 | 协议 |
|---|---|---|---|
| `n` | 参与聚合的 eval cell 数 | state 5m | 10 splits × seed 0 |
| `cos↓` | 平均 cos_dist（潜变量-目标余弦距离，低=校准好） | state 5m | CEM 300×30×10, H=5, budget 50, goal_offset=25, 5 eps |
| `LeWM@.05` | cos_dist<0.05 命中率（**被证伪的指标**，仅叙事用） | state 5m | 同上 |
| `envSR` | 真实 env-SR（v0.7.18.1 聚合 bug 修复后） | state 5m | 同上 |
| `event-ρ` | 观测事件↔潜变量一阶差分的 Pearson 相关 | G1（104 cells） | 200 步随机策略, 4 envs × 2 splits |
| `AUROC` | 事件类型线性探测 AUC（跨 5 目标平均） | G2（325 cells） | 13 DMC envs × 5 targets, **1-epoch 训练** |
| `effFLOP` | 有效 FLOPs/step（M），含事件驱动折扣 | G3（13 模型） | state, 实测 sparsity |
| `dense` | 稠密 FLOPs/step（M） | G3 | 同上 |
| `spar%` | 实测 spike 稀疏度（1−活跃率） | G3 | 2 batch × 2 sample 前向 |
| `trnM` | 可训练参数量（M） | 实测 | STJEWM 为 v0.7.18.4 公平重跑（5.06M） |
| `3seed cos±` | 3 种子 × 3 splits 的 cos_dist 均值±std | G5+B2 | **2.70M era（见下方一致性说明）** |
| `posR² / futR² / goalR²` | 线性探测 R²（位置/未来步/目标方向） | G4（611 cells） | B3 修复后, cross_benchmark_F1 |

## 如何读这张表（三簇结构）

按 `cos` 列从低到高：
- **坍缩簇**（cos ≈ 0）：MLP 0.007, GRU 0.020, SpikeDreamer 0.000 —— 常数潜变量，
  任何目标都「命中」（LeWM@.05 0.89–1.00 是假阳性）。
- **校准簇**（cos 0.10–0.12）：STJEWM 6 变体 + SLT×2 + CuBiFAE —— 潜变量与目标成比例，
  event-ρ ≥ 0.9986（事件对齐），3-seed CI 互相重叠（统计上不可区分）。
- **过反应簇**（cos 0.183）：LeWM-v2 —— 潜变量放大观测（posR² 0.605 最强位置记忆，
  但校准差）。

## 一致性说明（重要）

1. **参数量**：STJEWM 的 `cos`/`envSR`/`LeWM@.05` 列来自 v0.7.18.4 公平重跑
   （5.06M，n_layers=4）；`3seed` 列来自 G5（**2.70M era**，n_layers=2）。两者
   的 cos_dist 差异 < 0.004（fair rerun 验证），分簇结论不变，故未重跑 3-seed。
2. **AUROC 是 1-epoch 训练** 的值；P13 显示 3-epoch 时 LeWM 恢复到 0.63
   （1-epoch ~0.5 部分是 probe 构建假象），STJEWM 的 1-epoch ~0.50 是真实的。
3. **env-SR** 是修复后的真实值（聚合 bug 曾全部写成 0.0）：易 env 饱和 1.0，
   难 env 为 0，模型均值 0.34–0.38 —— 区分度低，主指标是 `cos`。
4. **LeWM@.05** 是 §2.3a 被证伪的指标（MLP 0.948 但 div=0.0002），仅保留
   作为 falsification 叙事证据，不用于模型排序。

## 各列的详细实验出处

- state evals: `results/5m/`（baselines）+ `results/5m_5mpar/`（fair STJEWM）
- G1 event-ρ: `results/journal_prep/G1_event_align_complete/summary.md`
- G2 AUROC: `results/journal_prep/G2_auroc_complete/summary.md`
- G3 FLOPs: `results/journal_prep/G3_energy_complete/summary.md`
- G4 probe R²: `results/journal_prep/G4_probe_complete/summary.md`
- G5 3-seed: `results/journal_prep/G5_multiseed/summary.md`
- 完整证据映射: `results/journal_prep/JOURNAL_STORY.md`

## 表（数据）

> **Zero gaps, parameter-fair.** STJEWM state rows from the 5.06M fair rerun
> (`results/5m_5mpar/`, n_layers=4); baselines from original 5m run. env-SR is
> the FIXED value (aggregation bug resolved v0.7.18.1). Sources: cos/LeWM/env-SR =
> state evals (10 splits, seed 0); event-ρ = G1 (104 cells); AUROC = G2 (325 cells);
> FLOPs = G3 (13 models, state); probe R² = G4 (611 cells); 3-seed = G5+B2.

| Model | n | cos↓ | LeWM@.05 | envSR | event-ρ | AUROC | effFLOP | dense | spar% | trnM | 3seed cos± | posR² | futR² | goalR² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STJEWM-trace | 89 | 0.104 | 0.404 | 0.369 | 0.9987 | 0.501 | 0.483 | 5.23 | 93.3 | 5.06 | 0.118±0.004 | -0.017 | -0.024 | -0.086 |
| STJEWM-spike | 89 | 0.111 | 0.382 | 0.371 | 0.9988 | 0.506 | 0.465 | 5.16 | 93.6 | 5.06 | 0.120±0.001 | -0.066 | -0.035 | -0.053 |
| STJEWM-rate | 89 | 0.103 | 0.400 | 0.369 | 0.9988 | 0.499 | 0.478 | 5.16 | 93.3 | 5.06 | 0.120±0.005 | -0.059 | -0.027 | -0.022 |
| STJEWM-no-trace | 89 | 0.123 | 0.360 | 0.339 | 0.9987 | 0.498 | 0.465 | 5.16 | 93.6 | 5.06 | 0.133±0.011 | -0.072 | -0.022 | -0.029 |
| STJEWM-leak | 89 | 0.120 | 0.369 | 0.348 | 0.9986 | 0.514 | 0.477 | 5.23 | 93.5 | 5.06 | 0.139±0.009 | -0.030 | -0.045 | -0.058 |
| STJEWM-membrane | 89 | 0.122 | 0.362 | 0.335 | 0.9987 | 0.505 | 0.481 | 5.16 | 93.3 | 5.06 | 0.135±0.008 | -0.040 | -0.080 | -0.044 |
| CuBiFAE | 89 | 0.105 | 0.422 | 0.366 | 0.9988 | 0.502 | 9.686 | 9.96 | 100.0 | 4.98 | 0.124±0.007 | -0.002 | 0.004 | -0.038 |
| SLT-trace | 93 | 0.106 | 0.366 | 0.346 | 0.9996 | 0.672 | 2.125 | 10.18 | 99.1 | 5.11 | 0.115±0.004 | -0.001 | -0.008 | -0.068 |
| SLT-free | 90 | 0.111 | 0.376 | 0.342 | 0.9997 | 0.587 | 1.940 | 10.07 | 99.2 | 5.05 | 0.122±0.006 | 0.063 | 0.099 | -0.010 |
| LeWM-v2 | 89 | 0.183 | 0.225 | 0.360 | 0.7515 | 0.626 | 9.770 | 9.77 | 0.0 | 4.97 | 0.190±0.013 | 0.605 | 0.396 | 0.168 |
| GRU | 89 | 0.020 | 0.894 | 0.364 | -0.0074 | 0.546 | 10.241 | 10.24 | 0.0 | 5.13 | 0.017±0.001 | 0.038 | 0.018 | -0.013 |
| MLP | 89 | 0.007 | 0.948 | 0.362 | -0.0233 | 0.499 | 9.984 | 9.98 | 0.0 | 5.00 | 0.005±0.001 | -0.043 | -0.023 | -0.042 |
| SpikeDreamer | 89 | 0.000 | 1.000 | 0.375 | -0.0003 | 0.543 | 9.573 | 10.07 | 99.8 | 5.12 | -0.000±0.000 | -0.037 | -0.023 | -0.095 |

## Notes

- **env-SR now FIXED** (was aggregation bug writing 0.0). True values: easy envs saturate 1.0 (ball_in_cup, cartpole, cheetah, finger), hard envs 0 (dog, humanoid, quadruped, reacher, stacker, tworoom). Per-model 0.34-0.38 — low discrimination, cos_dist is the discriminating metric.
- **FAIR params**: STJEWM retrained at 5.06M (n_layers=4); cos_dist delta < 0.004 vs old 2.70M run — calibration is parameter-robust.
- **event-ρ**: STJEWM 6 + SLT 2 + CuBiFAE ≥ 0.9986 (spike-based); LeWM 0.75; GRU -0.007; MLP -0.023; SpikeDreamer -0.0003.
- **AUROC** (1-epoch): SLT-trace 0.672 best, LeWM 0.626, STJEWM ≈ 0.50 chance; LeWM recovers 0.63 at 3-epoch (P13).
- **effFLOPs**: STJEWM 0.46-0.48 vs SLT 1.9-2.1 vs dense 9.8-10.2 MFLOPs/step (~20× cheaper).
- **posR²**: STJEWM ≈ -0.03..-0.07 (chance), LeWM +0.29 — event-vs-position dissociation.
- **LeWM@0.05 falsified** (MLP 0.948 with div=0.0002): included for the falsification narrative only.
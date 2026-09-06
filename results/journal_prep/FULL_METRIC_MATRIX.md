# Full Metric Matrix — 13 models × all experiments, COMPLETE + FAIR (v0.7.18.4)


> <span style="color:red">**【数据作废公告 2026-09-06】** 因数据代际重置,本文件所有实验数字已作废并标记为待定——所有实验将统一重新训练+评测后回填。新数据落盘前请勿引用本文件任何数值。协议、模型与表格结构保留。</span>

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
| `n` | 参与聚合的 eval cell 数 | state 5m | <span style="color:red">**待定**</span> |
| `cos↓` | 平均 cos_dist（潜变量-目标余弦距离，低=校准好） | state 5m | CEM 300×30×10, H=5, budget 50, goal_offset=25, 5 eps |
| <span style="color:red">**待定**</span> | cos_dist<0.05 命中率（**被证伪的指标**，仅叙事用） | state 5m | 同上 |
| `envSR` | 真实 env-SR（v0.7.18.1 聚合 bug 修复后） | state 5m | 同上 |
| `event-ρ` | 观测事件↔潜变量一阶差分的 Pearson 相关 | G1（104 cells） | 200 步随机策略, 4 envs × 2 splits |
| `AUROC` | 事件类型线性探测 AUC（跨 5 目标平均） | G2（325 cells） | 13 DMC envs × 5 targets, **1-epoch 训练** |
| `effFLOP` | 有效 FLOPs/step（M），含事件驱动折扣 | G3（13 模型） | state, 实测 sparsity |
| `dense` | 稠密 FLOPs/step（M） | G3 | 同上 |
| `spar%` | 实测 spike 稀疏度（1−活跃率） | G3 | 2 batch × 2 sample 前向 |
| `trnM` | 可训练参数量（M） | 实测 | STJEWM 为 v0.7.18.4 公平重跑（5.06M） |
| `3seed cos±` | 3 种子 × 3 splits 的 cos_dist 均值±std | <span style="color:red">**待定**</span> | **2.70M era（见下方一致性说明）** |
| `posR² / futR² / goalR²` | 线性探测 R²（位置/未来步/目标方向） | G4（611 cells） | B3 修复后, cross_benchmark_F1 |

## 如何读这张表（三簇结构）

按 `cos` 列从低到高：
- **坍缩簇**（cos ≈ 0）：MLP <span style="color:red">**待定**</span>, GRU <span style="color:red">**待定**</span>, LIFTransformer <span style="color:red">**待定**</span> —— 常数潜变量，
  任何目标都「命中」（LeWM@.05 <span style="color:red">**待定**</span> 是假阳性）。
- **校准簇**（cos <span style="color:red">**待定**</span>）：STJEWM 6 变体 + Stacked-LIF×2 + ALIF-timecell —— 潜变量与目标成比例，
  event-ρ ≥ <span style="color:red">**待定**</span>（事件对齐），3-seed CI 互相重叠（统计上不可区分）。
- **过反应簇**（cos <span style="color:red">**待定**</span>）：LeWM-v2 —— 潜变量放大观测（posR² <span style="color:red">**待定**</span> 最强位置记忆，
  但校准差）。

## 一致性说明（重要）

1. **参数量**：STJEWM 的 `cos`/`envSR`/`LeWM@.05` 列来自 v0.7.18.4 公平重跑
   （5.06M，n_layers=4）；`3seed` 列来自 G5（**2.70M era**，n_layers=2）。两者
   的 cos_dist 差异 < <span style="color:red">**待定**</span>（fair rerun 验证），分簇结论不变，故未重跑 3-seed。
2. **AUROC 是 1-epoch 训练** 的值；P13 显示 3-epoch 时 LeWM 恢复到 <span style="color:red">**待定**</span>
   （1-epoch ~<span style="color:red">**待定**</span> 部分是 probe 构建假象），STJEWM 的 1-epoch ~<span style="color:red">**待定**</span> 是真实的。
3. **env-SR** 是修复后的真实值（聚合 bug 曾全部写成 <span style="color:red">**待定**</span>）：易 env 饱和 <span style="color:red">**待定**</span>，
   难 env 为 0，模型均值 <span style="color:red">**待定**</span> —— 区分度低，主指标是 `cos`。
4. **LeWM@.05** 是 §2.3a 被证伪的指标（MLP <span style="color:red">**待定**</span> 但 div=<span style="color:red">**待定**</span>），仅保留
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
| STJEWM-trace | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-spike | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-rate | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-no-trace | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-leak | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-membrane | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| ALIF-timecell | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-trace | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-free | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LeWM-v2 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| GRU | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| MLP | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LIFTransformer | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |

## Notes

- **env-SR now FIXED** (was aggregation bug writing <span style="color:red">**待定**</span>). True values: easy envs saturate <span style="color:red">**待定**</span> (ball_in_cup, cartpole, cheetah, finger), hard envs 0 (dog, humanoid, quadruped, reacher, stacker, tworoom). Per-model <span style="color:red">**待定**</span> — low discrimination, cos_dist is the discriminating metric.
- **FAIR params**: STJEWM retrained at 5.06M (n_layers=4); cos_dist delta < <span style="color:red">**待定**</span> vs old 2.70M run — calibration is parameter-robust.
- **event-ρ**: STJEWM 6 + Stacked-LIF 2 + ALIF-timecell ≥ <span style="color:red">**待定**</span> (spike-based); LeWM <span style="color:red">**待定**</span>; GRU <span style="color:red">**待定**</span>; MLP <span style="color:red">**待定**</span>; LIFTransformer <span style="color:red">**待定**</span>.
- **AUROC** (1-epoch): Stacked-LIF-trace <span style="color:red">**待定**</span> best, LeWM <span style="color:red">**待定**</span>, STJEWM ≈ <span style="color:red">**待定**</span> chance; LeWM recovers <span style="color:red">**待定**</span> at 3-epoch (P13).
- **effFLOPs**: STJEWM <span style="color:red">**待定**</span> vs Stacked-LIF <span style="color:red">**待定**</span> vs dense <span style="color:red">**待定**</span> MFLOPs/step (~20× cheaper).
- **posR²**: STJEWM ≈ <span style="color:red">**待定**</span>..<span style="color:red">**待定**</span> (chance), LeWM +<span style="color:red">**待定**</span> — event-vs-position dissociation.
- **LeWM@0.05 falsified** (MLP <span style="color:red">**待定**</span> with div=<span style="color:red">**待定**</span>): included for the falsification narrative only.
- **External baseline: Spiking-WM (PNAS 2025, Brain-Cog-Lab)** — 唯一真实外部竞品（28.5M, pixels-free proprio 配置）。Its native episode returns and event-ρ (<span style="color:red">**待定**</span> over 12 DMC tasks, mean <span style="color:red">**待定**</span>, vs STJEWM ≥ <span style="color:red">**待定**</span> on every task) live in `MAIN_TABLE_5M_STATE_FULL.md` §"External comparison" — 指标语义不同（native return vs CEM env-SR/cos），故不并入本横截面表。
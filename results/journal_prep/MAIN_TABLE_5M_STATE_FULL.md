# Experiment 1: 5M-aligned STATE — FULL per-env main table (FAIR, STJEWM 5.06M)


> <span style="color:red">**【数据作废公告 2026-09-06】** 因数据代际重置,本文件所有实验数字已作废并标记为待定——所有实验将统一重新训练+评测后回填。新数据落盘前请勿引用本文件任何数值。协议、模型与表格结构保留。</span>

## 实验是什么

**5M 参数对齐的低维状态观测跨任务实验**：13 个世界模型（6 个 STJEWM
readout + 7 个 baseline）在 10 个任务组合（split）下训练并闭环评估，
每个 split 是不同环境的训练/留出组合。这是论文的**主实验**——
回答「在可训练参数量对齐（~5M）时，不同架构的潜状态校准质量如何」。

## 方法（训练）

- **模型**：13 个，全部可训练参数 4.97–5.13M（±3.2%）
  - STJEWM 6 readout（trace/spike/rate/no-trace/leak/membrane），n_layers=4，
    **5.06M**（v0.7.18.4 公平重跑；原 2.70M 版本结果一致，delta<<span style="color:red">**待定**</span>）
  - ALIF-timecell 4.98M, Stacked-LIF-trace 5.11M, Stacked-LIF-free 5.05M, LeWM-v2 4.97M,
    GRU 5.13M, MLP 5.00M, LIFTransformer 5.12M
- **数据**：`configs/oodc_5m/<split>.json` 指定的 dm_control 离线数据集
  （250k 步 npz），低维状态向量（pad 到 128 维）+ 56 维动作
- **训练协议**：1 epoch, batch 32, AdamW lr=3e-4, seed 0, history_size=1,
  goal_offset=25, loss = pred + 0.09·sigreg + 0.5·goal（STJEWM 系）

## 方法（评估）

- **CEM 规划器**：300 样本 × 30 elites × 10 迭代, horizon=5, budget=50
- **goal 来源**：数据集内**同轨迹 t+25 步的真实状态**（动态目标，需长程预测）
- **每 cell**：1 个（split, 模型, env）组合 = 5 episodes × 1 seed
- **每 split 的 env 集**：F1/F2/F3 各 14 envs（含 1 个留出族），oodc 系列 5–10 envs，
  G16 为 15 envs 并集

## 指标（每个 cell 两个数，格式 `env-SR / cos_dist`）

| 指标 | 定义 | 方向 | 备注 |
|---|---|---|---|
| **env-SR** | 闭环成功率：最终状态是否在 goal 的 tolerance 内 | 高=好 | v0.7.18.1 修复聚合 bug 前的真实值；易 env 饱和 1.0，难 env 为 0 |
| **cos_dist** | 最终潜状态与目标潜状态的余弦距离（1−cos）/2 | 低=好 | **主指标**；不受 env-SR 饱和影响，能区分三簇 |

## 怎么读这张表

- **每行一个模型**（含可训练参数 Trn(M)），每列一个 env，cell = `env-SR / cos_dist`
- **按 cos_dist 看三簇**（跨 env 一致）：
  - 校准簇（cos <span style="color:red">**待定**</span>）：STJEWM 6 + Stacked-LIF 2 + ALIF-timecell——潜状态与目标成比例
  - 坍缩簇（cos ≈ <span style="color:red">**待定**</span>）：GRU / MLP / LIFTransformer——常数潜变量，env-SR 的
    <span style="color:red">**待定**</span> 是静态可达的假象，cos_dist 才是真相
  - 过反应簇（cos <span style="color:red">**待定**</span>）：LeWM-v2——潜变量放大观测差异
- **按 env 看两级分化**：易 env（ball_in_cup/cartpole/cheetah/finger）所有模型
  env-SR≈<span style="color:red">**待定**</span>（CEM 5 步够得着）；难 env（dog/humanoid/quadruped/reacher/stacker/
  tworoom）所有模型 env-SR=0（需要长程协调，5 步规划够不到）——env-SR 不区分
  模型，区分度在 cos_dist

## 数据说明了什么（结论）

1. **三簇分界跨 10 splits 稳定**：校准/坍缩/过反应在任何 split 都不交换位置
2. **校准是 SNN 家族属性**：STJEWM（6 readout）、Stacked-LIF（2 变体）、ALIF-timecell 都校准，
   与 readout 协议无关（校准不变性）
3. **连续 RNN/Transformer 失败模式不同**：GRU/MLP 坍缩，LeWM 过反应——
   单指标（如 LeWM-SR）无法区分这两种失败，需要多指标包
4. **参数公平后结论不变**：STJEWM 从 2.70M 提到 5.06M，cos_dist 变化 < <span style="color:red">**待定**</span>
   ——校准是架构属性，不是参数量红利
5. **env-SR 是「易/难」二分，不是模型质量**：所有模型在易 env 成功、难 env
   失败——env-SR 测的是 CEM 规划能力天花板，不是潜状态质量

## 重要 caveat

- **env-SR 全部经过 v0.7.18.1 修复**（聚合 bug 曾把顶层写成 <span style="color:red">**待定**</span>）；per_seed
  数值一直正确，修复后顶层与 per_seed 一致
- **STJEWM 行来自 5.06M 公平重跑**（`results/5m_5mpar/`），baseline 行来自原
  5m 运行——同一协议，可横向对比
- **与 pixel 表不可直接比 env-SR**（pixel 用静态 goal，此处用 t+25 动态 goal）；
  cos_dist 可比（同一目标函数下的潜状态质量）

## 数据出处

- evals: `results/5m/<split>/<model>/seed_0/eval_<env>.json`（baselines）+
  `results/5m_5mpar/<split>/stjewm_<readout>/seed_0/`（fair STJEWM）
- 训练: `results/5m/_logs/` + `results/5m_5mpar/_logs/`
- 聚合逻辑: `results/journal_prep/`（FULL_METRIC_MATRIX.md 有 13 模型 × 14 指标横截面）

> Protocol: CEM 300×30×10, H=5, budget 50, goal_offset=25, 5 eps × 1 seed. Cell: **env-SR** / cos_dist.
> **FAIR rerun (v0.7.18.4)**: STJEWM 6 readouts retrained at n_layers=4 (trainable 5.06M,
> matching baselines ~5M). Original 2.70M run showed identical cos_dist (delta < <span style="color:red">**待定**</span>) —
> calibration is parameter-robust. Baselines unchanged from original 5m run.

## F1 (PushT held out) (14 envs)

| Model | Trn(M) | ball_in_cup | cartpole_2d | cheetah | dog | finger | fish | hopper | humanoid | pendulum_2d | quadruped | reacher | stacker | tworoom | walker |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STJEWM-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-spike | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-rate | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-no-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-leak | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-membrane | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| ALIF-timecell | 4.98 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-trace | 5.11 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-free | 5.05 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LeWM-v2 | 4.97 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| GRU | 5.13 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| MLP | 5.00 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LIFTransformer | 5.12 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |

## F2 (TwoRoom held out) (14 envs)

| Model | Trn(M) | ball_in_cup | cartpole_2d | cheetah | dog | finger | fish | hopper | humanoid | pendulum_2d | pusht | quadruped | reacher | stacker | walker |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STJEWM-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-spike | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-rate | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-no-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-leak | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-membrane | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| ALIF-timecell | 4.98 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-trace | 5.11 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-free | 5.05 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LeWM-v2 | 4.97 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| GRU | 5.13 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| MLP | 5.00 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LIFTransformer | 5.12 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |

## F3 (Reacher held out) (14 envs)

| Model | Trn(M) | ball_in_cup | cartpole_2d | cheetah | dog | finger | fish | hopper | humanoid | pendulum_2d | pusht | quadruped | stacker | tworoom | walker |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STJEWM-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-spike | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-rate | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-no-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-leak | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-membrane | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| ALIF-timecell | 4.98 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-trace | 5.11 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-free | 5.05 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LeWM-v2 | 4.97 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| GRU | 5.13 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| MLP | 5.00 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LIFTransformer | 5.12 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |

## oodc_F1 (5 envs)

| Model | Trn(M) | ball_in_cup | cartpole_2d | cheetah | finger | pendulum_2d |
|---|---|---|---|---|---|---|
| STJEWM-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-spike | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-rate | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-no-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-leak | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-membrane | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| ALIF-timecell | 4.98 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-trace | 5.11 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-free | 5.05 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LeWM-v2 | 4.97 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| GRU | 5.13 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| MLP | 5.00 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LIFTransformer | 5.12 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |

## oodc_F1F2 (10 envs)

| Model | Trn(M) | ball_in_cup | cartpole_2d | cheetah | dog | finger | hopper | humanoid | pendulum_2d | quadruped | walker |
|---|---|---|---|---|---|---|---|---|---|---|---|
| STJEWM-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-spike | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-rate | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-no-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-leak | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-membrane | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| ALIF-timecell | 4.98 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-trace | 5.11 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-free | 5.05 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LeWM-v2 | 4.97 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| GRU | 5.13 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| MLP | 5.00 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LIFTransformer | 5.12 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |

## oodc_F1F3 (5 envs)

| Model | Trn(M) | ball_in_cup | cartpole_2d | cheetah | finger | pendulum_2d |
|---|---|---|---|---|---|---|
| STJEWM-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-spike | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-rate | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-no-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-leak | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-membrane | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| ALIF-timecell | 4.98 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-trace | 5.11 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-free | 5.05 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LeWM-v2 | 4.97 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| GRU | 5.13 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| MLP | 5.00 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LIFTransformer | 5.12 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |

## oodc_F2 (5 envs)

| Model | Trn(M) | dog | hopper | humanoid | quadruped | walker |
|---|---|---|---|---|---|---|
| STJEWM-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-spike | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-rate | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-no-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-leak | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-membrane | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| ALIF-timecell | 4.98 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-trace | 5.11 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-free | 5.05 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LeWM-v2 | 4.97 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| GRU | 5.13 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| MLP | 5.00 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LIFTransformer | 5.12 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |

## oodc_F2F3 (6 envs)

| Model | Trn(M) | cheetah | dog | hopper | humanoid | quadruped | walker |
|---|---|---|---|---|---|---|---|
| STJEWM-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-spike | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-rate | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-no-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-leak | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-membrane | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| ALIF-timecell | 4.98 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-trace | 5.11 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-free | 5.05 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LeWM-v2 | 4.97 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| GRU | 5.13 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| MLP | 5.00 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LIFTransformer | 5.12 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |

## oodc_F3 (1 envs)

| Model | Trn(M) | cheetah |
|---|---|---|
| STJEWM-trace | 5.06 | <span style="color:red">**待定**</span> |
| STJEWM-spike | 5.06 | <span style="color:red">**待定**</span> |
| STJEWM-rate | 5.06 | <span style="color:red">**待定**</span> |
| STJEWM-no-trace | 5.06 | <span style="color:red">**待定**</span> |
| STJEWM-leak | 5.06 | <span style="color:red">**待定**</span> |
| STJEWM-membrane | 5.06 | <span style="color:red">**待定**</span> |
| ALIF-timecell | 4.98 | <span style="color:red">**待定**</span> |
| Stacked-LIF-trace | 5.11 | <span style="color:red">**待定**</span> |
| Stacked-LIF-free | 5.05 | <span style="color:red">**待定**</span> |
| LeWM-v2 | 4.97 | <span style="color:red">**待定**</span> |
| GRU | 5.13 | <span style="color:red">**待定**</span> |
| MLP | 5.00 | <span style="color:red">**待定**</span> |
| LIFTransformer | 5.12 | <span style="color:red">**待定**</span> |

## G16 (15 envs)

| Model | Trn(M) | ball_in_cup | cartpole_2d | cheetah | dog | finger | fish | hopper | humanoid | pendulum_2d | pusht | quadruped | reacher | stacker | tworoom | walker |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STJEWM-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-spike | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-rate | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-no-trace | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-leak | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| STJEWM-membrane | 5.06 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| ALIF-timecell | 4.98 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-trace | 5.11 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| Stacked-LIF-free | 5.05 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LeWM-v2 | 4.97 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| GRU | 5.13 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| MLP | 5.00 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| LIFTransformer | 5.12 | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |

## External comparison: Spiking-WM（PNAS 2025，唯一真实外部竞品）

**Spiking-WM**（Sun, Zhao, Lv & Zeng，Brain-Cog-Lab，中科院自动化所；PNAS 2025,
doi:10.1073/pnas.2513319122；arXiv:2503.00713；开源代码 Brain-Cog-Lab/Spiking-WM, MIT）
是 13 个模型中唯一有论文、有公开代码、可独立验证的第三方系统：full spiking Dreamer，
循环状态为 multi-compartment 神经元（MCN）发出的 spike 序列——事件驱动动力学满足
协议要求，但规划器读由 spike 序列导出的**连续 posterior mean**（违反 membrane-forbidden
读出合约）。

**协议差异（为什么不并入上面的 per-env 表）**：Spiking-WM 评测用其原生
episode return（max 1000，500 步 time limit），训练预算按难度 2–5×10⁵ env steps
（pendulum/cup/reacher 2×10⁵，hopper/fish 3×10⁵，其余 5×10⁵），seed 0，28.5M
可训练参数，MCRNN spike rate 0.8%；ST-JEWM 列为 goal-conditioned CEM 的
env-SR / cos_dist（F1 checkpoint，walker 为其 held-out family）。两套指标语义
不同，**数字不可直接比**——本节只做定向定性对照。

| Task | SpWM return | ST env-SR | ST cos-dist | SpWM ρ | SpWM ρ_stoch | SpWM ρ_spike | SpWM rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| cartpole_swingup | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| cheetah_run | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| walker_walk | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| finger_spin | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| pendulum_swingup | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| cup_catch | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| reacher_easy | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| hopper_hop | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| quadruped_walk | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| dog_walk | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| fish_swim | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |
| humanoid_run | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> | <span style="color:red">**待定**</span> |

\* walker_walk 是 F1 checkpoint 的 held-out family（ST-JEWM 未见过 walker 数据；
Spiking-WM 直接训练 walker）。

**两条结论**：

1. **控制侧 mixed**：Spiking-WM 解易任务（cup_catch <span style="color:red">**待定**</span>、reacher <span style="color:red">**待定**</span>）并学习
   中间任务（cartpole <span style="color:red">**待定**</span>、walker <span style="color:red">**待定**</span>、finger <span style="color:red">**待定**</span>、cheetah <span style="color:red">**待定**</span>，且只用了
   其发表预算的一半 + 降频更新），难任务失败（pendulum <span style="color:red">**待定**</span>、hopper <span style="color:red">**待定**</span>、
   humanoid <span style="color:red">**待定**</span>、dog <span style="color:red">**待定**</span>）。
2. **对齐诊断跨全部 12 任务成立**：Spiking-WM latent event-ρ = <span style="color:red">**待定**</span>（均值
   <span style="color:red">**待定**</span>），全面低于 ST-JEWM 家族在任一任务上的 ≥ <span style="color:red">**待定**</span>；两系统的 raw spike-rate
   对齐均为 chance（SpWM −<span style="color:red">**待定**</span>，ST-JEWM <span style="color:red">**待定**</span>）——差异不在"是否
   脉冲"，而在协议暴露的 gated trace 上。MCN spike 序列（均值 0.8% 激活）比协议
   trace 稀疏一个数量级。

数据出处：SpWM return = `results/spiking_wm/logs_<task>/metrics.jsonl` 最后一条
eval_return（12/12 与 NMI Table 2 逐格精确）；SpWM ρ/ρ_stoch/ρ_spike =
`results/spiking_wm/` 协议评测（2000 随机策略步）；ST env-SR/cos = F1 checkpoint
对应 env cell（见上文 F1 表）。

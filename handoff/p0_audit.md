# P0 审计决议(2026-10-21,对照第二轮外部审计)

审计的 7 个 P0 项逐条决议。状态:✅ 已闭合 / 📝 已文档化(写作约束)。

## P0.1 参数口径 ✅ 已闭合(实测)

按 ckpt 逐张量统计(脚本内联,可复现:载入 final.pt 按 top-level 前缀求和):

| 模型 | total | 冻结 encoder | trainable |
|---|---|---|---|
| stjewm_trace_only (state) | 10.523M | 5.459M(ViT-Tiny) | **5.064M** |
| stjewm trace (pixel, F3) | 10.461M | 5.459M | **5.002M** |
| lewm_baseline_v2 | 4.970M | —(全可训练) | 4.970M |
| gru_baseline | 5.132M | — | 5.132M |
| mlp_baseline | 5.001M | — | 5.001M |

**结论**:"5M 口径" = **可训练参数**,全模型 4.97–5.13M(±3.2%)闭合;STJEWM 额外携带 5.459M 冻结 ImageNet ViT(不计入匹配预算)。论文 "≈5.06M trainable" 正确;旧 handoff 的 "16.57M/10.54M" 作废(errata.md)。state 与 pixel 的 STJEWM 组件差异:pixel 无 state_projector,基线编码器可训练且计入 5M——**所有模型同一口径(可训练参数),state/pixel 分别成立**。

## P0.2 "Event history" 语义 📝 已文档化(protocol.md §2.8)

trace 的事件历史 = 单次前向内部的时间整合(训练 53 帧窗;想象 rollout 25 步自回归);真实时间规划边界无记忆(history_size=1)。标题/示意图禁止暗示跨真实时间的 history state。

## P0.3 Fig.2 c20 / Fig.3 ρ=0.905 溯源 ✅ 已闭合

出处 = `results/mech_0917_b/`(2026-09-17 21:12 生成;JSON 内嵌 ckpt 路径 = `/data/lx/tmp/results/5m_5mpar/.../final.pt` = **最终修复代际 ✓**):
- c(20) pooled:trace 0.032、membrane 6.966、gru 0.040、lewm 0.205(`rollout/rollout_summary.md`,与初稿 Fig.2 的 0.032/6.97/0.040/0.205 精确一致);
- CSR↔env-SR Spearman +0.905(p=0.002)(`scaling/scaling_summary.md:40`;n=8 架构,关联性措辞);
- scaling_summary 自带 provenance 引用链(rollout/gain/action_ema/env_sr 四源)。
命名:初稿 Fig.2 "S1 perturbation protocol" = `mech_0917_b` 的 **S1 closed-loop rollout stability**;与旧 a 批次 `mech_0917/history` 无关。该批次 3.0M 已 -f 入库(commit 见 git log)。

## P0.4 E9 ablate_all 形式化 📝 已文档化(protocol.md §2.6)

`ablate_all` = 在全部规划步用 `zero_stjewm_trace(model)` 包裹 `model.predict`(**仅 imagination 期间置零 trace 通路**);不触及 CEM 初始编码 z_t 的 trace 读出、不触及训练。正确表述:"规划期想象中的 trace 内容非 env-SR 必要条件(cartpole 0.60→0.70)"。逐 cell 明细在 ⚠️库外 `trace_ablation_final/`;如需 paired per-cell 表,由该目录再聚合(脚本存在)。

## P0.5 S3 绝对 gain 与 CSR 分母 📝 已文档化 + 数据在库

绝对 g_obs/g_a(含量纲、逐模型)= `results/mech_0917_b/gain/gain_summary.{json,md}`(如 mlp 0.000/6.97e-06 量级差异可见)。写作要求:CSR 与绝对幅度并列;不得把低 g_obs 写成"滤除不相关观测"(扰动任务相关性未证明);action 注入路径(predict 通路)须注明(protocol.md §2.1/§3)。

## P0.6 评测数据独立性 📝 已文档化(protocol.md §2.7)

in_dist 的 (init,goal) 与训练同源 250k 轨迹 → **轨迹层面重叠存在**;成功率表述 = "dataset-derived, goal-conditioned, physical-environment success"。E4 `heldout_*` = 环境组合 held-out(训练 split 补集),非轨迹级。规模轴混杂因子声明保留。

## P0.7 Pixel vs State 协议混杂 📝 已文档化(protocol.md §2.9)

H25×2 段 vs H5×10 段;跨模态陈述限定"协议内重复观察"。

## P1 处置

claims_evidence.md §1 的分级表已含审计要求的全部降级/删除项(event-sync 删除、readout-causal 降级为 C、CSR=探索性关联、E9 限定、pixel 分别报告 trace/spike、20× 禁用等);与第二轮审计 P1 表逐行一致。

## P2 交接材料状态

| 审计要求 | 状态 |
|---|---|
| 1. journal_prep 原表入库 | ✅ commit bbacd08(EXP_Report v4.0 + MAIN×2 + E13/E14 等);G5/B2 汇总在 ⚠️库外 `agg_final/`(路径已在 numbers.md 标注) |
| 2. Fig source_data + 逐 panel 出处 | 部分:mech_0917_b 表(本日入库)+ scaling provenance 链 ✅;**逐 panel source_data.csv 未成** —— Fig.1-5 由 `make_fig{1,2}.py`/`make_nmi_figures.py` 生成,重跑即得;列入待办 |
| 3. 代码文件 | ✅ 全部在库(README Experiment→code map;strict-load 模式已写入 Conventions) |
| 4. 参数拆分表 | ✅ 本文件 P0.1(state+pixel;组件级前缀统计) |
| 5. 配对 success/cos 原始数据 | cell JSON 在 ⚠️库外 state_final_corrected_20260916/(audit/aggregated_state_cells.json);journal_prep 表已入库;**逐 cell CSV 化未做**(列入待办) |
| 6. Fig.2 c20 / Fig.3 ρ 代际声明 | ✅ 本文件 P0.3(最终代际,可由 mech_0917_b + 脚本复现) |
| 7. E9 干预路径说明 | ✅ protocol.md §2.6 |

## 遗留待办(P2 未尽项)

1. Fig.1–5 逐 panel `source_data.csv` 生成(驱动图脚本已有,需一次重跑落盘);
2. E4 held-out 逐模型效果量/置信区间表(546 cells 在 ⚠️库外,未聚合入 journal_prep);
3. G5/B2 汇总从库外 `agg_final/` -f 入库(小文件,待确认后执行);
4. E9 逐 cell paired 表从 `trace_ablation_final/` 聚合入库。

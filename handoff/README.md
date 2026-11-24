# HANDOFF — 论文写作数据交接(读我第一个)

> 目标:写论文的 agent 在本目录内可以找到**每一个数字的出处、每一条主张的证据强度、每一个协议的精确定义**,不再对数据产生任何疑问。任何本目录没有的内容 = 该实验不存在或未完成,不得写入论文。

## 三条铁律

1. **数字只用两代来源:(a) 最终修复代际(final repaired generation, 2026-09-16/17)**——训练 manifest `4376bb14…`、协议 `causal_grad_sigreg_B_20260916`、freeze v7;**(b) 2026-11-20~23 审计轮新增**(oodc_F1 seed1/2 的 trace/GRU/LIF-Tx、λ=0 继续训练 control 及 λ 臂闭环)——以审计包内的 manifest/SHA256 为准。此前一切代际的数字一律作废(README "Headline results" 节与 EXP_Report v4.0 已声明)。
   数字来源优先级:`results/journal_prep/` 内已入库表格 > 审计包汇总表(`/home/lx/audit_20261120/`,OBS `obs://lixiang01/STJEWM_NMI/audit/20261120/`)> `/data/lx/tmp/results/agg_final/HEADLINE_NUMBERS.md` > 各实验原始 cell JSON。**永远不要从旧草稿、旧摘要或记忆里抄数字;写作前先过一遍 `errata.md` §1(作废清单,2026-11-23 新增 12 条)。**

2. **协议以 `protocol.md` 为准**(冻结常数 + 精确的 goal 构造 + success 判定)。写作中所有 Methods 描述与本目录冲突时,以本目录为准(它直接从代码 `closed_loop.py`/`cem.py`/`eval_one.sh` 核对过,包括 2026-10-02 的协议审计)。

3. **每条主张先查 `claims_evidence.md` 的证据分级**:
   - `A`(多实验交叉支持,可作核心主张)
   - `B`(单实验/单环境支持,需限定措辞)
   - `C`(机制性关联,禁止因果/普适措辞)
   - `VOID`(无数据,**禁止写入**)

## 文件索引

| 文件 | 内容 |
|---|---|
| `protocol.md` | 训练协议;state/pixel 闭环评测的逐步流水线(o_t→z_t→CEM→执行→度量);goal 构造;**success 判定的精确答案**(审稿人问题 2);编码约定(a≡0 的依据) |
| `experiments.md` | 全部实验清单 E1–E14、G1/G3、P12、机制探针 S/M 系列:驱动脚本、配置、输出位置、关键数字、已知坑 |
| `claims_evidence.md` | 主张→证据映射(分级);审稿人三个科学问题的直接回答;§6 现行五图结构(v2);**§7 2026-11 审计轮硬约束(可写/禁写/图注须改)** |
| `errata.md` | 作废数字清单、随机权重事故史、未完成实验清单、代际纪律 |
| `numbers.md` | 常用数字速查(每个数字标注文件出处);含 37-cell 三 seed、λ matched control、独立 probe、三 seed 动力学、剂量匹配遮蔽 |
| `p0_audit.md` | 2026-10-21 第二轮外部审计的 7 个 P0 决议与交接材料状态 |

## 外部审稿意见的对应关系(2026-10 收到)

审稿人提出的三个科学问题在 `claims_evidence.md` §3 逐条回答:
1. "trace 优越性是否来自 latent collapse" → §3.1(非退化证据包,8 条独立证据);
2. "env-SR 到底测什么" → §3.2(答案:**真实执行后的环境状态**,非模型预测;`closed_loop.py` 的 `physical_state()` → `check_success`);
3. "CSR↔success 相关是否普遍规律" → §3.3(n=8 架构关联,禁用 "general principle")。

(历史)2026-10 审稿人建议的五图结构(旧初稿图号)的逐图映射在 `claims_evidence.md` §4;**现行图结构以 §6(v2 五图:Fig.1 解离 → Fig.2 可及性 → Fig.3 潜动力学 → Fig.4 选择性响应 → Fig.5 CSR 关联)与 §7 为准**,旧新图号映射见 `experiments.md`。

## 仓库入口(代码级 ground truth)

- 闭环评测:`code/eval/closed_loop.py`(CLI 严格加载,打印 "trained weights loaded strictly")
- CEM:`code/core/cem.py`;编码:`code/core/encode.py`;训练:`code/train/train.py`
- 复现命令与逐实验脚本映射:`README.md` "Experiment → code file map"(2026-10-02 已提交)
- 权威表格:`results/journal_prep/`(EXP_Report.md v4.0 为 hub 文档,§12.1–12.9 每个实验一节)

## 主图 v2(2026-11-20 编者重排)

面板改名式重排(零重绘):`/home/lx/figs_nmi/20261120/`(14 面板 + `nmi_figures_v2.tex` + sha256 manifest);映射与措辞边界见 `claims_evidence.md` §6。执行语义红线(state 分段开环 vs pixel first-action MPC)见 `protocol.md` §2.1/§2.4——Methods 禁止统一写成 receding-horizon。

## 2026-11 审计轮(必读;细节在 `claims_evidence.md` §7、`numbers.md`、`errata.md` §1)

三轮外部审计(11-20 / 11-22 / 11-23)后的状态:

- **最关键(2026-11-24 LeWM 核查,`audit.md` v9)**:表里的 "LeWM" **不是论文的 LeWM**——是自写的状态输入版,只训了 1 epoch / 5,000 步 / 16 环境共享 16 万窗口,编码器尾部与预测器共用权重;13 个模型的单步 latent 预测 R² 全为负。官方权重 + 官方规划器走我们的环境循环得 **88%**(官方循环 88%)→ 环境侧 harness 正确;但 (1) `train.py` 在 `history_size>1` 时预测目标错位(**已修**,H=1 结果不变);(2) 我们的 CEM 在原生动作空间按 N(0,1) 提议,官方在标准化空间; (3) DMC 只在末态判成功; (4) DMC 数据无速度 → 初态速度清零,运动环境 80–99% 的目标不可达,另 6 个环境 49–100% 的目标起点即在容差内。**全部闭环 env-SR 与一切以 LeWM 为对照的排序作废**,待用户决定重做方案(`errata.md` 第 34–39 行、`claims_evidence.md` §10)。
- **最终定位(2026-11-24,AGR)**:要求主动运动的目标协议下,随机地板 0.031,**全部 13 个模型 0.012–0.049**——没有任何模型表现出目标到达控制能力;"控制/规划能力"类主张全部禁写(`claims_evidence.md` §9)。可保留的是表示层结论与规划通路诊断。
- **最关键(2026-11-23)**:该 goal-reaching 协议的成功判据主要被"保持不动"满足——**零动作 0.477 > 随机 0.356 > Trace 0.311**;所有"闭环控制能力"类措辞撤回,改写为"相对 GRU/LeWM 更少偏离不动地板";Fig.1b 须叠加两条地板(`claims_evidence.md` §8、`numbers.md` 参照地板节)。
- ~~"已核实并可写":Trace 对 GRU/LeWM 的 env-SR 排序在 37-cell 配对网格跨三训练 seed 复现~~ → **作废(2026-11-24)**:LeWM 对照不成立,且 DMC 闭环协议有缺陷。仍可写(表示层):oodc_F1 同 split 动力学(pooled c20、输入响应不对称)跨 seed 复现(环境依赖);剂量匹配后 Trace 对历史位置掩码更敏感;λ 辅助目标改变表示(可访问性、收缩)。
- **已撤回**:"预测发散但控制成功"(实为 z* 分母塌缩)、"SNN 天然不坍缩"、"Trace 优于 LIF-Tx"、"SLIF CSR=2.8e4 为真实敏感度"、"λ=1 相对 control 额外下降"(未跨 seed 复现)。
- **全部遗留实验已于 2026-11-23 补齐**(随机/零动作参照、剂量响应、E9 扩样、λ 多 seed、P13 多 epoch、G2 事件 AUROC、utility 4 driver 含 18 个新训 ckpt、P22 零动作操作化、E4 审计),清单见 `experiments.md` 末表;仍不存在的只有"需主动运动的 goal 协议"与"固定权重 readout 交换"(`errata.md` §3)。
- **面板现行版**:`/home/lx/figs_nmi/20261120/` 的 `fig4b_panel_v3.*` 与 `fig5b_panel_v2.*`;其余面板沿用;审计汇总表入库副本在 `verification_bundle/audit_20261120/`。

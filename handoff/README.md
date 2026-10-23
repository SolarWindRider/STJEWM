# HANDOFF — 论文写作数据交接(读我第一个)

> 目标:写论文的 agent 在本目录内可以找到**每一个数字的出处、每一条主张的证据强度、每一个协议的精确定义**,不再对数据产生任何疑问。任何本目录没有的内容 = 该实验不存在或未完成,不得写入论文。

## 三条铁律

1. **只用最终修复代际(final repaired generation, 2026-09-16/17)的数字。**
   训练 manifest `4376bb14…`、协议 `causal_grad_sigreg_B_20260916`、freeze v7。
   此前一切代际的数字一律作废(README "Headline results" 节与 EXP_Report v4.0 已声明)。
   数字来源优先级:`results/journal_prep/` 内已入库表格 > `/data/lx/tmp/results/agg_final/HEADLINE_NUMBERS.md` > 各实验原始 cell JSON。**永远不要从旧草稿、旧摘要或记忆里抄数字。**

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
| `claims_evidence.md` | 主张→证据映射(分级);审稿人三个科学问题的直接回答;逐图(Fig.1–4)问题与可用的修复数据 |
| `errata.md` | 作废数字清单、随机权重事故史、未完成实验清单、代际纪律 |
| `numbers.md` | 常用数字速查(每个数字标注文件出处) |

## 外部审稿意见的对应关系(2026-10 收到)

审稿人提出的三个科学问题在 `claims_evidence.md` §3 逐条回答:
1. "trace 优越性是否来自 latent collapse" → §3.1(非退化证据包,8 条独立证据);
2. "env-SR 到底测什么" → §3.2(答案:**真实执行后的环境状态**,非模型预测;`closed_loop.py` 的 `physical_state()` → `check_success`);
3. "CSR↔success 相关是否普遍规律" → §3.3(n=8 架构关联,禁用 "general principle")。

审稿人建议的五图结构(Fig.1 可读性≠控制力 → Fig.2 动力学差异 → Fig.3 选择性机制 → Fig.4 可解码性≠效用 → Fig.5 泛化)所需的全部数据**都已存在**,逐图映射见 `claims_evidence.md` §4。

## 仓库入口(代码级 ground truth)

- 闭环评测:`code/eval/closed_loop.py`(CLI 严格加载,打印 "trained weights loaded strictly")
- CEM:`code/core/cem.py`;编码:`code/core/encode.py`;训练:`code/train/train.py`
- 复现命令与逐实验脚本映射:`README.md` "Experiment → code file map"(2026-10-02 已提交)
- 权威表格:`results/journal_prep/`(EXP_Report.md v4.0 为 hub 文档,§12.1–12.9 每个实验一节)

## 主图 v2(2026-10-22 编者重排)

面板改名式重排(零重绘):`/home/lx/figs_nmi/20261022/`(14 面板 + `nmi_figures_v2.tex` + sha256 manifest);映射与措辞边界见 `claims_evidence.md` §6。执行语义红线(state 分段开环 vs pixel first-action MPC)见 `protocol.md` §2.1/§2.4——Methods 禁止统一写成 receding-horizon。

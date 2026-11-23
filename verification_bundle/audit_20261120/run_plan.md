# run_plan.md —— P0/P1 后的证据缺口与最小补跑批次(2026-11-20)

P0(核查)与 P1(回收)已完成,见 audit.md。以下为仍缺证据与建议执行的最小批次。**总预算:0 次新训练,全部为仅推理评估**。

## 结论速览

| 清单项 | 状态 | 说明 |
|---|---|---|
| P2.A 补训练 seed | **不需要** | E2 已含 13 模型 × seeds{1,2} × 37 cells(含 trace/GRU/LeWM/membrane),配对差值方向跨 seed 一致(audit.md P1.1);仅当需要 seed0 网格(E1 的 100 cells)上的严格配对时,补"用 seed1/2 checkpoint 跑 E1 网格"的纯推理评估(可选,非必须) |
| P2.B λ sweep 闭环 | **需要(仅推理)** | 见批次 1 |
| P2.C 固定权重动作通路剂量响应 | **建议(仅推理)** | 见批次 2 |
| P2.D 外推轴 | **不需要新实验** | 像素(E3,1690 cells)与 held-out(E4,546 cells)数据已在库;按 audit.md 的限定措辞直接写入正文即可 |

## 批次 1(P2.B):辅助解码 λ 的闭环总效应 —— 仅推理

- **输入**:4 个 checkpoint(λ=0 → 5m_5mpar trace final.pt;λ=0.1/1/10 → accessibility_20260917 三 ckpt),全部已存在。
- **协议**:完全复用主结果闭环协议(`closed_loop.py`,state 分段开环、CEM 300/30/10、H25、budget 50、goal=轨迹 t+25、env-SR 终点判定、eval seed 同主结果)→ 与 E1 逐 cell 可配对。
- **规模**:4 arms × E1 网格(100 cells × 5 episodes)= 400 cells ≈ 0.6–1 GPU·h。
- **产出**:每 λ 的成功率、相对 λ=0 的逐 cell 配对差值、decoder score(access_summary)、c20(dose 口径,已有)三者并排 → 回答"解码↑时规划成功率是否同步变化/是否存在解码改善而成功率持平或下降"。
- **风险**:fine-tune 只 1 epoch,λ=0.1 可能与参考几乎相同;如 paired diff 全 0,如实报告(总效应为零也是结论)。

## 批次 2(P2.C):冻结 trace 的动作通路剂量响应 —— 仅推理,小规模先行

- **输入**:冻结 trace checkpoint;**无需任何训练**。
- **协议**(按清单 C):同一次 CEM 搜索的所有候选共用参考动作序列 ā(= 该步 planner 当前的 EMA 参考或全零合法动作,写入配置);仅在模型想象 rollout 中输入 a′=ā+γ(a−ā),**干预仅进入候选打分;环境执行原始选定的完整 25 步动作序列,两段之间按原 runner 刷新真实观测**;γ∈{1,0.75,0.5},先 γ=1 少量实例验证复现原始 planner(成功率与选定动作一致),再全量配对。
- **规模**:先导 5–10 cells(按预先规则选取;GPU 分钟级);γ=1 必须先重现原候选得分排序、选定序列、段末状态与成功判定,再全量 100 cells × 3 γ ≈ 0.5 GPU·h。
- **产出**:每档 planner 所见动作响应/g_obs、c20(S1 口径)、成功率、终点距离、候选得分并列数;解释范围=**冻结模型在规划干预下对候选动作差异的依赖**,观测通路保持正常刷新。
- **注意**:此干预与 E9(trace 置零=评分通路消融)互补,写作时并列;仍属描述性机制证据,不支持 causal-universal 措辞。

## 批次 3(可选,低成本):随机动作 floor 参考曲线 —— 仅推理

- **动机**:audit.md P0.3 判【缺材料】:0.31 的绝对水平无参照。
- **协议**:同 E1 网格,动作=均匀随机合法动作(或 fixed-goal random planner),≥3 env seeds;100 cells × 5 ep ≈ 0.2 GPU·h。
- **产出**:每 env 的随机成功率 floor,使 Fig.1b/正文的绝对水平可解释。

## 不执行项(明示)

- 不补任何新训练(E2 已覆盖 A 的科学问题;G5/B2 汇总已 3-seed 满格)。
- 不做轨迹级 held-out 与逐 episode 交集统计(除非审稿人强制;成本高、预期不改变结论)。
- 不新增强基线;像素轴的视觉编码器/预处理信息已可从 eval_pixel_ckpt.py + protocol.md 回答。

## 需收窄的正文句子(主张-证据核对结果)

1. 任何"提高可读性 → 规划性能下降"的句子:删(Fig.2c 无闭环指标;批次 1 完成前不得写)。
2. "CSR 是因果决定因素/普适原理":收窄为"across the eight architecturally dependent models, a strong descriptive rank association(ρ≈0.90;对聚合口径稳健 +0.89–0.93)"。
3. "10 task splits × 10 environments"(Fig.1b 图注):改为"10 splits, 100 cells, per-split envs 2–16"。
4. "one-step prediction error"(Fig.1c):改为"teacher-forced closed-loop one-step relative L2",并引 ρ(线性解码, SR)=−0.79。
5. **执行语义**:不得写 state 为 receding-horizon MPC;分 modality 表述(state 分段开环 2 次 planning/episode;pixel first-action 50 次/episode)。
6. SNN"天然不坍缩"类表述:按 E7 事实写(λ=0 表示层塌缩 cos 0.022;失败模式不对称+λ 敏感性是合法形式)。

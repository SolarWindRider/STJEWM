# 主张→证据映射(claims_evidence.md)

分级:**A** 多实验交叉;**B** 单实验/需限定;**C** 关联性,禁因果措辞;**VOID** 无数据禁写。

## 1. 核心主张分级

| 主张 | 级别 | 证据 |
|---|---|---|
| "trace 读出支撑闭环目标到达控制" | **VOID——两种协议下均不成立** | 原 dataset-derived 协议:零动作 0.477 > 随机 0.356 > Trace 0.311(目标可被动满足)。**主动运动目标协议(AGR,2026-11-24)**:目标 ≥2×容差且零动作不可达;零动作地板 = 0.0000(有效性检查);**随机地板 0.031;全部 13 模型 0.012–0.049**(Trace 0.040,LIF-Tx/rate/MLP 0.049,GRU 0.015,LeWM 0.012)。Trace 对 GRU/LeWM 仅剩 +2.5~+2.7 pp 残差,对 LIF-Tx/SLIF-trace/spike/rate/MLP 为 0。**任何"控制/规划能力"主张禁写**(`active_goal_summary.csv`) |
| 传统表征指标(可解码性/预测精度)与原协议 env-SR 不共线 | **A**(env-SR 本身受地板主导,见 §9) | E8(position-R² vs env-SR 反例)+ E1(cos 可被塌缩平凡取得)+ E7(λ=0:cos 0.005 但 env-SR 平)+ E9(窗口消融不动成败) |
| trace 读出的动力学是"慢迹不变性":对瞬时扰动长尾持续,对持续输入正常响应 | **A** | M4(d_peak 极小 + nr100% + 持续对照 0.044≫0.017)+ S3(action 27× LeWM)+ 位置敏感性(全窗口 0.23–1.27) |
| 塌缩在 pixel 与 λ=0 维度上是**分级**现象,单指标隐藏之 | **A** | E3(1690 cells 连续谱 0.003–0.177)+ E7 |
| 能效 ~3× | **B** | G3(稀疏加权**假想代理**;措辞必须保留 "proxy, not measured hardware") |
| 外部 Spiking-WM 对照 | **B** | E11(严格子树加载;ρ 0.10–0.84 与本地同量级;**禁跨协议排名**) |
| 规模不变性 | **B**(负面结果) | E5 严格判据 10/12 失败、失败模式跨尺度切换;E14:trace in-union 保持、cheetah@G16 全体 0.00 |
| event synchronization 高 ρ 是 STJEWM 优势 | **VOID——勘误** | G1 终值:readout ρ 0.004–0.527,trace 0.034 属最低档;**此主张必须删除**(README §3 勘误) |
| "仅替换 readout 即翻转动力学" | **C(降级)** | readout 家族是**分别训练**的 ckpt(stjewm_trace/spike/rate/no_trace/hidden_leak/membrane)→ 只支持"读出设计与动力学共变",**不支持单变量因果**;仓库中不存在固定权重换 readout 的干预实验(E9 是窗口/掩码消融,不是 readout 交换) |
| CSR→success 是普遍设计原理 | **C** | n=8 架构(非独立样本)+ 部分近常数 latent + CSR 分母(obs 敏感度)极小;只能写 "cross-architecture association" |
| trace/GRU 动力学差异(都衰减;输入响应不对称)跨训练 seed 复现 | **A**(oodc_F1 同 split,3 seeds)+ B(cross_benchmark_F1 补充) | pooled c20 trace<gru 3/3;trace g_a≫g_o、GRU 反向 3/3(numbers.md);**环境依赖**:cheetah trace<gru 3/3、cartpole trace≥gru 3/3——不得写成"每环境同向",非因果机制 |
| trace 对历史位置的输入掩码更敏感 | **B**(representation displacement;无闭环) | 剂量匹配遮蔽:同总元素数下 scatter25/last25 位移比 Trace 2.2–3.2×;GRU/LeWM 对 27× 剂量不敏感(C/A≈1.0);cartpole 仅置零 1 维、cheetah 2 维 |
| "辅助解码目标使规划下降" | **C(限定)→ 三 seed 后:无一致效应** | 相对原 ckpt:seed0 三臂 −4.0/−8.6/−5.6,seed1 +3.2/+0.6/+4.0,seed2 −6.4/−6.6/−1.0;对 matched control:λ=1 为 −3.6 [−5.8,−1.2] / **+1.4 [+0.4,+2.4]** / −0.4 [−4.4,+3.6] → **不跨 seed 复现**。独立 probe:线性 R² 从强负(orig −1.8/−30/−2.1)→ ≈0(≤0.17,非单调),MLP 0.10–0.34(三 seed 一致) |

## 2. 两套中心论点的证据归属(回应审稿人"论点 A vs 论点 B")

- **论点 A(temporal calibration)**:div/resp/ρ 三元组 = E6/E12;regime 分类 = FULL_METRIC_MATRIX;event synchronization = G1(**勘误后只剩 obs-embedding ρ≈0.814,readout 层不成立**)。
- **论点 B(control-selective dynamics)**:S3(gain)+ S4(obs 1 步衰减)+ M4(瞬态/持续对照)+ 位置敏感性 + E9 + E13/E14。
- **审稿人建议以 B 为中心**:与数据吻合——B 的每一环都有多个独立实验;A 的三大支柱之一(event sync)在 readout 层已被勘误。**建议**:B 作中心主张,A 的 div/resp 作为"塌缩诊断包"保留(它们本来就是为了排除塌缩解释而设计的)。

## 3. 审稿人三个科学问题的直接回答

### 3.1 "trace 优越性是否只是 latent collapse?"——不是;非退化证据包(8 条)

1. **env-SR 非常数 latent 可达**(排除"常数 latent 即可得同样分数"):Trace 100-cell seed0 env-SR 0.310(全场最高,与 LIF-Tx 0.308 并列;MLP 等常数/塌缩 latent 0.20–0.28);37-cell 三 seed 0.263±0.011;cartpole G4/G8/G16 恒 1.00(E14)。**但这不是控制能力的证据**:零动作地板 0.477、随机 0.356 均高于全部模型(§8)——本条只用于反驳"trace 与常数 latent 无差别",不得引申为"闭环有效";
2. **动作响应真实存在**:位置敏感性全窗口 0.23–1.27(真权重,imag_position_sensitivity.json);
3. **对持续输入正常响应**(M4 持续对照 0.044–0.048 ≫ 瞬态残差 0.017–0.028)→ 是"慢"不是"死";
4. **S3**:g_a 相对 LeWM ≈27×,且跨窗口持续;
5. **encoder 层未退化**:obs-embedding event_rho ≈0.998(verify_align.json 实测 0.9982;G1 均值 0.814)——退化的是 readout **信息量选择**,不是感知;
6. **训练信号依赖**:λ=0 时 trace 塌缩(E7)——trace 内容随训练目标变化,非架构常数;
7. **div 非严格零**:1e-4–1e-5 量级,内容感知门控(mech: 界 [0,1]);
8. **λ→content 因果**:SIGReg sweep 显示 cos 与 div 随 λ 系统变化。

**写作要求**:CSR 必须与绝对动作响应幅度并列报告(承认 ratio 分母小);必须引用上面 ≥3 条作反塌缩论证。

### 3.2 "env-SR 测什么?"——真实执行后的环境状态

`closed_loop.py`:`env.step(action)` 真实执行 → `final_state = physical_state()`(原生未掩码)→ `physical_env.check_success(final_state, goal_state)`。**不是模型预测状态**。草稿 Methods 的 `x_pred` 记号必须改为 `x_final` 并注明"executed-rollout terminal state"。协议限定(goal-conditioned、25 步开环段、goal=真实轨迹 t+25)须如实声明,不得写成通用 MPC。

### 3.3 "CSR↔success 普适?"——只能作关联

Spearman ρ≈0.90 的 n=8 是八种架构(**n = 模型配置数,不是 episode 数**;高度非独立,共享训练管线/数据);部分模型近常数。措辞:"across the eight architecturally distinct models we observe a strong rank association (ρ≈0.90)";**禁止** "general principle / universal law"。若要因果,需新实验(预注册多架构多任务),本文不做此主张。
**CSR 必须并列两种口径**(2026-11-20 核定):CSR_env = mean_e[g_a,e/(g_o,e+1e-8)](逐 env 比值再平均,Fig.5b 所绘)与 CSR_pooled = mean(g_a)/(mean(g_o)+1e-8)(Fig.4a 坐标比值)。ρ 对口径稳健:+0.905 / +0.929(去 SLIF-trace +0.893)。**SLIF-trace 的 CSR_env=2.81e4 是伪影**(cheetah g_o=0 触 1e-8 加数;均值比口径 4.65),Fig.5b 该点须标 floor;MLP(g_o、g_a 均 <1e-6)为 degenerate。

## 4. 逐图问题与可用修复数据(2026-10 旧初稿图号;⚠️ 现行图号为 §6 的 v2 五图结构,旧新映射见 experiments.md)
| 草稿位置 | 问题 | 修复数据(已存在) |
|---|---|---|
| Fig.1 | capability 结论依赖 env-SR 语义 | protocol.md §2.3 澄清 + E1 cos-vs-envSR 反差(塌缩模型 cos 低 env-SR 高) |
| Fig.2 标题不符 | 说三维 regime map 实为 PCA 组合 | 重画:div/resp/ρ 三轴 = E6/E12 数据(`FULL_METRIC_MATRIX.md`、`LATENT_ROLLOUT_SUMMARY.md`);现有 PCA/contraction 面板移至 Extended Data |
| Fig.2c readout-flip | 分别训练的 ckpt 不能作因果 | 降级措辞(§1 表);或补固定权重干预实验(**不存在**,不得虚构) |
| Fig.3d 文图不符 | 说 success 实为位移 | 两套数据都有:M4/S 系列(位移)+ E1/E9(success)→ 并列展示 |
| Fig.4c(旧)= 新 Fig.2c:decoder 强度 | v1 只有(非 held-out 的)decoder score | **已补闭环**:λ=0/0.1/1/10 + matched control 的 100-cell 配对(§7);独立 probe 见 numbers.md |
| Fig.3/4 caption 为 "Caption" | 未写 | 依 §1/§3 素材补写 |
| Methods 重复(式 22/23、goal 节) | 排版 | 以 protocol.md §2.2 为唯一版本 |

## 5. Abstract 中声称但 Results 未展开的项目——数据全部存在,补进 Results 即可

| Abstract 声称 | 数据位置 |
|---|---|
| variance regularization 作用 | E7 `SIGREG_SWEEP.md` |
| trace content causality | E9(每条件 10 episodes,**探索性**)+ M4(重测版);剂量匹配遮蔽(仅 representation) |
| pixel 泛化 | E3(1690 cells) |
| G4/G8/G16 规模 | E5 + E14 |
| held-out 环境 | E4(546 cells) |
| seed 稳健性 | E2(G5/B2)cos 簇;37-cell env-SR 三 seed 配对(numbers.md) |

**唯一必须删除的 Abstract 内容**:event synchronization 优势主张(G1 勘误)。

## 6. 主图重排 v2(2026-11-20,编者定稿的 5 图结构)

**原则:最小改动,只重编组,面板像素与数据零改动。** 实现 = 面板改名 + LaTeX 层重排(字母在 `\subcaption` 分配,**不烙进面板**);旧 20260918 bundle 原样保留作 provenance。

### 新旧映射(14 主面板)

| 新位置 | v1 来源 | 内容 |
|---|---|---|
| Fig.1a/b/c | fig1a/b/c | 架构 / 闭环成功率 / 传统指标与成功率(不变) |
| Fig.2a/b/c | fig4a/b/c | 线性 vs 非线性探针 / 上下文长度 / 解码器强度 vs 收缩 |
| Fig.3a/b/c | fig2a/b/c | 潜轨迹 / 扰动传播 c(h) / readout 决定收缩 |
| Fig.4a/b/c | fig3a/c/d | 输入增益平面 / 脉冲响应 / 末帧 vs 整窗 |
| Fig.5a/b | fig2d + fig3b | 收缩 vs 成功率(ρ=−0.19)/ CSR vs 成功率(ρ=+0.905) |

文件:`/home/lx/figs_nmi/20261120/`(fig1a-fig5b_panel.{pdf,png,svg} + `nmi_figures_v2.tex` + `rename_manifest.json` 42 文件 sha256 与 v1 逐一相符)。v1 的 fig3e/fig4d 示意图、v1 fig5 四面板未进主图,留 bundle 作 Extended Data 候选。

### 与 Results 的对应(编者建议标题)

1. The dissociation between predictive metrics and closed-loop capability — Fig.1
2. Latent accessibility does not determine world-model utility — Fig.2
3. Spike-derived representations exhibit distinct latent dynamics — Fig.3
4. Latent states exhibit asymmetric responses to observations and actions — Fig.4
5. Control selectivity is associated with closed-loop performance — Fig.5

### 证据优先级链(写作顺序约束)

图中实际结果与真实代码实现 → 图注 → Results 正文 → Abstract/Discussion。**按图写论文,不为维持旧解释改图。**

### 两条措辞边界(已写入图注)

- **Fig.2c(旧 4c)**:闭环数据已补(2026-11-22~23)。**正确表述**:"相对匹配的继续训练对照,仅 λ=1 伴随额外成功率下降(−3.6 pp,95% CI [−5.8, −1.2]);继续训练本身对应 −5.0 pp([−8.2, −2.0]);λ=0.1/10 与对照不可区分;单训练 seed,无单调剂量效应。"**不得写**:"提高可读性导致控制下降"、"下降全由继续训练解释"。上子图的 decoder score 来自共训 head 的训练尾(非 held-out),须改注或换独立 probe。
- **Fig.5**:n=8 描述性关联(ρ=−0.19 p=0.65 ns;ρ=+0.905 p=0.002),**禁止** causal/universal 措辞(§3.3 不变)。

## 7. 2026-11-20~23 审计轮结论(写作阶段的硬约束;原始数据在审计包)

**可写**
1. 性能排序:37-cell 配对三 seed 上 Trace 高于 GRU/LeWM(+10.8~+14.6 pp);LIF-Transformer 与 Trace 相近(100-cell seed0 0.310 vs 0.308);100-cell 主图仍是 seed 0。**所有模型均低于不动地板(0.477)与随机地板(0.356)**,Fig.1b 必须叠加两条地板。
2. 动力学:oodc_F1 中 Trace/GRU 的 pooled c20 与输入响应差异跨 seed 复现;**不是每环境同向**(cartpole 3/3 trace≥gru),不构成因果机制,亦**不得解释为更好的控制**。
3. 预测精度:Trace seed2 的 E_lat 极值是归一化尺度问题(z* 分母塌缩),**不写成预测发散**。
4. 解码干预:辅助目标改变表示(线性可访问性强负→≈0;收缩翻转),闭环成功率**无跨 seed 一致效应**。
5. Masking:支持"对历史位置的不同响应",**不是**"历史记忆造成规划提升"。
6. 规划通路:想象期 trace 置零在 4 envs × 50 eps 下无一处显著下降,finger 上 **+40 pp [+24,+56]**;planner 可见动作差异减半(γ=0.5)+2.9 pp(ns)→ 规划期候选评分对 trace/动作差异的依赖很弱。
7. 可复现性:同一会话内闭环评测逐 episode 确定(4 次重跑一致);9 月 E1 与当前环境相差 +0.82 pp(9% episode 翻转)——跨会话比较须注明。
8. utility:冻结编码器线性策略的样本效率 Trace 1%≈100%(+2.5 pp),LeWM +24.2 pp;潜代价梯度与物理梯度 signed 相关 ≤0(Trace −0.135,LeWM −0.633);数据预算 0.5×→2× 对 Trace 无清晰效应;walker/humanoid 对所有模型为 0.00(零样本外推不可测)。
9. 事件类型 AUROC:Trace 0.57–0.58,低于 LeWM/LIF-Tx(0.59–0.65),MLP≈0.5。

**禁写**:"Trace 支撑闭环控制 / closed-loop competence"(绝对主张);"SNN 天然不坍缩";"Trace 优于 LIF-Tx";CSR 普适/因果;SLIF CSR=2.8e4 为真实敏感度;state 为 receding-horizon MPC;"10 splits × 10 envs";"λ=1 相对 control 额外下降";"历史记忆→规划提升";E9 作必要性结论;n=8 当作 episode 数;跨 split 结果写成同 split 的 seed replication;episode 数当训练 seed 数。

**图注须改**:Fig.1b(叠加零动作/随机地板;100 cells = 10 splits、每 split 2–16 envs);Fig.1c("teacher-forced latent rollout relative L2",非 one-step prediction error);Fig.2c(协议/checkpoint/训练状态说明;上子图 decoder score 口径;三 seed 结果);Fig.4b(64 窗/env × 2 env = 128 窗;h=5 用直接池化增益 → 用 v3 面板 `figs/fig4b_panel_v3.*`);Fig.5b(SLIF 点 floor 标记 → 用 v2 面板;两种 CSR 公式、ε、估计单位=模型配置)。

**面板文件**:`/home/lx/figs_nmi/20261120/`(fig1a–fig5b;fig4b_panel_v3、fig5b_panel_v2 为现行版);审计包 `/home/lx/audit_20261120/`(OBS `obs://lixiang01/STJEWM_NMI/audit/20261120/`),入库副本 `verification_bundle/audit_20261120/`。

## 8. 2026-11-23 补齐实验后新增的作废/改写(优先级高于 §7)

- **"闭环控制能力"类措辞全部撤回**;中心论点须从"Trace 支撑控制"改为"表示性质差异(动力学/可访问性)+ 在地板主导的协议下更少偏离不动地板"。
- ~~推测:需主动运动的 goal 构造下排序可能改变(未做)~~ → **已于 2026-11-24 以 AGR 协议验证:所有模型 ≈ 随机(§9)**。

## 9. 主动运动目标协议(AGR,2026-11-24)后的最终定位(优先级高于 §7/§8)

- **中心事实**:要求主动运动时,所有模型 ≈ 随机(≤5%);原协议的排序与绝对值由不动地板主导。论文若保留"闭环"章节,只能作为**负面/诊断性结果**报告:"latent-goal CEM 规划在该框架下未转化为目标到达"。
- **可保留的贡献**:表示层结论——(a) 膜电位禁用协议下的 trace 读出具有可跨 seed 复现的收缩与输入响应不对称性质;(b) 线性可访问性与 env-SR(原协议)解离;(c) trace 对历史位置掩码的剂量匹配敏感性;(d) 规划通路诊断集:潜-物理梯度不对齐、动作通路剂量响应不敏感、想象期 trace 置零不降反升(finger +40 pp)。
- **Trace 相对 GRU/LeWM 的 env-SR 排序(原协议三 seed 复现;AGR 下 ≤2.7 pp 残差)**只能作为行为描述,且必须与随机地板并列。
- **论文标题/摘要不得出现** control、planning competence、goal-reaching capability 类措辞;Fig.1b 的定位从"闭环能力"改为"地板主导的成功率 + AGR 下的坍缩"。
- 仍属推测、不得入正文:换用其他规划器(非 CEM)、更长训练、或 goal 表示学习后是否能恢复控制——均未验证。

## 10. LeWM 核查(2026-11-24;优先级高于 §7–§9;细节 `audit.md` v9、`errata.md` 第 34–39 行)

- **"LeWM" 基线无效**:论文的 LeWM 是每任务单独训练的 18M 像素模型(10 epoch),只在 Two-Room / Reacher / PushT / Cube 上验证(87 / 86 / 78–96 / 79%),从未在 DMC 上评测。我们的 `lewm_baseline_v2` 是自写的状态输入版(3 层、4.97M、1 epoch、5,000 步、16 环境共享 16 万窗口),结构上编码器尾部与预测器共用权重。**任何"Trace 高于 LeWM"及以 LeWM 为对照的排序一律禁写。**
- **全部闭环 env-SR 冻结**:§7–§9 引用的 Fig.1b、E1/E4/E13/E14、AGR、utility 成功率项、SpWM 对比的 ST env-SR 列,都建立在"13 个模型的单步 latent 预测 R² 全为负 + CEM 提议分布失配 + DMC 目标一半平凡一半不可达 + 只在末态判成功"的评测上。§9 的"所有模型 ≈ 随机"不能单独归因于模型,是评测失败模式的混合结果。
- **已验证正确的部分(可写)**:环境封装、状态恢复、目标设定、成功判定与 episode 循环——官方权重 + 官方规划器走我们的 `PushTEnv` 循环得 86–90%,官方循环 88%;官方栈在本机复现 PushT 44/50、TwoRoom 44/50。
- **已修**:`train.py` 在 `history_size>1` 时的预测目标错位(`prediction_terms`);H=1 的全部基准结果不变。
- **不受影响的数值**:表示层度量(div/resp/ρ、probe、rollout、SIGReg sweep)本身;但其 LeWM 对照需在基线重做后重新比较。
- **待用户决定**:是否 (a) 以忠实结构重做状态版 LeWM 与全部基线,(b) 把 CEM 提议改到标准化动作空间,(c) DMC 改为任一时刻成功并采集带速度的数据,(d) 全量重跑闭环评测。在此之前闭环章节不得定稿。

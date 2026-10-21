# 主张→证据映射(claims_evidence.md)

分级:**A** 多实验交叉;**B** 单实验/需限定;**C** 关联性,禁因果措辞;**VOID** 无数据禁写。

## 1. 核心主张分级

| 主张 | 级别 | 证据 |
|---|---|---|
| 膜电位禁用协议下,trace 状态可支撑闭环控制且优于连续基线 | **A** | E1(1300 cells)+ E2(3-seed 0 翻转)+ E13(逐 env 配对)+ E14(三规模)+ E3(pixel) |
| 传统表征指标(可解码性/预测精度)与控制效用不共线 | **A** | E8(position-R² vs env-SR 反例)+ E1(cos 可被塌缩平凡取得)+ E7(λ=0:cos 0.005 但 env-SR 平)+ E9(窗口消融不动成败) |
| trace 读出的动力学是"慢迹不变性":对瞬时扰动长尾持续,对持续输入正常响应 | **A** | M4(d_peak 极小 + nr100% + 持续对照 0.044≫0.017)+ S3(action 27× LeWM)+ 位置敏感性(全窗口 0.23–1.27) |
| 塌缩在 pixel 与 λ=0 维度上是**分级**现象,单指标隐藏之 | **A** | E3(1690 cells 连续谱 0.003–0.177)+ E7 |
| 能效 ~3× | **B** | G3(稀疏加权**假想代理**;措辞必须保留 "proxy, not measured hardware") |
| 外部 Spiking-WM 对照 | **B** | E11(严格子树加载;ρ 0.10–0.84 与本地同量级;**禁跨协议排名**) |
| 规模不变性 | **B**(负面结果) | E5 严格判据 10/12 失败、失败模式跨尺度切换;E14:trace in-union 保持、cheetah@G16 全体 0.00 |
| event synchronization 高 ρ 是 STJEWM 优势 | **VOID——勘误** | G1 终值:readout ρ 0.004–0.527,trace 0.034 属最低档;**此主张必须删除**(README §3 勘误) |
| "仅替换 readout 即翻转动力学" | **C(降级)** | readout 家族是**分别训练**的 ckpt(stjewm_trace/spike/rate/no_trace/hidden_leak/membrane)→ 只支持"读出设计与动力学共变",**不支持单变量因果**;仓库中不存在固定权重换 readout 的干预实验(E9 是窗口/掩码消融,不是 readout 交换) |
| CSR→success 是普遍设计原理 | **C** | n=8 架构(非独立样本)+ 部分近常数 latent + CSR 分母(obs 敏感度)极小;只能写 "cross-architecture association" |

## 2. 两套中心论点的证据归属(回应审稿人"论点 A vs 论点 B")

- **论点 A(temporal calibration)**:div/resp/ρ 三元组 = E6/E12;regime 分类 = FULL_METRIC_MATRIX;event synchronization = G1(**勘误后只剩 obs-embedding ρ≈0.814,readout 层不成立**)。
- **论点 B(control-selective dynamics)**:S3(gain)+ S4(obs 1 步衰减)+ M4(瞬态/持续对照)+ 位置敏感性 + E9 + E13/E14。
- **审稿人建议以 B 为中心**:与数据吻合——B 的每一环都有多个独立实验;A 的三大支柱之一(event sync)在 readout 层已被勘误。**建议**:B 作中心主张,A 的 div/resp 作为"塌缩诊断包"保留(它们本来就是为了排除塌缩解释而设计的)。

## 3. 审稿人三个科学问题的直接回答

### 3.1 "trace 优越性是否只是 latent collapse?"——不是;非退化证据包(8 条)

1. **闭环有效**:env-SR 0.31 tied-best(E1);cartpole G4/G8/G16 恒 1.00(E14)——常数 latent 无法引导控制;
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

Spearman ρ≈0.90 的 n=8 是八种架构(高度非独立,共享训练管线/数据);部分模型近常数。措辞:"across the eight architecturally distinct models we observe a strong rank association (ρ≈0.90)";**禁止** "general principle / universal law"。若要因果,需新实验(预注册多架构多任务),本文不做此主张。

## 4. 逐图问题与可用修复数据(审稿人 §一 表格)

| 草稿位置 | 问题 | 修复数据(已存在) |
|---|---|---|
| Fig.1 | capability 结论依赖 env-SR 语义 | protocol.md §2.3 澄清 + E1 cos-vs-envSR 反差(塌缩模型 cos 低 env-SR 高) |
| Fig.2 标题不符 | 说三维 regime map 实为 PCA 组合 | 重画:div/resp/ρ 三轴 = E6/E12 数据(`FULL_METRIC_MATRIX.md`、`LATENT_ROLLOUT_SUMMARY.md`);现有 PCA/contraction 面板移至 Extended Data |
| Fig.2c readout-flip | 分别训练的 ckpt 不能作因果 | 降级措辞(§1 表);或补固定权重干预实验(**不存在**,不得虚构) |
| Fig.3d 文图不符 | 说 success 实为位移 | 两套数据都有:M4/S 系列(位移)+ E1/E9(success)→ 并列展示 |
| Fig.4c 缺闭环指标 | 只有 probe score | probe = E8 `g2_summary.json`;闭环 = E1 → 同图双轴即可 |
| Fig.3/4 caption 为 "Caption" | 未写 | 依 §1/§3 素材补写 |
| Methods 重复(式 22/23、goal 节) | 排版 | 以 protocol.md §2.2 为唯一版本 |

## 5. Abstract 中声称但 Results 未展开的项目——数据全部存在,补进 Results 即可

| Abstract 声称 | 数据位置 |
|---|---|
| variance regularization 作用 | E7 `SIGREG_SWEEP.md` |
| trace content causality | E9 + M4(重测版) |
| pixel 泛化 | E3(1690 cells) |
| G4/G8/G16 规模 | E5 + E14 |
| held-out 环境 | E4(546 cells) |
| seed 稳健性 | E2(G5/B2) |

**唯一必须删除的 Abstract 内容**:event synchronization 优势主张(G1 勘误)。

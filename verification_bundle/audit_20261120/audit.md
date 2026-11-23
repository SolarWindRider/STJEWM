# ST-JEWM 实验核查报告(audit.md,2026-11-20)

索引:新版 Fig.1–5(`figs_nmi/20261120/`面板)+ results.csv(逐条数值与来源)。标记:【已复核】=与现稿一致且有代码/日志证据;【需补跑】;【缺材料】;【与现稿不符】=需要改正文/图注/图。

---

## P0.1 Fig.4a / Fig.5b 输入扰动与 CSR —— 【已复核,但需改一处图注】

**四个 panel 的原始表、脚本、配置**(全部定位):
- 数据:`results/mech_0917_b/scaling/scaling_summary.json`(table + spearman_n8/n6),由 `code/scripts/cls0917_scaling.py` 从 FD 增益(`code/scripts/stab0917_gain.py` + 自身 FD 块)聚合。
- FD 协议(`cls0917_scaling.py:140-175`):paired ±0.1×per-dim std 扰动**最后上下文帧**;action 通道直接改 a[-1];obs 通道改 o[-1] 后**重编码整个 27 帧窗**;dza/dzo=‖z₊−z₋‖ 按 window,归一化 = /‖z_baseline‖(+EPS)→"rel"增益;h=1..5 为扰动后 rollout 步。
- **聚合顺序(审计问题直答)**:`CSR = gain_a(h1)/(gain_o(h1)+1e-8)` **先逐 env 取比值(+1e-8 floor),再对 env 取均值**(ratio-then-aggregate,`cls0917_scaling.py:9,165,200`);而 Fig.4a 坐标 gain_o_rel/gain_a_rel 是**先均值再比值**(aggregate-then-ratio,`:196-199`)。Fig.4a 与 Fig.5b **同 checkpoint、同协议、同 JSON**,差别仅在聚合序与 floor。

**SLIF trace 两处读数不一致(审计重点)——判定:聚合伪影,非导出/标签错误**:
- 原始数值:slif g_o_rel=1.56e-4,g_a_rel=7.26e-4 → 均值比 **4.65**(=Fig.4a 读数);逐 env CSR 因 gain_o≈0 触 1e-8 floor,比值爆到 ~5e4,env 均值后 **CSR=2.81e4**(=Fig.5b 点)。
- 其余 7 模型两口径差 1.1–2.2×(聚合序的正常差异),仅 slif 因分母触 floor 被放大 ~6000×。
- **建议修改**:Fig.5b 给 slif 点加 floor 标记(如空心+†"denominator at measurement floor"),图注写明 CSR 对近零 g_o 不定义;正文引用 ρ 时并列两种口径(见下)。

**Spearman 复算(并列秩=平均秩;env_sr 无并列)**:
| 口径 | n=8 | n=6(去 MLP/SLIF) | n=7(只去 SLIF) |
|---|---|---|---|
| CSR as published | +0.905(p=0.002) | +0.886 | +0.893 |
| 均值比口径 g_a/g_o | +0.929 | +0.886 | +0.893 |
结论:关联对聚合口径稳健;Fig.5a c20 口径 ρ=−0.190(p=0.65 ns;去 MLP/SLIF 后 +0.086)。

**MLP 近下限模型**:gains 双双 <1e-6 → 脚本 `:203` 已标 degenerate(图中空心);分子分母原始值见 results.csv(g_o=5.97e-8,g_a=1.3e-7);floor 规则=分母 +1e-8。

**Fig.4b 定义**:impulse 加在**最终上下文位置**:action 通道 = a[-1] ± paired delta;obs 通道 = o[-1] ± paired delta 后重编码;未来观测/动作 = 教师强制真实序列;h=1,5 = 扰动注入后的第 1/5 个 rollout 步;热图值 = mean dz(h5)/mean dz(h1)(`:172-173`)。**†=underflow 伪影**:slif obs h5 的分母低于数值分辨率,图中显示值 17.9 是**展示占位符,非实测**——需在图注写明(现稿图注已写"underflow artefact",建议补"17.9 为占位")。

**Fig.4c 定义**:操作=**置零(masking),非噪声、非替换**:同一 25% 维度仅在末帧置零(last25)vs 整窗 27 帧逐帧置零(win25);3 个固定 mask seed **跨模型共享**(配对);位移=rel_l2=‖Δz‖/(‖z_clean‖+1e-8),cos 为辅助量。末帧 vs 整窗对照的正确解读:trace 对末帧信息不敏感(×1.1)但对整窗历史破坏敏感(×4.4)→ 状态由历史承载;LeWM/GRU 两者都不敏感 → 逐帧编码。

## P0.2 Fig.2/3/5a probe 与收缩因子 —— 【已复核】

- **Fig.3b/3c/5a 的 c20 同源**:scaling.table.c20 与 rollout per-env 均值**逐位相等**(trace (0.04976+0.01405)/2=0.03190;gru 0.04031;membrane 6.96641)→ 三处同一批 S1 评估(128 windows/env,cartpole+cheetah)。逐 env 值已入 results.csv。
- **Fig.2c lam0 参考 c20=1.361e-4(=1.4e-4 ✓)与 Fig.3c trace 0.0319 是同一 checkpoint**:`access_eval.py:33` lam0_ref 路径 = 5m_5mpar trace final.pt,与 Fig.3c/5a 的 checkpoint 相同。数值差 ~230× 完全由协议差解释:dose 协议 ε=0.05×**std(z*)**(32 windows,单一随机单位方向,`access_eval.py:131-136`)vs S1 ε=0.05×**mean‖z*‖**(128 windows,paired twin rollout,`stab0917_rollout.py:23-24,72`)。日志可判定,**无需重测**;两图图注均已声明协议不可比 ✓。
- Fig.3a:unified 协议 dump(latent_rollout.py,20260916 final repair 代际);no_trace dump 与 membrane byte-identical(md5 同、ckpt 不同)→ 排除并在图注声明 ✓(遗留问题已披露,不修改)。
- **Fig.2a**:解码目标=s_t+伪速度(s_t−s_{t−1}),128 维 pad 目标的非常数列;linear=Ridge(α=1.0,标准化);MLP probe=2 层+早停(训练 episodes 内部 val 切分,patience 20);**3 seeds=probe seed,非世界模型训练 seed**;误差条=probe seed s.d.;hold-out=episode 级 75/25;R²=multioutput uniform_average。
- **Fig.2b**:benefit(Δ)=R²(K=27)−R²(K=1);K∈{1,4,16,27,100} 零填充窗;K>27=超出训练窗(标记为外推);Δ∈{1,5,10,20},Ridge α=1.0;panel 展示 Δ=5。**"非单调改善"如实呈现**(K=4 的凹点=ridge 过拟合伪影,已复现并标注)。
- Fig.2c λ→checkpoint:λ=0 → 原 trace final.pt(参考,未重训);0.1/1/10 → accessibility_20260917 三个 fine-tuned ckpt;decoder score=uncentred training-tail 重构分(非 held-out R²),λ=0 无定义。

## P0.3 Fig.1/5 成功率与训练可比性 —— 【已复核;一处图注与现稿不符;审计前提一处与代码不符】

- **CEM 执行语义(审计清单第 3.1 条的前提需修正)**:清单假设"CEM 优化序列但环境只执行首动作"。**代码事实:state=分段开环**(`closed_loop.py:277` 整条 H=25 序列逐步执行后才重规划;50 步=2 次 planning);**pixel=first-action MPC**(`eval_pixel_ckpt.py:5,219`)。协议常数:CEM 300 候选/30 elite/10 iters/H25(state)或 H5(pixel);goal=同一真实训练轨迹 t+25 的记录状态;成功=执行后真实终点状态判 `check_success`(终点判定,非任一时点);`x_pred` 记号已废,用 x_final。**Methods 按此分 modality 表述(已在 protocol.md §2.1/§2.4 写红线)。**
- **Fig.1b 的 100 cells**:E1=10 splits × 每模型 100 cells,但**每 split 环境 2–16 个不等**(5/6/2/11/7/8/15/15/15/16),求和=100;非 10×10 全交叉。每 cell=5 episodes;eval seed 相同 → `closed_loop.py:245` `rng=default_rng(seed*7919+42)` 保证**所有模型共享同一批 (init, goal) 实例**。**【与现稿不符,需改图注】**:"10 task splits x 10 environments"应改为"10 task splits(每 split 环境数 2–16,合计 100 cells)"。
- **Fig.1c 复算**(从 fig1_source.json 的 24 条记录):
  - 一步闭环误差 vs SR:ρ_all=+0.238(Pearson +0.432;去空心 n6 ρ=+0.886)。
  - linear probe R² vs SR:**ρ_all=−0.786**(n6 −0.943)→ 解码越好成功率越低,解离主张成立。
  - 注:该"一步误差"=teacher-forced 闭环 rollout 的 h=1 相对 L2,非开环单步预测误差;图注措辞需与此定义一致。
- **逐 env 配对差(E1 seed0,5 envs)**:trace−GRU:cartpole +0.40、finger +0.80、pendulum −0.20(反转)、其余 0;trace−LeWM:cartpole +1.00、finger +0.60、其余 0。优势集中在 cartpole/finger(短视野局部控制)。
- **训练可比性**:同 manifest(4376bb14)同代际、1 epoch/batch 32/AdamW 3e-4、参数 4.97–5.13M;损失与架构绑定=设计性混淆(报告 §1.2.1 已声明)。
- **随机动作/planner 参考曲线**:**【缺材料】**全仓无 random-action/goal-shuffle/constant-predictor 评测 → 0.31 的绝对水平无 floor/ceiling 参照(与 verification_bundle INDEX Q1 一致)。

## P1 回收 —— 全部完成

1. **多训练 seed**【已复核】:E2=13 模型 × 3 splits × 37 cells × seeds{1,2}(共享评测实例,配对差值):
   - trace−GRU:seed1 **+0.108**(8胜/29平/0负),seed2 **+0.130**(13/22/2)。
   - trace−LeWM:seed1 **+0.124**(10/26/1),seed2 **+0.124**(12/23/2)。
   - 方向跨两个训练 run 一致 → 差异在训练重复间复现;membrane 也有两 seed → **P2.A 无需新训练**。
2. **像素/held-out/PushT/TwoRoom/部分观测**【已复核】:E3=1690 cells(pixel env-SR trace 0.172/spike 0.177/LeWM 0.091/MLP 0.178);E4=546 cells(环境组合 held-out,**非轨迹级**;调参与训练同源,是否"排除在调参外"无独立记录→标【缺材料】);cross_benchmark F1–F3 中 LeWM 的 pusht/tworoom/reacher 全 0.00(stacker 0.80、ball_in_cup 1.00 饱和);S4 部分观测=时间校准性质(单独注明)。
3. **Trace intervention 与 CEM rollout ablation**【已复核】:干预=包裹 `model.predict`(=CEM 候选评分函数,`event_window_ablation.py:139-150`)→ **检验的是"想象 rollout/评分通路"对 trace 的依赖**,不触真实观测编码、目标编码、planner 读出端。逐条件 checkpoint/SR/定义已在 `e9_ablation_paired.csv`(含 intervention_definition/causal_scope 列)。零效应(trace 置零 SR 不降)与 E9 cartpole 结果保持原判。
4. **辅助解码器 λ sweep**【缺材料→P2.B】:仅 receipts(decoder score+c20),**无闭环评估**;λ=0 参考即主 checkpoint,可比性成立 → 直接补推理评估即可。

## 需要修改的正文/图注汇总

| 位置 | 问题 | 修改 |
|---|---|---|
| Fig.1b 图注 | "10 task splits x 10 environments" | "10 task splits(每 split 2–16 envs,共 100 cells)" |
| Fig.5b | slif 的 CSR=2.8e4 是 floor 伪影 | slif 点加 floor 标记;图注写 CSR 在 gain_o≈0 时不定义;ρ 并列两口径(+0.905/+0.929) |
| Fig.4b 图注 | † 的 17.9 为展示占位 | 补一句"display placeholder, not measured" |
| Fig.1c 图注 | "one-step error" 定义 | 写明=teacher-forced 闭环 rollout h=1 相对 L2;可引 ρ=−0.79(线性解码 vs SR) |
| Methods(MPC) | 审计清单假设 receding-horizon | state=分段开环、pixel=first-action,分 modality 写(protocol.md 红线) |

旧版图面(20260918 bundle)与新导出(20261120 面板)均保留,差异来源=本文件+results.csv。

---

# v2 增补(2026-11-20 第二轮,响应审计复核)

## Fig.2c 数值核定 —— 【与 v1 CSV 不符处已修正】
- access_summary arm 键为 `lam0_ref/lam0.1/lam1.0/lam10.0`;均值 c20 = **1.36e-4 / 34.705 / 46.335 / 20.820**,与稿图 34.7/46.3/20.8 **一致**。v1 results.csv 中的问号猜测值(2.9e-4 等)**作废**,以 `fig2c_arms.csv` 为准。
- 结论修正:非零 λ 不是"衰减降低"而是**翻转为放大**(c20 从 1.4e-4 → 35–46,放大两个数量级以上);λ=10 略回(20.8)但仍 ≫1。Fig.2c 表述应写"任何解码压力都摧毁深度收缩并翻转为放大"。
- 三个非零 ckpt 配方(train_accessibility.py):Linear(192→128) 辅助解码器(标准化 pad state、全位置)+ **全模型参数** 1-epoch 微调,λ∈{0.1,1,10};产出模型 sd 去掉 decoder 键(drop-in)。**无同日程零权重微调对照**(λ=0=原 checkpoint)→ 批次 1 组间解释限定为"辅助解码微调配置 vs 原 checkpoint"。

## Fig.4b 逐格口径 —— 【已复核,v2 面板已出】
- h1 格 = 池化 FD 增益 gain(h1)(paired ±0.1×per-dim std,末位上下文帧);h5 格 = **gain(h1) × mean dz(5)/mean dz(1)**(≈gain(h5),差在基线范数 ‖z_base(h)‖ 随 h 变化);Trace action 0.12→0.26 = ×2.17 ✓ 与图注 ×2.1 一致。
- 未来步供给:**动作 = recorded future actions(教师强制)**;观测在想象 rollout 中**不再重编码**(纯 latent 自滚)。
- **v2 面板**(`figs/fig4b_panel_v2.*`,生成器 `figs/make_fig4b_v2.py`):SLIF obs h5 的 17.9 占位改为**灰格 "n/a †"**;其余像素不变。v1 面板保留于 20260918 bundle。

## Fig.5b —— 【v2 面板 + 逐环境原值表】
- **逐环境 FD 原值未持久化**(cls0917_scaling 只落盘 pooled 表);已用**同脚本、SEED_WIN=0、n_win=64(协议值)**确定性重导出:`fd_perenv.csv`(重导出 pooled 与原表逐位一致,见下)。
- 两口径并列:`fig5b_two_aggregations.csv` + `fd_perenv.csv`(逐模型×逐环境 g_a/g_o/csr + ratio_of_pooled_means)。
- **v2 面板**(`figs/fig5b_panel_v2.*`):沿用原口径,加 † 图内定义("hollow: degenerate latent; SLIF-trace CSR sits at the measurement floor (g_o≈0, +1e-8 addend)")。

## Fig.1c / Fig.2b 命名与索引 —— 【已复核】
- Fig.1c 左:**E_lat(h)=mean_i‖ẑ_ih−z*_ih‖/(‖z*_ih‖+1e-8)**(`stab0917_rollout.py:9-14,225`);教师强制 recorded actions;z* = warm 全 episode 前向的真实潜变量;想象期观测不重供给 → 命名 **"teacher-forced latent rollout relative L2"**(不得称 open-loop one-step prediction error)。
- Fig.2b:输入窗 = 帧 [t−K+1..t](K=100 跨 episode 时前置零填充),目标 = **obs[t+Δ]**(Δ∈{1,5,10,20};panel 取 Δ=5);t+max(Δ) 须在 episode 内(`mech0917_history.py:99-103`)。Fig.2a 误差条 = probe seed s.d.(3 个 MLP probe seed,非世界模型 seed)。

## Fig.1a —— 【无需修改】
示意为线性接口管线,**不含任何重规划循环**,对 modality 中性;不暗示逐步重规划。(可选:图注加"execution semantics are modality-specific, see Methods"。)

## E9 / E1 逐实例导出
- `e9_per_condition.csv`:6 模式 × 2 env,基准=显式 `baseline` 模式;统计单位=episode(跨模式共享同 (init,goal));cartpole baseline SR 0.60,全部干预模式 0.6–0.7(零负效应);cheetah 全 0–0.1(地板)。结论限定于被包裹的 `model.predict` 候选评分通路。
- `e1_per_episode.csv`:**6500 行**(1300 cells × 5 eps,0 缺失),含 episode_idx / initialization_seed / init/goal/final_state / actions_taken=50 / **planning_calls=2(分段执行直接证据)** / 三种 success 判定 / phys_dist。

## FD 重导出的协议差异说明(第一版重导出作废)
第一次重导出误用 n_win=128(协议为 **64**/env)→ 与原表差 10–40%;改 64 后 **pooled 与 scaling_summary 逐位一致**。教训已写入脚本注释;`fd_perenv.csv` 现为 64-窗版本。

## 结构化重导出
`export_audit_tables.py`(入库于本包)生成 **results.csv(118 行 × 14 列,0 错位)** 及 fig2c_arms / fig4b_heatmap / fig5b_two_aggregations / fig1c_values / maskobs_table / e2_paired_summary;E1/E2/像素/held-out 由 verification_bundle CSV 与本包 e1_per_episode.csv 覆盖,以 cell_id/checkpoint/episode_idx 连接。

---

# v3 增补(2026-11-20 P0-A:同网格三 seed 成功率)

## 权重覆盖与独立性 —— 【已复核】
- `checkpoint_manifest.csv`:4 模型(trace/gru/lewm/lif_tx)× 3 splits(cross_benchmark_F1、generalist_16env、oodc_F2)× 3 训练 seed = **36 个 final.pt,0 缺失**,含 sha256。
- 独立训练核验:ckpt `args.seed` = 0/1/2 与训练 run 一一对应(抽样三载验证);均为 1 epoch。训练 seed 记录 = ckpt args(目录名统一为 seed_0,以 5m_seed1/5m_seed2 根目录区分 run)。
- seed 语义:训练 seed = args.seed;**评测 seed = 0(三 seed 全部同一评测流)**;CEM 无独立随机源(确定性给定 ctx)。环境 seed 由 episode 采样 rng(seed×7919+42)决定,跨模型/跨训练 seed 共享 → 配对成立。

## 实例对齐 —— 【已复核】
E1(train seed 0)与 E2(train seeds 1/2)在同 (split, env) 的 **episode_idx 与 init_state 完全一致**(例:oodc_F2/全 6 env 集合,167365…逐位相同)→ 37-cell 网格跨训练 seed 逐实例配对有效,无需补跑。

## 37-cell 三 seed 主表(per_seed_summary.csv / paired_episodes.csv,2220 episode 行)

| 模型 | seed 0 | seed 1 | seed 2 | 均值±sd |
|---|---|---|---|---|
| Trace | 0.2595 | 0.2541 | 0.2757 | **0.263 ± 0.011** |
| GRU | 0.1135 | 0.1459 | 0.1459 | 0.135 ± 0.019 |
| LeWM | 0.1297 | 0.1297 | 0.1514 | 0.137 ± 0.013 |
| LIF-Tx | 0.2757 | 0.2595 | 0.2649 | **0.267 ± 0.008** |

配对差(pp,37 cells,win/tie/lose):

| 对比 | seed 0 | seed 1 | seed 2 | 均值±sd |
|---|---|---|---|---|
| Trace−GRU | **+14.6**(9/27/1) | +10.8(8/29/0) | +13.0(13/22/2) | **+12.8 ± 1.9** |
| Trace−LeWM | **+13.0**(8/29/0) | +12.4(10/26/1) | +12.4(12/23/2) | **+12.6 ± 0.3** |
| Trace−LIF-Tx | **−1.6**(2/29/6) | −0.5(4/29/4) | **+1.1**(4/31/2) | −0.3 ± 1.4 |

**审计参照核对**:seed 0 三值 +14.6/+13.0/−1.6 与任务单给出的参照**逐位一致**;seed1/2 +10.8/+13.0 与 +12.4/+12.4 亦一致。

## 结论范围(可直接写入论文)
1. Trace 相对 GRU/LeWM 的优势在全部三个训练 seed 复现(量级 +10.8~+14.6 pp,无负 seed)。
2. **Trace 与 LIF-Tx 的接近程度跨 seed 不稳健**:差 −1.6~+1.1 pp,seed 间方向翻转,平局占 29–31/37——"LIF-Tx 与 Trace 相当"是跨 seed 的诚实表述,"Trace 优于 LIF-Tx"不可写。
3. 网格限定:37-cell 跨 seed 主比较(3 splits:15/16/6 cells);Fig.1b 的 100-cell 图仅为 train seed 0。

---

# v4 增补(2026-11-20 第三轮:P0 同网格三 seed + 跨 seed 动力学)

## P0-A 三 seed 成功率 —— 见 v3 节(已交付,数值与任务单参照逐位一致)

## P0-B 跨 seed 动力学(dynamics_per_window.csv 57600 行 / dynamics_summary.csv 36 行 / dynamics_per_seed_summary.csv)

**口径声明**:诊断矩阵 = cross_benchmark_F1 训练的 checkpoints(三 seed 权重齐全;oodc_F1 的 seed1/2 权重不存在 → 按任务单"选覆盖 split 重测"规则执行,**标注为补充复现**,主图 Fig.3–4 仍用 oodc_F1 原代际)。窗口/流与已发表协议逐位同源(SEED_WIN=0,paired ±0.1×std,obs 重编码,recorded future actions)。

**三观察的跨 seed 判定**:

| 观察 | 判定 | 数值 |
|---|---|---|
| ① 两者都衰减(c20<1) | **复现 9/9** | trace 0.0050/0.0222/0.0000;gru 0.0421/0.0295/0.0570;lif_tx 0.0002–0.0003 |
| ② 合并衰减量接近(且 trace≤gru) | **复现 3/3** | 每 seed 均有 trace ≤ gru(0.005<0.042;0.022<0.030;0.00002<0.057) |
| ③ 动作/观测响应不同 | **复现 3/3** | trace g_a/g_o = 2.0e4–3.3e4(g_o 触 floor,标记);gru = 0.0019–0.0025(逐 seed 极稳:lif_tx 15–77,波动) |

**异常标记(必须写入正文/图注候选)**:trace seed2(cross_benchmark_F1)E_lat(h1)=1550/1919(latent rollout 尺度发散)且 c20≈2.2e-5,但同 ckpt 的 37-cell 闭环成功率 **0.2757(三 seed 最高)**——"latent 动力学病理 ≠ 闭环失效"的直接实例,与 P12/collapse-robust 主线一致;建议正文提及并纳入 Fig.1c 空心点族。

**并排成功率**(cross_benchmark_F1,15 cells/seed):trace 0.347/0.320/0.333;gru 0.133/0.160/0.200;lif_tx 0.320/0.320/0.320(`dynamics_models_sr_crossbenchmarkF1.csv`)。

## P0-C Fig.4b 窗口数与显示口径 —— 【已修正】
- n 口径:**64 windows/env × 2 envs = 128 个逐窗差分值**(每窗的 ± 为两次前向分支,dz=z₊−z₋ 每窗一个值)。v1 CSV 的 256 是错误标注,已撤回。
- **逐环境 per-env 聚合伪影全链确认**:SLIF obs h1 在 cheetah = 0(触 1e-8 floor)→ per-env dzo_decay5_over_1 = **229,281** → 池化均值被污染 → v1 面板该格无有效值,v1 脚本用 17.9 占位。
- **同源逐窗表**:`fd_perwindow_oodcF1_seed0.csv`(10240 行,oodc_F1 seed0 8 模型 = Fig.4b 确切 provenance,SEED_WIN=0 确定性)。
- **Fig.4b CSV 重导**(`fig4b_heatmap.csv`,n=128,同时给出 direct h5 与面板口径 h1×ratio 两列)。
- **Fig.4b v3 面板**(`figs/fig4b_panel_v3.*`):全部格子改用**逐窗直接池化增益**(mean dz(h)/base(h));SLIF obs h5 = **2e-04(实测)**,17.9 占位与 n/a 均退役;其余定性模式不变(trace obs 4e-3→2e-5 衰减、action 0.12→0.25 增长)。v1/v2 面板保留于 20260918 与本包。

## P1 辅助解码器规划终点 —— 【新实验完成,仅推理】

4 条件 × E1 全网格(100 cells × 5 eps;λ=0 复用 E1 trace run,逐实例配对;2 个新配置漂移格剔除,配对 n=100):

| 条件 | c20(Fig.2c 协议) | env-SR | 对 λ=0 配对差 |
|---|---|---|---|
| λ=0(原 checkpoint) | 1.36e-4 | **0.310** | — |
| λ=0.1 | 34.7 | 0.2700 | **−4.0 pp**(n=100) |
| λ=1.0 | 46.3 | 0.2240 | **−8.6 pp** |
| λ=10 | 20.8 | 0.2540 | **−5.6 pp** |

- 300 个配对 arm-cells 中 65 个成功率下降;三条臂**全部**为负。
- **可写结论**:辅助解码微调在摧毁深度收缩(1.4e-4→35–46× 放大)的同时,在**同一批评测实例**上使闭环成功率一致下降 4.0–8.6 pp——"买可读性伤规划"从 v2 的"无闭环数据"升级为**实测配对证据**。
- 归因限定(不变):λ=0 为原 checkpoint,非零臂含额外 1-epoch 微调 → 结论写"含辅助目标的微调配置相对原 checkpoint";**无同日程 λ=0 微调对照**,单独归因辅助损失仍为待决定项(decoder score 定义 = uncentred training-tail 重构分,λ=0 无定义)。
- 文件:`lambda_closedloop_summary.csv`、`lambda_paired.csv`(100 配对/臂);评估输出 `/data/lx/tmp/results/accessibility_closedloop_20261120/`。
- 运行备注:masked-cheetah 3 格首跑因 CUDA_VISIBLE_DEVICES/cuda:1 冲突失败,已修正重跑;2 个新配置漂移格(F3 splits 的 plain cheetah)从配对剔除。

---

# v5 增补(2026-11-22 任务单执行)

## P0-1 oodc_F1 同 split 三 seed 动力学 —— 【已完成,seed0 逐位复现已发表值】

oodc_F1 seed1/2 权重原不存在(已如实报告)→ 按 G5 原配方(train.py,1 epoch/batch 32/lr 3e-4/n-layers 2/spec oodc_F1.json/seed 1,2)补训练 6 个 ckpt(`results/5m_seed{1,2}/oodc_F1/<model>/seed_0/`,仓库内路径;loss_log/train.log 齐全)。S1+FD 同协议同窗口流重测三 seed(trace/gru/lif_tx),seed0 S1 c20=0.0498/0.0140 与已发表 Fig.3 **逐位一致**(provenance 复现 ✓)。

**文件**:`oodc_F1_dynamics_per_window.csv`(57600)、`oodc_F1_dynamics_summary.csv`(36)、`dynamics_summary.csv`(cross 口径重算恢复)。

**验收四问**:
1. c20 排序(pooled):trace 0.0319/0.0064/0.0071 vs gru 0.0403/0.0293/0.0887 → **trace ≤ gru 3/3 复现**;
2. 不对称:trace g_a≫g_o 3/3(21–30×;seed2 cheetah g_o=6.7e-6 近 floor,已标记);gru g_o≫g_a 3/3(0.21–0.30 vs 0.0004–0.0017)→ **复现**;
3. 环境依赖:cheetah 3/3 trace<gru;**cartpole seed0 反转**(trace 0.0498 > gru 0.0146)——须如实报告"pooled 排序 3/3,但 cartpole 单环境 seed0 反转";
4. 数值下限:lif_tx seed1 g_o=0 触 floor(标记);trace seed2 cheetah g_o 近 floor。

## P0-2 Trace seed2 异常分解 —— 【已定性:分母塌缩,非预测发散】

`latent_error_decomposition.csv`(6144 行)+ `latent_error_summary.csv`(48 行)。`latent_error_summary` 关键行(trace,cross_benchmark_F1):

| seed | env | h | num_mean | den_median | ratio_mean | ratio_median | den<1e-3 占比 |
|---|---|---|---|---|---|---|---|
| 0 | cartpole | 1 | 0.0132 | 0.0136 | 1.01 | 1.00 | 0% |
| 2 | cartpole | 1 | **0.0048(最小)** | **0.0027** | 1172 | 1.70 | **21.1%** |
| 2 | cheetah | 20 | 0.0037 | **6.7e-5** | 8050 | 54.9 | **70.3%** |

**判定(审计分类学的"分母接近零 + latent scale collapse")**:误差分子反而三 seed 最小(0.0037–0.0048 vs 0.013);是 z* 参考潜变量自身尺度塌缩(den_median 0.0136→0.0094→0.0027 跨 seed 渐进);ratio 均值被近零分母子群抬起(中位数仅 1.7–54.9;top1 窗占比 ≤1.4%,非单窗主导);gru 对照 den 健康(ratio 0.19–0.29 稳定)。**v4 审计中"预测发散但控制成功"的表述撤回**,改为"参考潜尺度塌缩使相对误差指标在该 ckpt 上不可解释;闭环成功率不受影响"。不作为论文核心发现,仅作指标适用性脚注。

## P0-3 matched continued-training control —— 【训练+闭环已完成,探针/bootstrap 见 CSV】

- control ckpt:`accessibility_20260917/oodc_F1/stjewm_trace_only_lam00/seed_0/final.pt`(λ_decode=0,312 步,与三臂同 seed/rng/优化器/数据顺序;`val_decode_r2=−0.0317`=未训练解码器的预期值)。
- 闭环(control vs λ=0,E1 100 cells):见 `lambda_closedloop_summary.csv`(lam00 行)。
- held-out decoder score 统一口径:各臂 receipt `val_decode_r2`(held-out uncentred R²,λ=0.1=0.094);control=−0.032(未训练解码器);**NaN 原因**=v1 digest 未读 receipt,已修正。
- bootstrap CI(lambda_bootstrap_ci.csv):λ=0.1 [−8.00,−0.20];λ=1.0 [−12.20,−5.20];λ=10 [−9.60,−2.00](cell 级 10k 重采样)。
- 线性/非线性 probe 对照与 control 的 probe 差值:v5 声称的 `probe_control_comparison.csv` **当时并未生成**(2026-11-23 复核指出);v6 补交独立 probe 结果,见文末 v6 节。

---

# v5 终版:P0-3 matched control 结果与结论改写(2026-11-22)

## 五条件终表(同实例配对,E1 100 cells;`lambda_closedloop_summary.csv`/`lambda_paired.csv`/`lambda_bootstrap_ci.csv`)

| 条件 | held-out decoder R² | c20(dose) | env-SR | 配对差 | 95% bootstrap CI |
|---|---|---|---|---|---|
| λ=0(原 ckpt,E1 复用) | —(无 decoder) | 1.36e-4 | 0.310 | — | — |
| **λ=0 继续训练 control**(新增) | **−0.099** | **2.4e-7** | 0.260 | **−5.00** | **[−8.2, −2.0]** |
| λ=0.1 | 0.198 | 34.7 | 0.270 | −4.00 | [−8.0, −0.2] |
| λ=1.0 | 0.444 | 46.3 | 0.224 | −8.60 | [−12.2, −5.2] |
| λ=10 | 0.531 | 20.8 | 0.254 | −5.60 | [−9.6, −2.0] |

(control c20 为 32 窗重测值,与 Fig.2c 的 128 窗取样略有 MC 差,已标注;方向结论不受影响。)

## 结论改写(v5 原文已撤回,以下为 2026-11-23 复核后的正确版本)

> **撤回声明**:v5 曾写"三臂与 control 无法区分""在 control 之上无额外 SR 惩罚""SR 变化由继续训练本身解释"——**均为错误结论**,已删除。以 λ=0 同日程继续训练 control 为主对照的正确统计如下。

1. 对原始 ckpt:三臂 −4.0/−8.6/−5.6 pp,control −5.0 pp(均 vs 原 ckpt)。
2. **对 matched control(主对照)**:λ=0.1 **+1.0 pp**(cell bootstrap 95% CI [−1.2, +3.4]);λ=1 **−3.6 pp**(**[−5.8, −1.2],区间不含 0**);λ=10 −0.6 pp([−2.2, +1.0])。split-cluster bootstrap(10 splits):[0.0, +2.0] / [−4.5, −2.6] / [−2.0, +0.3]。
3. **只有 λ=1 显示相对 control 的明确额外负差异**;无单调剂量效应(λ=0.1 为正、λ=10≈0);全部基于**单一训练 seed**,且 100 cells 存在任务层级依赖(10 splits)——不作因果或剂量结论。
4. 可写:"相对匹配的继续训练对照,仅 λ=1 伴随额外成功率下降(−3.6 pp,95% CI [−5.8, −1.2]);继续训练本身对应 −5.0 pp([−8.2, −2.0])"。**不可写**:"下降全由继续训练解释"或"辅助目标无额外代价"。
5. 文件:`lambda_vs_continued_control.csv`(逐 cell)、`lambda_vs_control_bootstrap.csv`(20,000 次 cell bootstrap 与 split-cluster bootstrap,seed=0)。

## P0-1 五问 #2 之外补:oodc_F1 同 split 三 seed 表(见 v5 P0-1 节)

## 五问终答(对应任务单 §7)

1. **Performance robustness**:成立(GRU/LeWM 三 seed 全正 +10.8~+14.6 pp);与 LIF-Tx 差异方向翻转(−1.6~+1.1),只能写"相当"。
2. **Dynamics robustness**:成立(oodc_F1 同 split 3/3;pooled c20 trace≤gru;不对称复现;cartpole seed0 环境级反转如实报告)。
3. **Predictive fidelity**:trace seed2(cross_benchmark_F1)异常 = **z* 分母尺度塌缩**(分子反而最小;den<1e-3 占 21–70%);非预测发散、非实现错误;撤回"预测发散但控制成功"表述。
4. **Accessibility versus utility**:相对原模型各辅助条件均下降(−4.0~−8.6 pp),但 matched control 本身 −5.0 pp;**以 control 为主对照时仅 λ=1 存在额外下降(−3.6 pp,CI 不含 0)**,λ=0.1/10 与 control 不可区分;无单调剂量效应,单训练 seed。线性可访问性的独立 probe 证据见 v6。
5. **History dependence**:**通过剂量匹配检验**——trace 对同总量散布历史置零的位移是末帧置零的 2.2–3.2×;gru/lewm/membrane 对历史掩蔽剂量完全不敏感(C/A≈1.0–1.6)。claim 保留(限定 representation displacement)。

---

# v6(2026-11-23 复核修复:R1–R4)

## R1 λ matched-control 统计(已改为以 control 为主对照)
见上文"结论改写(v5 原文已撤回)"节。方法:cell-level paired bootstrap,20,000 次重采样,`random.Random(0)`,百分位 2.5/97.5;split-cluster bootstrap 对 10 个 split 有放回重采样后合并其 cells(`lambda_vs_control_bootstrap.csv`)。边界:单一训练 seed;100 cells 嵌套于 10 splits;不做因果/剂量结论。

## R2 独立 probe 与 decoder receipts
- **`probe_control_comparison.csv` 已补交**(`export_independent_probes.py`,冻结 latent,5 ckpt 共用 128 窗/env 与 episode 级 75/25 划分 rng(123),leakage 断言通过;逐样本预测/真值在 `probe_predictions.npz`):

| ckpt | linear ridge R² | MLP(3 probe seeds) |
|---|---|---|
| orig | −1.828 | −0.231 ± 0.089 |
| control(λ=0 继续训练) | 0.013 | −0.171 ± 0.391 |
| λ=0.1 | −0.085 | 0.104 ± 0.021 |
| λ=1 | 0.103 | 0.236 ± 0.039 |
| λ=10 | −0.102 | 0.260 ± 0.012 |

  判读:独立 probe 下**线性可访问性增益小且非单调**(≤0.10;λ=10 为负),MLP probe 呈温和单调(0.10→0.26)。此前用共训 decoder head 得到的 0.20/0.44/0.53 **不能**作为架构无关的可访问性证据;v5 "剂量依赖的线性可读性提升"措辞收窄为上表。
- **两套 decoder score 的 reconciliation**(`decoder_score_reconciliation.csv`):receipt `val_decode_r2`(λ=0.1 0.094/λ=1 0.304/λ=10 0.331/control −0.032)取自 fine-tune 窗口表的**末 5%**,但训练 shuffle 覆盖全部窗口 → **非 held-out(样本泄露)**;我方重测(0.198/0.444/0.531/−0.099)用 32 个**从未参与训练**的新窗口、窗口末端 latent、held-out 半区(n=16/env)。两者均为 uncentred R²(1−SSE/SST,s=state/state_std,未中心化),同一 co-trained head(λ 臂);control 的 head 未训练(λ=0 无梯度)。audit v5 中 −0.0317/0.094 与 −0.0986/0.198 两处不一致即源于此。

## R3 checkpoint 与评测 provenance
- `checkpoint_manifest_oodcF1_new.csv`:6 个 oodc_F1 新 ckpt 的 SHA256、args、loss_log/train.log 路径(仓库内 `results/5m_seed{1,2}/oodc_F1/`,git 忽略目录)。
- **层数核对(重大更正)**:已发表 seed0 trace = `n_layers=4`(stack 4.732M,总 10.52M)。我方 P0-1 首次补训误用 G5 launch 脚本文本的 `--n-layers 2`(2 层,8.16M)→ **与正文"four-layer"不等价**;已用 4 层配方重训 trace seed1/2(现 10.52M,与 seed0 一致),2 层版归档于 `<dir>__2L_wrong_archive`。**既有 G5/B2 的 seed1/2 trace ckpt(cross_benchmark_F1/oodc_F2/generalist_16env)本就是 4 层**(修复重训代际),故 37-cell 三 seed 表与 cross_benchmark 动力学**不受影响**;gru/lif_tx 内部层数固定,不受该参数影响。
- **P0-1 以 4 层 trace 重测的终值**(oodc_F1,pooled c20):trace **0.0319/0.0164/0.0185** vs gru **0.0403/0.0293/0.0887**(trace<gru 3/3);不对称 trace g_a/g_o = 30/20/83×,gru 反向 329/573/245×(3/3)。**环境依赖**:cheetah 3/3 trace<gru(0.0140/0.0162/0.0083 vs 0.0660/0.0485/0.1483);**cartpole 3/3 trace≥gru**(0.0498/0.0165/0.0288 vs 0.0146/0.0102/0.0290)——此前"cartpole 仅 seed0 反转"更正为"cartpole 系统性反向"。v5 P0-1 节的 2 层数值(0.0064/0.0071)作废。
- `lambda_episodes.csv`:5 条件 × 100 cells × 5 eps = 2500 行,含 episode_idx/initialization_seed/init 与 goal 的 md5;**100/100 cells 五条件完全一致**(逐 episode 配对核验,非仅 cell 名匹配)。
- `lambda_arm_checkpoints.csv`:每个 λ arm 用于全部 100 cells 的**单一** ckpt 路径 + SHA256(训练 split 均为 oodc_F1;评测覆盖 10 splits)。

## R4 masking 边界
- `maskdose_per_window.csv`(v2)新增逐窗 `input_l2_clean/input_l2_masked/clean_latent_norm/latent_abs_displacement`。汇总:输入变化幅度 A≈B(ΔL2%:cartpole −1.2 vs −0.9;cheetah −0.4 vs −0.4),win25 为 27× 剂量(−37.2%/−13.0%)→ **A/B 在输入层面剂量匹配成立**。clean latent norm:trace 0.0139/0.0087 vs gru 32/22、lewm 13.8/11.9、membrane 22.8/17.8(trace 小 ~1000×);绝对位移 trace 0.0039–0.0197 vs 其他 0.17–15——**trace 的相对位移比由小分母放大,跨模型只宜比较绝对量或注明尺度**;trace 内部 B/A 比值(2.3×/3.2×)在绝对位移下同样成立(同窗分母恒定)。
- 置零维度数:0.25D 四舍五入且至少 1 → **cartpole 置零 1 维(native D=4~5),cheetah 置零 2 维**;并非每环境严格 25%。
- 结论限定:同数量置零 ≠ 同输入变化幅度;结果支持"不同时间位置的输入掩码对 latent 位移影响不同",**不证明**历史记忆造成规划提升;未做闭环评估。

## 写作阶段约束(与复核一致)
性能:37-cell 三 seed Trace 优于 GRU/LeWM,LIF-Tx 与 Trace 相近;100-cell 主图 seed 0。动力学:oodc_F1 pooled c20 与输入响应差异跨 seed 复现,非每环境同向,非因果机制。预测精度:seed2 E_lat 极值为归一化尺度问题。解码干预:相对原模型各辅助条件下降,matched control 后 **λ=1 仍有额外下降(−3.6 pp,CI 不含 0)**;线性可访问性以独立 probe 为准(小且非单调)。Masking:支持对历史位置的不同响应,非"历史记忆→规划提升"。

---

# v7(2026-11-23 补齐全部未完成实验;第一批结果)

## A. 随机动作参照(P1-2)——【完成;改变对绝对成功率的解释】

同一 E1 实例 manifest(eval seed 0)、同 action box、同 50 步预算与终点判据;每 cell 10 个随机种子 × 5 episodes(`code/scripts/extra_closed_loop_evals.py --mode random`;`random_baseline_per_cell.csv`)。

- **随机动作 pooled env-SR = 0.358 > Trace 0.311**(97 cells + 3 masked-cheetah cells);**58/97 cells 上随机 ≥ 全部 13 个模型**。
- 逐环境随机 SR:ball_in_cup 1.00、cartpole **0.99**、fish 0.985、stacker 0.80、finger 0.61、delayed_t_maze 0.53、humanoid_CMU 0.41、cheetah 0.25(模型全为 0.00)、pendulum 0.20、hopper 0.15、tworoom 0.12;walker/reacher/pusht/humanoid/quadruped/dog ≈ 0。
- **结论**:该 dataset-derived goal-reaching 协议的成功判据在多数环境可被随机动作满足(25 步目标态与起点相近);env-SR 的绝对水平(0.31)**低于随机地板(0.36)**。因此:(i) env-SR 绝对值不能证明控制能力;(ii) 模型间差异应表述为"相对随机地板的增减",且多数模型在多数环境上**劣于随机**;(iii) Fig.1b 必须叠加随机地板,正文须改写"closed-loop competence"的措辞。

## B. 动作通路剂量响应(P1-1)——【完成;无显著效应】

`cem_predict_hook` 内 a′ = ā + γ(a − ā),ā = 0(CEM 初始均值);环境执行原始 25 步序列;100 cells(`dose_response_summary.csv`)。

| γ | env-SR | 相对 γ=1 配对差 | cell bootstrap 95% CI |
|---|---|---|---|
| 1(sham) | 0.303 | — | — |
| 0.75 | 0.307 | +0.41 pp | [−2.68, +3.71] |
| 0.5 | 0.332 | +2.89 pp | [−0.62, +6.19] |

把 planner 看到的动作差异减半**不降低**成功率(点估计反而升高,ns)→ 与 E9、随机地板一致:该协议下成功与否对候选动作的精细区分依赖很弱。模型本身未改,g_a/g_o/E_lat 不变(冻结权重)。

## C. 闭环评测的 run-to-run 可复现性——【新发现,影响所有配对结论的误差界】

γ=1 sham 与 E1 基线(同 ckpt、同实例、同 seed、同代码路径)逐 episode 对比:**成功一致率 441/485 = 90.9%**(24 次 T→F、20 次 F→T);90% 的 episode 终点距离不同(32% 相差 >0.1);cell 级 SR 平均差 5.4 pp(21/97 cells 不同);总体 0.311 → 0.303。原因:GPU 非确定性核 + CEM 精英选择对微小数值差敏感。
→ **所有单次运行的配对差都带有这一噪声底**;已追加 3 次基线重跑(sham_rerun_1-3)以估计总体 SR 的 run-to-run 分布(结果见 v7 续)。

## D. Held-out 与 Pixel 协议审计(P1-4)——【完成;发现 26 cells 的 spec 泄漏】

- **评测 manifest 共享**:E1 的 13 个模型在同一 cell 上 (episode_idx, init) 完全一致;pixel 的 13 个模型在 cross_benchmark_F1 与 oodc_F2 全部 13 envs 上 (episode_seeds, goal_state, episode ids) 完全一致。
- **Pixel 协议**(eval_summary.protocol):H=5、`replan_every=1`(first-action)、frame_skip 1、episode_seeds/planner_seeds=[0..4];与 state(H25 分段、2 次规划)**不同**,须分别报告。
- **E4 held-out 单位**:546 cells 中 **520 = 未见环境**(评测 env ∉ 该模型训练 split 的 env 集);**26 cells 为 spec 泄漏**(oodc_F3 / oodc_F2F3 训练的模型被评 `cheetah`,而 cheetah 在其训练 split 内)——`e4_leaked_cell_ids.json`。排除后(`e4_per_model_ci.csv`,n=40):Trace **0.305 [0.19, 0.43]**、stacked_lif_trace 0.295、LIF-Tx 0.275、MLP 0.255、GRU 0.170、LeWM 0.160;CI 重叠。
- **调参独立性**:超参固定(1 epoch/batch 32/lr 3e-4,LeWM 协议),`train.py` 无验证集早停或 checkpoint 选择(取 final);E7/λ 扫描只用 in-dist split(oodc_F1/cross_benchmark_F1/oodc_F2)。可证明"无脚本化的 held-out 调参";**无法证明**不存在非脚本化的人工选择。

## E. λ 多训练 seed(P0-3 续)——【完成,三 seed × 100 cells】`lambda_multiseed_summary.csv`

| 训练 seed | orig | control | λ=0.1 | λ=1 | λ=10 |
|---|---|---|---|---|---|
| 0 | 0.310 | 0.260 | 0.270 | 0.224 | 0.254 |
| 1 | 0.266 | 0.258 | 0.298 | 0.272 | 0.306 |
| 2 | 0.322 | 0.260 | 0.258 | 0.256 | 0.312 |

对 matched control 的配对差(cell bootstrap 95% CI):

| 臂 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| λ=0.1 | +1.0 [−1.2, +3.4] | **+4.0 [+2.2, +5.8]** | −0.2 [−3.8, +3.4] |
| **λ=1** | **−3.6 [−5.8, −1.2]** | **+1.4 [+0.4, +2.4]** | −0.4 [−4.4, +3.6] |
| λ=10 | −0.6 [−2.2, +1.0] | **+4.8 [+2.4, +7.4]** | **+5.2 [+2.8, +8.0]** |

**结论**:seed 0 的"λ=1 相对 control 额外下降"**不跨训练 seed 复现**(seed 1 方向相反);辅助解码目标对闭环成功率**没有跨 seed 一致的效应**(三 seed 中方向不一致)。继续训练 control 相对 orig:−5.0 / −0.8 / −6.2 pp。原 ckpt 自身的跨 seed 波动 0.266–0.322。此前"仅 λ=1 存在额外下降"须再收窄为"单 seed 观察,未复现"。

**独立 probe(冻结 latent,同划分;`probe_control_comparison_multiseed.csv`)**:

| | orig linear / MLP | control | λ=0.1 | λ=1 | λ=10 |
|---|---|---|---|---|---|
| seed 0 | −1.83 / −0.23 | 0.01 / −0.17 | −0.09 / 0.10 | 0.10 / 0.24 | −0.10 / 0.26 |
| seed 1 | −30.1 / −9.9 | −4.76 / −0.32 | 0.17 / 0.18 | 0.06 / 0.30 | 0.10 / 0.34 |
| seed 2 | −2.13 / −0.47 | −0.78 / 0.11 | 0.03 / 0.17 | 0.03 / 0.24 | −0.29 / 0.22 |

→ 三 seed 一致:辅助目标把线性可访问性从强负拉到 ≈0(≤0.17,非单调),非线性(MLP)R² 到 0.10–0.34;**可访问性变化可复现,闭环成功率变化不可复现**。

## F. 可复现性的进一步澄清(同日重跑)

今日的 3 次同配置运行(sham γ=1、rerun1、rerun2)**逐 episode 完全一致**(agreement 1.000,aggregate 0.3031);9 月的 E1 原始运行与今日运行 agreement 0.909(0.3113 vs 0.3031)。即评测在**同一环境内是确定性的**,差异来自 9 月与当前环境之间的偏移(GPU/驱动/库状态),约 −0.8 pp 总体、9% episode 翻转。E1 的 13 模型比较在同一次会话内完成,内部一致。以"今日重跑"为 seed0 基线重算 λ 配对差,结论不变(`lambda_seed0_vs_session_baseline.csv`:control −4.12 [−7.84, −0.62];λ=1 −7.22 [−11.96, −3.09])。

## G. 零动作参照(P22 "cheetah edge" 的可操作化)——【完成;最关键】

同 manifest、同预算,`CEM.plan` 恒返回零动作(`--mode zero`;`zero_action_per_env.csv`,99 cells):

| | pooled env-SR |
|---|---|
| **零动作(保持不动)** | **0.477** |
| 随机动作 | 0.356 |
| Trace(CEM) | 0.311 |

| env | 零动作 | 随机 | Trace |
|---|---|---|---|
| ball_in_cup / cartpole / fish / humanoid_CMU | 1.00 / 1.00 / 1.00 / **1.00** | 1.00 / 0.99 / 0.985 / 0.41 | 1.00 / 1.00 / 0.90 / **0.11** |
| cheetah | **0.80** | 0.26 | **0.07** |
| finger / hopper / reacher | 0.80 / 0.60 / 0.20 | 0.61 / 0.15 / 0.01 | 0.54 / 0.14 / 0.00 |
| delayed_t_maze / tworoom | 0.00 / 0.00 | 0.53 / 0.12 | 0.53 / 0.00 |

**结论**:该协议的成功判据(到达轨迹 t+25 的记录态,容差 ‖Δqpos‖/√nq ≤ 0.1)在多数环境可由**保持不动**满足;Trace 在 **33/99 cells 低于零动作**。因此:
1. env-SR 的绝对水平与多数模型间差异反映的是"控制器把系统扰动离目标态的程度",不是控制能力;GRU/LeWM 的 0.16/0.15 远低于不动地板 0.48——它们主动偏离目标。
2. 仅 delayed_t_maze(Trace 0.53 vs 零动作 0.00)显示明确高于地板的行为。
3. **论文主张须重述**:Fig.1b 必须叠加零动作与随机地板;"Trace 支撑闭环控制"改为"在该 goal-reaching 协议下 Trace 相对 GRU/LeWM 的优势表现为更少偏离不动地板";所有基于 env-SR 绝对值的"competence"措辞撤回。
4. "cheetah edge"的解释:零动作 0.80、模型 ~0.00–0.07 → cheetah 地板来自 CEM 主动把系统推离目标态,与"潜空间-物理梯度不对齐"(I 节)一致。

## H. P13 多 epoch(3 vs 1,oodc_F1 训练,97 cells;`p13_multiepoch_summary.csv`)

Trace 0.303 vs 0.311(−0.8 pp,等于今日 1-epoch 重跑 0.303 → 无增益);GRU **0.210 vs 0.163(+4.7 pp)**;LeWM 0.167 vs 0.159(+0.8)。更多训练对 Trace 无效,对 GRU 有改善但仍低于 Trace。

## I. utility 4 driver(最终代际 manifest;`utility_final_gen/`)

- **sample_efficiency**(冻结编码器+线性策略,4 envs,1%→100% 数据):Trace 0.392→0.417(+2.5 pp,平台);LeWM 0.159→0.400(**+24.2**)、MLP +12.5、SLIF-free +10.0、GRU +6.7;其余 SNN 读出 ±2 pp(`utility_sample_efficiency_summary.csv`)。
- **latent_env_grad**(潜代价梯度 vs 物理目标梯度的相关,4 envs 均值):Trace −0.135、GRU −0.064、ALIF −0.035、LeWM **−0.633**、no_trace/membrane/leak −0.48(`utility_latent_env_grad_summary.csv`;修复了 cuDNN RNN backward 需 train 模式的 bug)→ 潜空间优化方向与物理目标不对齐甚至反向,与 G 节互证。
- **budget_scaling / cross_env_gen**:官方 driver 要求 0.5×/2× 预算 ckpt 与 G14(minus walker/humanoid)ckpt——最终代际从未训练。已补训(6 + 12 个)并以独立评估完成(见 v7 续)。

## J. G2 事件类型 AUROC(最终代际 oodc_F1 ckpt,676 probes,52 skipped;`g2_event_auroc_summary.csv`)

Trace:contact 0.583、high-motion 0.572、low-motion 0.569(略高于 0.5);GRU 0.589/0.585/0.568;LeWM 0.623/0.610/0.593;LIF-Tx 0.654/0.618/0.602;MLP 0.502/0.509/0.496(塌缩签名)。delayed_t_maze cue_state:Trace 0.198、GRU 0.881、LeWM 0.945。→ Trace 的事件类型线性可解码性**低于连续基线**;"event synchronization 优势"勘误维持。

## K. E9 扩样(50 episodes/条件;`e9_expanded_per_condition.csv`)

cartpole(H5 协议 baseline 0.68):全部消融模式 0.70–0.72,无下降;cheetah(baseline 0.14):−2~−8 pp,CI 全含 0;pendulum(baseline 0.08):≥ baseline。**5 倍样本后结论不变且更稳:规划期想象中的 trace 内容不是成功的必要条件。**

## L. 训练数据预算(0.5×/1×/2×,G16,6 envs × 5 eps;`budget_scaling_summary.csv`)——【完成;新训 6 ckpt】

| 模型 | 0.5× | 1.0× | 2.0× |
|---|---|---|---|
| Trace | 0.500 | 0.433 | 0.467 |
| Spike-only | 0.267 | 0.333 | 0.467 |
| MLP | 0.467 | 0.400 | 0.500 |

Trace 无清晰预算效应(差异为单 cell 翻转量级);Spike-only 随数据单调上升。每点仅 30 episodes,只作描述。新 ckpt:`/data/lx/tmp/results/generalist_G16_compression/<model>/<frac>/seed_0/`(post-freeze,非 manifest 内)。

## M. 零样本跨环境(G14 = G16 minus walker/humanoid,12 模型;`cross_env_gen_summary.csv`)——【完成;新训 12 ckpt】

held-out envs(walker、humanoid)**全部模型、两种 regime 均 0.00**(零动作地板亦为 0.00)→ 无可测的零样本外推信号,该比较不具信息量(floor effect)。G14 的 in-domain(5 envs)env-SR:Trace 0.56、rate 0.56、spike 0.52、SLIF-trace 0.48、LeWM 0.40、GRU/MLP/SLIF-free 0.32、hidden_leak 0.24。

## N. E9 扩样终表(4 envs × 6 模式 × 50 episodes;`e9_expanded_per_condition.csv`)

| env | baseline | ablate_all | Δ [95% CI] |
|---|---|---|---|
| cartpole_2d | 0.68 | 0.72 | +4 [0, +10] |
| cheetah | 0.14 | 0.10 | −4 [−14, +6] |
| pendulum_2d | 0.08 | 0.12 | +4 [−4, +12] |
| **finger** | **0.20** | **0.60** | **+40 [+24, +56]** |

finger 上把想象期 trace 置零使成功率**翻三倍**(CI 不含 0;event_window +16、non_event +24 亦显著)。四环境合并:**规划期想象中的 trace 内容不是成功的必要条件,在 finger 上反而有害**——与其零动作地板(0.80)一致:带 trace 的候选评分让 CEM 把系统推离目标态,消融后候选评分退化、动作趋近于零。该结论限定于 CEM 候选评分通路(`model.predict`)。

---

# v7 总综合:本轮对论文主张的影响(2026-11-23)

## 观察(全部可复算)
1. **不动地板 0.477 > 随机 0.356 > Trace 0.311**:协议成功判据主要被"保持不动"满足;Trace 在 33/99 cells 低于零动作;仅 delayed_t_maze(0.53 vs 0)高于地板。
2. Trace 相对 GRU/LeWM 的 env-SR 优势跨三训练 seed 复现(+10.8~+14.6 pp),相对 LIF-Tx 相当;但所有模型均低于不动地板。
3. 动作差异减半(γ=0.5)不降低成功率(+2.9 pp,ns);想象期 trace 置零在 finger 上 +40 pp。
4. 潜空间代价梯度与物理目标梯度不对齐(signed corr ≤0;LeWM −0.63)。
5. 同会话评测确定性一致;与 9 月 E1 存在 +0.8 pp 的环境偏移。
6. λ 辅助解码:线性可访问性从强负→≈0(三 seed 一致),闭环 SR 无跨 seed 一致效应。
7. 动力学(oodc_F1 三 seed)、剂量匹配遮蔽、E_lat 分母塌缩等结论见 v5/v6,不变。

## 解释(与观察一致但不唯一)
- 该 dataset-derived goal-reaching 协议下,CEM 在潜空间的候选优化没有转化为物理目标达成;成功主要取决于"少动"。Trace 的"优势"= 其候选评分比 GRU/LeWM 产生更小的偏离;消融/减半动作通路使其更接近"不动",因而在部分环境提升成功率。
- 因此动力学差异(c20、输入响应不对称)是真实的、可跨 seed 复现的**表示性质**,但不能解释为"更好的控制机制"。

## 推测(需新实验,不得入正文)
- 若换成需要主动运动的 goal 构造(目标态远离起点)或更长的 budget,模型间排序可能改变;验证需新协议(未做)。

## 写作硬约束(覆盖此前所有版本)
- **撤回**:"Trace 支撑闭环控制 / closed-loop competence"作为绝对主张;Fig.1b 必须叠加零动作与随机地板。
- **改写为**:"在该协议下 Trace 的成功率高于 GRU/LeWM(三 seed),但所有模型均低于不动地板;差异反映偏离地板的程度"。
- **撤回**:"λ=1 相对 control 额外下降"(单 seed,未复现);"历史记忆造成规划提升"。
- **保留**:表示层结论(动力学、可访问性、dose-matched masking、seed 稳健的性能排序、E_lat 分母塌缩定性)。

---

# v8 主动运动目标协议(AGR,2026-11-24)——【完成;决定性】

## 协议
原协议全部不变,只替换评测实例集合:窗口合格 ⇔ (1) 目标离起点 ≥ **2×成功容差**(环境自身度量)**且** (2) 从起点零动作走满 50 步**不**成功。合格集合与模型无关、带缓存,所有模型与地板共用(`protocol.md` §6;`code/eval/closed_loop.py --active-goal-disp 2`)。13 模型 × 各自 E1 ckpt × 81 cells(12 个 DMC 环境;ball_in_cup 无任何主动目标,delayed_t_maze/pusht/tworoom/reacher 无标量容差,均不纳入)= 1053 cells × 5 episodes。**有效性检查:零动作地板 = 0.0000(78/78 cells)**。

## 结果(`active_goal_summary.csv`、`active_goal_paired_vs_trace.csv`、`active_goal_per_env.csv`)

| 模型 | AGR env-SR | vs 随机地板 |
|---|---|---|
| **随机动作地板** | **0.031** | — |
| LIF-Tx / rate-only / MLP | 0.049 | +1.84 pp |
| ALIF | 0.044 | +1.34 |
| **Trace** / spike-only / SLIF-trace | **0.040** | +0.85 |
| SLIF-free | 0.022 | −0.88 |
| membrane / no_trace | 0.027 | −0.39 |
| GRU | 0.015 | −1.62 |
| LeWM / hidden_leak | 0.012 | −1.87 |

Trace 对基线的配对差(81 cells,cell bootstrap):vs GRU **+2.47 [+0.49, +4.69]**、vs LeWM **+2.72 [+0.99, +4.69]**、vs hidden_leak +2.72 [+0.74, +4.94];vs LIF-Tx −0.99 [−4.20, +1.98]、SLIF-trace 0.00、spike 0.00、rate −0.99、MLP −0.99、ALIF −0.49(均含 0)。
逐环境:成功仅出现在 cartpole(0.14–0.29,随机 0.08)、finger(0.06–0.20,随机 0.09)、cheetah(0–0.11,随机 0.08)、walker(0–0.06)、hopper/stacker(≤0.05);dog、fish、humanoid、humanoid_CMU、pendulum、quadruped **所有模型 0.00**。

## 结论
1. **当目标真正需要运动时,全部 13 个模型的闭环成功率坍缩到随机地板附近(0.012–0.049 vs 随机 0.031)**;每模型 405 episodes,SR≈0.04 的标准误≈1 pp——所有模型与随机的差 ≤1.8 pp。
2. 原协议的 0.31(以及全部模型间"优势")来自被动可达的目标;AGR 下 Trace 仅剩对 GRU/LeWM/hidden_leak 的 **≤2.7 pp** 残差,对 LIF-Tx/SLIF-trace/spike/rate/MLP 为 0。
3. **在该评测框架下,没有任何模型(包括 ST-JEWM)表现出目标到达控制能力**。论文不能主张闭环控制;可保留的是表示层结论,以及"Trace 的规划器比 GRU/LeWM 扰动更小"这一行为描述。
4. 诊断性含义:CEM 在 latent 空间的候选优化未转化为物理目标到达(与潜-物理梯度不对齐、剂量响应不敏感、E9 finger 置零反而更好互证)。问题在规划/目标表示链路,而非某个读出。

## 可写 / 不可写
- 可写:"在 dataset-derived goal-reaching 协议下成功率受不动地板主导;在要求主动运动的 AGR 协议下所有模型≈随机(≤5%)。"
- 不可写:任何"world model 支撑规划/控制"的结论;把原协议的模型排序解释为控制能力差异。

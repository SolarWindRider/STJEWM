# 勘误与已知问题(errata.md)

## 1. 作废数字(出现即错)

| 旧数字/旧说法 | 状态 | 替代 |
|---|---|---|
| "SNN family ρ≥0.95 vs continuous chance" | **作废(G1 勘误)** | readout ρ 0.004–0.527 全 13 模型;obs-embedding ρ 均值 0.814 |
| LeWM-SR@0.05 作为核心指标 | **证伪** | 塌缩模型得分最高(LIF-Tx 1.00@cos 0.000、ALIF 0.95、SLIF-trace 0.94、MLP 0.88),trace 0.37;只作证伪分析 |
| "能效 20×" | 作废 | 稀疏加权代理 ≈3×(3.37 vs 9.76–10.24),须带 "proxy" 限定 |
| "全 12 模型尺度不变" | 作废 | 严格判据下 10/12 失败(E5);失败模式跨尺度切换 |
| v0.7.19 及以前全部数字 | 作废 | 以 journal_prep 最终表格为准 |
| "STJEWM 总 16.57M / 可训练 10.54M"(旧 handoff/摘要) | **作废(实测更正)** | state trace 总 10.523M = 冻结 5.459M + 可训练 5.064M;pixel 10.461M/5.002M;5M 口径=可训练参数 |
| 早期摘要数字 trace 0.116±0.004 [0.107,0.125] | 过时统计口径 | G5 最终:pooled 0.118±0.012(13 模型 × 3 seeds;不同聚合方式,**两者都不要再引用旧值**) |
| pixel "Stacked-free 1.00" 旧说法 | 作废 | E3 真值:SLIF-free 0.000,pendulum 相关旧叙述删除 |
| "trace seed2 latent rollout 预测发散但控制成功"(2026-11-20 审计 v4) | **作废(2026-11-22 分解)** | 异常 = **z\* 参考潜尺度塌缩**:误差分子三 seed 最小(0.0048),den<1e-3 窗口占 21–70%,den_median 跨 seed 0.0136→0.0094→0.0027;相对误差指标在该 ckpt 不可解释,非预测发散、非实现错误(`latent_error_summary.csv`) |
| "辅助解码目标导致规划下降" / "买可读性伤规划" | **作废(2026-11-22 matched control)** | 见下行:control 本身 −5.0 pp |
| "三臂与 control 无法区分" / "下降全由继续训练解释" / "无额外 SR 惩罚"(审计 v5) | **作废(2026-11-23 复核)** | 以 control 为主对照:λ=1 **−3.6 pp [−5.8, −1.2]**(区间不含 0);λ=0.1 +1.0 [−1.2, +3.4];λ=10 −0.6 [−2.2, +1.0]。无单调剂量效应、单训练 seed |
| "共训 decoder head 的 val_decode_r2 = held-out 可读性"(0.094/0.304/0.331;Fig.2c 上子图) | **作废(2026-11-23)** | receipt 的验证尾在训练 shuffle 内(非 held-out);独立冻结-latent probe:线性 R² ≤0.10 且非单调,MLP ≤0.26(numbers.md) |
| "SNN 天然不坍缩 / 不需要 SIGReg" | **作废(E7)** | λ=0 时 STJEWM 表示层塌缩(cos 0.101→0.005,LeWM-SR→0.99),env-SR 不变;合法表述 = "失败模式不对称 + 控制对塌缩不敏感" |
| "Trace 优于 LIF-Transformer" | **作废(37-cell 三 seed)** | 配对差 −1.6/−0.5/+1.1 pp,方向翻转;只能写"相当"(100-cell seed0:0.310 vs 0.308) |
| "cartpole 的 c20 排序仅 seed 0 反转" | **更正(4 层重测)** | oodc_F1 三 seed 上 cartpole 均 trace≥gru(系统性);cheetah 均 trace<gru |
| "SLIF-trace CSR=2.81e4 是真实敏感度" / "SLIF obs h5 = underflow,17.9" | **作废** | 前者 = cheetah g_o=0 触 1e-8 加数的聚合伪影(均值比口径 4.65);后者 = per-env 比值聚合伪影,直接池化 g(5)=1.9e-4 可测(v3 面板) |
| "Fig.4b n=256 窗" / "Fig.1b = 10 splits × 10 envs" | **作废** | 64 窗/env × 2 env = 128 窗;100 cells = 10 splits、每 split 2–16 envs(5/6/2/11/7/8/15/15/15/16) |
| E9 表旧版(non_event_window 0.60、缺 cem_rollout_ablation 行) | **作废** | numbers.md 新表:non_event 0.70,6 模式齐全 |
| oodc_F1 trace seed1/2 首次补训(2 层,8.16M) | **作废(已归档)** | 误用 `--n-layers 2`;已以 4 层(10.52M)重训,2 层版在 `<dir>__2L_wrong_archive` |
| handoff 旧文"E1 = 10 splits × 10 envs(均)" / "E2 = F1 13env + oodc_F2 7env + generalist 17env" | **作废** | E1 每 split 2–16 envs;E2 = cross_benchmark_F1 15 + generalist_16env 16 + oodc_F2 6 = 37 |
| "Trace 支撑闭环控制 / closed-loop competence / 可支撑 goal-reaching"(2026-11-23 零动作与随机参照) | **作废(绝对主张)** | 零动作 pooled env-SR **0.477** > 随机 **0.356** > Trace 0.311;Trace 在 33/99 cells 低于零动作;仅 delayed_t_maze(0.53 vs 0)高于地板。改写为"相对 GRU/LeWM 更少偏离不动地板";Fig.1b 须叠加两条地板(numbers.md) |
| "λ=1 相对 matched control 额外下降 −3.6 pp(可写)" | **更正(三 seed)** | seed0 −3.6 [−5.8,−1.2] / seed1 +1.4 [+0.4,+2.4] / seed2 −0.4 [−4.4,+3.6] → 不跨 seed 复现,无一致效应 |
| E9 "探索性 10 episodes" 版数字 | **被扩样版取代** | 4 envs × 6 模式 × 50 eps:`e9_expanded_per_condition.csv`;finger ablate_all **+40 pp [+24,+56]** |
| "G2 事件 AUROC / P13 多 epoch / utility 4 driver / P22 未跑" | **已全部补跑(2026-11-23)** | G2 676 probes;P13 3 epochs;utility 四项(含新训 6 预算 ckpt + 12 G14 ckpt);P22 以零动作参照操作化;见 experiments.md |
| E4 "546 cells 均为 held-out" | **更正** | 26 cells(oodc_F3/oodc_F2F3 训练的模型评 cheetah)为 spec 泄漏,排除后 520 cells;`e4_leaked_cell_ids.json` |
| "ST-JEWM(或任一基线)具备闭环目标到达/规划控制能力" | **VOID(AGR,2026-11-24)** | 主动运动目标协议下随机地板 0.031,全部模型 0.012–0.049;零动作地板 0.0000(构造性检查通过)。原协议的 0.31 等数字仅为"被动可达目标上的成功率",不得解释为控制 |
| "AGR 是新协议所以旧数字仍可用" | **错误** | 旧协议数字须始终标注"dataset-derived goal reaching(可被动满足)"并与零动作/随机地板并列 |
| "`lewm_baseline_v2` 即 LeWM" / 任何 "LeWM" 行 | **无效(2026-11-24 核查)** | 自写状态输入版:3 层、4.97M、**1 epoch / 5,000 步 / 16 环境共享 16 万窗口**;编码器尾部与预测器共用同一组权重、AdaLN 条件是状态而非动作、SIGReg 不在规划用的嵌入上。13 个模型的单步 latent 预测 R² 相对"复制上一帧"**全为负**。论文的 LeWM 是每任务单独训练的 18M 像素模型(10 epoch);本机官方栈复现 PushT 44/50、TwoRoom 44/50(audit v9) |
| "Trace 高于 LeWM(+10.8~+14.6 pp)" 及一切以 LeWM 为对照的模型排序 | **无效** | 对照本身不成立;须以重做的基线重跑后再比较 |
| "LeWM-SR(cos_dist<0.1)是 LeWM 论文指标" | **出处更正** | 论文只报告环境原生成功率;该代理指标来自本机 `/home/lx/LeWM/README.md` 的 dataset-replay 代理 |
| `history_size>1` 的任何训练 | **目标错位 bug(2026-11-24 已修)** | `train.py::prediction_terms`:因果输出位置 k ↔ 帧 k+1;旧代码对齐到帧 H+k,训出 H 步 ahead 预测器而规划器按单步使用。**H=1 的全部基准结果不变**(两种写法相同) |
| "我们的 CEM 与官方一致" | **错误** | 我们在原生动作单位按 N(0,1) 提议并硬截断;官方在标准化动作空间提议。对照(直接动力学模型,PushT):20% → 36% |
| DMC 闭环协议(末态判定;速度清零初态) | **协议缺陷(2026-11-24)** | ①成功只在第 50 步末态判定,官方语义为预算内任一时刻:fish 0→0.3–0.9、finger →≤0.5。②数据集只有 qpos → 初态速度清零:记录动作从静止重放 25 步落入容差的比例 walker 17% / hopper 21% / quadruped 11% / humanoid 0.5%。③49–100% 的目标起点即已在容差内(cartpole 49%、humanoid_cmu 82%、finger 69%、fish 89%、stacker 84%、ball_in_cup 100%) |

## 2. 随机权重事故史(写作时防止同类错误,也可作为 rigor 叙述素材)

1. **M4 transient(2026-10-02 已修复)**:`mech0917_transient.py` 曾漏 `load_state_dict`,M4 数字来自随机网络(旧值:trace d_peak 0.130/lewm 7.4)。已补 strict 加载并全量重测(新值见 experiments.md),旧 JSON/PNG 已被覆盖,新数据已入库(commit bbacd08 前身 1345adb)。
2. **外部 Spiking-WM(更早,已修复)**:初版外部表在随机初始化网络上评测(strict-load 命中 0/82);修复为严格子树加载(82–96 tensors)。最终数字 ρ 0.10–0.84。**任何"早期外部对照数字"不得再现。**
3. **本日审计结论**(全仓 31 个 build_model 调用点):除上述两处,官方链路(eval_one.sh→closed_loop 等)全部严格加载;`ood1_path_c.py` 用 strict=False(历史 OOD Path-C 数字,若引用须注明)。

## 3. 未完成/未运行(VOID,不得写入任何主张)

- ~~G2 事件类型 AUROC~~ → 2026-11-23 已在最终代际完成(676 probes,52 skipped;`g2_event_auroc_summary.csv`)
- ~~P13 多 epoch、P22 cheetah~~ → 已完成(P13:3 epochs vs 1;P22 以零动作/随机参照操作化)
- ~~utility 4 drivers(sample_efficiency / latent_env_grad / cross_env_gen / budget_scaling)~~ → 已在最终代际完成(官方 driver 的 manifest 门控要求的 6 个预算 ckpt 与 12 个 G14 ckpt 已补训;后两项以独立评估完成)
- 16 个早期 single-env ckpt(数字存档 `docs/single_env_historical_eval.md`,非最终代际)
- P12 是**公开重建**(原始数据丢失,作者确认按规格重建),必须标注
- ~~随机动作 / 剂量响应 / E9 扩样 / λ 多 seed~~ → 2026-11-23 已全部完成(experiments.md)
- **需主动运动的 goal 构造或更长 budget 的协议**(检验模型排序是否随地板结构改变)——未做,相关推测不得入正文
- 固定权重 readout 交换干预(不存在;future work)
- Pixel/Held-out "无非脚本化人工调参"不可证明(只能证明无脚本化调参)
- E4 逐模型效果量的 seed 不确定性(单 seed)

## 4. 代际纪律

- 训练 manifest:`4376bb14…`(371 ckpts,`_repair_archive/20260916T102154Z/training_final_repair_manifest.json`)
- 协议名:`causal_grad_sigreg_B_20260916`;冻结源 freeze v7
- 每个 cell JSON 携带自身 provenance;表格带 provenance 头
- 引用规则:数字只从 `results/journal_prep/`(已入库,commit bbacd08)取;库外原始树仅用于核查

## 5. 已知的评测协议语义(写作时如实声明,审稿人已问)

- 上下文编码动作通道 a≡0(规划时动作未定;消融支持无影响;protocol.md §2.1)
- 25 步开环段执行(非逐步 MPC)、goal 为真实轨迹 t+25 态(protocol.md §2.2)
- CEM 在 latent 空间的代价 = ‖z−z_goal‖²(JEPA 式 latent goal matching,与 LeWM 同构)
- Spiking-WM 对照是"协议内重评",非其原始论文协议(禁跨协议排名)
- **state = 分段开环**(H25 整段执行,每 episode 2 次规划);**pixel = first-action MPC**(H5,每步重规划,50 次)——Methods 必须分 modality,禁统一写 receding-horizon(protocol.md §2.1/§2.4)
- 统计单位:n=8 是模型配置数(非 episode);三 seed 结果只覆盖 37-cell 网格;bootstrap 以 cell 为单位(100 cells 嵌套于 10 splits)

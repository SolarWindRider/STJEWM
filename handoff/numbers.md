# 常用数字速查(numbers.md)

**每个数字后括号 = 出处文件。**全部为最终修复代际(2026-09-16/17)。冲突时以 journal_prep 原表为准。

## State 闭环(E1/E2,G5 paired,13 模型 × 3 seeds)

| 模型 | pooled cos | 出处 |
|---|---|---|
| STJEWM-trace | 0.118±0.012 | G5_multiseed/ |
| STJEWM-spike | 0.213±0.016 | 同上 |
| LeWM-v2 | 0.196±0.018 | 同上 |
| LIF-Transformer | 0.001±0.002(塌缩) | 同上 |
| Stacked-LIF-trace | 0.017±0.009 | 同上 |
| ALIF-timecell | 0.028±0.021 | 同上 |
| Stacked-LIF-free | 0.030±0.002 | 同上 |

- seed 翻转:0(簇结构 seed 稳定)(⚠️库外 `agg_final/g5_multiseed.json`;**G5/B2 汇总不在 journal_prep**)
- trace env-SR **0.310**(100-cell,train seed 0)= 全场最高,与 LIF-Transformer **0.308** 并列(差 −0.2 pp;次高 stacked_lif_trace 0.282、rate 0.264;GRU 0.158、LeWM 0.154);readout ρ 0.034;div ~1e-4;resp 0.26(G1/E1;审计包 results.csv)
- **pooled cos 口径 = 冻结协议闭环终局 cos_dist**(split×env×seed cells 合并),≠ teacher-forced 单步预测精度,两者不可互换

## Fig.2/3 机制数字(mech_0917_b 批次,2026-09-17 最终权重,库内被忽略目录)

c(20) pooled(`results/mech_0917_b/rollout/rollout_summary.md`,ckpt=最终代际 ✓):trace **0.032**、membrane **6.966**、gru **0.040**、lewm **0.205**、mlp 0.000、slif_trace 0.169、spike 16.852、hidden_leak 6.143。逐 env(trace:cartpole 0.0498 / cheetah 0.0140;gru:0.0146 / 0.0660)——**cartpole 上 trace 的 c20 高于 gru,pooled 排序由 cheetah 驱动**。
CSR↔env-SR Spearman:**+0.905(p=0.002)**(`results/mech_0917_b/scaling/scaling_summary.md:40`;n=8 架构,只作关联)
逐模型绝对 gain(g_obs / g_a 含量纲):`results/mech_0917_b/gain/gain_summary.md`(写作时与 CSR 并列)

## 证伪分析(LeWM-SR@0.05)

LIF-Tx 1.00(cos 0.000)、ALIF 0.95、SLIF-trace 0.94、MLP 0.88、trace 0.37(README §1)

## Pixel(E3,1690 cells)

- trace 0.115 / spike 0.177(最高);ALIF 0.003;SLIF×2/MLP/LIF-Tx 0.000;rate/no_trace/leak/membrane/LeWM/GRU 0.020–0.044(MAIN_TABLE_5M_PIXEL_FULL.md)
- 跨模态配对(1105 pairs):state env-SR 0.252 / cos 0.080 / LeWM-SR 0.679 ↔ pixel 0.147 / 0.045 / 0.800(`agg_final/cross_modality_paired.json` ⚠️库外)

## 规模轴(E5/E14)

- 严格判据(div range<0.5 且 resp max/min<5):仅 STJEWM-spike 与 LeWM 通过;10/12 失败;ALIF resp 5.7/9.9→**1937.9**(G16);MLP div 1.4e8(G16)(SCALE_INVARIANCE.md)
- 闭环(E14,in-union 4 env pooled env-SR):trace 0.35/0.35/0.30;LeWM 0.10/0.15/0.15;cheetah@G16 全 12 模型 0.00;cartpole trace 恒 1.00(E14_scale_closed_loop/)

## SIGReg(E7)

λ=0(cross_benchmark_F1,15 cells):cos **0.101→0.005**,LeWM-SR 0.37→0.99,env-SR 0.35→0.32;oodc_F2:cos 0.178→0.063,env-SR 0.00→0.03。**两 split 合并 21 cells**:λ=0 cos 0.022 / env-SR 0.238,vs λ=0.09 0.123 / 0.248、λ=0.01 0.109 / 0.276、λ=0.001 0.141 / 0.248(paired t=+1.31 ns)→ **表示层塌缩但控制不变**;不得写"SNN 天然不坍缩"(SIGREG_SWEEP.md + 审计包 e7_sigreg_cells.csv)

## 能效代理(G3/P11)

trace 3.37;SLIF-free 3.66;SLIF-trace 5.05;GRU/MLP/LeWM 9.76–10.24 MFLOP-equiv/token;LIF-Tx soma sparsity 98.4%、ALIF 96.1%(**不转化为加权节省**)(P11_energy_final/)

## 机制探针(S3/S4/M4/位置敏感性)

- S3:trace g_a 相对 LeWM ≈27×;action 响应跨全窗口持续(mech_0917/gain/)
- S4:obs 脉冲 1 步内归零(mech_0917/maskobs/)
- M4(真权重,2026-10-02):trace d_peak 0.005–0.011 / T_recover >50(nr100%)/ 持续对照 0.044–0.048 vs 瞬态残差 0.017–0.028;LeWM d_peak 13.7–21.1 / 瞬时恢复;GRU 指数 4–13 步(`results/mech_0917/transient/transient_summary.md`)
- 位置敏感性:trace 0.23–1.27 全窗口;LeWM 末位 0.018、其余 1.5e-4(`results/actionctx_ablation/imag_position_sensitivity.json`)
- obs-embedding event_rho:0.9982(单 cell 实测);G1 批量均值 0.814

## Spiking-WM(E11)

严格子树加载 82–96 tensors、7.23–7.48M 参数、12 tasks;within-episode ρ 0.10–0.84(spiking_wm_final_20260916/ ⚠️库外)

## Trace 消融(E9,最终 ckpt × {cartpole, cheetah};每条件每环境 **10 episodes**,探索性)

| 模式 | cartpole envSR | cartpole cos | cheetah envSR | cheetah cos |
|---|---|---|---|---|
| baseline | 0.60 | 0.240 | 0.00 | 0.095 |
| event_window | 0.60 | 0.227 | 0.00 | 0.096 |
| non_event_window | 0.70 | 0.231 | 0.10 | 0.089 |
| random_window | 0.70 | 0.221 | 0.00 | 0.086 |
| ablate_all | 0.70 | 0.205 | 0.00 | 0.075 |
| cem_rollout_ablation | 0.70 | 0.205 | 0.00 | 0.075 |

(干预仅作用于 CEM 候选评分中的 `model.predict` 通路,见 protocol.md §2.6;样本量不足以证必要性/非必要性;`trace_ablation_final/` ⚠️库外,逐条件表在审计包 `e9_per_condition.csv`)

## 协议常数(数字审稿人会问)

pad_obs 128 / action_dim 56 / history_size 1 / horizon 25 / budget 50 / goal_offset 25 / CEM 300-30-10(pusht 30)/ embed 192 / 状态 128 维(零填充)/ **trace stack n_layers=4**(protocol.md §1)/ state 每 episode 2 次 planning;pixel H5 first-action、50 次 planning
**参数(2026-10-21 实测)**:state trace 总 10.523M = 冻结 ViT 5.459M + 可训练 **5.064M**;pixel trace 10.461M / 5.002M;基线全可训练 lewm 4.970M、gru 5.132M、mlp 5.001M;5M 口径 = 可训练参数(4.97–5.13M)

## 主图 Fig.5 相关系数(2026-10-22 从 scaling_summary.json 直接复算)

| 关联 | n=8 | n=6(去 MLP/SLIF) | p(n=8) | 来源 |
|---|---|---|---|---|
| ρ(c20, env-SR) | **−0.190** | +0.086 | 0.65(ns) | `results/mech_0917_b/scaling/scaling_summary.json` spearman_n8.c20 |
| ρ(CSR, env-SR) | **+0.905** | +0.886 | 0.002 | 同上 spearman_n8.csr / spearman_n6_no_mlp_slif.csr |

写作口径:小样本描述性关联,禁 causal/universal(claims_evidence §3.3、§6)。

## 37-cell 三训练 seed 配对闭环(2026-11-20;审计包 per_seed_summary.csv / paired_episodes.csv)

网格 = cross_benchmark_F1(15)+ generalist_16env(16)+ oodc_F2(6)= 37 cells;三 seed 共用同一评测流(eval seed 0,episode_idx/init 逐位一致)。**只代表该 37-cell 网格;Fig.1b 的 100-cell 仅 seed 0。**

| 模型 | seed0 | seed1 | seed2 | 均值±sd |
|---|---|---|---|---|
| Trace | 0.2595 | 0.2541 | 0.2757 | 0.263±0.011 |
| LIF-Tx | 0.2757 | 0.2595 | 0.2649 | 0.267±0.008 |
| LeWM | 0.1297 | 0.1297 | 0.1514 | 0.137±0.013 |
| GRU | 0.1135 | 0.1459 | 0.1459 | 0.135±0.019 |

配对差(pp,胜/平/负):Trace−GRU **+14.6(9/27/1)/+10.8(8/29/0)/+13.0(13/22/2)**;Trace−LeWM **+13.0(8/29/0)/+12.4(10/26/1)/+12.4(12/23/2)**;Trace−LIF-Tx **−1.6(2/29/6)/−0.5(4/29/4)/+1.1(4/31/2)**(方向翻转 → 只能写"相当")。

## λ 辅助解码器:matched continued-training control(2026-11-22~23;审计包 lambda_*.csv)

E1 同 100 cells 逐 episode 配对(5 条件 episode_idx/init/goal 100/100 一致)。control = 同日程 312 步继续训练、λ_decode=0。

| 条件 | env-SR | vs 原 ckpt | **vs control**(cell bootstrap 95% CI,20k,seed 0) |
|---|---|---|---|
| λ=0(原 ckpt) | 0.310 | — | — |
| λ=0 继续训练 control | 0.260 | −5.0 [−8.2, −2.0] | — |
| λ=0.1 | 0.270 | −4.0 [−8.0, −0.2] | +1.0 [−1.2, +3.4] |
| **λ=1** | 0.224 | −8.6 [−12.2, −5.2] | **−3.6 [−5.8, −1.2]** |
| λ=10 | 0.254 | −5.6 [−9.6, −2.0] | −0.6 [−2.2, +1.0] |

split-cluster bootstrap(10 splits):+1.0 [0.0, +2.0] / −3.6 [−4.5, −2.6] / −0.6 [−2.0, +0.3]。**只有 λ=1 相对 control 有额外下降;无单调剂量效应;单训练 seed。**

c20(dose 协议):原 1.36e-4 / control **2.4e-7**(继续训练保持收缩)/ λ=0.1 34.7 / λ=1 46.3 / λ=10 20.8(三臂翻转为放大)。

**独立 probe(冻结 latent,同划分,episode 级 75/25;审计包 probe_control_comparison.csv)**:

| ckpt | linear ridge R² | MLP R²(3 probe seeds) |
|---|---|---|
| orig | −1.828 | −0.231±0.089 |
| control | 0.013 | −0.171±0.391 |
| λ=0.1 | −0.085 | 0.104±0.021 |
| λ=1 | 0.103 | 0.236±0.039 |
| λ=10 | −0.102 | 0.260±0.012 |

线性可访问性增益**小且非单调**;共训 decoder head 的 receipt 分(0.094/0.304/0.331)**非 held-out**(验证尾在训练 shuffle 内),不得作可访问性证据。

## oodc_F1 同 split 三训练 seed 动力学(2026-11-23;4 层 trace;审计包 oodc_F1_dynamics_summary.csv)

pooled c20:trace **0.0319 / 0.0164 / 0.0185** vs GRU **0.0403 / 0.0293 / 0.0887**(trace<gru 3/3);逐 env:cheetah trace<gru 3/3(0.0140/0.0162/0.0083 vs 0.0660/0.0485/0.1483),**cartpole trace≥gru 3/3**(0.0498/0.0165/0.0288 vs 0.0146/0.0102/0.0290)——排序是环境依赖的,非因果机制。不对称:trace g_a/g_o = 30/20/83×;GRU 反向 329/573/245×(3/3)。seed0 与已发表值逐位一致。

## 剂量匹配遮蔽(2026-11-22;审计包 maskdose_*.csv)

同总元素数(0.25D,四舍五入≥1:cartpole 1 维、cheetah 2 维):scatter25/last25 位移比 Trace **2.23 / 3.20**(cartpole/cheetah)、GRU 0.09/0.06、LeWM 0.06/0.03、Membrane 0.25/0.55;win25(27× 剂量)/last25:Trace 4.3/4.6,GRU 1.1/1.0,LeWM 1.0/1.0。输入 ΔL2%:A≈B(−1.2/−0.9;−0.4/−0.4),win25 −37.2/−13.0。**仅 representation displacement,未做闭环。**

## CSR 两口径、floor 与 Fig.4b 口径(2026-11-20)

CSR_env(逐 env 比值再平均,Fig.5b)vs CSR_pooled(均值比,Fig.4a 坐标):trace 32.5 / 30.2;GRU 0.00292 / 0.00304;**SLIF-trace 2.81e4 / 4.65**(cheetah 的 g_o=0 → 逐 env 比值 2.3e5,1e-8 加数伪影,**不是真实有限敏感度**)。ρ(CSR, env-SR):+0.905(原)/ +0.929(均值比)/ +0.893(去 SLIF);ρ(c20, env-SR) −0.190(p=0.65)。n=8 = 模型配置数。
Fig.4b 统计单位 = 64 窗/env × 2 env = **128 窗**(± 为每窗两分支);h=5 用直接池化增益(v3 面板,SLIF obs h5 = 1.9e-4 实测;v1 的 17.9 为占位)。

## 像素 env-SR(E3;与 state 执行协议不同)

Trace 0.172 / spike-only 0.177 / MLP 0.178 / LeWM 0.091(pixel 为 first-action MPC,不得与 state 的分段开环混报)。

## Fig.1c 相关复算(2026-11-20)

teacher-forced rollout 相对 L2(h=1)vs env-SR:ρ=+0.238(全 8)/+0.886(去空心);线性 probe R² vs env-SR:**ρ=−0.786**(全 8)/−0.943(去空心)→ 解离核心成立。

## 参照地板(2026-11-23;同 E1 实例 manifest;`zero_action_per_env.csv`、`random_baseline_per_cell.csv`)

| 策略 | pooled env-SR |
|---|---|
| **零动作(保持不动 50 步)** | **0.477**(99 cells) |
| 随机均匀动作(10 seeds × 5 eps/cell) | 0.356(97 cells) |
| Trace(CEM,E1) | 0.311 |

零动作逐 env:ball_in_cup/cartpole/fish/humanoid_CMU 1.00;cheetah/finger/stacker 0.80;hopper 0.60;reacher/pendulum 0.20;其余 0.00(delayed_t_maze、tworoom 0.00)。随机逐 env:cartpole 0.99、ball_in_cup 1.00、fish 0.985、stacker 0.80、finger 0.61、delayed_t_maze 0.53、cheetah 0.25。**Trace 在 33/99 cells 低于零动作;58/97 cells 随机 ≥ 全部模型。**

## 动作通路剂量响应(γ 缩放 planner 可见的动作差异;`dose_response_summary.csv`)

γ=1(sham)0.303;γ=0.75 0.307(+0.41 pp [−2.68, +3.71]);γ=0.5 0.332(+2.89 pp [−0.62, +6.19])。

## 可复现性(`rerun_noise_summary.json`)

同会话 4 次同配置运行逐 episode 一致(agreement 1.000,aggregate 0.3031);9 月 E1 原始运行 0.3113,与今日运行 agreement 0.909(偏移 +0.82 pp)。

## λ 多训练 seed(`lambda_multiseed_summary.csv`;每 seed 100 cells)

| seed | orig | control | λ=0.1 | λ=1 | λ=10 |
|---|---|---|---|---|---|
| 0 | 0.310 | 0.260 | 0.270 | 0.224 | 0.254 |
| 1 | 0.266 | 0.258 | 0.298 | 0.272 | 0.306 |
| 2 | 0.322 | 0.260 | 0.258 | 0.256 | 0.312 |

对 control 的配对差:λ=1 −3.6 [−5.8,−1.2] / +1.4 [+0.4,+2.4] / −0.4 [−4.4,+3.6];λ=10 −0.6 / +4.8 [+2.4,+7.4] / +5.2 [+2.8,+8.0];λ=0.1 +1.0 / +4.0 [+2.2,+5.8] / −0.2。
独立 probe 线性 R²(orig/control/λ臂):seed0 −1.83/0.01/−0.09~0.10;seed1 −30.1/−4.76/0.06~0.17;seed2 −2.13/−0.78/−0.29~0.03;MLP R²(λ臂)0.10–0.34。

## E9 扩样(4 envs × 6 模式 × 50 eps;`e9_expanded_per_condition.csv`)

baseline → ablate_all:cartpole 0.68→0.72(+4 [0,+10]);cheetah 0.14→0.10(−4 [−14,+6]);pendulum 0.08→0.12(+4 [−4,+12]);**finger 0.20→0.60(+40 [+24,+56])**。

## P13 多 epoch(3 vs 1,97 cells;`p13_multiepoch_summary.csv`)

Trace 0.303 vs 0.311(−0.8);GRU 0.210 vs 0.163(+4.7);LeWM 0.167 vs 0.159(+0.8)。

## utility(最终代际;`utility_final_gen/`;摘要 `utility_*_summary.csv`)

- 样本效率(1%→100% 数据,4 envs):Trace 0.392→0.417;LeWM 0.159→0.400;MLP 0.000→0.125;GRU 0.150→0.217;spike 0.392→0.383;rate 0.342→0.359。
- 潜-物理梯度 signed 相关(4 envs 均值):Trace −0.135;spike −0.125;rate +0.027;no_trace/membrane −0.477;hidden_leak −0.486;ALIF −0.035;SLIF-trace −0.027;SLIF-free +0.054;LeWM −0.633;GRU −0.064;MLP −0.260。
- 数据预算(6 envs × 5 eps):Trace 0.500/0.433/0.467(0.5×/1×/2×);spike 0.267/0.333/0.467;MLP 0.467/0.400/0.500。
- 零样本跨环境(G14 → walker/humanoid):全部模型 0.00(full-G16 亦 0.00;零动作 0.00)。

## G2 事件类型 AUROC(676 probes;`g2_event_auroc_summary.csv`)

Trace contact/high/low = 0.583/0.572/0.569;GRU 0.589/0.585/0.568;LeWM 0.623/0.610/0.593;LIF-Tx 0.654/0.618/0.602;MLP 0.502/0.509/0.496;delayed_t_maze cue_state:Trace 0.198、GRU 0.881、LeWM 0.945。

## E4 held-out(排除 26 个泄漏 cells;`e4_per_model_ci.csv`)

Trace 0.305 [0.19, 0.43];stacked_lif_trace 0.295;LIF-Tx 0.275;SLIF-free/MLP 0.255;GRU 0.170;LeWM 0.160(n=40 cells/模型,CI 重叠)。

## 主动运动目标协议 AGR(2026-11-24;`active_goal_summary.csv`、`active_goal_paired_vs_trace.csv`、`active_goal_per_env.csv`)

实例:目标 ≥2×成功容差 且 零动作 50 步不可达;13 模型 × 81 cells(12 DMC envs)× 5 eps;零动作地板 **0.0000**(78/78,有效性检查);**随机地板 0.031**(78 cells × 50 eps)。

| 模型 | AGR env-SR | vs 随机 |
|---|---|---|
| LIF-Tx / rate-only / MLP | 0.049 | +1.84 |
| ALIF | 0.044 | +1.34 |
| Trace / spike-only / SLIF-trace | 0.040 | +0.85 |
| membrane / no_trace | 0.027 | −0.39 |
| SLIF-free | 0.022 | −0.88 |
| GRU | 0.015 | −1.62 |
| LeWM / hidden_leak | 0.012 | −1.87 |

Trace 配对差:vs GRU +2.47 [+0.49, +4.69];vs LeWM +2.72 [+0.99, +4.69];vs hidden_leak +2.72 [+0.74, +4.94];vs LIF-Tx −0.99 [−4.20, +1.98];vs SLIF-trace/spike 0.00;vs rate/MLP −0.99;vs ALIF −0.49;vs membrane/no_trace +1.23 [−1.23, +3.70];vs SLIF-free +1.73 [−0.49, +3.95]。每模型 405 episodes,SR≈0.04 时 SE≈1 pp。
逐环境(Trace):cartpole 0.23(随机 0.08)、finger 0.11(0.09)、walker 0.06(0.08)、hopper 0.03、cheetah 0.00(0.08);dog/fish/humanoid/humanoid_CMU/pendulum/quadruped 全模型 0.00。
各环境合格目标数:200(finger 102);ball_in_cup 0。

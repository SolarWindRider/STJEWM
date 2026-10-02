# ST-JEWM 实验总报告(最终修复重跑版,EXP_Report v4.2——新增 E13 配对控制与 E14 规模轴闭环)

> 生成:2026-09-17。本版覆盖 2026-09-16 最终修复代际:371 个受影响 checkpoint 全部重训(training manifest `4376bb14c50e30e12c3a1d47435b17489568f148b6f0abd57592efe5936e1f09`,训练协议 `causal_grad_sigreg_B_20260916`),全部评测/诊断网格在冻结源(source_freeze_manifest_v7 `4ba90480e09d93d61d442be9569a03c020f6bbed9906af915015b3eb2dded834`)下重跑并逐格校验通过。**本文所有数字均来自最终落盘数据,历史代际数字全部作废**;v3.2 及更早版本(含其勘误)仅存档于 `_repair_archive/20260916T102154Z/documents/`。
> 原始数据:`/data/lx/tmp/results/`(state `state_final_corrected_20260916/`、aux `_repair_auxiliary_grid/20260916_final_resume/`、pixel `5m_pixel_final_20260916/`、external `spiking_wm_final_20260916/` 等);主表与汇总:`results/journal_prep/`(全部由最终 JSON 重建,含 provenance 头)。

---

## 0. 实验总览

| 编号 | 实验 | 规模 | 状态 | 回答的问题 |
|---|---|---|---|---|
| E1 | state 主线(13 模型 × 10 splits) | 130 ckpt / 1300 cells | ✅ | 参数对齐下各架构潜状态质量 |
| E2 | seed 鲁棒性(seed 0/1/2 × 3 splits) | 39 ckpt / 962 新增 cells(含 seed0 共 2892 之内) | ✅ | 三簇结论是否依赖随机种子 |
| E3 | pixel 全量(13 模型 × 10 splits × 13 env) | 130 ckpt / 1690 env-cells | ✅ | 结论是否迁移到像素观测 |
| E4 | held-out 泛化(9 留出 spec) | 546 cells | ✅ | 分布外子族上 regime 是否保持 |
| E5 | 规模轴 G4/G8/G16 | 36 ckpt / 252 stats | ✅ | 任务多样性增加是否破坏校准 |
| E6 | 三轴诊断(div/resp/ρ) | state 910 + pixel 520 + probes 91 | ✅ | 校准的定量刻画与塌缩检测 |
| E7 | sigreg sweep(λ ∈ {0.09,0.01,0.001,0}) | 8 ckpt / 84 cells | ✅ | 防塌正则的敏感性 |
| E8 | position probe | 91 cells | ✅ | latent 线性可解码位置信息 |
| E9 | trace causality 消融 | 2 env × 6 模式 | ✅ | trace 是否是规划的因果杠杆 |
| E10 | utility:latent-goal MPC | 12 模型 × 4 env × 5 horizon | ✅ | 规划视程对潜状态利用的影响 |
| E11 | 外部竞品 Spiking-WM(PNAS 2025) | 12 任务(严格子树加载重测) | ✅(本次以修复后协议重测) | 与已发表 SNN 世界模型的正面对比 |
| E12 | 统一口径 latent rollout | 17 ckpt × 2 env × 2 split = 68 runs | ✅ | regime-map 同批口径、λ-sweep rollout 口径 |
| G3/P11 | 能量代理 vs 稠密账本 | 26 行(13 模型 × 2 模态) | ✅(本次重测) | 稀疏度加权代理(非硬件能耗) |
| P12 | 合成 encoder 验证(按文档规格重建) | 6 命名 encoder + 增益/噪声扫描 | ✅(重建,公开 seed/规则) | 诊断框架能否识别已知失败模式 |
| E13 | 配对控制分析 STJEWM-trace vs LeWM(state 85 对 + pixel 130 对) | 逐 split × env 配对 + episode 级 McNemar | ✅(分析新增 2026-09-17) | pooled env-SR 优势是否逐格成立;优势与 latent-goal cos 是否解离 |
| E14 | 规模轴闭环补测(G4/G8/G16 × 12 模型 × in-union 4 env) | 144 cells,E1 同协议 | ✅(2026-09-17 新增) | 多样性/数据量增加是否保持 in-union 控制效用(E5 只测了动力学) |

未重跑且仍标注红字的遗留:G2 事件类型 AUROC(本文 E8 为 position probe 替代)、多 epoch(P13)、utility 其余 4 driver(sample_efficiency/latent_env_grad/cross_env_gen/budget_scaling——代码已修复待跑,不在本文主张范围)、cheetah P22。

---

## 1. 公共设置

### 1.1 数据集与数据划分

| 数据集 | 类型 | 观测 | 规模 | 用途 |
|---|---|---|---|---|
| DMC 250k(cartpole_2d/pendulum_2d/finger/ball_in_cup/cheetah/walker/hopper/quadruped/humanoid/humanoid_CMU/dog/fish/stacker/reacher_4d/delayed_t_maze) | 低维状态(proprio) | 2–87 维 | 每环境 250k 步离线轨迹 | E1/E2/E5/E7/E8 训练+评测 |
| PushT(pusht_expert_train.h5) | 低维状态(7D,px 尺度,训练时 /500 归一化) | 7D | 20000 条专家轨迹 | E1 cross-benchmark / E4 |
| TwoRoom(tworoom.h5) | 低维状态(10D,/250 归一化) | 10D | 10000 条轨迹(每集末 NaN action 行已剔除) | E1 cross-benchmark / E4 |
| 同上数据集的像素渲染(84×84×3,frozen ViT-Tiny 编码) | 像素 | 21168 维展平 | 同上 | E3 |

**训练/验证/测试划分**:训练用每 split 训练环境全集离线轨迹滑窗(window = 1 + 25 + 1 = 27 帧,窗口不跨 episode 边界);无独立验证集(1 epoch 固定预算,无模型选择);测试为闭环 CEM 评测(seen)与 held-out 留出子族(E4)。原始数据完整性:51 个原始文件全部通过 SHA256 + 全量内容审计(2026-09-16,`_repair_archive/20260916T102154Z/raw_dataset_integrity.json`),TwoRoom 的 10000 个 NaN action 行确证为集末占位符且被 episode 安全窗口排除。

### 1.2 方法

**我们的方法 ST-JEWM**(Spiking Temporal Joint-Embedding World Model,5.06M trainable):
- 多室脉冲神经元堆叠(MultiCompStack,4 层 × 192)+ 门控 spike trace(GatedSpikeTrace)
- **membrane-forbidden 协议**:规划器只能读 6 种脉冲派生读出之一(trace/spike/rate/no_trace/leak/membrane),禁止读连续膜电位
- 损失(JEPA 三项):`pred + 0.09·sigreg + 0.5·goal`
- 6 个 readout 变体共享同一 backbone(消融读出接口)

**竞品方法**:
- **Spiking-WM**(PNAS 2025,Brain-Cog-Lab):full spiking Dreamer,multi-compartment 神经元——唯一已发表 SNN 世界模型竞品,开源实现直接运行。**实测 WorldModel 子树参数 7.23–7.48M(12 任务各自计,numel 求和;v3.x 文中的 28.5M 为对完整 agent 存档的误读,作废)**。

**Baselines**(7 个,全部参数对齐;LeWM embed 288 → 4.97M,与旧代 192 宽度不同,见 §1.3):

| Baseline | 循环载体 | 读出 | 损失 |
|---|---|---|---|
| ALIF-timecell | ALIF 脉冲 + 1D conv time-cell | conv 读出 | pred + 1e-3·L1(spike) |
| Stacked-LIF-trace/free | 8 层 LIF 堆叠 | trace / 无 trace | pred + 1e-4·L1(spike) |
| LeWM | 连续 Transformer(H=3) | 连续隐状态 | pred + sigreg + goal |
| GRU | 连续 GRU(hidden 560×2) | 连续隐状态 | pred + sigreg + goal |
| MLP | 无循环(hidden 640×12) | 前馈 | pred + sigreg + goal |
| LIF-Transformer | LIF 感知 + 连续 Transformer 记忆 | 连续隐状态 | pred + 1e-3·KL + 1e-3·L1(spike) |

### 1.2.1 损失函数的两组划分(controlled comparison 的边界)

核对代码后的精确事实:**损失函数分两组**——

**A 组|JEPA 三项损失**(`stjewm_loss`,8 个模型:STJEWM 6 readouts + LeWM + GRU + MLP):

$\mathcal{L} = \|\hat z_{t+1} - \mathrm{sg}(z^*_{t+1})\|^2 + 0.09\cdot\mathrm{SIGReg}(z^{pre}) + 0.5\cdot\|\hat z_g - \mathrm{sg}(z_g)\|^2$

- 预测 target 一律 stop-gradient,由同一在线编码器产生(无独立 target 网络、无 EMA)
- SIGReg 作用于各模型自己的 obs-only 投影
- STJEWM 6 readouts 之间损失完全相同,唯一差异 = 读出接口(严格受控消融)

**B 组|native loss**(4 个脉冲基线):ALIF `pred+1e-3·L1(spike)`;Stacked-LIF `pred+1e-4·L1(spike)`;LIF-Tx `pred+1e-3·KL+1e-3·L1(spike)`。

**对比较的约束**:四项公共条件严格受控(数据、窗口、预算、优化器、参数量);损失与架构绑定属设计性混淆,state 侧校准差异不单独归因于损失;pixel 塌缩的归因不受此混淆影响(A 组 LeWM/GRU/MLP 与 B 组 ALIF/SLIF/LIF-Tx 全部塌缩,唯 STJEWM-trace/spike 保持非零余弦,区分变量横跨两组损失);E7(λ=0 时 STJEWM 自身塌缩)证明 SIGReg 对 STJEWM 的必要性。

### 1.3 训练与评测协议(对齐 LeWM,arXiv:2603.19312 App.D/F.1)

- 训练:1 epoch,batch 32,AdamW lr 3e-4,seed 0/1/2,history H=1,goal_offset G=25,pad 128(action 56)/pixel 21168;**LeWM embed_dim=288**(修复记录:旧代误用 192,本代恢复与 5M 参数对齐;29 个 LeWM 条目在 manifest 中带 approved_arg_deltas)
- 评测(state):CEM 300 × 30 elites × 10 iters(PushT 30),horizon 25,budget 50,goal = 同轨迹 t+25 真实状态,5 episodes × 1 seed
- 评测(pixel):**horizon 5,每次执行后 replan(replan_every=1)**(协议变更,见 §4.1);成功判定环境原生,物理指标一律用未掩码原生状态

### 1.4 评价指标(三轴诊断 + 任务指标)

**三轴动力学诊断**(单独任何一轴都会被欺骗,联合构成三角不可能性):
| 轴 | 定义 | 刻画 |
|---|---|---|
| **div** | latent 逐维时间标准差的均值 | 塌缩检测:→0 = 常数 latent |
| **resp** | mean‖Δlatent‖ / mean‖Δobs‖ | 过反应检测:≫1 = 放大噪声 |
| **event-ρ** | Pearson(‖Δobs‖, ‖Δlatent‖)(episode 内) | 事件同步 |

**任务指标**:cos_dist ∈ [0,1](⚠️ 可被坍缩平凡取得);env-SR(环境原生);LeWM-SR(cos<0.05 命中率,证伪分析用);position-R²(E8)。ρ 的采样误差:198 个 episode 内转移下 SE≈0.07,接近 0 的 ρ 排序不作主张。

---

## 2. E1 state 主线:13 模型 × 10 splits(1300 cells)

### 2.1 设置与动机
参数量对齐下,13 个世界模型在 10 个任务组合上训练并闭环评测。每模型 100 cells(10 splits × 全 env,seed 0)。F1F3/F2F3/F3 split 的 env 覆盖在本代扩展(+cheetah_qpos_masked/+delayed_t_maze)。

### 2.2 结果:每模型汇总(100 cells/模型)

| Model | cos | env-SR | LeWM-SR@0.05 |
|---|---:|---:|---:|
| STJEWM-trace | 0.109 | 0.31 | 0.37 |
| STJEWM-spike | 0.208 | 0.25 | 0.16 |
| STJEWM-rate | 0.050 | 0.26 | 0.73 |
| STJEWM-no_trace | 0.086 | 0.20 | 0.70 |
| STJEWM-leak | 0.086 | 0.18 | 0.71 |
| STJEWM-membrane | 0.086 | 0.20 | 0.70 |
| ALIF | 0.010 | 0.21 | 0.95 |
| Stacked-LIF-trace | 0.017 | 0.28 | 0.94 |
| Stacked-LIF-free | 0.034 | 0.25 | 0.85 |
| LeWM | 0.203 | 0.15 | 0.23 |
| GRU | 0.084 | 0.16 | 0.58 |
| MLP | 0.070 | 0.25 | 0.88 |
| LIF-Transformer | 0.000 | 0.31 | 1.00 |

per-env 全表:`MAIN_TABLE_5M_STATE_FULL.md`(现覆盖 E1+E2+E4+E7 全部 2892 cells)。数据备注:no_trace 与 membrane 的逐值完全一致(两 readout 在最终训练下收敛为等价 checkpoint),两行均为真实落盘数据。

### 2.3 说明了什么
1. **env-SR 区分度有限且对塌缩不敏感**:LIF-Tx(cos 0.000,纯塌缩)env-SR 0.31 与 STJEWM-trace(0.31)持平——env-SR 不能单独判定潜状态质量
2. **LeWM-SR@0.05 被证伪(方向反转)**:本代中塌缩/近常数 latent 恰恰是该指标的最高分——LIF-Tx 1.00、ALIF 0.95、SLIF-trace 0.94、MLP 0.88,而 env-SR 并列最优的 STJEWM-trace 仅 0.37。该 proxy 指标与行为质量负相关,不可用
3. **cos 轴的读出依赖**:trace 读出(0.109)显著低于 spike/LeWM(0.208/0.203),rate/no_trace/leak/membrane 居中(0.050–0.086)——校准判定以 E6 三轴为准,单看 cos 不构成优劣结论

---

## 3. E2 seed 鲁棒性(3 seeds × 3 splits,962 cells)

### 3.1 设置
seed 1/2 独立训练(与 seed 0 同数据同超参),3 splits 全 env 评测(seed0 由 E1 覆盖)。

### 3.2 结果(3-seed pooled cos,mean ± std;全表 `agg_final/g5_multiseed.json`)

| Model | 3-seed cos ± std |
|---|---|
| STJEWM-trace | 0.118 ± 0.012 |
| STJEWM-spike | 0.213 ± 0.016 |
| STJEWM-rate | 0.047 ± 0.007 |
| STJEWM-no_trace | 0.074 ± 0.013 |
| STJEWM-leak | 0.079 ± 0.012 |
| STJEWM-membrane | 0.074 ± 0.013 |
| ALIF | 0.028 ± 0.021 |
| SLIF-trace | 0.017 ± 0.009 |
| SLIF-free | 0.030 ± 0.002 |
| LeWM | 0.196 ± 0.018 |
| GRU | 0.079 ± 0.008 |
| MLP | 0.061 ± 0.016 |
| LIF-Transformer | 0.001 ± 0.002 |

B2 五模型子集与 G5 全量在 cos 均值上一致(±0.003)。

### 3.3 说明了什么
- readout 间排序跨种子稳定(std 0.002–0.021);**塌缩簇(LIF-Tx/SLIF-trace/ALIF/SLIF-free/rate)与非塌缩簇(trace/spike/LeWM)之间无种子翻转**,置信区间不跨簇
- cos 的绝对值读出依赖(seed 下同样成立),结论以簇结构为准

---

## 4. E3 pixel 全量(130 ckpt × 13 env = 1690 cells)

### 4.1 设置
同一 13 模型以像素观测(84×84,frozen ViT-Tiny 5.459M 共用 + trainable ≈5.0M)重训,goal=静态 qpos(可达)。**协议变更(相对 v3.x):pixel 规划 horizon=5 且每步 replan(旧 horizon 25);env 集固定为 13 个 DMC 域(不含 pusht/tworoom/reacher)**。本代恢复原计划的第 13 域 humanoid_CMU(旧代 12 域系漏配,非设计)。

### 4.2 结果(130 cells/模型 = 10 splits × 13 env 池化)

| Model | cos | env-SR | LeWM-SR@0.05 |
|---|---:|---:|---:|
| STJEWM-trace | 0.115 | 0.21 | 0.29 |
| STJEWM-spike | 0.177 | 0.11 | 0.30 |
| STJEWM-rate | 0.020 | 0.18 | 0.90 |
| STJEWM-no_trace | 0.034 | 0.11 | 0.84 |
| STJEWM-leak | 0.029 | 0.10 | 0.87 |
| STJEWM-membrane | 0.031 | 0.11 | 0.86 |
| ALIF | 0.003 | 0.09 | 0.98 |
| Stacked-LIF-trace | 0.000 | 0.22 | 1.00 |
| Stacked-LIF-free | 0.000 | 0.18 | 1.00 |
| LeWM | 0.034 | 0.10 | 0.86 |
| GRU | 0.044 | 0.14 | 0.86 |
| MLP | 0.000 | 0.20 | 1.00 |
| LIF-Transformer | 0.000 | 0.12 | 1.00 |

pixel 三轴量化(E6,resp/div,n=40):STJEWM-trace 0.384/0.0024(小而有向);ALIF 184.8/0.0527;SLIF-free 175.6/0.0268;GRU 1176.8/0.7837(发散);LeWM 49.8/0.0230;MLP ≈0/0(绝对常数)。

### 4.3 说明了什么
1. **pixel 输入下大面积塌缩,梯度分层清晰**:STJEWM-trace/spike 保持最高 cos(0.115/0.177);rate/no_trace/leak/membrane/LeWM/GRU 居中(0.020–0.044);ALIF 近零(0.003);SLIF×2/MLP/LIF-Tx 恰 0.000(逐 split 全 0)
2. LeWM-SR 平凡满分 1.00 恰好集中在恰 0 的四家族(塌缩命中);trace/spike 的 0.29/0.30 为最低——该指标仍与质量负相关
3. 塌缩与 sigreg 无关(§1.2.1 两组损失均塌);frozen ViT(5.459M)共用,差异来自潜动力学

---

## 5. E4 held-out 泛化(546 cells)

### 5.1 设置
9 个留出 spec(heldout_ 前缀的 oodc 子族与 cross-benchmark),在未训练过的环境子族上评测全部 13 模型。

### 5.2 结果(seen → heldout,pooled cos)

| Model | seen | heldout |
|---|---:|---:|
| STJEWM-trace | 0.109 | 0.071 |
| STJEWM-rate | 0.050 | 0.018 |
| SLIF-free | 0.034 | 0.009 |
| LeWM | 0.203 | 0.146 |
| ALIF | 0.010 | 0.028(唯一明确上升) |

env-SR seen → heldout:trace 0.31→0.30;spike 0.25→0.20;LIF-Tx 0.31→0.26;SLIF-trace 0.28→0.29;GRU 0.16→0.16。

### 5.3 说明了什么
- **12/13 模型 heldout cos ≤ seen cos**:goal 条件化 latent 在留出 spec 下保持甚至更贴目标;env-SR 全面下降 ≤5pp
- regime 簇结构分布外保持;ALIF 的轻微上升(0.010→0.028)不改变其塌缩簇归属(E6)

---

## 6. E5 规模轴(G4/G8/G16,252 stats)——旧"规模不变性"结论被推翻

### 6.1 设置
12 模型分别在 4/8/16 env union 上训练,7 env × 200 随机策略步测 div/resp/ρ。**回答:任务多样性增加是否破坏校准。** 本版改用显式判据:div 相对极差 <0.5 且 resp max/min <5 才记 scale-invariant。

### 6.2 结果(节选;全表 `SCALE_INVARIANCE.md`,逐行带 source sha256;判据逐行重算)

| Model | resp G4/G8/G16 | div G4/G8/G16 | 判据 |
|---|---|---|---|
| STJEWM-trace | 0.207/0.205/0.207 | 0.0002/0.0002/0.0001 | **no**(rel range 0.60,略超阈;绝对值近常数) |
| STJEWM-spike | 0.200–0.207 | — | **判据 yes**(rel range 0.32,resp 比 1.50) |
| ALIF | 5.7/9.9/**1937.9** | 0.498(G16) | **no**:G16 resp 爆炸 ×196 |
| MLP | 0/0/**5.9e9** | 0/0/**1.4e8** | **no**:G4/G8 塌缩 → G16 爆炸 |
| SLIF-trace | 0.382/0.389/0.388 | 0.000/0.000/0.022 | **no**:G4/G8 塌缩 → G16 非塌 |
| LeWM | 55–114 | 0.378/0.398/0.357 | **判据 yes**(rel range 0.10,resp 比 2.07)——过反应但规模稳定 |
| GRU | 74.7–84.0 | 0.33→0.72 | no |

### 6.3 说明了什么
1. **10/12 模型未通过显式判据**(v3.x 的"全部 ✅"是判据缺失下的误读);通过的仅 STJEWM-spike(稳定校准)与 LeWM(过反应但稳定)。规模轴**切换失败模式**:塌缩(G4/G8)↔ 爆炸(G16)在 MLP/SLIF-trace 上双向出现
2. **STJEWM-trace 的 div 跨规模恒定于 1e-4 量级**——"近常数读出的规模稳定",不是校准的规模不变;其 resp ≈0.21 恒定
3. LeWM 的 div 最平稳(0.36–0.40)但 resp 始终过高(55–114);GRU div 随规模近乎翻倍
4. 诚实结论:**"任务多样性增加时校准自动保持"对多数模型不成立**;规模轴是失败模式的放大器

---

## 7. E6 三轴诊断(div/resp/ρ;state 910 + pixel 520 cells)

### 7.1 设置
div/resp:state 70 cells/模型(5m_stats 7 baselines + 5m_stats_fair 6 STJEWM)+ pixel 40/模型;ρ:同一批 70-cell 协议内计算(episode 内 Pearson,ρ 未定义记 null 不计均值)。

### 7.2 结果:readout event-ρ(70 cells/模型;v3.2 的 "0.948–0.962" 表连同其勘误一并作废,本表为最终重训值)

| Model | ρ mean | 区间 |
|---|---:|---|
| LeWM | 0.527 | — |
| STJEWM-no_trace | 0.377 | — |
| STJEWM-leak | 0.385 | — |
| STJEWM-membrane | 0.377 | — |
| ALIF | 0.221 | — |
| GRU | 0.225 | — |
| STJEWM-spike | 0.129 | — |
| SLIF-free | 0.130 | — |
| MLP | 0.138 | — |
| SLIF-trace | 0.071 | — |
| STJEWM-trace | 0.034 | — |
| STJEWM-rate | 0.004 | — |
| LIF-Tx | −0.011 | — |

同批 obs-embedding ρ 均值 0.814(区间 −0.06–1.00)——**不存在 ≈1 的常数上限**。逐格分布见 `DIAG_RELOAD_SUMMARY.md`(n/ρ-defined/mean/min/max)。

state 幅值:STJEWM-trace div 0.0001/resp 0.26(近常数);no_trace/leak/membrane div 0.21–0.22/resp 74–81;ALIF resp 221.8;GRU 278.4;SLIF-free 700.9;MLP div 1.9e7(爆炸);LIF-Tx div 0.0054/resp 19.2。

### 7.3 说明了什么
1. **event-ρ 的旧叙事(脉冲族高位档 vs 连续族 chance)不成立**:最终重训后全部模型的 readout ρ 落在 −0.01–0.53,无一家达到旧表 0.9+ 水平;LeWM(0.527)反而是最高。事件同步作为"STJEWM 家族属性"的主张撤回
2. 三轴三角仍然成立但分布重排:塌缩轴(div→0)由 MLP/LIF-Tx/SLIF-trace/近常数 trace 读出占据;过反应轴(resp≫1)由 SLIF-free/GRU/ALIF 占据;LeWM 位置重排为"div 健康 + resp 偏高 + ρ 最高"
3. **STJEWM-trace 的画像统一为:近常数 trace 读出(div 1e-4、resp 0.26、ρ 0.034)+ 最好的闭环 cos(0.109)与并列最优 env-SR(0.31)**——低方差、低响应、事件同步弱,但规划目标对齐最好;这正是"三轴联合不能单独评判"的实例,也是 §10 trace 非因果 lever 的机理基础

---

## 8. E7 sigreg sweep(84 cells)

### 8.1 设置
STJEWM-trace,λ ∈ {0.09, 0.01, 0.001, 0.0},2 splits(cross_benchmark_F1 + oodc_F2),84 env-cells。

### 8.2 结果

| λ | F1 cos | F1 env-SR | F1 LeWM-SR | oodc_F2 cos | oodc_F2 LeWM-SR |
|---|---:|---:|---:|---:|---:|
| 0.09(默认) | 0.101–0.132 | 0.35 | 0.37–0.56 | 0.178 | 0.067 |
| 0.01 | 0.080 | 0.37 | 0.37–0.56 | — | — |
| 0.001 | 0.132 | 0.31 | 0.37–0.56 | — | — |
| **0.0** | **0.005** | 0.32 | **0.987** | 0.063 | 0.50 |

(精确逐格见 `SIGREG_SWEEP.md`;λ>0 三档 F1 cos 0.080–0.132、env-SR 0.31–0.37。)rollout 口径:λ=0 div 0.0001/resp 0.083(`latent_rollout/`)。

### 8.3 说明了什么
1. **λ=0 塌缩在新数据上更极端**:cos 0.005(λ>0 的 1/16–1/26)+ LeWM-SR 0.987 平凡命中——常数 latent 机制
2. λ ∈ [0.001, 0.09] 平坦:防塌对强度不敏感,但对**有无**敏感
3. **env-SR 单独看不出塌缩**(λ=0 的 env-SR 0.32 与 λ>0 相当)——必须配 cos/LeWM-SR 对;这再次证明单一指标不可用

---

## 9. E8 position probe(91 cells)

### 9.1 设置
从 latent 线性回归物理位置,13 模型 × 7 DMC env。

### 9.2 结果(position-R²,7 env 均值;`journal_prep/g2_summary.json`)

| Model | R² |
|---|---:|
| LeWM | 0.729 |
| GRU | 0.465 |
| STJEWM-no_trace / membrane | 0.257 |
| STJEWM-leak | 0.254 |
| ALIF | 0.242 |
| STJEWM-spike | 0.158 |
| SLIF-free | 0.113 |
| STJEWM-rate | 0.048 |
| SLIF-trace | −0.013 |
| STJEWM-trace | −0.001 |
| MLP | −0.001 |
| LIF-Tx | −0.001 |

### 9.3 说明了什么
1. 塌缩模型(MLP/LIF-Tx/trace/rate)R²≈0:常数 latent 无位置信息——独立佐证
2. **LeWM 位置 R² 最高(0.729)且 ρ 也最高(0.527)**:v3.x 的"位置可解码 ≠ 事件同步"二分在本代弱化;但 LeWM 的 resp 偏高(E6)仍构成其过反应画像
3. **event-aligned trace 读出是 position-free 的**(−0.001):trace 的信息不在位置坐标,与其闭环最优 cos 并存——诊断维度与任务维度的解耦实例

---

## 10. E9 trace causality 消融(2 env × 6 模式,10 eps,H=5)

### 10.1 设置
STJEWM-trace(最终 ckpt)× {cartpole_2d, cheetah} × 6 模式(baseline/event_window/non_event_window/random_window/ablate_all/cem_rollout_ablation)。

### 10.2 结果(`trace_ablation_final/`)

| 模式 | cartpole envSR | cartpole cos | cheetah envSR | cheetah cos |
|---|---:|---:|---:|---:|
| baseline | 0.60 | 0.240 | 0.00 | 0.095 |
| event_window | 0.60 | 0.237 | 0.00 | 0.094 |
| non_event_window | 0.60 | 0.237 | 0.00 | 0.094 |
| random_window | 0.70 | 0.205 | 0.00–0.10 | 0.075–0.094 |
| ablate_all | 0.70 | 0.205 | 0.00 | 0.075 |
| cem_rollout_ablation | — | — | — | — |

### 10.3 说明了什么
1. **0/2 env 支持"trace 是规划优势的因果来源"**:cartpole 上清空 trace 甚至略升(0.60→0.70),cheetah 全模式 0.00——trace 不是规划因果杠杆(与近常数 trace 读出一致,§7.3)
2. trace 的价值重新定位:**协议暴露的可审计事件载体**,不是规划信息通道;诚实负面结果保持
3. cheetah 的全零基线说明该 env 在 H=5 预算下不可解,消融无区分空间

---

## 11. E10 utility:latent-goal MPC horizon sweep(48 cells)

### 11.1 设置
12 个 G16 模型 × 4 env × 5 horizon(H=1/3/5/10/20),CEM 100×10×10,3 eps,原生有界动作 + 观测重规划。成功阈值:cheetah/walker 0.1、reacher 0.05、finger 0.3(qpos RMS)。

### 11.2 结果(terminal cos 节选;全表 `UTILITY_MPC_TABLE.md`)

| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |
|---|---|---:|---:|---:|---:|---:|
| stjewm_trace_only | cheetah | 0.063 | 0.050 | 0.083 | 0.069 | **0.022** |
| stjewm_trace_only | walker | 0.206 | 0.046 | 0.046 | 0.036 | 0.034 |
| stjewm_trace_only | reacher | 0.390 | 0.195 | 0.254 | 0.112 | 0.144 |
| stjewm_trace_only | finger | 0.263 | 0.191 | 0.144 | 0.091 | 0.046 |
| LeWM | reacher | 0.0005→ | — | — | — | 0.375(随 H 恶化) |

### 11.3 说明了什么
1. **trace 读出(校准簇代表)的终端 cos 随视程改善或稳定**(0.02–0.39),而非塌缩读出(no_trace/leak/membrane/rate/MLP/SLIF-trace)终端 cos ≈0.0001–0.036 = 塌缩平凡命中
2. LeWM 的终端 cos 在 reacher/finger 上随 H 单调恶化(0.0005→0.375):过反应 latent 在深规划下放大误差——与 E6 画像一致
3. env_success 全表 ≤0.67,零星 1.00 格集中于塌缩模型的 finger H≤5(近目标初始化的平凡命中),不作优势主张

---

## 12. E11 外部竞品:Spiking-WM(PNAS 2025)——严格加载重测

### 12.1 设置
开源实现直接运行(proprio 配置,12 个 DMC 任务 × 2000 步随机策略,seed 0)。**本代修复了致命加载缺陷**:旧协议以 `strict=False` 加载完整 agent 存档,实际 0 个权重命中(82 缺失/318 多余),旧 E11 表评测的是随机初始化网络。修复后仅取根 `_wm.` 子树并剥离 `_orig_mod.`,**12/12 任务 strict 加载通过(82–96 张量,WorldModel 参数 7.23–7.48M)**;episode 内诊断(terminal-to-reset 跳变剔除),原始数组落盘并带 SHA256。审计存档:`_repair_archive/20260916T102154Z/external_checkpoint_load_audit.txt`。

### 12.2 结果(12 任务,episode 内 event-ρ)

| 任务 | posterior-mode ρ | encoder ρ |
|---|---:|---:|
| reacher_easy | 0.649 | 0.835 |
| finger_spin | 0.594 | 0.750 |
| hopper_hop | 0.536 | 0.771 |
| walker_walk | 0.408 | 0.651 |
| cup_catch | 0.389 | 0.775 |
| fish_swim | 0.368 | 0.749 |
| cheetah_run | 0.344 | 0.638 |
| humanoid_run | 0.243 | 0.528 |
| quadruped_walk | 0.227 | 0.261 |
| cartpole_swingup | 0.121 | 0.641 |
| dog_walk | 0.046 | 0.097 |
| pendulum_swingup | −0.483 | 0.554 |

(v3.x 的"latent ρ 0.05–0.80 均值 0.47"恰是误加载 encoder 层的产物;本表为修复后同批口径。actor 实际消费全部 spike-time 槽,不取时间均值。)

### 12.3 说明了什么
1. **修复后的 Spiking-WM 是事件同步强基线**:encoder ρ 0.10–0.84、posterior ρ −0.48–0.65——与 E6 修复后的本地 readout ρ(−0.01–0.53)同量级;v3.x "外部 0.47 vs 我们 0.95"的对比叙事作废
2. **诊断口径不可直接排名**:E11 为全 proprio RSSM 的 episode 内相关诊断,与本地 CEM env-SR/cos 协议不同;E11 保留为案例研究,不进入跨模型排名表
3. 方法论教训已固化为协议:外部 checkpoint 必须严格子树加载 + 哈希绑定 + 负载审计后才可引用

---

## 12.5 E12 统一口径 latent rollout(68 runs)

### 12.5.1 设置
`code/scripts/latent_rollout.py`,17 ckpt × 2 env × 2 split,各 200 步随机策略,同批输出 div/resp/ρ + 轨迹(`latent_rollout/{split}/{model}/{env}.npz`)。

### 12.5.2 结果(代表性值;全表 `LATENT_ROLLOUT_SUMMARY.md`)

| Model | div | resp | ρ 范围 |
|---|---:|---:|---|
| STJEWM-trace(λ=0.09) | 0.0001 | 0.080 | −0.176–0.368 |
| λ=0 | 0.0001 | 0.083 | −0.100–−0.021 |
| LeWM | 0.252 | 21.4 | 0.545–0.976 |
| GRU | 0.641 | 137.7 | — |
| MLP | 24.29 | 3952.8 | — |

### 12.5.3 说明了什么
1. **trace 读出在自由 rollout 下近常数**(div 1e-4):与 E6/E9 一致——低方差是 trace 读出的内在属性,非 λ=0 特有
2. **λ=0 的塌缩签名在闭环(E7 cos 0.005/LeWM-SR 0.99),不在开 rollout 标量**(λ=0 div 与 λ>0 同为 1e-4)——开环统计对"低方差但仍有微弱结构"不敏感
3. MLP 交互口径爆炸(div 24.29,非 v3.x 的 7.5e7;旧值系旧 ckpt/旧口径);LeWM 的 ρ 0.545–0.976 与 E6 的 0.527 互证
4. ρ 的 SE≈0.07(n=198):近零排序是噪声

---

## 12.6 P12 合成验证(按文档规格重建;ckpt 代际无关)

### 12.6.1 背景
原始 P12 producer/raw sweep/seeds 在全部本地归档与 OBS 中未找到(检索台账 `_repair_archive/.../raw-scan`);经作者确认按 `results_draft.md` §2.3 叙述规格**公开重建**(`code/scripts/p12_synthetic_validation.py`,protocol `p12_synthetic_reconstruction_20260917`,seed 20260917)。**这是重建实验,不冒充原实验数字**。

### 12.6.2 结果(199 步 episode 安全 cartpole 窗口,seeded)
- 6 命名 encoder 分类:constant→collapsed(div 3.8e-15)✓;noisy σ=0.1→noise ✓;uncorrelated→noise(ρ 0.005)✓;identity k≥0.1→over_reactive(原始观测尺度)
- **div 判据的尺度依赖(公开披露)**:div(k·obs)=k·div(obs);原叙述"k 0.3→0.5 越阈 0.05"隐含其观测逐维 std≈0.10–0.17,与本仓库原始 cartpole 尺度(div(identity)=1.149)不符。本尺度下越阈 k∈(0.04, 0.1),尺度无关越阈 k*=0.0435
- **ρ 判据复现文档区间**:σ* = 0.0405 ∈ (0.02, 0.05) ✓;div 单调随 k 递增 ✓;ρ 随 σ 在噪声地板(|ρ|<2SE)附近可逆序(披露)
- 判定规则、尺度选择、seed 全部写入输出 JSON(`p12_synthetic_reconstruction/summary.json`),4/6 文档级 claims 复验为真、2 为假(均披露)

### 12.6.3 说明了什么
诊断框架能在合成 ground truth 上区分 collapsed/calibrated/over_reactive/noise 四种失败模式,且边界可预测;但**边界数值是尺度相对的**,论文表述以本重建的显式规则与披露为准。

---

## 12.7 G3/P11 能量代理 vs 稠密账本(26 行,本次重测)

### 12.7.1 设置
`measure_energy.py`:13 模型 × {state, pixel},解析稠密操作账本(逐 Linear 2·in·out)+ 真实随机前向实测 soma spike 稀疏度(4×4 batch,seed 20260802)。**明确标注:这是假想的稀疏度加权代理(hypothetical sparsity-weighted proxy),不是硬件能耗、速度或实测 FLOPs**;pixel 冻结 ViT 不入账。

### 12.7.2 结果(state;全表 `P11_energy_final/energy_summary.md`)
- STJEWM-trace:代理 3.37 vs 稠密账本 9.95 MFLOP-equivalents/token(≈2.95×);STJEWM 6 readouts 代理 3.10–4.65
- 无节省基线:ALIF 9.70、LIF-Tx 9.57、GRU 10.24、MLP 9.98、LeWM 9.76;SLIF-trace 5.05 / SLIF-free 3.66 是仅有的其他节省者
- **稀疏度 ≠ 节省**:LIF-Tx sparsity 98.4%、ALIF 96.1%,但其代理无节省(加权分区不含其稠密路径)

### 12.7.3 说明了什么
- v3.x 的"0.46–0.48 vs 9.8–10.2 ≈20×"不成立;真实量级为 **≈3× 代理差**(STJEWM-trace vs GRU/MLP/LeWM),且该数字依赖"事件操作可免"的假想折扣,论文引用时必须带此限定

---

## 12.8 E13 配对控制分析:STJEWM-trace vs LeWM(state 85 对 + pixel 130 对,2026-09-17 新增)

### 12.8.1 设置
- **配对规则**:state 侧以 (split, env) 一对一配对 trace(`5m_5mpar`)与 LeWM(`5m`)的 E1 cell;pixel 侧以 (split, env) 配对 `stjewm` 与 `lewm_baseline` 的 pixel cell。episode 级配对身份逐一断言(state 用 `initialization_seed` 列表,pixel 用 `episode_idx` 列表,0 mismatch)。
- **canonical 环境集**:13 个 state/pixel 共有 DMC qpos 环境(pusht/tworoom/reacher/delayed_t_maze/cheetah_qpos_masked 不参与跨模态主张)。
- **统计**:split 级 = 每 split 等权 env 均值差,exact sign-flip 检验(2^10);episode 级 = 不一致对 exact McNemar。cos 为 `mean_cos_dist`(越低越好),`cos_adv = LeWM − trace`(正 = trace 更优)。
- **产物**:`journal_prep/E13_paired_control/{paired_control_summary.json,paired_control_table.md}`;脚本 `code/scripts/e13_paired_control.py`。

### 12.8.2 结果
**State(85 对,425 episodes)**:
- split 级:Δ env-SR **+0.167**(sign-flip p=0.0078);cos_adv **+0.083**(p=0.0039);解离 split **0/10**;corr(ΔSR, cos_adv) **+0.86**——state 侧 trace 在任务与 goal-cos 两轴同时占优,无解离。
- episode 级:only-trace 79 vs only-LeWM 8(McNemar p=8.4e-16)。
- 逐 env:cartpole_2d +0.74(7/7 wins;26/0,p=3e-8)、fish +0.85(18/1,p=7.6e-5)、finger +0.43(7/7;17/2,p=7e-4)、hopper +0.14、humanoid_CMU +0.11(其中 **3/7 对解离**:trace SR 更高但 LeWM cos 更低);ball_in_cup/stacker 全 tie;quadruped 双 0 且 cos_adv −0.047。
- 难度分层(两模型平均 SR):easy(>0.7)11 对 Δ=0;**medium(0.1–0.7)36 对 Δ=+0.394(30/5/1)**;hard(<0.1)38 对双 0。低维 env(nq<10)平均 ΔSR 0.204 vs 高维 0.161。

**Pixel(130 对,650 episodes)**:
- split 级:Δ env-SR **+0.114**(p=0.0020);cos_adv **−0.081**(p=0.0020);**解离 split 10/10**——全部 10 个 split 上 LeWM 的 goal-cos 更低(指标更优)而 trace 的 env-SR 更高。
- episode 级:only-trace 77 vs only-LeWM 3(McNemar p=1.4e-19)。
- 逐 env:cartpole +0.54(27/0,p=1.5e-8;9/10 对解离)、fish +0.52(26/0,p=3e-8;9/10)、pendulum +0.28(16/2,p=0.0013;7/10);高维 env(nq≥28)双模型 SR=0 且 LeWM cos 更低(cos_adv −0.15~−0.21)。
- 维度相关性:corr(nq, ΔSR) −0.35;低维(<10)平均 ΔSR 0.137 vs 高维 0.087。

### 12.8.3 说明了什么
1. **pooled 0.31 vs 0.15 的 state 优势是逐格成立的**,不是少数 env 撑起来的:10/10 split 方向一致(exact p=0.0078),episode 级 79:8。
2. **"More informative ≠ more useful for control"在 pixel 套件成立**:LeWM 在全部 10 个 split 上 goal-cos 更优却 env-SR 全面更差(10/10 解离 split,77:3 episodes)——latent-goal 相似度与控制效用解离的直接配对证据。
3. state 侧**无解离**(corr +0.86):trace 同时更贴 goal-cos 且更会控制;解离主张应限定在 pixel 模态(与 state 的 humanoid_CMU 个别 cell)。
4. 优势集中于**中等难度带**(Δ+0.394);easy 饱和、hard 双 0(CEM/H 预算上限),符合"差异出现在有区分空间处"的预期。
5. **限定**:本分析为事后(post-hoc)配对分析,非预注册主终点;p 值为 exact 但未做多重比较校正;每 cell 5 episodes。措辞用 "directionally consistent across all splits",不用 "proof"。

---

## 12.9 E14 规模轴闭环补测(G4/G8/G16 × 12 模型 × in-union 4 env = 144 cells,2026-09-17 新增)

### 12.9.1 设置
- **动机**:E5 规模轴只测了 div/resp/ρ(`scaling_table.json` 明记 `env_sr_evidence: not evaluated by this diagnostic grid`);本节补上"校准保持 ⇒ 控制是否也保持"的闭环半边。
- **设计**:E5 训练 union 嵌套(G4 ⊂ G8 ⊂ G16),取三尺度共同的 in-union env 核 {cartpole_2d, pendulum_2d, cheetah, pusht},消除 out-of-union 覆盖混淆;E1 完全同协议(CEM 300/30、horizon 25、budget 50、goal=t+25、5 eps、pad 128/act 56、pusht CEM 30 iters),checkpoint 为最终修复代际(manifest `4376bb14…`),0 重训。
- **注意**:规模轴同时混有"环境数"(4/8/16)与"每 env 数据量"(2000/2000/10000 windows)两个因子;且 env-SR 对塌缩不敏感(E1/E7),判读必须与 E5 动力学联用。
- **产物**:`journal_prep/E14_scale_closed_loop/`;原始 cells `/data/lx/tmp/results/scale_closed_loop_20260917/`(含逐 cell sha256 与 receipts);脚本 `code/scripts/e14_scale_closed_loop.py`。

### 12.9.2 结果(pooled 4 env;全表见 `scale_closed_loop_table.md`)
- **STJEWM-trace:0.35 → 0.35 → 0.30(G4/G8/G16)**——in-union 控制效用跨规模基本保持;cartpole 三个规模均 1.00/1.00/1.00。
- **LeWM:0.10 → 0.15 → 0.15**——每个规模都被 trace 压制(+0.25/+0.20/+0.15),且其 goal-cos 在本核上同样更差(0.336/0.308/0.325 vs trace 0.158/0.178/0.101)。
- **cheetah 是唯一随规模退化但全体一致的 env**:G4 最佳 0.40、G8 最佳 0.40、**G16 全部 12 模型 0.00**——与 E5 的"规模=失败放大器"在闭环端对齐(G16 动力学普遍劣化,ALIF resp 1937.9 等)。
- pendulum 全模型 ~0.20、pusht 几乎全 0:与规模无关的均匀难/未解(CEM/H 预算上限),不承载规模信息。
- **env-SR 不可单独排序再次复现**:SLIF-trace 在 G4/G8 以 **0.45** 的 pooled env-SR 高于 trace(0.35),但其 cos = 0.000(纯塌缩)、LeWM-SR 平凡 1.00;MLP(塌缩)也拿 0.35/0.25/0.35。
- trace 与 LeWM 逐 env 配对(G4):cartpole +0.8、cheetah +0.2、pendulum/pusht 平;方向与 E13 state 一致。

### 12.9.3 说明了什么
1. **规模轴的闭环结论与动力学结论互补**:E5 显示多数模型潜动力学跨规模切换失败模式;E14 显示即便如此,STJEWM-trace 在 in-union 易 env 上的控制效用大体保持(cartpole 恒 1.00),而 hard env(cheetah@G16)对全体模型失效——规模对控制的影响是 **env 难度依赖的**,不是全局塌方。
2. **trace ≥ LeWM 在规模轴每个尺度都成立**(pooled +0.15~+0.25),与 E13 state 逐格结论同向,延伸到训练多样性维度。
3. **限定**:每 cell 5 episodes、只有 4 个 in-union env、3 个尺度点;不构成规模效应的显著性的预注册检验;SLIF-trace 高分属塌缩平凡命中,不得作为"规模改善排序"引用。

---

## 13. 威胁与限制

1. **单 seed 主线**:E1/E3 主表 seed 0;E2(3 seeds)验证簇结构稳定
2. **held-out 覆盖**:E4 覆盖 9 spec;更远分布移位未测
3. **E5 判据敏感性**:scale-invariance 判据(div 极差 <0.5 且 resp max/min <5)为本版显式选择;不同判据会改变逐模型 yes/no,但"塌缩↔爆炸切换"的定性事实(ALIF/MLP/SLIF-trace)判据无关
4. **CEM 规划器天花板**:state 侧 horizon 25/budget 50;难 env 全 0 属规划器限制
5. **pixel 协议变更**:horizon 5 + 每步 replan(§4.1);pixel 与 state 的数字不可直接互比,比较经 cross_modality 配对表(`agg_final/cross_modality_paired.json`,1105 对:state env-SR 0.252/cos 0.080 vs pixel 0.147/0.045)
6. **E11 口径**:外部诊断为 episode 内相关,不进排名表;其修复史(0 权重命中)作为方法论警示记录
7. **P12 为重建实验**:规则/尺度/seed 为重建决策,已披露;原始 sweep 未寻获
8. **utility 其余 4 driver** 未跑,本文无其主张
9. **损失两组混淆**(§1.2.1):归因以横跨两组的共同行为与 E7 为依据
10. **G3 代理性质**(§12.7):非硬件测量

## 14. 可复现性

- 训练:`code/scripts/generalist_v0_7_5_5m/repair_training.py --manifest .../training_final_repair_manifest.json --execute`(371/371,status JSON 记录逐 ckpt 隔离/晋升)
- 评测:`repair_eval_grid.py`(2892 cells,plan `152a2568…`,audit/ 聚合)、`repair_auxiliary_grid.py`(1321 cells)、`run_pixel_eval_all.sh`(1690 cells,grid_status.json)
- 诊断/消融:`measure_energy.py --training-manifest …`、`event_window_ablation.py`、`latent_rollout.py`、`eval_spiking_wm_protocol.py`、`p12_synthetic_validation.py`
- 冻结源:`_repair_archive/20260916T102154Z/source_freeze_manifest_v7.json`(sha `4ba90480…`);训练/数据源注册表逐 ckpt 内嵌,精确等值校验
- 主表:`results/journal_prep/`(全部含 provenance 头)
- **命名对照**:报告写 **LeWM**;代码/目录 id `lewm_baseline_v2`(state)或 `lewm_baseline`(pixel)

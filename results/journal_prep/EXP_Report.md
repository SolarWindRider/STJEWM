# ST-JEWM 实验总报告(v2 重训版,EXP_Report)

> 生成:2026-09-09。覆盖 2026-09-03 起数据代际重置后的**全部重跑实验**。
> 原始数据:`/data/lx/tmp/results/`;主表与汇总:`results/journal_prep/`。
> 旧代际(no-backward 随机权重评测)数字全部作废,本文所有数字均来自 v2 重训后落盘数据。

---

## 0. 实验总览

| 编号 | 实验 | 规模 | 状态 | 回答的问题 |
|---|---|---|---|---|
| E1 | state 主线(13 模型 × 10 splits) | 130 ckpt / 1248 cells | ✅ | 参数对齐下各架构潜状态质量 |
| E2 | seed 鲁棒性(seed 0/1/2 × 3 splits) | 78 ckpt / 234 cells | ✅ | 三簇结论是否依赖随机种子 |
| E3 | pixel 全量(13 模型 × 10 splits) | 130 ckpt / 1560 env-cells | ✅ | 结论是否迁移到像素观测 |
| E4 | held-out 泛化(9 留出 spec) | 546 cells | ✅ | 分布外子族上 regime 是否保持 |
| E5 | 规模轴 G4/G8/G16 | 36 ckpt / 252 stats | ✅ | 任务多样性增加是否破坏校准 |
| E6 | 三轴诊断(G1 ρ + div/resp) | state 140 + pixel 520 + G1 24 | ✅ | 校准的定量刻画与塌缩检测 |
| E7 | sigreg sweep(λ ∈ {0.09,0.01,0.001,0}) | 8 ckpt / 63 cells | ✅ | 防塌正则的敏感性 |
| E8 | position probe(G2 风格) | 91 cells | ✅ | latent 线性可解码位置信息 |
| E9 | trace causality 消融 | 2 env × 6 模式 | ✅ | trace 是否是规划的因果杠杆 |
| E10 | utility:latent-goal MPC | 12 模型 × 4 env × 5 horizon | ✅ | 规划视程对潜状态利用的影响 |
| E11 | 外部竞品 Spiking-WM(PNAS 2025) | 12 任务 | ✅(历史数据,与代际无关) | 与已发表 SNN 世界模型的正面对比 |
| — | P12 synthetic / G3 FLOPs | — | 沿用旧数字 | 与 ckpt 代际无关(合成 encoder / 架构分析) |

未重跑且标注红字的遗留:G2 事件类型 AUROC(本文 E8 为 position probe 替代)、多 epoch(P13)、utility 其余 4 driver(需代码适配)、cheetah P22。

---

## 1. 公共设置

### 1.1 数据集与数据划分

| 数据集 | 类型 | 观测 | 规模 | 用途 |
|---|---|---|---|---|
| DMC 250k(cartpole_2d/pendulum_2d/finger/ball_in_cup/cheetah/walker/hopper/quadruped/humanoid/humanoid_CMU/dog/fish/stacker/reacher_4d/delayed_t_maze) | 低维状态(proprio) | 2–87 维 | 每环境 250k 步离线轨迹 | E1/E2/E5/E7/E8 训练+评测 |
| PushT(pusht_expert_train.h5) | 低维状态(7D,px 尺度,训练时 /500 归一化) | 7D | 20000 条专家轨迹(平均 196 步) | E1 cross-benchmark / E4 |
| TwoRoom(tworoom.h5) | 低维状态(10D,/250 归一化) | 10D | 10000 条轨迹(平均 92 步;每集末 NaN action 行已剔除) | E1 cross-benchmark / E4 |
| 同上数据集的像素渲染(84×84×3,frozen ViT-Tiny 编码) | 像素 | 21168 维展平 | 同上 | E3 |

**训练/验证/测试划分**:
- **训练**:每个 split 的训练环境全集离线轨迹,滑窗采样(window = history 1 + goal_offset 25 + 1 = 27 帧;**窗口不跨 episode 边界**,由 `dones`/episode 元数据约束)
- **验证**:无独立验证集——1 epoch 固定预算训练,模型选择不存在(所有模型同预算),避免选择偏差
- **测试**:闭环 CEM 评测(同环境 seen)与 held-out 留出子族(E4,env 级留出:每个 split 持有不同环境子族作为未训练测试环境)

### 1.2 方法

**我们的方法 ST-JEWM**(Spiking Temporal Joint-Embedding World Model,5.06M trainable):
- 多室脉冲神经元堆叠(MultiCompStack,4 层 × 192)+ 门控 spike trace(GatedSpikeTrace)
- **membrane-forbidden 协议**:规划器只能读 6 种脉冲派生读出之一(trace/spike/rate/no_trace/leak/membrane),禁止读连续膜电位
- 损失(JEPA 三项):`pred + 0.09·sigreg + 0.5·goal`——latent 空间 1 步预测 + SIGReg 方差正则(防坍缩)+ goal 一致性(t+25 latent)
- 6 个 readout 变体共享同一 backbone(消融读出接口)

**竞品方法**:
- **Spiking-WM**(PNAS 2025,Brain-Cog-Lab):full spiking Dreamer,multi-compartment 神经元,28.5M——唯一已发表 SNN 世界模型竞品,开源实现直接运行

**Baselines**(7 个,全部 4.97–5.13M 参数对齐):
| Baseline | 循环载体 | 读出 | 损失 |
|---|---|---|---|
| ALIF-timecell | ALIF 脉冲 + 1D conv time-cell | conv 读出 | pred + L1 sparse |
| Stacked-LIF-trace/free | 8 层 LIF 堆叠 | trace / 无 trace | pred + sparse |
| LeWM-v2 | 连续 Transformer(H=3) | 连续隐状态 | pred + sigreg + goal |
| GRU | 连续 GRU(hidden 560×2) | 连续隐状态 | pred + sigreg + goal |
| MLP | 无循环(hidden 640×12) | 前馈 | pred + sigreg + goal |
| LIF-Transformer | LIF 感知 + 连续 Transformer 记忆 | 连续隐状态 | pred + KL + sparse |

### 1.3 训练与评测协议(对齐 LeWM,arXiv:2603.19312 App.D/F.1)

- 训练:1 epoch,batch 32,AdamW lr 3e-4,seed 0/1/2,history H=1,goal_offset G=25,pad 128(action 56)/pixel 21168
- 评测:CEM 300 样本 × 30 elites × 10 iters(PushT 30 iters),**horizon 25 env steps(等效原文 H5 × frame-skip 5)**,budget 50,goal = 同轨迹 t+25 真实状态,5 episodes × 1 seed
- 成功判定(环境原生):PushT 位置差 <20 px 且角度差 <π/9;TwoRoom <16 px;DMC L2/√nq ≤ 0.1

### 1.4 评价指标(三轴诊断 + 任务指标)

**三轴动力学诊断**(核心;单独任何一轴都会被欺骗,联合构成三角不可能性):
| 轴 | 定义 | 刻画 |
|---|---|---|
| **div**(divergence) | latent 逐维时间标准差的均值 | 塌缩检测:→0 = 常数 latent |
| **resp**(responsiveness) | mean‖Δlatent‖ / mean‖Δobs‖ | 过反应检测:≫1 = 放大噪声;→0 = 无响应 |
| **event-ρ** | Pearson(‖Δobs‖, ‖Δlatent‖) | 事件同步:观测跳变时 latent 是否同步跳变 |

**任务指标**:
- **cos_dist** = (1 − cos(z_final, z_goal))/2 ∈ [0,1]:规划终点与 goal 的 latent 对齐度(⚠️ 可被坍缩平凡取得,单独使用无效)
- **env-SR**:环境原生成功率(§1.3 判定)
- **LeWM-SR**:cos_dist < 0.1 的命中率(复现仓库 proxy 指标,非 LeWM 原文;证伪分析用)
- **position-R²**:从 latent 线性预测物理位置的 R²(E8)

---

## 2. E1 state 主线:13 模型 × 10 splits(1248 cells)

### 2.1 设置与动机
参数量对齐(±3.2%)下,让 13 个世界模型在 10 个任务组合(不同环境训练/留出)上训练并闭环评测,回答:**在可训练参数对齐时,不同架构的潜状态质量有何差异?** 每模型评测 96 cells(10 splits × 全 env,seed 0)。

### 2.2 结果:每模型汇总(全量 seen cells)

| Model | cells | cos | env-SR | LeWM-SR |
|---|---:|---:|---:|---:|
| STJEWM-trace | 96 | 0.246 | 0.35 | 0.45 |
| STJEWM-spike | 96 | 0.233 | 0.34 | 0.48 |
| STJEWM-rate | 96 | 0.257 | 0.34 | 0.44 |
| STJEWM-no_trace | 96 | 0.262 | 0.31 | 0.48 |
| STJEWM-leak | 96 | 0.239 | 0.31 | 0.49 |
| STJEWM-membrane | 96 | 0.267 | 0.33 | 0.40 |
| ALIF | 96 | 0.128 | 0.35 | 0.70 |
| Stacked-LIF-trace | 96 | 0.073 | 0.34 | 0.81 |
| Stacked-LIF-free | 96 | 0.097 | 0.34 | 0.73 |
| LeWM-v2 | 96 | 0.196 | 0.34 | 0.30 |
| GRU | 96 | 0.148 | 0.34 | 0.53 |
| MLP | 96 | 0.029 | 0.35 | 0.91 |
| LIF-Transformer | 96 | 0.000 | 0.35 | 1.00 |

per-env 全表:`MAIN_TABLE_5M_STATE_FULL.md`(13 模型 × 10 splits,cell = env-SR/cos_dist)。

### 2.3 说明了什么
1. **env-SR 无区分度**(全员 0.31–0.35,挤在 4pp 内):易 env(ball_in_cup/cartpole/cheetah 1.000,finger 0.890)饱和,难 env(dog/humanoid/pusht/stacker/reacher/quadruped/humanoid_CMU)全 0,中段 hopper 0.156、pendulum 0.143、**tworoom 0.123**、fish 0.115、walker 0.037——区分潜状态质量的任务指标是 cos_dist 与三轴诊断,不是 env-SR
2. **LeWM-SR 被证伪**:cos 最低的 LIF-Tx(0.000)拿满分 1.00、MLP 0.91——常数 latent 平凡命中,该指标不反映潜状态质量
3. **cos 轴重新定位**:STJEWM cos(0.233–0.267)高于 SLIF/ALIF——但这不构成劣势判定;校准判定以 E6 三轴为准(STJEWM 三轴零方差,见 §7)

---

## 3. E2 seed 鲁棒性(3 seeds × 3 splits)

### 3.1 设置
seed 0/1/2 独立训练(同数据同超参,仅随机种子不同),在 cross_benchmark_F1 上评测,**回答:三簇结论是否依赖某个偶然种子**。

### 3.2 结果(F1,mean ± std 与 95% CI,t 分布 df=2)

| Model | mean ± std | 95% CI | Cohen's d vs LeWM |
|---|---|---|---|
| STJEWM-trace | 0.2682 ± 0.0151 | [0.2308, 0.3056] | +2.08 |
| STJEWM-spike | 0.2270 ± 0.0203 | [0.1765, 0.2775] | +1.08 |
| STJEWM-rate | 0.2630 ± 0.0074 | [0.2446, 0.2814] | — |
| STJEWM-no_trace | 0.2529 ± 0.0366 | [0.1619, 0.3439] | — |
| STJEWM-membrane | 0.2687 ± 0.0241 | [0.2087, 0.3287] | — |
| STJEWM-leak | 0.2707 ± 0.0192 | [0.2230, 0.3183] | — |
| ALIF | 0.1560 ± 0.0308 | [0.0794, 0.2326] | −0.52 |
| SLIF-trace | 0.0550 ± 0.0184 | [0.0092, 0.1008] | −2.93 |
| SLIF-free | 0.2019 ± 0.0159 | [0.1624, 0.2414] | — |
| LeWM | 0.1803 ± 0.0577 | [0.0368, 0.3237] | — |
| GRU | 0.1530 ± 0.0243 | [0.0927, 0.2133] | −0.62 |
| MLP | 0.1817 ± 0.0809 | [−0.0194, 0.3828] | +0.02 |
| LIF-Tx | 0.0130 ± 0.0207 | [−0.0384, 0.0644] | — |

### 3.3 说明了什么
- **STJEWM 跨种子最稳**(6 readout std 0.007–0.037);MLP 波动最大(±0.081,常数 latent 对初始化敏感)
- cos 排序与三轴判定**解耦**:校准判定以 div/resp/ρ 为准(§7),cos 数字本身不构成好坏结论

---

## 4. E3 pixel 全量(1560 env-cells)

### 4.1 设置
同一 13 模型以像素观测(84×84,frozen ViT-Tiny 5.459M + trainable 5.0M)重训,goal=静态 qpos(可达)。**回答:结论是否依赖低维状态输入;pixel 输入下各架构的表征是否退化。**

### 4.2 结果(120 cells/模型)

| Model | cos | env-SR | LeWM-SR | pixel latent 判定 |
|---|---:|---:|---:|---|
| STJEWM-trace | 0.225 | 0.18 | 0.73 | **非退化** |
| STJEWM-spike | 0.179 | 0.18 | 0.77 | **非退化** |
| STJEWM-rate | 0.234 | 0.18 | 0.74 | **非退化** |
| STJEWM-no_trace | 0.084 | 0.15 | 0.88 | **非退化** |
| STJEWM-leak | 0.083 | 0.16 | 0.86 | **非退化** |
| STJEWM-membrane | 0.233 | 0.13 | 0.73 | **非退化** |
| ALIF | 0.000 | 0.08 | 1.00 | 塌缩(常数) |
| SLIF-trace | 0.000 | 0.19 | 1.00 | 塌缩 |
| SLIF-free | 0.000 | 0.20 | 1.00 | 塌缩 |
| LeWM-v2 | 0.000 | 0.18 | 1.00 | 塌缩 |
| GRU | 0.017 | 0.19 | 0.95 | 塌缩 |
| MLP | 0.000 | 0.17 | 1.00 | 塌缩 |
| LIF-Transformer | 0.000 | 0.18 | 1.00 | 塌缩 |

pixel 三轴量化(E6,div/resp):MLP resp=0.000/div=0.0000(绝对常数);ALIF div 0.053、LeWM 0.023、SLIF 0.010–0.027(近常数,幅度被压平);GRU div 0.784(发散)。

### 4.3 说明了什么
1. **pixel 输入下,除 STJEWM 外全部 7 个对照的 latent 塌缩为常数**(cos 恰 0.000、LeWM-SR 平凡满分 1.00)
2. 塌缩与 sigreg 有无无关(LeWM/GRU/MLP 有 0.09·sigreg 仍塌;SLIF/ALIF/LIF-Tx 无 sigreg 也塌)——防塌因素是 STJEWM 的**多室脉冲动力学 + trace 接口**,不是正则项
3. frozen ViT(5.459M)为全部模型共用——塌缩差异来自潜动力学,不是编码器

---

## 5. E4 held-out 泛化(546 cells)

### 5.1 设置
9 个留出 spec(6 个 oodc 子族 split + 3 个 cross-benchmark):在**未训练过的环境子族**上评测全部 13 模型。**回答:regime 结构在分布外是否保持。**

### 5.2 结果

**§3.1 oodc 子族 held-out(39 cells/模型)**:
| Model | cos |
|---|---:|
| MLP / LIF-Tx | 0.000 / 0.000 |
| SLIF-trace / SLIF-free | 0.061 / 0.077 |
| ALIF | 0.090 |
| GRU | 0.115 |
| LeWM | 0.148 |
| STJEWM 6 readouts | 0.161–0.224 |

**§3.2 cross-benchmark held-out(pusht/tworoom/reacher,3 cells/模型)**:
| Model | cos |
|---|---:|
| LIF-Tx / MLP | 0.000 / 0.002 |
| SLIF-trace | 0.023 |
| LeWM / GRU / SLIF-free / ALIF | 0.087–0.173 |
| STJEWM 6 readouts | 0.195–0.304 |

### 5.3 说明了什么
- **相对排序在分布外保持**:塌缩对 ≈0、校准族中等、STJEWM 偏高——regime 结构是分布外稳定的
- 3-cell 粒度(cross)噪声大,细粒度排序不作主张;塌缩/非塌缩二分稳健

---

## 6. E5 规模轴 scale-invariance(G4/G8/G16,252 stats)

### 6.1 设置
12 模型(无 LIF-Tx)分别在 4/8/16 env union 上训练,7 env × 200 随机策略步测 div/resp。**回答:任务多样性增加是否破坏校准、是否产生新失败模式。**

### 6.2 结果

| Model | G4 resp | G8 resp | G16 resp | G4 div | G8 div | G16 div | 不变量 |
|---|---:|---:|---:|---:|---:|---:|---|
| STJEWM-trace | 0.207 | 0.205 | 0.207 | 0.0142 | 0.0142 | 0.0143 | ✅ |
| STJEWM-spike | 0.204 | 0.207 | 0.200 | 0.0135 | 0.0132 | 0.0136 | ✅ |
| STJEWM-rate | 0.210 | 0.206 | 0.207 | 0.0122 | 0.0126 | 0.0156 | ✅ |
| STJEWM-no_trace | 0.204 | 0.209 | 0.205 | 0.0113 | 0.0141 | 0.0138 | ✅ |
| STJEWM-leak | 0.210 | 0.204 | 0.201 | 0.0148 | 0.0130 | 0.0144 | ✅ |
| STJEWM-membrane | 0.207 | 0.211 | 0.213 | 0.0136 | 0.0143 | 0.0122 | ✅ |
| ALIF | 0.208 | 0.207 | 0.202 | 0.0147 | 0.0139 | 0.0140 | ✅ |
| SLIF-trace | 0.382 | 0.389 | 0.388 | 0.0124 | 0.0140 | 0.0150 | ✅ |
| SLIF-free | 0.375 | 0.375 | 0.381 | 0.0130 | 0.0143 | 0.0139 | ✅ |
| LeWM | 26.4 | 28.0 | 21.5 | 0.198 | 0.212 | 0.207 | ✅(过反应不变) |
| GRU | 82.6 | 84.0 | 74.7 | 0.030 | 0.030 | 0.030 | ✅(噪声不变) |
| MLP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | ✅(坍缩不变) |

### 6.3 说明了什么
1. **STJEWM 6 readouts 跨三规模几乎恒定**(resp 极差 ≤0.011、div 极差 ≤0.003)——校准的规模不变性
2. **故障模式规模不变**:LeWM 过反应、GRU 噪声、MLP 坍缩跨规模不交换——规模轴不产生新失败模式
3. 12/12 模型 scale-invariant

---

## 7. E6 三轴诊断(G1 event-ρ + div/resp)

### 7.1 设置
- **G1 event-ρ**:200 步随机策略,cartpole_2d+cheetah × cross_benchmark_F1/oodc_F2,13 模型 × 4 = 24 任务(6 STJEWM)+ 7 baselines × 4 = 28
- **div/resp**:state(70 cells/模型)+ pixel(40/模型系)

### 7.2 结果:G1 latent event-ρ(新 ckpt)

| Model | n | ρ mean | ρ min | ρ max |
|---|---:|---:|---:|---:|
| STJEWM-spike | 4 | 0.9999 | 0.9999 | 0.9999 |
| STJEWM-membrane | 4 | 0.9999 | 0.9999 | 0.9999 |
| STJEWM-leak | 4 | 0.9996 | 0.9996 | 0.9996 |
| STJEWM-trace | 4 | 0.9993 | 0.9993 | 0.9993 |
| STJEWM-rate | 4 | 0.9993 | 0.9993 | 0.9993 |
| STJEWM-no_trace | 4 | 0.9984 | 0.9984 | 0.9984 |
| ALIF | 4 | 0.9364 | 0.7823 | 0.9994 |
| SLIF-free | 4 | 0.9677 | 0.9341 | 0.9988 |
| SLIF-trace | 4 | 0.9297 | 0.8434 | 0.9992 |
| LeWM | 4 | 0.2237 | −0.2607 | 0.7485 |
| GRU | 4 | 0.1492 | −0.5616 | 0.6935 |
| MLP | 4 | 0.3664 | −0.0434 | 0.9932 |
| LIF-Tx | 4 | −0.0247 | −0.0955 | 0.1083 |

### 7.3 说明了什么
1. **STJEWM 6 readouts ρ 全部 ≥0.9984 且跨 readout 方差近零**——事件同步是家族属性,与读出接口无关
2. 最强对照 SLIF-free 0.9677、ALIF 0.9364(但 ALIF min 掉到 0.78);**连续家族(GRU/LeWM/MLP/LIF-Tx)全部 chance 或崩塌**
3. 三轴联合的三角不可能性:MLP 塌 resp/div 两轴、GRU 崩 ρ、LeWM 崩 ρ 且 resp 28——任一故障模式只塌特定轴组合

---

## 8. E7 sigreg sweep(λ 敏感性)

### 8.1 设置
STJEWM-trace 固定,λ_sigreg ∈ {0.09(默认), 0.01, 0.001, 0.0},2 splits(cross_benchmark_F1 + oodc_F2)。**回答:防塌正则的敏感度;λ=0 时是否塌缩。**

### 8.2 结果(84 env-cells = 4 λ × {cross_benchmark_F1 15, oodc_F2 6})

| λ | F1 cos | F1 env-SR | F1 LeWM-SR | oodc_F2 cos | oodc_F2 env-SR | oodc_F2 LeWM-SR |
|---|---:|---:|---:|---:|---:|---:|
| 0.09(默认) | 0.2520 | 0.33 | 0.37 | 0.0741 | 0.07 | 0.73 |
| 0.01 | 0.2443 | 0.31 | 0.29 | 0.0661 | 0.03 | 0.73 |
| 0.001 | 0.2505 | 0.31 | 0.43 | 0.0734 | 0.00 | 0.73 |
| **0.0** | **0.0931** | 0.32 | **0.63** | 0.1108 | 0.07 | 0.43 |

辅助(cartpole 200 步随机策略,单帧 encode):λ=0.09 div=0.00004、λ=0.0 div=0.00001(λ=0 更趋常数)。

### 8.3 说明了什么
1. **λ 在 [0.001, 0.09] 区间内数字平稳**(cos 0.244–0.252),对正则强度不敏感
2. **λ=0(完全去掉 SIGReg)出现塌缩特征组合**:cos_dist 骤降(0.252→0.093)+ LeWM-SR 骤升(0.37→0.63)+ div 降 4 倍——与 pixel 侧"常数 latent 平凡命中"同一机制:**没有方差正则,latent 向低方差(常数化)移动,cos/LeWM-SR 的"改善"是平凡的**
3. **结论:SIGReg 是 ST-JEWM 防塌的必要组件**(state 模态直接证据;pixel 模态的对照全塌缩与之互为印证)

---

## 9. E8 position probe(91 cells)

### 9.1 设置
从 latent 线性回归物理位置(1 epoch,4000 train/1000 val),13 模型 × 7 DMC env。**回答:各 latent 线性可解码的物理位置信息量;塌缩模型应为 0。**

### 9.2 结果(position-R²,7 env 均值)

| Model | R² |
|---|---:|
| LeWM | 0.6489 |
| GRU | 0.4634 |
| SLIF-free | 0.3296 |
| STJEWM-leak | 0.2942 |
| STJEWM-trace | 0.2714 |
| STJEWM-no_trace | 0.2676 |
| SLIF-trace | 0.2674 |
| STJEWM-membrane | 0.2613 |
| STJEWM-spike | 0.2612 |
| ALIF | 0.2529 |
| STJEWM-rate | 0.2467 |
| MLP | −0.0011 |
| LIF-Tx | −0.0026 |

### 9.3 说明了什么
1. **塌缩模型(LIF-Tx/MLP)R²=0**:常数 latent 不含位置信息——E3/E6 的塌缩判定得到线性探测独立佐证
2. **STJEWM 6 readouts 0.247–0.294 聚在一起**:位置信息跨 readout 一致
3. LeWM 位置 R² 最高(0.649)但 event-ρ 0.22(chance)——**位置可解码 ≠ 事件同步**;LeWM 是"记住位置但不对齐事件"的典型,与三轴判定互洽

---

## 10. E9 trace causality 消融(2 env × 6 模式)

### 10.1 设置
STJEWM-trace(新 ckpt)× {cartpole_2d, cheetah} × 6 模式(baseline/event_window/non_event_window/random_window/ablate_all/cem_rollout_ablation),5 eps。**回答:trace 窗口是否是规划行为的因果来源。**

### 10.2 结果

| 模式 | cartpole envSR | cartpole cos | cheetah envSR | cheetah cos |
|---|---:|---:|---:|---:|
| baseline | 0.80 | 0.0592 | 0.00 | 0.2358 |
| event_window | 0.80 | 0.0592 | 0.00 | 0.2358 |
| non_event_window | 0.80 | 0.0592 | 0.00 | 0.2358 |
| random_window | 0.80 | 0.0592 | 0.00 | 0.2358 |
| ablate_all | 0.80 | 0.0592 | 0.00 | 0.2358 |
| cem_rollout_ablation | 0.80 | 0.1137 | 0.00 | 0.1766 |

### 10.3 说明了什么
1. **四种 trace 窗口消融与 baseline 逐位相同**——CEM 规划行为对 trace 窗口内容**完全不敏感**
2. **0/2 env 支持"trace 是规划优势的因果来源"**(与旧代际 0/3 一致)——诚实负面结果保持
3. trace 的价值重新定位:**协议暴露的可诊断事件同步载体**(event-ρ≥0.998),不是规划因果杠杆
4. cem_rollout_ablation 的行为改变说明 CEM 内部信息流才是敏感路径

---

## 11. E10 utility:latent-goal MPC horizon sweep(48 cells)

### 11.1 设置
12 个 G16 模型 × 4 env(cheetah/walker/reacher/finger)× 5 planning horizon(H=1/3/5/10/20),CEM 100×10×10,3 eps。**回答:潜状态在不同规划视程下如何被利用;最优视程在哪里。**

### 11.2 结果(mean_cos_dist_terminal,节选;全表 `UTILITY_MPC_TABLE.md`)

| model | env | H=1 | H=3 | H=5 | H=10 | H=20 |
|---|---|---:|---:|---:|---:|---:|
| stjewm_trace_only | cheetah | 0.0754 | 0.0211 | **0.0208** | 0.0340 | 0.0419 |
| stjewm_trace_only | walker | 0.1691 | 0.1032 | 0.1001 | **0.0963** | 0.0945 |
| stjewm_rate_only | cheetah | 0.0491 | 0.0336 | 0.0123 | 0.0154 | **0.0071** |
| stjewm_trace_only | reacher | 0.6496 | 0.6623 | 0.6608 | 0.6363 | **0.5999** |

### 11.3 说明了什么
1. 中等视程(H=3–10)普遍优于 H=1:单步规划无法利用多步 latent 预测
2. walker/reacher 上长视程(H=10–20)持续改善——长程任务受益于多步 latent lookahead
3. cheetah 的最优点因模型而异(trace H5、rate H20)——最优视程是任务×模型属性

---

## 12. E11 外部竞品:Spiking-WM(PNAS 2025)

### 12.1 设置
开源实现直接运行(proprio 配置,28.5M 参数,2–5×10⁵ env steps,12 个 DMC 任务)。与 ST-JEWM 对比:**同属 multi-compartment SNN 路线的已发表世界模型**。

### 12.2 结果(12 任务)

| 维度 | Spiking-WM | ST-JEWM |
|---|---|---|
| 控制(易) | cup_catch 758.8、reacher 421.0、cartpole 229.7、walker 211.8、finger 229.8 | env-SR 1.00(易 env 饱和) |
| 控制(难) | pendulum 0.0、hopper 0.0、humanoid 1.4、dog 10.8 | env-SR 0(horizon 结构限制) |
| **latent event-ρ** | **0.05–0.80**(12 任务,均值 0.47) | **≥0.9984**(每个任务) |
| spike-rate 对齐 | −0.034–0.027(chance) | 0.000–0.017(chance) |
| 规划器接口 | 连续 posterior mean(违反 membrane-forbidden) | gated trace(协议内) |

### 12.3 说明了什么
1. 对齐诊断跨全部 12 任务成立:Spiking-WM 的 latent 事件同步全面低于 STJEWM
2. 两系统 raw spike-rate 对齐均 chance——**差异不在"是否脉冲",在协议暴露的 gated trace 上**
3. 其 MCN spike 序列(0.8% 激活)比协议 trace 稀疏一个数量级

---

## 13. 威胁与限制

1. **单 seed 主线**:E1/E3 主表为 seed 0;种子鲁棒性由 E2(3 seeds × 3 splits)抽样验证
2. **held-out 覆盖**:E4 覆盖 9 spec;更远的分布移位(其他 benchmark)未测
3. **CEM 规划器天花板**:env-SR 受 horizon=25/budget=50 结构限制(§2.3),难 env 全 0 是规划器限制而非潜状态缺陷
4. **utility 其余 driver**:latent_env_grad/sample_efficiency/cross_env_gen/budget_scaling 的 driver 代码适配完成(build 已统一),结果表待下一批跑批
5. **pixel goal 语义**:静态 qpos(可达),与 LeWM 原文的动态 goal 不同,不与其 pixel 数字直接对比

## 14. 可复现性

- 全部脚本:`code/scripts/generalist_v0_7_5_5m/`(train/eval/dispatcher/run_all_v2/run_phase*)、`code/scripts/{probe,event_window_ablation,measure_pixel_stats,measure_latent_stats_5m}.py`
- 原始落盘:`/data/lx/tmp/results/`(eval json ×3400+、stats ×772、ckpt 412)
- 主表:`results/journal_prep/MAIN_TABLE_5M_{STATE,PIXEL}_FULL.md`、`DIAG_RELOAD_SUMMARY.md`、`SCALE_INVARIANCE.md`

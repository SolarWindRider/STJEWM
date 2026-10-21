# verification_bundle INDEX(第三轮审计 §7C 验收包)

构建:`code/scripts/build_verification_bundle.py`(已入库),repo commit 见 `provenance_manifest.csv`。
代际:全部来自最终修复代际(manifest `4376bb14…`,freeze v7)。构建时间见 manifest `built_utc`。

## A. 文件 → 主张 → 图 panel 映射

| 文件 | 行数 | 支撑主张(claims_evidence.md 编号) | 图 panel |
|---|---|---|---|
| `state_eval_cells.csv` | 1300 | 主张"trace 支持受限协议闭环控制"(B 级限定写法);cos 可被塌缩平凡取得 | Fig.1b/c |
| `e2_seed_paired.csv` | 962 | seed 稳健性(0 翻转) | Fig.5a |
| `e4_heldout_cells.csv` | 546 | 环境组合 held-out(**非轨迹级**,见 Q 限定) | Fig.5d |
| `e7_sigreg_cells.csv` | 84 | λ=0 塌缩;SIGReg 降低低方差退化(**非控制必要条件**) | Fig.2a |
| `e8_probe_cells.csv` | 13 | position-R² 与 env-SR 不共线(trace R²≈−0.0009 仍 env-SR 0.31) | Fig.1c/Fig.4 |
| `e9_ablation_paired.csv` | 12 | 想象期 trace 置零不降 SR(**限定表述**) | Fig.4d |
| `c20_per_model.csv` | 16 | trace c20=0.032 持续收缩(membrane 6.966) | Fig.3a/b |
| `gain_per_model.csv` | 16 | 逐模型绝对 g_obs/g_a(含量纲)→ CSR 派生量 | Fig.4a/b |
| `m4_response_timeseries.csv` | 1434 | 瞬态/持续响应曲线(逐 τ) | Fig.4c |
| `params_all_models.csv` | 13 | 5M 口径 = 可训练参数;**encoder_forward_calls_state_mode=0**(Q4) | Methods |

每行均带 split/model/env/seed/checkpoint 路径/协议常数(numerator=success_rate_env 分母=n_episodes×n_seeds;cos 单位 [0,1])。

## B. 五问直答

### Q1(trace vs λ=0 vs 随机/换目标控制的配对增益)
**部分可答(已有数据)+ 部分不可用。** 已有最接近的对照 = E9 的 `cem_rollout_ablation`(CEM 想象中 trace 置零 ⇒ 候选判别退化的 CEM):cartpole SR 不降(见 e9_ablation_paired.csv)→ 在该干预下"目标判别带来的额外成功≈0"。**NOT AVAILABLE**:random-action、goal-shuffle、constant-predictor 三种对照无现存数据;如需请按 `event_window_ablation.py` hook 模式新增(小实验,未获授权不执行)。

### Q2(E9 全 trace 置零后候选 J 方差/动作差异/goal 依赖)
**NOT AVAILABLE(逐候选 J 未落盘)**。现存仅逐模式汇总 SR/cos(e9_ablation_paired.csv)。复现所需 hook 已在 `event_window_ablation.py:139-150`(`zero_stjewm_trace` 包裹 `predict`);补录需重跑并记录候选代价分布——未获授权不执行。

### Q3(M4 d_peak 与 residual 的口径)
**同一度量、同批 60 trials(cell)、同一 L2 范数(192 维 latent),不同时间点**:d_peak = 扰动窗 [t0, t0+m) 内最大位移;residual = 窗结束后 τ_rel=+50 的位移;曲线逐 τ(−6…+50)见 `m4_response_timeseries.csv`。trace d_peak(0.005)< residual(0.017)不是口径不一致,而是"窗内响应极小、窗后漂移持续增长"这一发现本身(定义源:`mech0917_transient.py:13-16,126-153`;阈值 T_recover = d<0.1·d_peak,cap 50)。c20(S1 rollout 扰动协议)与 M4(在线 obs 噪声)是**不同协议**,不得混写同一机制。

### Q4(state 是否运行冻结 ViT?基线输入路径?)
**state 模式:不运行。** 运行时 hook 实测:6 个 STJEWM state 模型 `encoder forward calls = 0`(`logs/param_count.log`;state 输入走 `state_projector`,ViT 仅存于 ckpt,无预训练先验进入 state 计算)。pixel 模式 ViT 为激活编码器。基线:state 路径用各自可训练小编码器(lewm state_encoder 0.122M 等);pixel 基线走其自身管线(`eval_pixel_ckpt.py`)。13 模型组件级参数表见 `params_all_models.csv`(total/frozen/trainable/逐组)。

### Q5(Fig.4 "提高可解码性→收缩变差、SR 不变" 的同模型干预?)
**VOID。** `cls0917_nonlinear_probe.py` 是事后探针(post-hoc probe),不存在"同一模型附加 decoder 训练干预"实验。该句必须从稿件删除(或列为 future work);可保留的只有 E8 关联版:"linear decodability 不预测 env-SR"(e8_probe_cells.csv)。

## C. NOT AVAILABLE 清单(审计 §7C 要求显式标记)

| 项 | 状态 |
|---|---|
| Q1 random-action / goal-shuffle / constant-predictor 对照 | NOT AVAILABLE(需小实验) |
| Q2 E9 逐候选 J 分布 | NOT AVAILABLE(需带日志重跑) |
| Fig.4 decoder-training 干预 | NOT AVAILABLE → 命令 VOID(Q5) |
| 轨迹级 held-out(非重叠轨迹评测) | NOT AVAILABLE(E4 是环境组合 held-out) |
| 逐 episode 训练/评测 ID 交集统计 | 部分:cell 级 `path`/`source_entry` 相同即同源(in_dist);episode 级统计未做 → 需要时可从 npz dones 切分重算 |
| E4 逐模型效果量/CI | 原 546 cells 已入 `e4_heldout_cells.csv`;聚合 CI 未算(可从此 CSV 直接算) |

## D. 复算入口

- E1–E7 cells:上游 `aggregated_state_cells.json`(路径+sha256 见 provenance_manifest.csv EXTERNAL 行)
- c20/gain/M4:JSON → CSV 的映射即 `scripts/build_verification_bundle.py`(逐字段可读)
- 权威表格:`results/journal_prep/`(EXP_Report.md v4.0);G5/B2 汇总 ⚠️库外 `agg_final/`

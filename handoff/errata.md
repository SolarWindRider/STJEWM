# 勘误与已知问题(errata.md)

## 1. 作废数字(出现即错)

| 旧数字/旧说法 | 状态 | 替代 |
|---|---|---|
| "SNN family ρ≥0.95 vs continuous chance" | **作废(G1 勘误)** | readout ρ 0.004–0.527 全 13 模型;obs-embedding ρ 均值 0.814 |
| LeWM-SR@0.05 作为核心指标 | **证伪** | 塌缩模型得分最高(LIF-Tx 1.00@cos 0.000、ALIF 0.95、SLIF-trace 0.94、MLP 0.88),trace 0.37;只作证伪分析 |
| "能效 20×" | 作废 | 稀疏加权代理 ≈3×(3.37 vs 9.76–10.24),须带 "proxy" 限定 |
| "全 12 模型尺度不变" | 作废 | 严格判据下 10/12 失败(E5);失败模式跨尺度切换 |
| v0.7.19 及以前全部数字 | 作废 | 以 journal_prep 最终表格为准 |
| 早期摘要数字 trace 0.116±0.004 [0.107,0.125] | 过时统计口径 | G5 最终:pooled 0.118±0.012(13 模型 × 3 seeds;不同聚合方式,**两者都不要再引用旧值**) |
| pixel "Stacked-free 1.00" 旧说法 | 作废 | E3 真值:SLIF-free 0.000,pendulum 相关旧叙述删除 |

## 2. 随机权重事故史(写作时防止同类错误,也可作为 rigor 叙述素材)

1. **M4 transient(2026-10-02 已修复)**:`mech0917_transient.py` 曾漏 `load_state_dict`,M4 数字来自随机网络(旧值:trace d_peak 0.130/lewm 7.4)。已补 strict 加载并全量重测(新值见 experiments.md),旧 JSON/PNG 已被覆盖,新数据已入库(commit bbacd08 前身 1345adb)。
2. **外部 Spiking-WM(更早,已修复)**:初版外部表在随机初始化网络上评测(strict-load 命中 0/82);修复为严格子树加载(82–96 tensors)。最终数字 ρ 0.10–0.84。**任何"早期外部对照数字"不得再现。**
3. **本日审计结论**(全仓 31 个 build_model 调用点):除上述两处,官方链路(eval_one.sh→closed_loop 等)全部严格加载;`ood1_path_c.py` 用 strict=False(历史 OOD Path-C 数字,若引用须注明)。

## 3. 未完成/未运行(VOID,不得写入任何主张)

- G2 事件类型 AUROC(本文 E8 position probe 是其替代)
- P13 多 epoch、P22 cheetah
- utility 4 drivers:sample_efficiency / latent_env_grad / cross_env_gen / budget_scaling(代码在,标记 pending)
- 16 个早期 single-env ckpt(数字存档 `docs/single_env_historical_eval.md`,非最终代际)
- 固定权重 readout 交换干预(不存在;若审稿要求只能列为 future work)
- P12 是**公开重建**(原始数据丢失,作者确认按规格重建),必须标注

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

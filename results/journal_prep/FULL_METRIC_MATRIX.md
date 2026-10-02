# Full Metric Matrix — 13 models × 11 metric columns, COMPLETE + FAIR(final repair 回填版)

> **Regenerated 2026-09-17** from the final repaired evidence (20260916 final repair, `causal_grad_sigreg_B_20260916`). All values are copied from the source files listed below — no legacy numbers.
> Training manifest: `/data/lx/tmp/results/_repair_archive/20260916T102154Z/training_final_repair_manifest.json` — sha256 `4376bb14c50e30e12c3a1d47435b17489568f148b6f0abd57592efe5936e1f09`.
> Freeze v7: `4ba90480e09d93d61d442be9569a03c020f6bbed9906af915015b3eb2dded834`.

> Evidence sources:
> - `/data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json` — sha256 `723e09a40dca4b478ebd9685987ce5848c60628791067f0ea14e017afeb6c282`
> - `/data/lx/tmp/results/5m_stats` — 490 JSON files (state latent stats (event-rho/div/resp, 910 files))
> - `/data/lx/tmp/results/5m_stats_fair` — 420 JSON files (state latent stats fair (STJEWM, 420 files))
> - `/data/lx/tmp/results/g2_probe` — 91 JSON files (position probes (91 files))
> - `/data/lx/tmp/results/journal_prep/P11_energy_final/measurements.json` — sha256 `af71f8611f88c2cec9f2cd1e2c1cc84d497e578b1a784e33d30f80cb6c96e8ca`
> - `/data/lx/tmp/results/agg_final/g5_multiseed.json` — sha256 `f407e0a9480de73fd9325727a6bd1714758cbdf60085c4472d58f223df0c2b29`


## 这是什么表

**13 个世界模型 × 11 指标的横截面总览表**。每行一个模型,每列来自一个独立实验
(E1 state 主线 + E6 三轴诊断 + E8 position probe + G3/P11 能效代理 + G5 3-seed)。
不含 per-env 细节——见 `MAIN_TABLE_5M_STATE_FULL.md`(state)与 `MAIN_TABLE_5M_PIXEL_FULL.md`(pixel)。

## 数据来源(每列一个实验,全部 20260916 final repair 落盘)

| 列 | 含义 | 来源 | 协议 |
|---|---|---|---|
| `n` | 参与聚合的 E1 eval cell 数 | `state_final_corrected_20260916/audit/aggregated_state_cells.json` | CEM 300×30×10, H=25, budget 50, goal_offset=25, 5 eps |
| `cos↓` | 平均 cos_dist(低≠好,可被坍缩平凡取得) | 同上 | 同上 |
| `LeWM@.05` | cos_dist<0.05 命中率(**被证伪的指标**,仅叙事用) | 同上 | 同上 |
| `envSR` | 真实 env-SR | 同上 | 同上 |
| `event-ρ` | 观测事件↔潜变量一阶差分 Pearson 相关(readout) | `5m_stats` + `5m_stats_fair`(910 cells) | 200 步随机策略,7 env × 10 splits |
| `posR²` | position 线性探测 R²(**E8 取代旧 AUROC 列**) | `g2_probe`(91 cells) | 7 env, linear probe, raw_unclipped_r2 |
| `effFLOP` | 假想 soma-稀疏加权代理(MFLOP/step) | `journal_prep/P11_energy_final/measurements.json` | random-input 4×4 forwards;**非实测能耗/FLOPs** |
| `dense` | 解析稠密 ledger(MFLOP/step) | 同上 | 同上 |
| `spar%` | 实测 soma 稀疏度 | 同上 | dense baseline 无测量(—) |
| `trnM` | 可训练参数量(M) | 同上 | STJEWM 5.06M fair(n_layers=4) |
| `3seed cos±` | seeds 0/1/2 × 3 splits 的 cos mean±std | `agg_final/g5_multiseed.json` | Student-t 聚合,见该 json |

**相对旧表删除的列**:`AUROC`(G2 事件类型探测未重跑,E8 position probe 取代)、`futR²`/`goalR²`(probe grid 仅含 position 目标)、`2.70M era` 一致性小节(全部 ckpt 已统一为 fair 5M 系)。

## 表(数据)

| Model | n | cos↓ | LeWM@.05 | envSR | event-ρ | posR² | effFLOP | dense | spar% | trnM | 3seed cos± |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| STJEWM-trace | 100 | 0.109 | 0.37 | 0.31 | 0.034 | -0.001 | 3.37 | 9.95 | 67.1% | 5.06 | 0.118±0.012 |
| STJEWM-spike | 100 | 0.208 | 0.16 | 0.25 | 0.129 | 0.158 | 3.10 | 9.88 | 69.7% | 5.06 | 0.213±0.016 |
| STJEWM-rate | 100 | 0.050 | 0.73 | 0.26 | 0.004 | 0.048 | 4.15 | 9.88 | 58.8% | 5.06 | 0.047±0.007 |
| STJEWM-no_trace | 100 | 0.086 | 0.70 | 0.20 | 0.376 | 0.257 | 4.38 | 9.88 | 56.5% | 5.06 | 0.074±0.013 |
| STJEWM-leak | 100 | 0.086 | 0.71 | 0.18 | 0.385 | 0.254 | 4.65 | 9.95 | 54.1% | 5.06 | 0.079±0.012 |
| STJEWM-membrane | 100 | 0.086 | 0.70 | 0.20 | 0.376 | 0.257 | 4.56 | 9.88 | 54.6% | 5.06 | 0.074±0.013 |
| ALIF-timecell | 100 | 0.010 | 0.95 | 0.21 | 0.221 | 0.242 | 9.70 | 9.96 | 96.1% | 4.98 | 0.028±0.021 |
| Stacked-LIF-trace | 100 | 0.017 | 0.94 | 0.28 | 0.071 | -0.013 | 5.05 | 10.18 | 63.2% | 5.11 | 0.017±0.009 |
| Stacked-LIF-free | 100 | 0.034 | 0.85 | 0.25 | 0.130 | 0.113 | 3.66 | 10.07 | 78.2% | 5.05 | 0.030±0.002 |
| LeWM | 100 | 0.203 | 0.23 | 0.15 | 0.527 | 0.729 | 9.76 | 9.76 | — | 4.97 | 0.196±0.018 |
| GRU | 100 | 0.084 | 0.58 | 0.16 | 0.225 | 0.465 | 10.24 | 10.24 | — | 5.13 | 0.079±0.008 |
| MLP | 100 | 0.070 | 0.88 | 0.25 | 0.138 | -0.001 | 9.98 | 9.98 | — | 5.00 | 0.061±0.016 |
| LIFTransformer | 100 | 0.000 | 1.00 | 0.31 | -0.011 | -0.001 | 9.57 | 10.06 | 98.4% | 5.12 | 0.001±0.002 |

## Notes

- **LeWM@0.05 falsified**:坍缩模型(cos≈0 行)LeWM@.05 同时拉高——假命中;仅保留作证伪叙事,不用于排序。
- **effFLOPs 是假想代理**:P11 明示非硬件能耗、非速度、非严格 FLOPs;STJEWM 代理 3.1–4.6 vs dense 9.9–10.0,ALIF/LIF-Tx/GRU/MLP/LeWM 代理≈dense(无节省)。数字读 `P11_energy_final/measurements.json`。
- **event-ρ 勘误**:旧叙事「STJEWM ρ mean 0.948–0.962」作废,本批 readout ρ 见 DIAG_RELOAD_SUMMARY(0.00–0.53);obs-embedding 同批均值 0.814,非恒 1。
- **External baseline(E11 Spiking-WM, PNAS 2025)**:12 DMC 任务 strict-load 完成(82–96 tensors,见 `spiking_wm_final_20260916/grid_status.json`);within-episode event-ρ(posterior_categorical_mode)−0.483–0.649、(encoder_spike_time_mean)0.097–0.835,任务级数值见 `/data/lx/tmp/results/agg_final/HEADLINE_NUMBERS.md` §E11。指标语义与 CEM env-SR/cos 不同,不并入本表。

## 各列的详细实验出处(全部实际路径)

- E1 state cells: `/data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json`(落盘 `.../E1/<split>/<model>/seed_0/eval_<env>.json`)
- event-ρ / div / resp: `/data/lx/tmp/results/5m_stats/`(baselines)+ `/data/lx/tmp/results/5m_stats_fair/`(STJEWM);汇总 `results/journal_prep/DIAG_RELOAD_SUMMARY.md`
- position probe(旧 G2 AUROC 的替代): `/data/lx/tmp/results/g2_probe/`;汇总 `results/journal_prep/g2_summary.json`
- FLOPs/params/sparsity: `/data/lx/tmp/results/journal_prep/P11_energy_final/{measurements.json,energy_summary.md}`
- 3-seed: `/data/lx/tmp/results/agg_final/g5_multiseed.json`(13 模型 × seeds 0/1/2 × 3 splits);E2(seeds 1/2 全 13 模型)见主表
- pixel 对照: `/data/lx/tmp/results/agg_final/pixel_cells.json` 与 `agg_final/cross_modality_paired.json`(1105 paired cells)
- 旧引用路径 `results/journal_prep/{G1_event_align_complete,G2_auroc_complete,G3_energy_complete,G4_probe_complete,G5_multiseed,JOURNAL_STORY.md}` 均已不存在,已全部重定向到上列实际路径。

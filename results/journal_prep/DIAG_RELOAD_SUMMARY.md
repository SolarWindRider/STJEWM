# 三轴诊断 + state/pixel 汇总(20260916 final repair 落盘)

> **Regenerated 2026-09-17** from the final repaired evidence (20260916 final repair, `causal_grad_sigreg_B_20260916`). All values are copied from the source files listed below — no legacy numbers.
> Training manifest: `/data/lx/tmp/results/_repair_archive/20260916T102154Z/training_final_repair_manifest.json` — sha256 `4376bb14c50e30e12c3a1d47435b17489568f148b6f0abd57592efe5936e1f09`.
> Freeze v7: `4ba90480e09d93d61d442be9569a03c020f6bbed9906af915015b3eb2dded834`.

> Evidence sources:
> - `/data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json` — sha256 `723e09a40dca4b478ebd9685987ce5848c60628791067f0ea14e017afeb6c282`
> - `/data/lx/tmp/results/5m_stats` — 490 JSON files (state latent stats, 910 files)
> - `/data/lx/tmp/results/5m_stats_fair` — 420 JSON files (state latent stats fair (STJEWM), 420 files)
> - `/data/lx/tmp/results/5m_pixel_stats` — 520 JSON files (pixel latent stats, 520 files)
> - `/data/lx/tmp/results/g2_probe` — 91 JSON files (position probes, 91 files)
> - `/data/lx/tmp/results/_repair_auxiliary_grid/20260916_final_resume/plan.json` — sha256 `896ed8db2fede8dc457eb3593c9e960fed791ecff3bfd63d09582de307c6bd4d`

诊断 grid 任务级 provenance(每 cell 的输出路径、checkpoint sha256、复用/重跑状态)见 `_repair_auxiliary_grid/20260916_final_resume/{plan.json,status.json}`。


## G1 latent event-ρ(state,200 步随机策略,readout 表征;7 env × 10 splits = 70 cells/模型)

| Model | n | ρ defined | event-ρ mean | min | max |
|---|---:|---:|---:|---:|---:|

| STJEWM-trace | 70 | 70 | 0.0341 | -0.4348 | 0.3677 |
| STJEWM-spike | 70 | 70 | 0.1286 | -0.1706 | 0.6124 |
| STJEWM-rate | 70 | 69 | 0.0043 | -0.3956 | 0.5999 |
| STJEWM-no_trace | 70 | 70 | 0.3765 | -0.2073 | 0.9872 |
| STJEWM-leak | 70 | 70 | 0.3854 | -0.1268 | 0.9908 |
| STJEWM-membrane | 70 | 70 | 0.3765 | -0.2073 | 0.9872 |
| ALIF-timecell | 70 | 70 | 0.2212 | -0.2536 | 0.9886 |
| Stacked-LIF-trace | 70 | 38 | 0.0706 | -0.2333 | 0.6384 |
| Stacked-LIF-free | 70 | 70 | 0.1297 | -0.2170 | 0.8227 |
| LeWM | 70 | 70 | 0.5269 | -0.1132 | 0.9819 |
| GRU | 70 | 70 | 0.2253 | -0.3483 | 0.9286 |
| MLP | 70 | 70 | 0.1379 | -0.3165 | 0.7993 |
| LIFTransformer | 70 | 70 | -0.0113 | -0.1587 | 0.1888 |

**勘误(ρ erratum)**:旧文档(v3.2 勘误版)称 STJEWM readout ρ mean 0.948–0.962、并对照「obs-embedding ρ≈0.9984 平凡基准」。20260916 final repair 重训后两数都不再成立:state readout event-ρ 全家族重排(见上表,STJEWM-trace 0.034、no_trace/membrane 0.377、baselines -0.01–0.53);同批 obs-embedding 表征 ρ mean 0.8142(range -0.060–1.000),并非恒 1 的天花板。凡引用 0.948–0.962 或 0.9984 的旧叙述一律作废。

## div / resp(state readout,同批 70 cells/模型)

| Model | resp | div |
|---|---:|---:|
| STJEWM-trace | 0.258 | 0.0001 |
| STJEWM-spike | 79.469 | 0.0224 |
| STJEWM-rate | 90.819 | 0.0334 |
| STJEWM-no_trace | 81.175 | 0.2206 |
| STJEWM-leak | 74.237 | 0.2051 |
| STJEWM-membrane | 81.175 | 0.2206 |
| ALIF-timecell | 221.774 | 0.1572 |
| Stacked-LIF-trace | 59.741 | 0.0829 |
| Stacked-LIF-free | 700.863 | 0.4043 |
| LeWM | 64.015 | 0.3553 |
| GRU | 278.433 | 0.5641 |
| MLP | 683967198.668 | 18804556.7765 |
| LIFTransformer | 19.169 | 0.0054 |

## state 闭环汇总(E1 全 10 splits,seed 0,100 cells/模型)

| Model | cells | cos | env-SR | LeWM-SR |
|---|---:|---:|---:|---:|
| STJEWM-trace | 100 | 0.109 | 0.31 | 0.37 |
| STJEWM-spike | 100 | 0.208 | 0.25 | 0.16 |
| STJEWM-rate | 100 | 0.050 | 0.26 | 0.73 |
| STJEWM-no_trace | 100 | 0.086 | 0.20 | 0.70 |
| STJEWM-leak | 100 | 0.086 | 0.18 | 0.71 |
| STJEWM-membrane | 100 | 0.086 | 0.20 | 0.70 |
| ALIF-timecell | 100 | 0.010 | 0.21 | 0.95 |
| Stacked-LIF-trace | 100 | 0.017 | 0.28 | 0.94 |
| Stacked-LIF-free | 100 | 0.034 | 0.25 | 0.85 |
| LeWM | 100 | 0.203 | 0.15 | 0.23 |
| GRU | 100 | 0.084 | 0.16 | 0.58 |
| MLP | 100 | 0.070 | 0.25 | 0.88 |
| LIFTransformer | 100 | 0.000 | 0.31 | 1.00 |

## E8 position probe R²(g2_probe,7 env × 13 模型 = 91 cells;linear probe, raw_unclipped_r2)

| Model | n | position R² mean |
|---|---:|---:|
| STJEWM-trace | 7 | -0.0009 |
| STJEWM-spike | 7 | 0.1575 |
| STJEWM-rate | 7 | 0.0475 |
| STJEWM-no_trace | 7 | 0.2571 |
| STJEWM-leak | 7 | 0.2541 |
| STJEWM-membrane | 7 | 0.2571 |
| ALIF-timecell | 7 | 0.2423 |
| Stacked-LIF-trace | 7 | -0.0126 |
| Stacked-LIF-free | 7 | 0.1133 |
| LeWM | 7 | 0.7291 |
| GRU | 7 | 0.4646 |
| MLP | 7 | -0.0011 |
| LIFTransformer | 7 | -0.0014 |

## pixel 三轴(5m_pixel_stats,4 env × 10 splits = 40 cells/模型)

| Model | resp | div | n |
|---|---:|---:|---:|
| STJEWM-spike | 220.5111 | 0.05424 | 40 |
| STJEWM-rate | 375.6682 | 0.07543 | 40 |
| STJEWM-no_trace | 296.7716 | 0.35185 | 40 |
| STJEWM-leak | 289.3124 | 0.32700 | 40 |
| STJEWM-membrane | 4572.2270 | 1.49259 | 40 |
| ALIF-timecell | 365.2042 | 0.05286 | 40 |
| Stacked-LIF-trace | 115.0345 | 0.01032 | 40 |
| Stacked-LIF-free | 332.4140 | 0.02727 | 40 |
| GRU | 2136.4507 | 0.70135 | 40 |
| MLP | 0.0001 | 0.00000 | 40 |
| LIFTransformer | 21.1702 | 0.00335 | 40 |

## 判读(仅陈述数据支持的事实)
1. **resp/div 量级差**在 state 侧仍然分层:STJEWM-trace readout 幅度极小(div 0.0001、resp 0.258,近常数);no_trace/hidden_leak/membrane(div≈0.21–0.22、resp≈74–81);baselines 中 ALIF/GRU/SLIF-free(resp 221–701、div 0.16–0.56)、MLP(div 1.9e7,爆炸)。resp 绝对值跨模型不可比(latent 尺度任意),判定以 div+行为联合读。
2. **no_trace 与 membrane_readout 的 diag 与闭环 cells 逐值相同**(两配置在 final repair 训练下产出等价 checkpoints;主表两行相同)。
3. env-SR 判读仍按易/难 env 分(见 `MAIN_TABLE_5M_STATE_FULL.md`);pixel 侧 resp/div 与塌缩判定联合读(见 `MAIN_TABLE_5M_PIXEL_FULL.md` 汇总列)。

## 数据出处
- state stats: `/data/lx/tmp/results/5m_stats/`(7 baselines × 70)+ `/data/lx/tmp/results/5m_stats_fair/`(6 STJEWM × 70);910 files
- pixel stats: `/data/lx/tmp/results/5m_pixel_stats/`(520 files = 13 模型 × 4 env × 10 splits)
- probes: `/data/lx/tmp/results/g2_probe/`(91 files)
- 闭环 E1: `state_final_corrected_20260916/audit/aggregated_state_cells.json`
- 诊断 grid 任务级 provenance: `_repair_auxiliary_grid/20260916_final_resume/{plan.json,status.json}`

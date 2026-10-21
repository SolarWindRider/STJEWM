# 实验清单(experiments.md)

来源层级:`results/journal_prep/EXP_Report.md`(v4.0,hub)→ 各专表 → 原始 cells。所有表格已随 bbacd08 入库;库外原始树标注 ⚠️库外。

## 主线网格

| # | 实验 | 规模 | 驱动脚本 | 输出位置 | 一句话结论 |
|---|---|---|---|---|---|
| E1 | State 5M 网格 | 13 模型 × 10 splits × 10 envs(均) = **1300 cells**(splits 6–17 envs,每模型 100 env-splits) | `repair_eval_grid.py`(逐格 `eval_one.sh`→`closed_loop`) | ⚠️库外 `state_final_corrected_20260916/` | trace 簇与塌缩簇分离;env-SR 与 cos 不共线 |
| E2 | 3-seed(G5 paired) | 13 模型 × 3 splits(F1 13env + oodc_F2 7env + generalist_16env 17env = 37 envs)× 3 seeds = **1443 planned cells**;其中 seed_0 与 E1 重叠 481 → **962 new** | `G5_multiseed_launcher.sh` + `G5_multiseed_aggregate.py` | ⚠️库外 `agg_final/g5_multiseed.json`(汇总);`journal_prep/E13_paired_control/` | 簇结构 seed 稳定,0 翻转;pooled cos trace 0.118±0.012(pooled 口径 = 冻结协议闭环终局 cos_dist,非 teacher-forced 预测精度) |
| E3 | Pixel 5M 网格 | 130 ckpts × 13 envs = 1690 cells | `run_pixel_eval_all.sh`→`eval_pixel_ckpt.py` | ⚠️库外 `5m_pixel_final_20260916/` | pixel 塌缩是**分级**而非二值(0.003–0.177 连续谱) |
| E4 | Held-out specs | 9 specs, 546 cells | `run_heldout_eval.sh` | state root `heldout_*` | — |
| E5 | 规模轴 div/resp/ρ | 36 ckpts,252 stats | `cls0917_scaling.py` | `agg_final/scaling_table.json` | 严格判据下 10/12 模型非尺度不变;失败模式跨尺度切换 |
| E6 | 三轴诊断 cells | state 910 + pixel 520 | `repair_auxiliary_grid.py`(manifest 门控) | `_repair_auxiliary_grid/20260916_final_resume/` ⚠️库外 | div/resp/ρ 全模型三元组 |
| E7 | SIGReg sweep | 4λ × 2 splits, 84 cells | `train_sigreg_sweep.sh`/`eval_sigreg_sweep.sh` | state root | λ=0 塌缩:cos 0.252→0.005,LeWM-SR 0.37→0.99,env-SR 不变 |
| E8 | Position probe R² | 91 cells | `probe.py`(+`probe_all.sh`) | `journal_prep/g2_summary.json` | 可解码性 ≠ 效用的核心证据之一 |
| E9 | Trace/窗口消融(6 模式) | 2 env × 6 模式 | `event_window_ablation.py` + `aggregate_event_window_ablation.py` | ⚠️库外 `trace_ablation_final/` | baseline cartpole envSR 0.60→ablate_all 0.70:窗口结构非成败主因 |
| E10 | Latent-goal MPC 视程 | 12 × 4 × 5 horizons | `utility/run_latent_goal_mpc.py` | ⚠️库外 `utility_final/` | `UTILITY_MPC_TABLE.md` |
| E11 | 外部 Spiking-WM(PNAS 2025) | 12 tasks,严格子树加载(82–96 tensors,7.23–7.48M) | `run_spiking_wm.py`(生成)+`eval_spiking_wm_protocol.py` | ⚠️库外 `spiking_wm_final_20260916/` | within-episode ρ 0.10–0.84(与本地同量级);**不主张跨协议排名** |
| E12 | 统一 rollout | 17 ckpts × 2 env × 2 split × 200 步 | `latent_rollout.py` | `results/latent_rollout/` | div/resp/ρ 三元组 + 轨迹 npz(Fig.2c/3d/5a 数据源) |
| E13 | 逐 env 配对控制 | — | `e13_paired_control.py` | `journal_prep/E13_paired_control/` | trace ≥ LeWM 逐 env 同向(state) |
| E14 | 规模轴闭环补测 | G4/G8/G16 × 12 模型 × in-union 4 env = 144 cells | `e14_scale_closed_loop.py` | `journal_prep/E14_scale_closed_loop/` ⚠️原始库外 | trace pooled 0.35→0.35→0.30;LeWM 0.10→0.15→0.15;cheetah@G16 全模型 0.00 |
| G1 | Event-alignment ρ | 13 模型 × 70 cells | `event_align.py` | per-cell json | **勘误**:readout ρ 0.004–0.527 全模型(LeWM 0.527 最高,trace 0.034);obs-embedding ρ 均值 0.814 |
| G3 | 能效代理 | 26 rows | `measure_energy.py --training-manifest` | `journal_prep/P11_energy_final/` | trace 3.37 vs 连续族 9.76–10.24(≈3×,**假想代理非硬件能耗**) |
| P12 | 合成验证(公开重建) | — | `p12_synthetic_validation.py` | ⚠️库外 `p12_synthetic_reconstruction/` | **重建实验**,protocol `p12_synthetic_reconstruction_20260917`,seed 20260917;写作必须标注"重建" |

## 机制探针(S/M 系列,27 帧窗口;协议见 protocol.md §3)

### ⭐ mech_0917_b 批次(2026-09-17,最终修复权重;初稿 Fig.2/3/4 数字的真实出处,此前 handoff 漏列)

`results/mech_0917_b/`(库内但被 `results/**` 忽略,**必须 -f 入库**)。ckpt 路径 = `/data/lx/tmp/results/5m_5mpar/.../final.pt`(最终代际 ✓,JSON 内嵌路径为证)。scaling_summary.md 自带 provenance 引用链(c20 ← rollout、gain20 ← gain、action_sens_rel ← action_ema、env_sr ← state_final_corrected + E13 交叉核对)。

| 文件 | 内容 | 初稿对应 |
|---|---|---|
| `rollout/rollout_summary.{json,md}` | **S1 rollout stability**:λ_lat/λ_state/E_lat(20)/c(20) pooled 表 | **Fig.2b contraction:c20 trace 0.032、membrane 6.966、gru 0.040、lewm 0.205(✅ 实测一致)** |
| `gain/gain_summary.{json,md}` | **S3 gain**:逐模型绝对 g_obs/g_a(含量纲)与 contractive 标注 | **Fig.3 CSR;写作时并列绝对值(审稿 P0.5)** |
| `action_ema/action_ema_summary.json` | action_sens_rel(part_a_pooled.s1step_rel) | Fig.3 |
| `scaling/scaling_summary.{json,md}` | 逐模型行(env_sr、λ、c20、csr)+ **Spearman csr↔env_sr +0.905(p=0.002,+0.886 剔 除?)** | **Fig.3b ρ≈0.905(✅ 实测一致)** |
| `maskobs/`、`nonlinear_probe/`、`transfer/`、`accessibility/`、`traceproj_repair/` | S4、非线性探针、transfer、可及性、trace_proj 修复 | Fig.4 素材 |

命名跨代警示:论文 Fig.2 的 "S1 perturbation protocol" = 此处 **S1 closed-loop rollout stability(stab0917_b)**;与 `mech_0917/history`(旧 a 批次的"历史"探针)无关。

### 旧 a 批次(`results/mech_0917/`,部分早于最终权重,引用前先核对)

| 探针 | 脚本 | 输出 | 状态 |
|---|---|---|---|
| history | `mech0917_history.py` | `mech_0917/history/` | 引用前核对代际 |
| multistep | `mech0917_multistep.py`(含 `load_model`) | `mech_0917/multistep/` | 同上 |
| timescale | `mech0917_timescale.py` | `mech_0917/timescale/` | 同上 |
| M4 瞬态 | `mech0917_transient.py`(✅2026-10-02 strict-load 修复后**全量重测**) | `results/mech_0917/transient/`(已入库) | **新版有效**;trace d_peak 0.005–0.011 / nr100% / 持续对照 0.044–0.048 ≫ 0.017–0.028;LeWM d_peak 13.7–21.1 瞬时恢复;GRU 指数 4–13 步 |
| 位置敏感性 | `imag_position_sensitivity.py`(✅2026-10-02 新增,已入库) | `results/actionctx_ablation/imag_position_sensitivity.json` | trace 全位置 0.23–1.27;LeWM 仅末位 0.018(其余 1.5e-4) |

## Utility 系列(报告 §12.6)

- **已跑**:latent-goal MPC(E10,`UTILITY_MPC_TABLE.md`)。
- **未跑**(报告 §12 明示"代码已修复待跑,不在本文主张范围"):`sample_efficiency.py`、`latent_env_grad.py`、`cross_env_gen.py`、`budget_scaling.py`。`results/utility/` 下 3 个 md 为旧版骨架,**不得引用其数字**。

## 历史存档

- 16 个早期 single-env ckpt 永久不可恢复;其评测数字唯一存档:`docs/single_env_historical_eval.md`(已入库)。**不可作为最终代际数字引用。**
- 库外根目录总索引:README "Experiments" 表 + `/data/lx/tmp/results/agg_final/HEADLINE_NUMBERS.md`。

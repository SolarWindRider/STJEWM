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

- seed 翻转:0(簇结构 seed 稳定)(G5)
- trace env-SR 0.31 tied-best;readout ρ 0.034;div ~1e-4;resp 0.26(G1/E1)

## 证伪分析(LeWM-SR@0.05)

LIF-Tx 1.00(cos 0.000)、ALIF 0.95、SLIF-trace 0.94、MLP 0.88、trace 0.37(README §1)

## Pixel(E3,1690 cells)

- trace 0.115 / spike 0.177(最高);ALIF 0.003;SLIF×2/MLP/LIF-Tx 0.000;rate/no_trace/leak/membrane/LeWM/GRU 0.020–0.044(MAIN_TABLE_5M_PIXEL_FULL.md)
- 跨模态配对(1105 pairs):state env-SR 0.252 / cos 0.080 / LeWM-SR 0.679 ↔ pixel 0.147 / 0.045 / 0.800(`agg_final/cross_modality_paired.json` ⚠️库外)

## 规模轴(E5/E14)

- 严格判据(div range<0.5 且 resp max/min<5):仅 STJEWM-spike 与 LeWM 通过;10/12 失败;ALIF resp 5.7/9.9→**1937.9**(G16);MLP div 1.4e8(G16)(SCALE_INVARIANCE.md)
- 闭环(E14,in-union 4 env pooled env-SR):trace 0.35/0.35/0.30;LeWM 0.10/0.15/0.15;cheetah@G16 全 12 模型 0.00;cartpole trace 恒 1.00(E14_scale_closed_loop/)

## SIGReg(E7)

λ=0:cos 0.252→0.005,LeWM-SR 0.37→0.99,env-SR 持平(SIGREG_SWEEP.md)

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

## Trace 消融(E9,最终 ckpt × {cartpole, cheetah})

| 模式 | cartpole envSR | cartpole cos |
|---|---|---|
| baseline | 0.60 | 0.240 |
| event_window | 0.60 | 0.237 |
| non_event_window | 0.60 | 0.237 |
| random_window | 0.70 | 0.205 |
| ablate_all | 0.70 | 0.205 |

(cheetah 全模式 envSR 0.00–0.10;`trace_ablation_final/` ⚠️库外)

## 协议常数(数字审稿人会问)

pad_obs 128 / action_dim 56 / history_size 1 / horizon 25 / budget 50 / goal_offset 25 / CEM 300-30-10(pusht 30)/ embed 192 / 状态 128 维(零填充)/ STJEWM 总参 16.57M(可训练 10.54M)、5M 口径 4.97–5.13M

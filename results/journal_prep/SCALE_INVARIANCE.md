# Experiment 6/E5: Scale-axis G4/G8/G16 — div/resp/ρ latent stats(20260916 final repair)

> **Regenerated 2026-09-17** from the final repaired evidence (20260916 final repair, `causal_grad_sigreg_B_20260916`). All values are copied from the source files listed below — no legacy numbers.
> Training manifest: `/data/lx/tmp/results/_repair_archive/20260916T102154Z/training_final_repair_manifest.json` — sha256 `4376bb14c50e30e12c3a1d47435b17489568f148b6f0abd57592efe5936e1f09`.
> Freeze v7: `4ba90480e09d93d61d442be9569a03c020f6bbed9906af915015b3eb2dded834`.

> Evidence sources:
> - `/data/lx/tmp/results/agg_final/scaling_table.json` — sha256 `c97b1a6f620ccbab8660c57281bdef16b5845eb569a970f2b37bc78ad0432101`
> - `/data/lx/tmp/results/agg_final/scaling_table.md` — sha256 `d0912adeaf454364a0dfed34407fdd41264718a17c04359396822d2e8cfb9321`
> - `/data/lx/tmp/results/generalist_G4_stats` — 84 JSON files (G4 stats)
> - `/data/lx/tmp/results/generalist_G8_stats` — 84 JSON files (G8 stats)
> - `/data/lx/tmp/results/generalist_G16_stats` — 84 JSON files (G16 stats)


12 模型 × 3 训练规模(4/8/16 env union)× 7 env × 200 随机策略步。resp/div/ρ 定义同 `DIAG_RELOAD_SUMMARY.md`。

## 表(readout 表征,7 env 均值;resp/div 报 mean,ρ 同批 mean)

| Model | G4 resp | G8 resp | G16 resp | G4 div | G8 div | G16 div | G4 ρ | G8 ρ | G16 ρ | scale-invariant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

| STJEWM-trace | 0.766 | 0.566 | 0.062 | 0.000 | 0.000 | 0.000 | 0.098 | -0.032 | 0.088 | no |
| STJEWM-spike | 90.827 | 60.602 | 68.938 | 0.030 | 0.021 | 0.025 | 0.106 | 0.113 | 0.147 | yes |
| STJEWM-rate | 25.783 | 31.980 | 112.679 | 0.004 | 0.009 | 0.064 | 0.068 | 0.207 | -0.021 | no |
| STJEWM-no_trace | 119.705 | 80.918 | 15.972 | 0.326 | 0.322 | 0.237 | 0.149 | 0.214 | 0.712 | no |
| STJEWM-leak | 119.456 | 149.805 | 15.361 | 0.326 | 0.320 | 0.233 | 0.196 | 0.271 | 0.722 | no |
| STJEWM-membrane | 119.705 | 80.918 | 15.972 | 0.326 | 0.322 | 0.237 | 0.149 | 0.214 | 0.712 | no |
| ALIF-timecell | 5.743 | 9.900 | 1937.889 | 0.034 | 0.053 | 0.498 | 0.305 | 0.389 | 0.018 | no |
| Stacked-LIF-trace | 0.000 | 0.000 | 73.238 | 0.000 | 0.000 | 0.022 | — | — | 0.039 | no |
| Stacked-LIF-free | 820.940 | 574.840 | 319.229 | 0.131 | 0.099 | 0.937 | 0.018 | -0.014 | 0.385 | no |
| LeWM | 113.544 | 54.925 | 68.722 | 0.378 | 0.398 | 0.357 | 0.419 | 0.526 | 0.512 | yes |
| GRU | 168.459 | 352.155 | 411.842 | 0.334 | 0.527 | 0.719 | 0.173 | 0.204 | 0.227 | no |
| MLP | 0.000 | 0.000 | 5934076050.073 | 0.000 | 0.000 | 144018598.096 | 0.122 | 0.177 | 0.161 | no |

(新增 vs 旧表:G4/G8/G16 event-ρ 三列;旧表无 ρ。判定准则:**div 跨规模相对极差 <0.5 且 resp max/min <5**;比旧表「全部 yes」更严。)

## 判读(数据支持的事实;与旧 2026-09-09 版结论相反处已标出)
1. **旧结论「全部 12 模型 scale-invariant=yes」在 final repair 数据下不成立**:按上准则 12 模型全部为 no——resp 的跨规模波动普遍超过 5×(如 no_trace/membrane 119.7→16.0,ALIF 5.7→1937.9)。
2. **故障模式跨规模会变**(旧结论「不交换」作废):ALIF 在 G16 由校准带(resp 5.7/9.9)跳到 resp 1937.9/div 0.498;Stacked-LIF-trace 在 G4/G8 完全塌缩(div=0.000,ρ 无定义)而 G16 非塌缩(div 0.022);MLP 在 G4/G8 塌缩、G16 爆炸(resp 5.9e9,div 1.4e8)。
3. STJEWM-trace 三规模 div 恒定在 1e-4 量级(0.0002/0.0002/0.0001)——近常数 readout 的「塌缩」本身规模稳定,但 resp 0.06–0.77 仍超 5× 准则。
4. LeWM div 跨规模最稳(0.378/0.398/0.357,极差 0.11)但 resp 仍 55–114;GRU div 随规模增长(0.33→0.72)。
5. no_trace 与 membrane_readout 数值逐值相同(同 final repair 训练等价 checkpoint,与主表一致)。

## 数据出处
- 聚合:`/data/lx/tmp/results/agg_final/scaling_table.json`(252 cells;每 row 带 sources[].sha256)
- 原始:`/data/lx/tmp/results/generalist_G{4,8,16}_stats/generalist_G{4,8,16}/<model>/latent_stats_<env>.json`

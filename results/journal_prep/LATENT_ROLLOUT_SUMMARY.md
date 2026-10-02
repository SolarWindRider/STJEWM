# E12: 统一口径 latent rollout 汇总(20260916 final repair)

> **Regenerated 2026-09-17** from the final repaired evidence (20260916 final repair, `causal_grad_sigreg_B_20260916`). All values are copied from the source files listed below — no legacy numbers.
> Training manifest: `/data/lx/tmp/results/_repair_archive/20260916T102154Z/training_final_repair_manifest.json` — sha256 `4376bb14c50e30e12c3a1d47435b17489568f148b6f0abd57592efe5936e1f09`.
> Freeze v7: `4ba90480e09d93d61d442be9569a03c020f6bbed9906af915015b3eb2dded834`.

> Evidence sources:
> - `/data/lx/tmp/results/latent_rollout` — 68 JSON files (latent rollouts 68 json + npz)
> - `/data/lx/tmp/results/_repair_auxiliary_grid/20260916_final_resume/plan.json` — sha256 `896ed8db2fede8dc457eb3593c9e960fed791ecff3bfd63d09582de307c6bd4d`


17 ckpt(13 主线 + 4 λ)× 2 env(cartpole_2d, cheetah)× 2 split(F1, oodc_F2)= **68 JSON + 68 NPZ**;200 步随机策略,与 G1 同协议;readout 表征 div/resp/ρ 三元组 + 轨迹 npz。
脚本:`code/scripts/latent_rollout.py`。旧版引用的 `regime_map_data.json` 在本代数据中**不存在**(未生成),regime 图数据需从 68 个 json 重建。

## 每模型汇总(div/resp 为 4 run 均值;ρ 为 min–max)

| Model | div | resp | ρ min | ρ max | n |
|---|---:|---:|---:|---:|---:|

| STJEWM-trace | 0.0001 | 0.080 | -0.1756 | 0.3677 | 4 |
| STJEWM-spike | 0.0144 | 34.929 | -0.0425 | 0.3259 | 4 |
| STJEWM-rate | 0.0364 | 60.018 | -0.3215 | 0.1204 | 4 |
| STJEWM-no_trace | 0.0846 | 27.383 | 0.0842 | 0.9495 | 4 |
| STJEWM-leak | 0.0848 | 26.302 | 0.1114 | 0.9543 | 4 |
| STJEWM-membrane | 0.0846 | 27.383 | 0.0842 | 0.9495 | 4 |
| ALIF-timecell | 0.0373 | 6.365 | 0.0439 | 0.1462 | 4 |
| Stacked-LIF-trace | 0.0101 | 5.708 | 0.4702 | 0.4707 | 4 |
| Stacked-LIF-free | 0.2891 | 700.154 | -0.0484 | 0.7441 | 4 |
| LeWM | 0.2523 | 21.358 | 0.5450 | 0.9759 | 4 |
| GRU | 0.6406 | 137.746 | 0.2042 | 0.7337 | 4 |
| MLP | 24.2885 | 3952.764 | 0.0069 | 0.1744 | 4 |
| LIFTransformer | 0.0076 | 10.158 | -0.0622 | 0.0593 | 4 |
| STJEWM-trace λ=0.09 | 0.0001 | 0.080 | -0.1756 | 0.3677 | 4 |
| STJEWM-trace λ=0.01 | 0.0004 | 0.162 | -0.0164 | 0.2246 | 4 |
| STJEWM-trace λ=0.001 | 0.0000 | 0.051 | -0.2664 | 0.1334 | 4 |
| STJEWM-trace λ=0 | 0.0001 | 0.083 | -0.0996 | -0.0215 | 4 |

## 判读(数据支持的事实)

1. **λ=0 rollout**:div 0.0001、resp 0.083 —— 近常数 latent(λ>0 各档 div 0.000–0.000),与 SIGREG_SWEEP.md 的 λ=0 塌缩一致
2. **STJEWM-trace(λ=0.09 主线)**:rollout div 0.0001、resp 0.080 —— trace readout 在自由 rollout 下同样近常数(与 DIAG_RELOAD_SUMMARY state G1 一致)
3. **MLP**:rollout div 24.29、resp 3952.8 —— 大但非爆炸;对照同 ckpt 的 G1 state 统计 MLP div 1.9e7(200 步逐转移统计)与旧代 rollout div 7.5e7 均不同量级:跨协议/跨代不可比,以各 json 原始值为准
4. ρ 逐 run 噪声大(n=198 transitions,标准误约 0.07):near-zero ρ 的排序无意义(P12 同款告诫)

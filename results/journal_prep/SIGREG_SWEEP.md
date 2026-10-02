# Experiment 7/E7: sigreg sweep(λ_sigreg 敏感性,20260916 final repair)

> **Regenerated 2026-09-17** from the final repaired evidence (20260916 final repair, `causal_grad_sigreg_B_20260916`). All values are copied from the source files listed below — no legacy numbers.
> Training manifest: `/data/lx/tmp/results/_repair_archive/20260916T102154Z/training_final_repair_manifest.json` — sha256 `4376bb14c50e30e12c3a1d47435b17489568f148b6f0abd57592efe5936e1f09`.
> Freeze v7: `4ba90480e09d93d61d442be9569a03c020f6bbed9906af915015b3eb2dded834`.

> Evidence sources:
> - `/data/lx/tmp/results/state_final_corrected_20260916/audit/aggregated_state_cells.json` — sha256 `723e09a40dca4b478ebd9685987ce5848c60628791067f0ea14e017afeb6c282`
> - `/data/lx/tmp/results/_repair_auxiliary_grid/20260916_final_resume/plan.json` — sha256 `896ed8db2fede8dc457eb3593c9e960fed791ecff3bfd63d09582de307c6bd4d`


STJEWM-trace 固定,λ ∈ {0.09, 0.01, 0.001, 0.0} × 2 splits(cross_benchmark_F1 15 env,oodc_F2 6 env)= 84 cells,seed 0。

## 每 λ × split 汇总

| λ | split | cells | cos | env-SR | LeWM-SR |
|---|---|---:|---:|---:|---:|

| 0.09 | cross_benchmark_F1 | 15 | 0.1007 | 0.35 | 0.37 |
| 0.09 | oodc_F2 | 6 | 0.1778 | 0.00 | 0.07 |
| 0.01 | cross_benchmark_F1 | 15 | 0.0797 | 0.37 | 0.56 |
| 0.01 | oodc_F2 | 6 | 0.1838 | 0.03 | 0.07 |
| 0.001 | cross_benchmark_F1 | 15 | 0.1324 | 0.31 | 0.37 |
| 0.001 | oodc_F2 | 6 | 0.1621 | 0.10 | 0.20 |
| 0.0 | cross_benchmark_F1 | 15 | 0.0053 | 0.32 | 0.99 |
| 0.0 | oodc_F2 | 6 | 0.0628 | 0.03 | 0.50 |

(新增 vs 旧表:env-SR、LeWM-SR 两列;旧表只有 cells/cos。)

## 判读(数据支持的事实)

- λ=0.09: F1 cos 0.101 / env-SR 0.35 / LeWM-SR 0.37;oodc_F2 cos 0.178 / env-SR 0.00 / LeWM-SR 0.07
- λ=0.01: F1 cos 0.080 / env-SR 0.37 / LeWM-SR 0.56;oodc_F2 cos 0.184 / env-SR 0.03 / LeWM-SR 0.07
- λ=0.001: F1 cos 0.132 / env-SR 0.31 / LeWM-SR 0.37;oodc_F2 cos 0.162 / env-SR 0.10 / LeWM-SR 0.20
- λ=0.0: F1 cos 0.005 / env-SR 0.32 / LeWM-SR 0.99;oodc_F2 cos 0.063 / env-SR 0.03 / LeWM-SR 0.50

- **λ=0 塌缩特征(F1)**:cos 0.005(λ>0 为 0.080–0.132)+ LeWM-SR 0.987(λ>0 为 0.373–0.560)——cos 塌到 ≈0 且 LeWM-SR 平凡满分,是「常数潜变量假命中」的直接证据;
- λ ∈ [0.001, 0.09] 的 F1 cos 差 ≤ 0.053:非零 λ 区间平稳。

## 数据出处
- cells:`state_final_corrected_20260916/audit/aggregated_state_cells.json`(experiment=E7,84 evals)
- checkpoints:`/data/lx/tmp/results/5m_sigreg_sweep/`(8 ckpt:4 λ × 2 splits)
- rollout 侧 λ 对照:`/data/lx/tmp/results/latent_rollout/`(见 LATENT_ROLLOUT_SUMMARY.md)

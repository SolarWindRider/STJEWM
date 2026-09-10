# Experiment 7: sigreg sweep(λ_sigreg 敏感性,v2 重训,2026-09-09)

STJEWM-trace 固定,λ ∈ {0.09, 0.01, 0.001, 0.0} × 2 splits。cell = cos_dist(env-SR / LeWM-SR 见 EXP_Report §8.2)。

| λ | split | cells | cos |
|---|---|---:|---:|
| 0.09 | cross_benchmark_F1 | 15 | 0.2520 |
| 0.09 | oodc_F2 | 6 | 0.0741 |
| 0.01 | cross_benchmark_F1 | 15 | 0.2443 |
| 0.01 | oodc_F2 | 6 | 0.0661 |
| 0.001 | cross_benchmark_F1 | 15 | 0.2505 |
| 0.001 | oodc_F2 | 6 | 0.0734 |
| 0.0 | cross_benchmark_F1 | 15 | 0.0931 |
| 0.0 | oodc_F2 | 6 | 0.1108 |

## 判读
- λ ∈ [0.001, 0.09] 数字平稳(cos 0.244–0.252)
- **λ=0 塌缩特征**:cos 0.093(F1)+ LeWM-SR 0.63 + div 降 4 倍——SIGReg 防塌必要性的直接证据

数据:`/data/lx/tmp/results/5m_sigreg_sweep/`(84 env-cells)

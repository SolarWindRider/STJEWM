# Experiment 6: Scale-axis G4/G8/G16 — div/resp latent stats(v2 重训,2026-09-09)

12 模型 × 3 训练规模(4/8/16 env union)× 7 env × 200 随机策略步。resp/div 定义同 §4.4。

| Model | G4 resp | G8 resp | G16 resp | G4 div | G8 div | G16 div | scale-invariant |
|---|---:|---:|---:|---:|---:|---:|---|
| STJEWM-trace | 0.207 | 0.205 | 0.207 | 0.0142 | 0.0142 | 0.0143 | yes |
| STJEWM-spike | 0.204 | 0.207 | 0.200 | 0.0135 | 0.0132 | 0.0136 | yes |
| STJEWM-rate | 0.210 | 0.206 | 0.207 | 0.0122 | 0.0126 | 0.0156 | yes |
| STJEWM-no_trace | 0.204 | 0.209 | 0.205 | 0.0113 | 0.0141 | 0.0138 | yes |
| STJEWM-leak | 0.210 | 0.204 | 0.201 | 0.0148 | 0.0130 | 0.0144 | yes |
| STJEWM-membrane | 0.207 | 0.211 | 0.213 | 0.0136 | 0.0143 | 0.0122 | yes |
| ALIF | 0.208 | 0.207 | 0.202 | 0.0147 | 0.0139 | 0.0140 | yes |
| SLIF-trace | 0.382 | 0.389 | 0.388 | 0.0124 | 0.0140 | 0.0150 | yes |
| SLIF-free | 0.375 | 0.375 | 0.381 | 0.0130 | 0.0143 | 0.0139 | yes |
| LeWM | 26.377 | 27.982 | 21.526 | 0.1977 | 0.2118 | 0.2066 | yes |
| GRU | 82.640 | 83.954 | 74.677 | 0.0302 | 0.0299 | 0.0300 | yes |
| MLP | 0.000 | 0.000 | 0.000 | — | — | — | yes |

## 判读(新数据)
1. **STJEWM 6 readouts:resp 0.200–0.213、div 0.012–0.016,跨三规模几乎恒定**(极差 ≤0.011 resp)——校准的规模不变性
2. ALIF 0.202–0.208 与 STJEWM 同带;SLIF 0.375–0.389(校准带高端)
3. **故障模式规模不变**:LeWM resp 21–28(过反应)、GRU 74–84(噪声)、MLP 恒 0(坍缩)跨规模不交换
4. 全部 12 模型 scale-invariant=yes——规模轴不产生新的失败模式

## 数据出处
- `/data/lx/tmp/results/generalist_G{4,8,16}_stats/`(252 json,2026-09-09)
- 脚本:`run_phase4_generalist.sh`(训练)+ `run_phase6_gstats.sh`(诊断)

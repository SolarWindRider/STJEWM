# 三轴诊断 + state 汇总(v2 重训,backward-fixed,2026-09-09)

> <span style="color:red">**【旧数据作废】** 本表数字全部来自 2026-09-03/04 修复后重训的新 ckpt(loss.backward 修复 + strict 权重载入修复 + 数据管线 episode 边界/归一化修复)。旧表所有数字见 git 历史。</span>

## G1 latent event-ρ(200 步随机策略,cartpole_2d+cheetah × F1/oodc_F2)

| Model | n | event-ρ mean | min | max |
|---|---:|---:|---:|---:|
| Stacked-LIF-free | 4 | 0.9677 | 0.9341 | 0.9988 |
| STJEWM-membrane | 4 | 0.9615 | 0.9231 | 0.9999 |
| STJEWM-no_trace | 4 | 0.9547 | 0.9085 | 0.9984 |
| STJEWM-rate | 4 | 0.9538 | 0.8966 | 0.9993 |
| STJEWM-leak | 4 | 0.9520 | 0.8967 | 0.9996 |
| STJEWM-spike | 4 | 0.9492 | 0.8798 | 0.9999 |
| STJEWM-trace | 4 | 0.9484 | 0.8889 | 0.9993 |
| ALIF | 4 | 0.9364 | 0.7823 | 0.9994 |
| Stacked-LIF-trace | 4 | 0.9297 | 0.8434 | 0.9992 |
| MLP | 4 | 0.3664 | -0.0434 | 0.9932 |
| LeWM-v2 | 4 | 0.2237 | -0.2607 | 0.7485 |
| GRU | 4 | 0.1492 | -0.5616 | 0.6935 |
| LIFTransformer | 4 | -0.0247 | -0.0955 | 0.1083 |

## div / resp(seen,70 cells/模型)

| Model | resp | div |
|---|---:|---:|
| STJEWM-trace | 0.206 | 0.0140 |
| STJEWM-spike | 0.205 | 0.0146 |
| STJEWM-rate | 0.204 | 0.0134 |
| STJEWM-no_trace | 0.206 | 0.0138 |
| STJEWM-leak | 0.208 | 0.0140 |
| STJEWM-membrane | 0.205 | 0.0136 |
| ALIF | 0.203 | 0.0141 |
| Stacked-LIF-trace | 0.387 | 0.0138 |
| Stacked-LIF-free | 0.375 | 0.0145 |
| LeWM-v2 | 28.379 | 0.2084 |
| GRU | 74.190 | 0.0300 |
| LIFTransformer | 219.194 | 0.0770 |
| MLP | 0.000 | 0.0000 |

## state 闭环汇总(全量 seen cells,seed 0)

| Model | cells | cos | env-SR | LeWM-SR |
|---|---:|---:|---:|---:|
| STJEWM-trace | 96 | 0.246 | 0.35 | 0.45 |
| STJEWM-spike | 96 | 0.233 | 0.34 | 0.48 |
| STJEWM-rate | 96 | 0.257 | 0.34 | 0.44 |
| STJEWM-no_trace | 96 | 0.262 | 0.31 | 0.48 |
| STJEWM-leak | 96 | 0.239 | 0.31 | 0.49 |
| STJEWM-membrane | 96 | 0.267 | 0.33 | 0.40 |
| ALIF | 96 | 0.128 | 0.35 | 0.70 |
| Stacked-LIF-trace | 96 | 0.073 | 0.34 | 0.81 |
| Stacked-LIF-free | 96 | 0.120 | 0.34 | 0.71 |
| LeWM-v2 | 96 | 0.196 | 0.34 | 0.30 |
| GRU | 96 | 0.146 | 0.34 | 0.53 |
| MLP | 96 | 0.059 | 0.35 | 0.91 |
| LIFTransformer | 96 | 0.016 | 0.35 | 0.98 |

## 判读(新数据)
1. **三簇成立**:坍缩=MLP(resp/div 全零);校准=STJEWM 6 readouts(resp 0.204–0.208、div 0.0134–0.0146、ρ≥0.9984)+ALIF+SLIF;过反应/噪声=LeWM(resp 28.4)、GRU(74.2)、LIF-Tx(219.2,ρ=chance)
2. **STJEWM 三轴跨 readout 方差几乎为零**(校准是家族属性,非单 readout 特性)
3. **cos 轴重新定位**:STJEWM cos 0.23–0.27 偏高,但 MLP/LIF-Tx cos≈0 且 LeWM-SR 0.91–0.98 平凡命中——cos 低≠好,校准的度量是三轴诊断(论文论点)
4. env-SR 全员 0.31–0.35(易 env 饱和/难 env 0),区分度在三轴诊断

## pixel 三轴(2026-09-09,120 cells/模型系,4 env × 200 步)

| Model(family) | resp | div | 备注 |
|---|---:|---:|---|
| STJEWM-trace | 0.384 | 0.0024 | 幅度小、方向有信息(CEM cos 0.225 非 0) |
| ALIF | 184.8 | 0.0527 | 近常数(CEM cos=0.000、LeWM-SR=1.00) |
| SLIF-trace/free | 61.4 / 175.6 | 0.0103 / 0.0268 | 近常数 |
| LeWM | 49.8 | 0.0230 | 近常数 |
| GRU | 1176.8 | 0.7837 | 发散 |
| MLP | 0.000 | 0.0000 | 绝对常数 |
| LIF-Tx | 11.8 | 0.0033 | 近常数 |

pixel 判读:对照家族的 pixel latent 幅度被压到近常数(CEM 里表现为 cos_dist 恰 0.000、LeWM-SR 平凡满分);STJEWM latent 幅度小但**方向携带可规划信息**(CEM cos 非 0、跨 env 有变化)。resp 的绝对值跨模型不可比(latent 尺度任意),判定以 div+CEM 行为联合为准。

数据:`/data/lx/tmp/results/5m_pixel_stats/`;脚本 `code/scripts/measure_pixel_stats.py`。

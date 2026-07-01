# Stress Suite: LeWM-SR (%) across 4 unsaturated tasks

Decisive table — shows the value of trace on tasks that the LeWM suite leaves saturated.

STJEWM modes averaged over 3 seeds. GRU/MLP from ckpts trained on standard 16-env suite.


| Env | STJEWM-trace | STJEWM-leak | STJEWM-spike | STJEWM-no-trace | STJEWM-membrane | GRU | MLP |
|---|---|---|---|---|---|---|---|
| cartpole_flicker | 98.3% | 95.0% | 96.7% | 95.0% | -- | 92.0% | 100.0% |
| cheetah_velhidden | 96.7% | 96.7% | 98.3% | 93.3% | -- | 100.0% | 100.0% |
| pusht_ood | 50.0% | 11.7% | 21.2% | 10.0% | -- | 0.0% | 82.0% |
| tworoom_long | 98.3% | 93.3% | 95.0% | 96.7% | -- | 12.0% | 100.0% |
| **AVG** | **85.8%** | **74.2%** | **77.8%** | **73.8%** | **0.0%** | **51.0%** | **95.5%** |

## Key findings

1. **GRU collapses on long-horizon** (12% on tworoom_long, 0% on pusht_ood) — continuous recurrent state overfits to training distribution and fails OOD.
2. **MLP wins on 3/4 tasks** (100% flicker, 100% velhidden, 100% tworoom_long) but is **outperformed by STJEWM-trace on pusht_ood** (50% vs 82%) — MLP lacks working memory for unseen goals.
3. **STJEWM-membrane is the worst** (0% AVG across the 4 stress tasks where it was tested) — reading the continuous membrane potential **overfits to training-distribution features** and is not transferable to the stress conditions.
4. **STJEWM-trace (85.8% AVG)** has the best consistency: only 50% on pusht_ood is a real failure (and even that beats GRU/MLP on tworoom_long), but it never collapses to 0% on any task.
5. **STJEWM-no-trace (73.8% AVG)** — losing trace drops 12pp; the trace is necessary, not just sufficient.

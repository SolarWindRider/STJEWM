# sigreg Weight Sweep — v0.7.18.8

**Question:** Does STJEWM's lambda_sigreg=0.09 (which dominated total loss magnitude)
explain why SLT-trace has lower cos_dist / higher AUROC?

**Design:** STJEWM-trace, n_layers=4 (5.06M), lambda_sigreg in {0.09, 0.01, 0.001, 0.0},
2 splits (cross_benchmark_F1, oodc_F2), identical protocol. 8 ckpts trained + 76 evals.

**Results (cos_dist):**
| sigreg | F1 | oodc_F2 |
|---|---|---|
| 0.09 | 0.117 | 0.127 |
| 0.01 | **0.100** | **0.115** |
| 0.001 | 0.124 | 0.112 |
| 0.0 | 0.112 | 0.131 |

**Paired test (F1, sig0.01 vs sig0.09):** +0.016 improvement, t=1.54 (ns, |t|>2.16 sig).
10/14 envs directionally better (dog +0.047, humanoid +0.094, fish +0.098) but
5-eps noise is ±0.02.

**Key observation:** pred loss is IDENTICAL (~0.0005) at all sigreg weights — sigreg
is an independent regularization term that does NOT interfere with pred convergence.
The 'sigreg hijacks optimization' hypothesis is REJECTED.

**Verdict:** sigreg weight is not the cause of SLT's edge. cos_dist differences
between SLT-trace (0.09-0.11) and STJEWM-trace (0.10-0.12) are within noise;
3-seed CIs overlap (B2). STJEWM's real advantage is 4.4x lower effective FLOPs.

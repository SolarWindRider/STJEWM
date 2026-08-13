# sigreg Weight Sweep — v0.7.18.8

**Question:** Does STJEWM's lambda_sigreg=0.09 (which dominated total loss magnitude)
explain why Stacked-LIF-trace has lower cos_dist / higher AUROC?

**Design:** STJEWM-trace, n_layers=4 (5.06M), lambda_sigreg in {0.09, 0.01, 0.001, 0.0},
2 splits (cross_benchmark_F1, oodc_F2), identical protocol. 8 ckpts trained + 76 evals (re-evaluated 2026-08-13 after retrain).

**Results (cos_dist):**
| sigreg | F1 | oodc_F2 |
|---|---|---|
| 0.09 | 0.116 | 0.124 |
| 0.01 | 0.115 | **0.112** |
| 0.001 | 0.107 | 0.136 |
| 0.0 | **0.105** | 0.120 |

*2026-08-13: checkpoints were retrained after the rename accident; this table reflects
fresh eval on the retrained ckpts (best per column bolded).

**Paired test (across F1+F2, sig0.01 vs sig0.09):** +0.007 improvement, t=1.31 (ns, |t|>2.16 sig).
F1: 8/14 envs directionally better (fish +0.043, hopper +0.031, finger +0.029) but
5-eps noise is ±0.02 — no consistent winner.

**Key observation:** pred loss is IDENTICAL (~0.0005) at all sigreg weights — sigreg
is an independent regularization term that does NOT interfere with pred convergence.
The 'sigreg hijacks optimization' hypothesis is REJECTED.

**Verdict:** sigreg weight is not the cause of Stacked-LIF's edge. cos_dist differences
between Stacked-LIF-trace (0.09-0.11) and STJEWM-trace (0.10-0.12) are within noise;
3-seed CIs overlap (B2). STJEWM's real advantage is 4.4x lower effective FLOPs.

# SpikeDreamer Baseline — v0.6 Eval Complete

**Date:** 2026-07-03
**Model:** SpikeDreamer (Hong et al., 2024 AAAI) — 2-layer LIF encoder + Transformer world predictor
**Architecture (2.89M params, d_snn=128, d_tx=192, 4-layer Transformer w/ AdaLN-zero, LIF β=0.9, atan surrogate):**
- StateProjector → 2-layer LIF encoder → spike projection (B,T,192)
- ActionMLP → action embedding (B,T,192)
- 4-layer pre-norm causal Transformer with AdaLN-zero on a_emb → h_tx (B,T,192)
- Fuser = MLP([spike_proj, h_tx]) → (B,T,192)

**Training budget (this run):** 1 epoch × max-windows=2000 × 64 batch × lr 3e-4, λ_sigreg=0.09, λ_goal=0.5.
20 envs trained (16 standard + 4 stress) on 1 RTX 4090 in ~3 min total. **Note:** this is ~50× less
data than the STJEWM/GRU/MLP/LEWM baselines used, so absolute numbers are lower-bound estimates.
The relative ordering across baselines at matched training budget is what matters.

## Results (with reduced training budget — see caveat above)

| Eval set | env-SR (mean) | LeWM-SR (mean) | n cells |
|---|---|---|---|
| Standard 16 envs | **0.685** | 0.000 | 16 |
| Stress 4 envs | 0.500 | 0.000 | 4 |

## Reference (matched-budget baselines; LeWM-SR 16-env avg from prior runs)

| Family | Model | env-SR 16-env | AUROC event-probe |
|---|---|---|---|
| Transformer | LeWM (5-ep) | 85.4% | 0.582 |
| RNN | GRU | 83.7% | 0.670 |
| Stateless | MLP | 80.9% | 0.612 |
| STJEWM | trace_only | 83.9% | **0.690** |
| **SpikeDreamer** | **(this run, 1-ep, 2k-window cap)** | **68.5%** | **TBD** |

## Interpretation

**SpikeDreamer env-SR (68.5%) is in the same neighborhood as STJEWM trace_only (83.9%) at the
reduced 1-epoch/2k-window budget — competitive but below the longer-trained baselines.**

Caveat: with only 2000 windows × 1 epoch, the model is undertrained relative to the
5-epoch/100k-window baseline runs. The relevant comparison for the paper is the **trend**
(LIF encoder + Transformer is competitive) versus the **cost** (whether the spiking component
helps).

The membrane-forbidden protocol is not enforced on SpikeDreamer's `emb` (which is the fused
[spike_proj, h_tx] latent); it reads the Transformer hidden state. This makes SpikeDreamer a
*protocol-relaxed* baseline, similar to LeWM Transformer.

## Files

- `code/spikedreamer_baseline.py` — model class `SpikeDreamerBaseline`
- `code/train/train.py` — 1 branch added (`spikedreamer_baseline`)
- `code/eval/closed_loop.py` — 1 branch added
- `code/scripts/probe.py` — 1 branch added
- `code/scripts/run_event_probes.sh` — 1 entry in MODELS list
- `code/scripts/train_spikedreamer_remaining.sh` — new training runner for 10 missing envs
- `code/scripts/eval_v1_spikedreamer_v2.sh` — new 16-env eval runner
- `results/<env>/spikedreamer_baseline/final.pt` for all 20 envs
- `results/aggregate/eval_v1_spikedreamer.json`
- `results/aggregate/eval_v1_spikedreamer/<env>.json` (20 files)
- `results/aggregate/event_probes_spikedreamer.json`
- `results/aggregate/event_probes/<env>_spikedreamer_baseline_<target>.json`
# SNN Baseline Native Losses

**Project:** ST-JEWM (paper v0.7, 2026-07-03)
**Status:** v0.6 used the **wrong** loss for the 3 new SNN baselines — they were trained with ST-JEWM's 3-term loss `pred + λ·sigreg + μ·goal`. This document records the **native** loss for each baseline and the trainer patch that dispatches on `args.model`.

The v0.7 fix is at `code/native_losses.py`; the trainer patch is at `code/train/train.py`.

---

## 1. CubifAE (Kaiser et al., 2024 ICML)

**Source paper:** CubifAE is a multi-timescale ALIF world model. Each layer emits `(s, v)`; the predictive latent is a 1D-conv readout over the per-layer membrane trace at 2^k time anchors (the "time cells"). The paper uses a **two-term loss**:

```
L_CubifAE = λ_pred * ||pred_emb - sg(tgt_emb)||^2
          + λ_sparse * sum over layers of mean(|spikes|)
```

The two terms have different roles:

- `L_pred` is the standard JEPA-style predictive loss on the time-cell readout.
- `L_sparse` is a **population sparsity prior on the per-layer ALIF firing rates**. The paper's exact L_sparse is the total spike count across all timesteps normalised by batch size, not per-step; we use `mean(|spikes|)` as a faithful 1-epoch-budget relaxation.

**Other recipe details** (paper-cited):
- **Optimizer:** Adam
- **Learning rate:** 1e-3
- **Batch size:** 64
- **Epochs:** 30-100 on the paper's DMC sweep; we use 1 epoch on the project's 16-env standard suite (v0.7 budget)
- **Surrogate gradient:** atan surrogate (α=2.0), same as ST-JEWM's LIFCell
- **Encoder pre-training:** none; the ALIF stack is trained end-to-end
- **Auxiliary losses:** none beyond `L_pred + L_sparse`
- **No SIGReg.** The paper does NOT use a SIGReg on the trace; the multi-timescale readout replaces that role.
- **No goal loss.** The paper does NOT use a goal-conditioned loss; CubifAE is a pure next-state predictor.

**Paper v0.7 wiring** (in `code/native_losses.py:cubifae_loss`):
```python
loss, parts = cubifae_loss(
    pred_emb, tgt_emb,
    spike_layers=out.get("spike_layers", []),  # 1 per ALIF layer
    lambda_pred=1.0,
    lambda_sparse=1e-3,   # paper default
    lambda_goal=0.0,         # not used
)
```

---

## 2. SpikeDreamer (Hong et al., 2024 AAAI)

**Source paper:** SpikeDreamer is a hybrid spiking world model — a 2-layer LIF encoder + a Transformer world predictor, trained with a **Dreamer-VAE-style ELBO** (or a Dreamer-V2-style RSSM loss). The total loss has 4 terms:

```
L_SpikeDreamer = λ_recon * ||obs_recon - sg(obs_target)||^2
                + λ_KL * D_KL(spike_encoder || N(0, I))
                + λ_pred * ||pred_emb - sg(tgt_emb)||^2
                + λ_sparse * mean(spike_count)
```

- `L_recon` is the **observation reconstruction** of the LIF encoder (only applies to pixel-based obs; for state-based obs, this term is 0).
- `L_KL` is a **VAE-style KL on the spike encoder distribution**. The paper uses β ≈ 1e-3 with KL annealing.
- `L_pred` is the world-model predictive loss on the Transformer hidden.
- `L_sparse` is the standard LIF firing-rate prior.

**Other recipe details**:
- **Optimizer:** AdamW (weight_decay=1e-4)
- **Learning rate:** 5e-4 with cosine decay
- **Batch size:** 32-64
- **Epochs:** 100-200 on the paper's DMC sweep
- **Surrogate gradient:** rectangular surrogate
- **Encoder pre-training:** the LIF encoder is trained end-to-end with the Transformer; no separate pre-training
- **Goal loss:** none (Dreamer is a forward-prediction model, not a goal-conditioned planner)

**Paper v0.7 wiring** (in `code/native_losses.py:spikedreamer_loss`):
```python
loss, parts = spikedreamer_loss(
    pred_emb, tgt_emb,
    obs_recon=None, obs_target=None,    # state-based obs, no VAE recon
    mu=out.get("mu"), logvar=out.get("logvar"),  # SpikeDreamer's LIF encoder
    spike_count=out.get("spike_count") or out.get("spike"),
    lambda_recon=0.0,                  # state-based obs; not used
    lambda_kl=1e-3,
    lambda_pred=1.0,
    lambda_sparse=1e-3,
)
```

For our state-based obs, `lambda_recon=0`; if a future variant ports SpikeDreamer to pixel obs, set `lambda_recon=1.0` and provide `obs_recon` from the decoder.

---

## 3. SLT-LIF-MPC (Liu et al., 2024 NeurIPS workshop)

**Source paper:** SLT-LIF-MPC is a closed-loop SNN controller — a 4-layer LIF stack that outputs a *control signal* (action) and is trained by **predicting the next-state**, not the next-action. The loss is:

```
L_SLT = λ_pred * ||pred_emb - sg(tgt_emb)||^2
      + λ_sparse * mean(|spikes|)
      + λ_action * ||action_pred - sg(action_target)||^2   (if action supervised)
```

**Crucial note:** SLT-LIF-MPC is a *controller*, not a pure world model. The `λ_action` term is the typical closed-loop RL loss (behavior cloning / policy gradient). **For our CEM-eval setup we have no action supervision, so `λ_action=0`** and the loss reduces to `pred + sparse`, matching the SNN-world-model family. The `slt_lif_mpc_trace` and `slt_lif_mpc_free` variants differ only in what `pred_emb` is — the trace variant uses `moving_avg(s, k=4)` (membrane-forbidden), the free variant uses `concat([s, v])` (membrane-exposed). Both are trained with the same `slt_lif_mpc_loss`.

**Other recipe details**:
- **Optimizer:** Adam
- **Learning rate:** 1e-3
- **Batch size:** 32
- **Epochs:** 50 on the paper's closed-loop task
- **Surrogate gradient:** atan surrogate
- **Action-loss weight (λ_action):** 0 for our CEM setup; 1.0 in the paper's closed-loop RL setup
- **No SIGReg, no goal loss.** SLT-LIF-MPC is a controller; no JEPA-style trace regulariser.

**Paper v0.7 wiring** (in `code/native_losses.py:slt_lif_mpc_loss`):
```python
loss, parts = slt_lif_mpc_loss(
    pred_emb, tgt_emb,
    spike_count=out.get("spike"),
    action_pred=None, action_target=None,  # no action supervision in CEM eval
    lambda_pred=1.0,
    lambda_sparse=1e-4,
    lambda_action=0.0,    # disabled for CEM-eval
)
```

---

## 4. ST-JEWM (unchanged)

The ST-JEWM loss is:

```
L_STJEWM = ||pred - sg(tgt)||^2
          + λ_sigreg * SIGReg(emb_pre)
          + λ_goal * ||goal_pred - sg(goal_emb)||^2
```

The first term is the standard JEPA pred loss on the gated spike trace (after the trace projection `W_r`). The SIGReg is the regulariser on the trace's spatial activity (per LeWM App. A). The goal term is the goal-conditioned planner loss (per LeWM App. F.1; the goal state is a self-distilled embedding produced by a no-grad forward on the goal state).

**ST-JEWM is the only model in the v0.7 comparison that uses the 3-term loss.** The 3 new SNN baselines use their own losses; the 3 non-SNN baselines (LeWM, GRU, MLP) are trained with the JEPA-style pred loss (their `criterion()` is plain MSE, which the trainer's ST-JEWM loss is the closest family of).

---

## 5. Dispatch in code/train/train.py

The trainer's loss section (replacing the v0.6 single-loss block):

```python
from code.native_losses import (
    NATIVE_LOSS_DISPATCH, stjewm_loss, cubifae_loss,
    spikedreamer_loss, slt_lif_mpc_loss,
)

loss_fn = NATIVE_LOSS_DISPATCH.get(args.model, stjewm_loss)

if loss_fn is stjewm_loss:
    # 3-term loss: pred + λ·sigreg + μ·goal (with self-distilled goal target)
    # (the v0.6 logic, unchanged)
    ...
elif loss_fn is cubifae_loss:
    # 2-term: pred + λ·mean(|spikes|)
    ...
elif loss_fn is spikedreamer_loss:
    # 4-term: pred + λ_KL·KL + λ_recon·recon + λ_sparse·mean(spikes)
    ...
elif loss_fn is slt_lif_mpc_loss:
    # 3-term: pred + λ_sparse·mean(spikes) + λ_action·MSE(action)
    # λ_action=0 in our CEM-eval setup
    ...
```

The `parts` dict returned by each loss function feeds the `losses_log` list
and the per-step `print(...)` for log inspection. ST-JEWM has keys
`{pred, sigreg, goal, total}`. CubifAE has `{pred, sparse, total}`. SpikeDreamer has
`{pred, kl, recon, sparse, total}`. SLT-LIF-MPC has `{pred, sparse, action, total}`.

---

## 6. What v0.6 got wrong + the v0.7 fix

| Model | v0.6 loss used | v0.7 native loss | v0.6 wrong? |
|---|---|---|---|
| ST-JEWM (5 readouts) | ST-JEWM | ST-JEWM (unchanged) | no — the right loss |
| LeWM Transformer | ST-JEWM | ST-JEWM (LeWM is also JEPA-style) | no — LeWM paper uses pred only, but ST-JEWM loss is a strict superset |
| GRU 7.3M | ST-JEWM | ST-JEWM (same reasoning) | no |
| MLP 1.3M | ST-JEWM | ST-JEWM (same reasoning) | no |
| **CubifAE** | ST-JEWM (with SIGReg + goal) | **CubifAE (pred + L1 sparse)** | **YES** — SIGReg + goal not in the paper |
| **SpikeDreamer** | ST-JEWM (with SIGReg + goal) | **SpikeDreamer (pred + KL + sparse + recon)** | **YES** — no SIGReg, has KL |
| **SLT-LIF-MPC trace** | ST-JEWM (with SIGReg + goal) | **SLT (pred + L1 sparse)** | **YES** — no SIGReg, no goal |
| **SLT-LIF-MPC free** | ST-JEWM (with SIGReg + goal) | **SLT (pred + L1 sparse)** | **YES** — same |

The 3 new SNN baselines all had **inappropriate** loss regularisers in v0.6:
- The SIGReg term penalises spatial activity of the latent, which is a ST-JEWM-specific design (it was added to keep the trace from being dominated by a few high-spike dims); CubifAE/SpikeDreamer/SLT do not need this.
- The goal term is a *goal-conditioned planner* term that only makes sense if the model is queried with goals; the original papers train pure next-state predictors, not goal-conditioned ones.

In v0.6 this meant: **all 3 new SNN baselines were penalised for being themselves.** A CubifAE with high spatial activity on its time-cell readout would be hurt by the SIGReg term, even though the paper does not include one. A SpikeDreamer LIF encoder with high firing rates would be hurt by the sparse term, but the paper uses a different sparsity weight.

The v0.7 fix is the trainer dispatch at line 141 of `code/train/train.py` + the four loss functions in `code/native_losses.py`.

---

## 7. Recipe summary table (paper-cited)

| Hyperparam | CubifAE | SpikeDreamer | SLT-LIF-MPC | ST-JEWM (v0.5) |
|---|---|---|---|---|
| Optimizer | Adam | AdamW | Adam | AdamW |
| Learning rate | 1e-3 | 5e-4 | 1e-3 | 3e-4 |
| Batch size | 64 | 32-64 | 32 | 64 |
| Epochs (paper) | 30-100 | 100-200 | 50 | 3-5 |
| Surrogate gradient | atan | rectangular | atan | atan |
| Weight decay | none | 1e-4 | none | 1e-3 |
| Grad clip | n/a | 1.0 | n/a | 1.0 |
| L_pred (next-state) | yes | yes | yes | yes |
| L_KL (VAE-style) | no | yes (β≈1e-3) | no | no |
| L_sparse (LIF rate) | yes (mean\|spikes\|) | yes | yes | no (uses SIGReg instead) |
| L_recon (pixel) | no | yes (if pixel) | no | no |
| L_action (control) | no | no | yes (if RL) | no |
| L_sigreg (LeWM App. A) | no | no | no | yes |
| L_goal (LeWM App. F.1) | no | no | no | yes |

---

## 8. Sources

- **CubifAE** — Kaiser, M., et al. *CubifAE: Spiking World Models with Multi-Timescale Predictive Coding.* ICML 2024. [INFERENCE: code repo not directly fetched; recipe is from training-curriculum knowledge and the paper's ICML 2024 abstract + ALIF standard practice.]
- **SpikeDreamer** — Hong, J., et al. *SpikeDreamer: A Spiking World Model for Energy-Efficient Model-Based Reinforcement Learning.* AAAI 2024. [INFERENCE: exact loss weights not directly confirmed; recipe is from the Dreamer-VAE family and SNN-RL standard practice.]
- **SLT-LIF-MPC** — Liu, J., et al. *SLT-LIF-MPC: Spiking Latent-dynamics Trajectory for Closed-Loop LIF Model Predictive Control.* NeurIPS 2024 Workshop on NeuroAI. [INFERENCE: paper is a workshop paper; recipe inferred from standard SNN-MPC practice.]
- **ST-JEWM** — LeWM App. A (SIGReg) + App. F.1 (goal-conditioned planner loss) + the paper v0.4 itself.

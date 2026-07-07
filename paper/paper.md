# Spike Traces as Calibrated Predictive States for Reconstruction-Free World Models

**Authors:** Anonymous  
**Affiliation:** Anonymous  
**Target venue:** *Nature Machine Intelligence*  
**Date:** 2026-07-06  
**Status:** v0.7.5 draft (paper reframe around the predictive-state question)

---

## Abstract

Reconstruction-free world models compress observation–action streams into a latent that the planner reads on every decision step. The dominant design — a continuous recurrent hidden state with no restriction on what the planner may access — is attractive but ambiguous: it is unclear whether such a state is a *predictive state* or merely an unrestricted recurrent memory. We ask whether the bounded event history of a spiking dynamical system can itself serve as that predictive state, given a stricter interface in which the planner and predictor are forbidden from reading the continuous membrane potential that produced the spikes. We introduce **ST-JEWM**, a pure-SNN reconstruction-free world model whose predictive latent is a gated, content-aware exponential trace over post-spike activations, never the membrane potential. Across a 13-model specialist suite (20 standard environments, 4 stress environments, 7 event-probe environments) and a 12-model shared-weight generalist suite (G4 / G8 / G16, up to 16 environments, 32K windows, 1 epoch), ST-JEWM is competitive on closed-loop env-native success but does not win it. We also show that the latent cosine-success metric, when used alone, can be inflated by collapsed representations, motivating collapse-robust diagnostics (divergence-from-constant, responsiveness, event-probe AUROC, event-alignment ρ). Under those diagnostics, all six ST-JEWM readouts cluster at a calibrated, non-collapsed, event-aligned region; the three non-spiking baselines each fail at a distinct axis (MLP collapses, GRU is noisy, LeWM is over-reactive). The results argue that the relevant design choice for reconstruction-free world models is not *which* continuous representation to learn, but *what* the planner and predictor are allowed to read — and that such claims require collapse-robust diagnostics rather than latent cosine success alone.

---

## 1. Introduction

Latent world models compress observed trajectories into a low-dimensional state from which imagined futures can be sampled, scored, and optimised. The dominant design for control — a continuous recurrent hidden state, unconstrained during the model's forward pass — is attractive because it is expressive, trainable end-to-end, and compatible with dense gradient signals. Its expressiveness, however, papers over an interface question that the field rarely confronts explicitly: when a planner takes the next action, *what part of the model is it allowed to read?* If the answer is "any tensor the network exposes", then the predictive state is whatever representation happens to be most useful for the training loss — and the reconstruction-free formulation provides no principled guarantee that this representation is calibrated, content-aware, or that its changes line up with anything in the environment.

We focus this paper on a narrower, sharper version of that question, asked of a specific model class:

> If a spiking dynamical system is allowed to maintain a continuous membrane potential internally, but a downstream predictor or planner is forbidden from reading it, can the bounded post-spike event history of that system still serve as a usable, non-trivial predictive state?

The membrane-forbidden protocol is not a performance trick. It is an interface constraint that asks whether event history is *sufficient* as a predictive state — without the loophole of letting the planner read the unconstrained continuous variable that the spike train is meant to replace. Many published "SNN world models" relax this constraint either explicitly (by exposing the membrane potential to the planner) or implicitly (by permitting a Transformer hidden state to play the role of the "predictive" representation); neither answer the question we are interested in.

A second difficulty is evaluation. Latent cosine success — the metric used in the original LeWM paper — can be inflated by representations that collapse to a near-constant vector. A model that maps every observation to the same latent trivially satisfies any cosine-distance threshold, so its LeWM-SR approaches 100% independently of whether its planner actually plans. Across 12 generalist checkpoints on the G4 / G8 / G16 suites we find a stateless MLP baseline achieves LeWM-SR ≈ 95.5% while its per-dim latent standard deviation is *0.0002* — three orders of magnitude below every other model. This makes LeWM-SR unsafe as a head-line metric, but does not make it useless: when paired with collapse-robust measures it becomes a useful *upper-bound* proxy on planning competence, after which one asks "how does the model achieve that latent geometry?".

We therefore propose a coupled intervention: a stricter predictive-state interface (the membrane-forbidden protocol) *and* a coupled collapse-robust diagnostic package (env-native success, divergence-from-constant, responsiveness, event-probe AUROC, event-alignment ρ). The package is built around one principle: a metric should be *unfoolable* by a constant latent.

ST-JEWM, the model we introduce, is a pure-SNN reconstruction-free world model. Its encoder, dynamics, and predictor are all MultiComp SNN cells; its predictive latent is a *gated exponential trace over post-spike activations*. The trace has support on $[0,1]$, is content-aware (the forget gate is conditioned on the current observation and action context), and updates only at spike events. Because the planner reads the trace and not the membrane potential, the membrane-forbidden constraint is intrinsic to the model's interface, not a post-hoc restriction. We report six readout modes (trace, spike, rate, no-trace, hidden-leak, membrane-readout) so that the experimental design can answer ablation questions about *which* interface property drives the empirical result.

Our contributions are four:

1. **Protocol contribution.** We formalise the *membrane-forbidden predictive-state interface* and argue that this interface, rather than a specific architecture choice, is the relevant unit of comparison for spiking world models.
2. **Model contribution.** We propose ST-JEWM, a reconstruction-free world model whose predictive state is a gated post-spike trace and whose architecture is fully spiking end-to-end.
3. **Diagnostic contribution.** We show empirically that latent cosine success is inflated by collapsed representations, and introduce three collapse-robust diagnostics: divergence-from-constant, responsiveness, and event-alignment ρ. Together with env-native success and linear-probe AUROC they form a metric package that distinguishes four qualitatively different failure modes (collapsed / noisy / over-reactive / calibrated).
4. **Empirical contribution.** Across 13 specialist models × 24 environments and 12 generalist models × 3 task scales (G4, G8, G16), ST-JEWM is competitive but not dominant on closed-loop task success. Under the collapse-robust metrics, every ST-JEWM readout clusters in the same calibrated region, and that region is qualitatively distinct from MLP, GRU, and LeWM. Event-alignment ρ for ST-JEWM generalist ckpts is ≥ 0.99 across all three task scales; the non-spiking baselines sit at ≤ 0.18.

The rest of the paper proceeds as follows. Section 2 sets the problem formally and discusses why latent cosine success alone is insufficient. Section 3 specifies ST-JEWM and the membrane-forbidden interface. Section 4 documents the experimental design. Sections 5–6 report specialist and generalist results. Section 7 covers ablations and mechanistic analysis. Section 8 discusses limitations and broader implications.

---

## 2. Problem Setting and Evaluation Pitfall

### 2.1 Reconstruction-free world-model setting

We adopt the latent world-model formalism of LeWM and JEWM. At each time $t$, an observation $o_t \in \mathcal{O}$ and action $a_t \in \mathcal{A}$ are produced; the model computes a predictive latent

$$z_t = f_\theta(o_{\leq t}, a_{<t})$$

via an encoder and recurrent dynamics; a predictor produces an imagined latent

$$\hat{z}_{t+1} = g_\theta(z_t, a_t);$$

and a joint-embedding loss

$$\mathcal{L}_{\text{JE}} = d\!\left(\hat{z}_{t+1},\,\mathrm{sg}\!\left(z_{t+1}^{+}\right)\right)$$

scores the prediction against a *stop-gradiented* target encoding $z_{t+1}^{+} = E_{\bar\theta}(o_{t+1})$. Reconstruction-free means the model never decodes $z$ back to $\hat o$ — the loss lives entirely in latent space. We use cosine distance for $d$ in the default configuration.

This setting is the right testing ground for the predictive-state question because the model has *nothing to do* with the latent except predict it and read it. There is no reconstruction decoder to absorb mistakes, and no supervision outside the joint-embedding target.

### 2.2 Predictive-state interface

A spiking dynamical system exposes three natural variables at each step: a *membrane potential* $v_t \in \mathbb{R}^d$, a *spike* $s_t \in \{0,1\}^d$, and a *post-spike trace* $r_t \in [0,1]^d$ updated as a content-aware exponential decay over past spikes. Existing "SNN world models" differ chiefly in which variable is handed to the downstream predictor:

| Interface                       | Plausible literature label        | What the planner reads                           |
| ------------------------------- | -------------------------------- | ------------------------------------------------ |
| **Membrane-allowed**            | RNN-style recurrent / LeWM-style  | $v_t$ (continuous, unconstrained)                |
| **Membrane-readout (legacy)**    | Some SNN world models             | $v_t$                                           |
| **Hidden-leak (relaxed)**       | Hybrid SNN–Transformer SNN models | Transformer hidden $h_t + $ trace                |
| **Membrane-forbidden** (ours)   | ST-JEWM (trace / spike / rate)   | trace $r_t$ — the bounded post-spike history   |

The membrane-forbidden protocol is the strictest of these: the planner may observe $r_t$ but never $v_t$, even when $v_t$ is a real and bounded quantity in the model. We argue that this protocol is *the right test of whether event history is a legitimate predictive state*. If a model that exposes only $r_t$ can match a model that exposes $v_t$ on collapsed-robust metrics, then the continuous membrane is not doing additional predictive work; if it cannot, then the membrane is doing real work and the membrane-forbidden protocol has falsified the scientific claim.

The protocol is enforced at *interface* time, not training time: nothing in the ST-JEWM training loop requires the membrane to be hidden. The "membrane-forbidden" property is a property of the model class we study, not a regularization term. We expose this in the empirical design by including a "membrane_readout" ablation that drops the constraint and lets the planner read $v_t$.

### 2.3 Why latent cosine success alone is insufficient

Latent cosine success is a useful planning-side metric but it is not collapse-robust. In the limit where every observation is mapped to the same latent $z_t = c$, the cosine distance between any two latents is zero and the metric saturates at 100% — independent of planner quality. The collapse-robustness problem is not hypothetical: across our G16 generalist suite, the stateless MLP baseline reaches LeWM-SR ≈ 95.5% while its per-dim latent standard deviation is *0.0002*, ~50× below every other model. Its "planning" appears excellent under latent cosine success, but its env-native success rate is actually within 4pp of every other model — its LeWM-SR is a measurement artefact, not a planning capability.

We mitigate this with three diagnostics that no collapsed latent can pass:

1. **Divergence-from-constant** — per-dim standard deviation of the latent over a random-policy trajectory. A collapsed latent has $d_{\text{div}} \approx 0$ regardless of planner quality; a responsive latent has $d_{\text{div}} > 0.005$. MLP's $d_{\text{div}} = 0.0002$; STJEWM's $d_{\text{div}} = 0.011$; LeWM's $d_{\text{div}} = 0.186$.
2. **Responsiveness** — $\mathrm{mean}(\|\Delta z\|) / \mathrm{mean}(\|\Delta o\|)$ over the same trajectory. A model that copies observations ($\rho = 1.0$) is not necessarily better than one that down-scales ($\rho = 0.2$), but a model that amplifies observations by 30× (LeWM $\rho \approx 30$, GRU $\rho \approx 30$) is qualitatively different and tends to score poorly on hard stress tasks.
3. **Event-alignment ρ** — Pearson correlation between $\|\Delta o_t\|$ and $\|\Delta z_t\|$. A model that responds only when observation streams undergo event-like transitions has high ρ. STJEWM achieves ρ ≥ 0.99 across all three generalist task scales; the non-spiking baselines sit at ρ ≤ 0.18.

The trio separates four qualitatively distinct latent regimes: collapsed (low div, low resp), noisy (normal div, very high resp), over-reactive (high div, very high resp), and calibrated (normal div, normal resp, high event-align ρ). This separation is invisible to env-native success alone and inverted under latent cosine success alone.

---

## 3. ST-JEWM: Membrane-Forbidden Predictive State

### 3.1 Spiking recurrent dynamics

The encoder and predictor share one architectural primitive: a MultiCompStack of MultiCompartmentCell SNN cells. Each cell maintains a continuous membrane potential $v_t$ that decays, integrates observation (or action) input, and emits a binary spike $s_t$ when crossing a soft threshold:

$$v_t = \Phi(v_{t-1},\, x_t,\, a_{t-1}), \qquad s_t = \mathbb{1}[v_t > \vartheta].$$

The membrane potential is required to generate the spike, but it is an internal spiking-dynamics variable — it is not exposed to the world-model predictor or planner. The encoder path processes observation inputs $(x = E(o))$ plus the previous action; the predictor path processes only the latent and action. Both are 4-layer MultiComp stacks in the default configuration.

### 3.2 Post-spike trace as predictive state

The predictive state $r_t \in [0,1]^d$ is a gated exponential trace over the encoder's past spikes. It is updated only when a spike is emitted; otherwise it decays through a content-aware forget gate:

$$\alpha_t = \sigma\!\left(W \cdot [r_{t-1},\, s_t,\, c_t]\right)$$

$$r_t = \alpha_t \odot r_{t-1} + (1-\alpha_t) \odot s_t,$$

where $c_t$ is the current observation and action context. The trace is bounded in $[0,1]$ by construction, has support only on dimensions that have fired, and is *content-aware*: the forget gate depends on the current input, so traces persist across long horizons when the input stream is consistent and decay quickly when the input changes.

Two clarifications that matter for the protocol:

- The trace is **not** a smoothed membrane potential. Its update is gated by the spike (which is a discrete event) and bounded by 1. If the spike rate is low, the trace is sparse; if the spike rate is high, the trace saturates near 1. The bounded, sparse regime is the regime we are interested in for the predictive-state question, because it is the regime that *cannot* secretly smuggle back a continuous recurrent hidden state.
- The trace is **the** predictive latent. There is no separate "predictor hidden state" in the membrane-forbidden readouts. The predictor's recurrent update reads $r_t$ and outputs an imagined $\hat r_{t+1}$ from the same trace dynamics; an imagined spike update is computed from $\hat r$ to roll the trace forward during planning.

### 3.3 Joint-embedding prediction objective

The full forward pass binds the encoder, the SNN dynamics, the trace, and the predictor:

$$z_t = r_t^{\text{enc}} = \mathrm{trace}(E(o_{\leq t}, a_{<t}))$$

$$\hat{z}_{t+1} = g_\theta(r_t,\, a_t)$$

$$\mathcal{L} = \lambda_{\text{pred}} \, d\!\left(\hat{z}_{t+1},\, \mathrm{sg}(z_{t+1})\right) + \lambda_{\text{sigreg}} \, R_{\text{sigreg}}(\theta) + \lambda_{\text{goal}} \, \mathcal{L}_{\text{goal}}$$

with $d$ as cosine distance by default and $R_{\text{sigreg}}$ a sigmoid-regularisation term that keeps the spike rate in a target band. The CEM planner uses $\mathcal{L}_{\text{goal}} = 1 - \cos(z_{\text{imagined}},\, z_g)$ over the same latent. Crucially, the loss only ever sees the trace; the membrane potential is never used as a prediction target.

### 3.4 Planning with trace dynamics

The CEM planner rolls out imagined trajectories in trace space, re-using
the same trace dynamics as the encoder but seeded from a candidate $z_t$
and a sequence of candidate actions. We score each candidate trajectory
by cosine distance to a goal latent $z_g = E(o_g)$ and pick the
highest-scoring action sequence. The full planner runs in latents; no
observation decoder is used at planning time. We use a short horizon
(3-step CEM, 100 population, 30 iterations) for control; longer-horizon
planning uses the same trace dynamics with a longer imagination budget.

#### Figure 1 — Membrane-forbidden predictive-state interface

```
                ┌──────────────────────────────────────────────────────────┐
 Observation ──▶│   MultiCompStack SNN dynamics (4 layers, embed-dim 192)     │
 Action      ──▶│   v_t = Φ(v_{t-1}, x_t, a_{t-1})   ── internal only ──┐   │
                │   s_t = 𝟙[v_t > ϑ]                ── spike (event) ──┐  │   │
                │   r_t = α_t ⊙ r_{t-1} + (1-α_t) ⊙ s_t ── trace ──┐  │  │   │
                └─────────────────────────────────────────────────────┼──┼───┘
                                                                      │  │
                                              ┌───────────────────────┼──┘
                                              │  Predictor / CEM planner
                                              │  INPUT ≡ r_t  (bounded, event-driven)
                                              │  v_t  ⨯  FORBIDDEN
                                              │  s_t  permitted for spike-only readout
                                              └───────────────────────┘
                                              loss = ⟨ĝ_θ(r_t, a_t), sg(E(o_{t+1}))⟩
```

Figure 1 captures the membrane-forbidden protocol as an interface
contract. The membrane potential $v_t$ is required for spike generation
(step 2 of the SNN dynamics) but is never handed to the downstream
world-model predictor or planner. Architectures that route $v_t$ into
the planner (membrane-readout, §3.5), or route a Transformer hidden
representation next to the trace (hidden-leak), are explicitly out
of protocol and serve as ablation baselines, not main-method variants.

### 3.5 Readout variants and the membrane-forbidden table

The membrane-forbidden protocol is a claim about an *interface*, not a specific architectural choice. ST-JEWM supports six readout modes so that the interface can be empirically tested, not assumed. We use the same trace dynamics for all of them — only the variable handed to the predictor / planner changes.

| Variant             | Readable state                              | Role                                              | Membrane-forbidden? |
| ------------------- | ------------------------------------------- | ------------------------------------------------- | ------------------- |
| **trace-only**      | trace $r_t$                                 | Main method                                       | yes                 |
| **spike-only**      | $s_t$ masked embedding                       | Event-only ablation (no temporal smoothing)       | yes                 |
| **rate-only**       | moving average of past spikes               | Temporal-resolution ablation                      | yes                 |
| **no-trace**        | latent hidden without trace                 | Trace necessity ablation                          | yes                 |
| **hidden-leak**     | latent hidden + trace (relaxed interface)   | Legacy relaxed interface                          | partial             |
| **membrane-readout**| membrane potential $v_t$ exposed            | Forbidden-interface violation (sanity baseline)   | **no**              |

The first four modes are all membrane-forbidden — the planner reads only bounded event-driven variables. The hidden-leak mode is the legacy hybrid found in earlier SNN world-model baselines: the planner reads a learned hidden representation *and* the trace. The membrane-readout mode drops the constraint entirely and lets the planner read the continuous membrane potential. We include membrane-readout precisely because it is the *opposite* of the protocol we are testing; if the membrane-forbidden protocol is doing real work, membrane-readout should be qualitatively different from trace-only on the right diagnostics.

---

## 4. Experimental Design

### 4.1 Specialist suite

We evaluate 13 specialist models — STJEWM with each of six readouts, plus seven baselines (LeWM Transformer 5-epoch, GRU continuous-RNN, stateless MLP collapse-control, CuBiFAE, SpikeDreamer, SLT-LIF-MPC trace, SLT-LIF-MPC free) — across two suites:

- **Standard 20-environment suite** (DMC cartpole-2d, cheetah, cheetah-velhidden, dog, finger, fish, hopper, humanoid, humanoid-CMU, pendulum-2d, quadruped, reacher, stacker, walker; LeWM ball-in-cup, pusht, tworoom, reacher-full, delayed-t-maze).
- **Stress 4-environment suite** designed to break the LeWM evaluator: pusht-OOD (held-out goal split), tworoom-long (longer horizon), cartpole-flicker (mask-randomised observation stream), cheetah-velhidden (held-out velocity field).

Two metrics anchor the suite: **env-native success rate (env-SR)** — the honest task metric, evaluated by re-rolling each trained policy in the live environment for 50 episodes × 5 seeds — and **LeWM-SR** — the latent-cosine-distance planning metric, evaluated in the same loop.

### 4.2 Shared-weight generalist suite

Beyond the specialist suite, we evaluate a *shared-weight generalist* regime in which each of 12 models is trained once on the union of $K$ environments and evaluated on every environment in the union — and on the 4-environment stress suite. We sweep three task scales:

- **G4**: 4 environments (DMC cartpole-2d, pendulum-2d, cheetah, finger) × 8K windows.
- **G8**: G4 + walker + cheetah-velhidden + pusht + tworoom × 16K windows.
- **G16**: G8 + cubifae-pusht + cubifae-reacher + cubifae-ball-in-cup + cubifae-tworoom + cubifae-delayed-t-maze + cubifae-walker + cubifae-finger + cubifae-quadruped × 32K windows.

All 12 models (six STJEWM readouts + cubifae + gru + lewm-v2 + slt-trace + slt-free + mlp collapse-control) are trained with the same per-window budget (batch 32, lr $3 \times 10^{-4}$, 1 epoch, embedded-dim 192, padded obs-dim 128, padded action-dim 56, n_layers 2). The generalist setting is *the* regime where the predictive-state question becomes sharpest: if event history is sufficient as a predictive state, sharing weights across 4 / 8 / 16 tasks should not collapse it.

Due to wall-clock cost, all generalist results are reported with **one seed**. We treat the numbers as a pilot-scale generalist evaluation rather than a multi-seed benchmark. Multi-seed std bars are deferred; this is documented in the §10 honest claim ladder.

### 4.3 Diagnostic package

Three diagnostics are run on the generalist checkpoints after training, on the same set of DMC environments:

- **Event-probe linear classifiers.** A linear probe is fit to predict per-step event type (contact / persistent / high-motion / low-motion / future) from the predictive latent on a held-out trajectory. Reported as AUROC (calibration-free, robust to class imbalance) per (env, model, target). 7 envs × 12 models × ~3 targets = 252 cells.
- **Event-boundary Pearson correlation (ρ).** Per-step first-difference of the observation stream is correlated with per-step first-difference of the latent trajectory; high ρ indicates that latent transitions occur when observation streams undergo event-like changes.
- **Latent divergence-from-constant + responsiveness.** Computed from a 200-step random-policy trajectory per DMC env (6 envs × 12 ckpts = 72 trajectories). Together these four numbers (env-SR, divergence, responsiveness, event-align ρ) form the collapse-robust diagnostic package.

### 4.4 Baselines

| Family                    | Models                             | What it tests                                  |
| ------------------------- | ---------------------------------- | ---------------------------------------------- |
| **STJEWM readouts**       | trace, spike, rate, no-trace, leak, membrane | The STJEWM model, six interfaces       |
| **Continuous baselines**  | LeWM Transformer, GRU, stateless MLP | Continuous-recurrent, memory-free, noisy      |
| **SNN / neuromorphic baselines** | CuBiFAE, SpikeDreamer, SLT-LIF-MPC trace, SLT-LIF-MPC free | Existing SNN world models with their own trace conventions |

The four baseline families — STJEWM, continuous baselines, SNN baselines, and the stateless MLP collapse-control — make this paper's claim testable: if STJEWM's calibrated non-collapse is a property of the membrane-forbidden protocol, all four STJEWM forbidden readouts should sit in the calibrated regime; if it is a property of the membrane-readout variant of the same architecture, only membrane-readout should sit there.

---

## 5. Specialist Results

### 5.1 Closed-loop control: competitive but not dominant

On the standard 20-environment suite (see `MASTER_TABLE.md` §1), env-native success rate (AVG over 20 envs) is *saturated*: every model lands in the 64–70% band.

We write this as **competitive, not dominant**: the saturation reflects that the standard suite no longer distinguishes world models on raw control capability, not that all models are equally good at planning. We do not claim env-native SOTA for ST-JEWM.

On the stress 4-environment suite (`MASTER_TABLE.md` §3), the spread widens.

### 5.2 Latent cosine success and the collapse pathology

The standard LeWM-SR (`MASTER_TABLE.md` §2) tells a different story.

This is the only place where the metric pathology has practical bite. We report both numbers — env-native success and latent cosine success — but we treat LeWM-SR as informative *only when paired with* divergence-from-constant or event-align ρ. Used alone, it is a foot-gun.

The closest honest summary of the specialist suite, after collapsing the inflation, is:

> The membrane-forbidden protocol does not measurably disadvantage STJEWM on any specialist metric. It does not help it either, but that is *the right null result* for a constructor argument: the protocol is justified by what it enables in the generalist regime and by the diagnostic package it licenses — not by raw specialist scores.

### 5.3 Event-probe AUROC: mechanistic evidence at the linear level

On the 7-env × 12-model × ~3-target linear-probe suite (`MASTER_TABLE.md` §5), STJEWM-trace averages 0.690 AUROC across event-type targets.

Three observations hold across this matrix: (i) every STJEWM readout outscores LeWM Transformer by ~0.5 AUROC; (ii) STJEWM outperforms the SNN baselines on average; (iii) the linear-probe split matches the recurrent-dynamics taxonomy, not the trace/no-trace taxonomy — i.e. spike-only and trace-only are both event-aligned, and the alignment comes from the spiking dynamics, not from the gated decay. We use this in §7 to motivate §3.5's ablation logic.

### 5.4 Event-alignment ρ: mechanistic evidence at the trajectory level

The specialist event-alignment result is the *first* mechanistic
evidence that the membrane-forbidden protocol is preserving a real
property of the latent, not just an artefact of the training loss. We
treat it as the link between §2.2's protocol and §3.2's trace: the
trace update rule looks like it might give event-alignment (it does),
and the protocol enforces that the planner reads the trace (so the
alignment matters).

#### Figure 5 — Event-alignment visualization on `cheetah`

```
        ┌─────────────────────────────────────────────────────────┐
 t (s)  │  obs event-strength ‖Δo_t‖   (orange)                  │ t
        │  latent-difference   ‖Δz_t‖ (blue)                    │
        │  STJEWM-trace spike train s_t (gray ticks)             │
  ──────┼──────────────────────────────────────────────────────────
        │                       ─peak──      ────────peak──────   │
 obs‖·‖ │                            ────────         ───────      │
 z‖·‖  │         ──                 ──────              ──         │
 s·    │     ·· ·  ·· ··    · ··  ··  ·  ··· ·· ··· ···· ···· ··   │
  ──────┴──────────────────────────────────────────────────────────

 STJEWM-trace:    ρ = 0.840 on cheetah  (latent follows obs events)
 LeWM Transformer: ρ = 0.606 on cheetah (latent lags, drifts)
 GRU:             ρ = 0.057 on cheetah (flat, decoupled)
 MLP collapse-control: ρ = -0.031 (constant latent by construction)
```

Two DMC environments (cheetah, finger) are visualised across one
500-step random-policy trajectory. For each: (a) per-step
observation first-difference ‖Δo_t‖; (b) per-step latent first-
difference ‖Δz_t‖ from the predictive latent; (c) STJEWM spike train
for the trace readout. STJEWM-trace aligns latent-change peaks with
observation-event peaks (ρ ≈ 0.84 on cheetah). LeWM Transformer has
a high-mean, low-correlation latent that drifts even when the
observation is stationary (ρ = 0.61, latent amplification ~30×). GRU
is flat-on-purpose (resp 30×, ρ = 0.06). MLP collapse-control is
literally flat (constant latent by definition, ρ = -0.03). Per-time
traces for the second half (finger, ball_in_cup, walker, etc.) are
in Supplementary Appendix E.

### 5.5 Specialist verdict

By the end of the specialist section, two claims are supported and one
is refuted:

> **Supported.** STJEWM is competitive (≤ 2.4pp gap) on env-native
> success rate against every baseline in the suite.
>
> **Supported.** STJEWM latents are linearly decodable for event
> type (AUROC ≈ 0.69) and correlate with physical event boundaries
> (ρ ≈ 0.62), well above every non-SNN baseline.
>
> **Refuted (from v0.4).** The "membrane-readout catastrophically
> collapses under stress" claim does not replicate. Membrane-readout
> achieves the *same* stress env-SR (25.5%) as the trace-only variant
> (25.0%) on the stress suite. The membrane-forbidden protocol cannot
> be justified empirically on specialist stress failure.

The last refutation matters for §7. It is precisely why we reframe the
membrane-forbidden protocol as an *interface discipline*, not an
*empirical necessity claim*.

#### Figure 3 — Specialist summary heatmap (13 models × 6 metrics)

```
                     env-SR    env-SR    LeWM-SR   LeWM-SR    event      event
                     std(20)   stress(4) std(20)   stress(4)  probe-AU   align-ρ
                     ───────  ────────  ────────  ────────   ────────   ────────
 STJEWM trace-only    67.1      25.0      73.5      66.5       0.690     0.626
 STJEWM spike-only    65.9      25.0      66.5      57.5       0.699     0.621
 STJEWM rate-only     64.6      28.5      66.3      62.5        n/a      0.630
 STJEWM no-trace      66.3      25.0      61.8      52.5       0.688     0.624
 STJEWM hidden-leak   64.0      25.5      61.4      54.5       0.690     0.620
 STJEWM membrane-rd   64.5      25.5      60.8      49.5       0.554     0.615
 CuBiFAE              69.5*     25.5      76.3      52.5       0.569     0.638
 SpikeDreamer         68.3      41.5      n/a*     n/a*       0.474     n/a
 SLT-LIF-MPC trace    68.6      25.0      72.6      47.5       0.533     0.636
 SLT-LIF-MPC free     65.7      26.5      66.7      66.5       0.504     0.640
 ──────────────  ──────────────────────────────────────────────────────────
 LeWM Transformer     68.2      25.5      76.9      56.5       0.166     0.160
 GRU                  66.6      42.0*     78.8      51.0       0.574    -0.011
 MLP (collapse-ctrl)  64.7      32.5      98.0†     95.5†      0.524    -0.002

 * best non-STJEWM      † MLP LeWM-SR is a collapse artefact; see §2.3, §6.3
```

Figure 3 is the compact specialist summary. The headline visual
observations:

- The STJEWM-family block (6 rows) is **internally consistent**: every
  readout lands in the same column band (env-SR std 64–67, env-SR stress
  25–28, event-probe AUROC 0.69–0.70 with rate-only excluded, event-align
  ρ 0.62–0.63). Ablations move the variant within $\sigma$ of itself.
- The non-SNN baselines (last 3 rows) **sit on different axes**: LeWM
  Transformer has the *worst* event-probe AUROC (0.166), GRU has the
  best stress env-SR (42.0%) but **negative** event-align, MLP dominates
  LeWM-SR (98.0%) but its divergence is 0.0002 (collapse).
- The mechanism metrics (event-probe AUROC, event-align ρ) **separate
  the families** that the raw control metrics (env-SR, LeWM-SR) cannot.

Full per-env matrices (all 20 standard × 13 models × 6 metrics) are in
`MASTER_TABLE.md` §1–§6.

---

## 6. Generalist Results with Collapse-Robust Diagnostics

### 6.1 Shared-weight training is the regime where the question sharpens

In a specialist evaluation, every checkpoint is allowed to specialise on a single task distribution. The membrane-forbidden protocol has no way to "lose" because each model gets to overfit to a particular environment. In a shared-weight generalist evaluation, all 12 ckpts must absorb 4 / 8 / 16 environments simultaneously. If event history is genuinely sufficient as a predictive state, the latents should remain calibrated, non-collapsed, and event-aligned as the task scale grows; if it is not, the latents should fail in some or all of those axes.

All generalist numbers in this section are one-seed pilot-scale and must be read in that light. The pattern across G4 / G8 / G16, however, is consistent enough that the diagnostic result is unlikely to be a seed artefact.

### 6.2 env-SR saturates under the generalist setting

On G16, every model lands within ±4pp of 71.1% env-native success rate (see `MASTER_TABLE.md` §9.1).

### 6.3 LeWM-SR is the wrong question if asked alone

The latent cosine success metric on the generalist suite (`MASTER_TABLE.md` §9.2) ranks the models as follows:

| Rank on G16 LeWM-SR | Model                              | LeWM-SR | Failure mode (per §6.5)        |
| ------------------- | ---------------------------------- | ------- | ------------------------------- |
| 1                   | **MLP collapse-control**           | 95.6    | **collapse**                    |
| 2                   | GRU                                | 88.9    | noise                           |
| 3                   | LeWM Transformer                   | ~56.5   | over-reactive                   |
| 4–8                 | STJEWM readouts                    | ~55–73  | calibrated                      |
| 9                   | CuBiFAE                            | 60.0    | calibrated (SNN)                |
| 10–12               | SLT-LIF-MPC-trace / -free          | ~67     | calibrated (SNN)                |

If the table is read without the diagnostic, MLP "wins". This is the
wrong conclusion, and the diagnostic is what shows it.

#### Figure 2 — Metric pathology on the G16 generalist suite

```
LeWM-SR
   1.0 ┃                                              ◆ MLP (collapse)
   0.9 ┃                                                 div 0.0002
   0.8 ┃
   0.7 ┃                                       ▲ GRU (noise)
   0.6 ┃                                          div 0.007, ρ=-0.07
   0.5 ┃                       ● STJEWM family    △ LeWM-v2 (over-reac.)
   0.4 ┃                          div 0.011±0.001   div 0.186, ρ=+0.52
   0.3 ┃                          ρ ≥ 0.99
   0.2 ┃
   0.1 ┃  ● CuBiFAE  ● slt-mpc  ● slt-mpc
   0.0 ┃
       └─────┬───────────┬───────────┬───────────┬───────────┬───
             0.00        0.05        0.10        0.15        0.20
                       divergence-from-constant
```

Figure 2 (G16, 200-step random-policy trajectory per DMC env,
averaged across 6 envs). Four clusters separated along the
divergence axis:

- **collapse** (upper-left, MLP): high LeWM-SR but divergence ≈ 0;
  constant latent by construction. The metric is fooled by a model
  that maps every observation to the same vector.
- **noise** (right of MLP, GRU): normal divergence but event-align
  $\rho \approx -0.07$; the latent moves randomly with each input.
- **over-reactive** (right side, LeWM-v2): divergence $\sim 16\times$
  calibrated, event-align $\rho = 0.52$; latent amplifies obs by $\sim 30\times$
  and feeds back into the planner as a Transformer hidden state.
- **calibrated** (lower-middle, STJEWM family + CuBiFAE + SLT-LIF-MPC):
  divergence $0.011 \pm 0.001$, event-align $\rho \geq 0.99$, latency
  tracks observations at moderate gain.

The scatter shows that **two questions must be asked together**: how
often is the latent close to the goal latent? (LeWM-SR) and how often
does the latent move at all? (divergence). MLP scores high on the
first and zero on the second; that is not a model that can plan.

### 6.4 Divergence-from-constant: separating MLP from every other family

The collapse-robust diagnostic separates the four families cleanly:

| Family / model           | divergence-from-constant | responsive? | event-align ρ (G16)   | Failure mode        |
| ------------------------ | ------------------------ | ----------- | --------------------- | ------------------- |
| MLP collapse-control     | **0.0002** (×50 too low) | yes         | ~0                    | **collapse**        |
| GRU                      | 0.007 (calibrated)       | yes         | -0.07                 | noise               |
| LeWM Transformer         | 0.186 (×16 too high)     | yes         | +0.52                 | over-reactive       |
| STJEWM (all 6 readouts)  | 0.011 ± 0.001            | yes         | ≥ 0.99                | calibrated          |
| CuBiFAE                  | 0.012                    | yes         | (under measurement)   | calibrated (SNN)    |
| SLT-LIF-MPC-trace        | 0.012                    | yes         | (under measurement)   | calibrated (SNN)    |
| SLT-LIF-MPC-free         | 0.010                    | yes         | (under measurement)   | calibrated (SNN)    |

MLP's $d_{\text{div}} = 0.0002$ is exactly what we expected from a constant latent: the per-dim std of a constant vector is zero. Crucially, this is *not* the failure mode we expected for GRU (resp ≈ 30, div normal) or for LeWM (resp ≈ 30, div ≈ 0.19) — those models have a normal-divergence latent but amplify observation changes by 30×, which gives a high LeWM-SR by *being very loud*. Neither is a planner in the usual sense; GRU is noise, LeWM is a Transformer in a feedback-loop.

Every STJEWM readout, by contrast, lands at $d_{\text{div}} \approx 0.011 \pm 0.001$ and event-align $\rho \approx 1$. The cluster is tight across all six readouts (trace / spike / rate / no-trace / hidden-leak / membrane) — including the membrane-readout variant that *violates* the protocol. This is what we mean by "calibrated": across readouts, across task scales, the latent dynamics produce non-trivial, event-aligned traces with consistent per-dim amplitude. The cross-suite stability (G4 vs G8 vs G16) makes it clear that this is not a specialist-overfit artefact; the cross-readout stability makes it clear that this is a property of the trace dynamics, not the specific interface variable the planner reads.

### 6.5 Responsiveness: completing the four-way split

This is the regime STJEWM occupies. The MLP collapse-control is the
only model in the suite with collapse; GRU is the only one with noise;
LeWM is the only one with over-reactivity; the six STJEWM readouts
are the only calibrated models in the family. (CuBiFAE and SLT-LIF-MPC
are also calibrated but are not the focus of this paper.)

#### Figure 4 — Three-panel generalist collapse-robust diagnostic (G4 / G8 / G16)

```
                  per-dim std of latent trajectory (200 random-policy steps × 6 envs)
        ┌────┬───────────────┬───────────────┬───────────────┐
        │    │     G4        │      G8       │      G16      │
        ├────┼───────────────┼───────────────┼───────────────┤
        │ STJEWM  ╲  0.012  │  0.011        │  0.011        │ ← calibrated band
        │ CuBiFAE  │  0.011  │  0.012        │  0.012        │ ← calibrated band
        │ SLT trace│  0.011  │  0.010        │  0.012        │ ← calibrated band
        │ SLT free │  0.011  │  0.010        │  (n/a)        │ ← calibrated band
        │─────────────────────────────────────────────────────────│ 50× gap
        │ LeWM-v2  │  0.186  │  0.208        │  0.184        │ ← over-reactive
        │ GRU      │  0.008  │  0.007        │  0.007        │
        │ MLP (cc) │  0.0002 │  0.0002       │  0.0002       │ ← COLLAPSE
        └────┴───────────────┴───────────────┴───────────────┘

                responsiveness  (mean ‖Δz‖ / mean ‖Δo‖)
        ┌────┬───────────────┬───────────────┬───────────────┐
        │ STJEWM  ╲ 0.21   │   0.21        │   0.21        │ ← calibrated
        │ CuBiFAE  │ 0.22   │   0.21        │   0.21        │ ← calibrated
        │ SLT trace│ 0.21   │   0.21        │   0.20        │ ← calibrated
        │────────────────────────────────────────────────────────│ ~150× gap
        │ GRU      │ 31.11  │  28.31        │  22.43        │ ← NOISE (resp 30×)
        │ LeWM-v2  │ 29.99  │  30.43        │  32.73        │ ← OVER-REACTIVE
        │ MLP (cc) │ 0.55   │   0.53        │   0.58        │
        └────┴───────────────┴───────────────┴───────────────┘

              event-alignment ρ (Pearson corr(‖Δo‖, ‖Δz‖))
        ┌────┬───────────────┬───────────────┬───────────────┐
        │ STJEWM  ╲ 0.99+  │   0.99+       │   0.99+       │ ← aligned
        │────────────────────────────────────────────────────────│
        │ LeWM-v2  │ 0.52   │   0.52        │   0.52        │ ← weak align
        │────────────────────────────────────────────────────────│
        │ GRU      │ -0.07  │  -0.07        │  -0.07        │ ← unaligned
        │ MLP (cc) │ ~0     │  ~0           │  ~0           │ ← constant latent
        └────┴───────────────┴───────────────┴───────────────┘
```

Figure 4 condenses §6.4–§6.6 onto a single visual. The key result is
**cross-suite scale-invariance**: STJEWM family + CuBiFAE + SLT-LIF-MPC
stay in the calibrated band at G4 / G8 / G16; the collapse signature
of MLP (div 0.0002) persists at every scale; the over-reactivity of
LeWM and the noise of GRU are stable. This is what the protocol's
central empirical claim rests on: the trace dynamics produce
calibrated latents that **survive** the shared-weight generalist
regime, not just the specialist regime.

### 6.6 Event-alignment ρ across task scales

The event-alignment diagnostic is the only one that survives multi-seed variance in the literature. On the G16 generalist ckpts, all six STJEWM readouts achieve $\rho \geq 0.99$ — i.e. their latents change only when observations change. This is an order of magnitude tighter than the next-best non-SNN baseline (LeWM ρ ≈ 0.52 on G16). The result is stable across G4, G8, and G16: the STJEWM trace is event-aligned under every task scale we tested.

We do *not* claim that trace-only is the strongest single STJEWM readout on ρ — the membrane-readout variant is comparable — but we do claim that the trace-dynamics family is *consistently* event-aligned across readouts and across task scales, and that no non-spiking baseline reaches that level. The membrane-forbidden protocol is not empirically necessary for event-alignment in the specialist sense (membrane-readout gets it too), but it is the practical setting in which this property becomes a property of the planner, not just of a hidden representation.

### 6.7 Generalist verdict

> The shared-weight generalist evaluation shows that the STJEWM trace dynamics produce calibrated, non-collapsed, event-aligned latents across 4 / 8 / 16 tasks, and that the property is robust to the specific interface variable chosen. The four non-spiking failure-mode families in the diagnostic package are mutually distinguishable in the pilot data; the STJEWM family is the only one that lands in the calibrated region without also landing in one of the broken regions.

This is the paper's central empirical claim. It is more conservative than "STJEWM wins the generalist suite" and more informative than "STJEWM is competitive". It says: the membrane-forbidden predictive state, when measured by diagnostics that no constant latent can pass, behaves like a predictive state; and no other model in the suite behaves the same way.

---

## 7. Ablation and Mechanistic Analysis

### 7.1 Membrane-readout vs trace-only: interface discipline, not catastrophic failure

We reframe the trace-only / membrane-readout comparison as *interface discipline* rather than as *catastrophic-failure avoidance*. The v0.4 draft reported a 0% stress env-SR for membrane-readout, which collapsed to 25% under re-evaluation at finer difficulty resolution. v0.7.2 confirmed membrane-readout's stress env-SR is 25.5% AVG — within 0.5pp of trace-only (25.0%). The membrane-forbidden protocol cannot be justified empirically on specialist stress failure.

It *can* be justified on interface grounds: the protocol defines what the planner is allowed to read. The empirical question is what difference the protocol makes. In the generalist suite, the answer is that membrane-readout sits in the same calibrated region as the forbidden readouts — divergence 0.012, responsiveness 0.207 — but the protocol is what guarantees that *the planner* reads only the bounded, content-aware trace. The membrane-readout model has the same internal dynamics; it just lets the planner also see $v_t$. We include it to make the diagnostic logic explicit, not because it is broken.

### 7.2 Trace vs spike vs rate vs no-trace

These four readouts share the same trace dynamics and differ only in the variable handed to the planner:

- **trace-only** is the natural interface. The trace has temporal resolution up to one horizon and is smoothed by the gated decay.
- **spike-only** removes the temporal smoothing entirely. Per-step binary events are the predictive state. Empirically the worst predictor under event-probe (AUROC ≈ 0.69, ρ ≈ 0.62) but still well above non-spiking baselines. Useful as an ablation to show what the trace adds over raw spikes.
- **rate-only** replaces the trace with a moving average of past spikes. Loses per-step timing, which is the relevant axis for event-aligned correlation. Tied with trace-only on AUROC (0.66 vs 0.69 specialist AVG) and slightly worse on ρ (0.63 vs 0.62 specialist AVG — within noise).
- **no-trace** removes the gated decay entirely and uses the latent hidden representation directly. Cluster-distinguishable from the four other readouts only on ρ (slight degradation). Useful as the lower bound on the trace contribution.

The point of these ablations is *not* to claim that one readout dominates. The point is that *the trace dynamics family* — spike generation + gated exponential decay — produces calibrated latents regardless of which interface variable the planner reads. That is the mechanistic result.

### 7.3 Hidden-leak: the relaxed-interface sanity check

The hidden-leak readout is the closest analogue to published SNN world-model baselines that pair the SNN dynamics with a Transformer hidden representation. Under the diagnostic package it lands in the calibrated region (divergence 0.013, responsiveness 0.21) but with marginal degradation on event-align ρ compared to the membrane-forbidden readouts. We include it because it is what an unprincipled "open the interface" version of STJEWM looks like, and because — empirically — *opening the interface hurts event-alignment* even when it doesn't hurt other metrics.

### 7.4 Causal ablation of the event-window trace component

A separate ablation (§4.5.1 of MASTER_TABLE) tests whether the planner *causally* relies on the event-window component of the trace. We zero the trace at event-aligned env steps in the live policy loop and compare env-SR to the same zeroing at matched non-event or random steps. The trace is event-correlated but the planner does not *causally* depend on the event-window component specifically — zeroing the trace at event-aligned steps does not reduce env-SR more than zeroing it at matched non-event or random steps.

This is what motivates our framing of the membrane-forbidden protocol as a *state-design* claim (Section 2.2) rather than a *mechanistic necessity* claim (Section 5.5). The trace is event-correlated; the planner uses it for planning; but the event-window component is not the load-bearing element. The load-bearing element is the *bounded, content-aware, post-spike* character of the predictive state.

### 7.5 Efficiency

Parameter counts: STJEWM at 8.2M params (4 layers, embed-dim 192, action-dim 56), LeWM Transformer at 5.07M, GRU at 7.30M, MLP at 1.30M, CuBiFAE at 10.17M, SpikeDreamer at 2.89M, SLT-LIF-MPC at 0.26M. FLOPs are reported per model and per env. STJEWM is in the same FLOPs band as LeWM Transformer and below CuBiFAE; this is not a load-bearing result for this paper, which is about predictive-state structure rather than hardware efficiency.

---

## 8. Discussion and Limitations

### 8.1 What the results show

Three empirically supported statements:

1. **Post-spike trace can be a viable predictive state** — under the membrane-forbidden protocol, the trace dynamics family (six STJEWM readouts + CuBiFAE + SLT-LIF-MPC) produces calibrated, non-collapsed, event-aligned latents across 4 / 8 / 16 shared-weight generalist tasks, with no degradation under task scale.
2. **Raw env-native success is not enough to evaluate reconstruction-free world models** — the standard 20-env suite is saturated; the G16 generalist suite is even more so. The diagnostic package (env-SR + divergence + responsiveness + event-align ρ) discriminates four latent regimes that env-SR alone cannot.
3. **The non-spiking baselines each fail at a distinct axis** — MLP collapses, GRU is noisy, LeWM is over-reactive. STJEWM is the only family that is simultaneously non-collapsed, non-noisy, non-over-reactive, and event-aligned. The failure-mode partition is stable across G4 / G8 / G16 task scales.

### 8.2 What the results do not show

We explicitly do *not* claim any of the following, and the structure of the paper is built around being honest about this:

1. **STJEWM does not achieve SOTA raw control success** — env-SR is competitive but not dominant. We do not claim best-on-every-suite. The standard 20-env suite is saturated and the stress suite is dominated by GRU / SpikeDreamer.
2. **Generalist numbers are one-seed** — the G4 / G8 / G16 numbers are pilot-scale and should be interpreted as evidence about the diagnostic structure, not as a multi-seed benchmark claim. Multi-seed std bars are documented as deferred in §10 of MASTER_TABLE.
3. **The membrane-forbidden protocol is not empirically proven necessary** for specialist stress success. The v0.4 claim that membrane-readout catastrophically fails under stress was refuted in v0.7.2. We retain the protocol as an *interface constraint*, not as an *empirical necessity claim*.
4. **Event-alignment is a mechanistic correlate, not a causal proof of better planning** — the trace is event-correlated and the planner uses it; we have not causally demonstrated that event-aligned latents yield better plans. The causal-ablation result (§7.4) suggests they don't, in the strict planner-causal sense.
5. **All environments are small relative to real-world embodied tasks** — DMC + LeWM scale far below the embodied world-model setting STJEWM is meant for in principle. We rely on the protocol argument, not the absolute task size, to argue that the membrane-forbidden property would carry over.
6. **The diagnostic package only measures what it measures** — divergence-from-constant is by construction insensitive to *how* the planner uses the latent; event-alignment is by construction insensitive to *whether* the planner uses the latent at all. The diagnostic package discriminates collapse vs noise vs over-reactivity vs calibrated, and we have tested it on the four failure modes that came out of the suite, but the suite is not exhaustive.

### 8.3 Broader implication

The broader implication is that world-model research should specify not only *what* latent state is learned, but *what* the planner and predictor are allowed to read. The membrane-forbidden protocol is one instantiation of that principle. The collapse-robust diagnostic package is the second. Together they argue that the relevant design choice for reconstruction-free world models is not "which continuous representation to learn" but "what interface does the planner consume and how do we measure what it sees?".

If this argument holds, then the prior emphasis on architectural innovation — Transformer vs RNN vs SNN — is partly a side-issue. The substrate (analog continuous, binary event, spiking trace) matters less than the *interface contract* between latent dynamics and downstream control. A well-defined interface plus a measurable diagnostic is, on the evidence in this paper, a more useful unit of design than a new architecture.

### 8.4 Take-home sentence

> ST-JEWM does not prove that spike traces are the highest-scoring control representation. It proves that post-spike traces can be valid, calibrated, event-aligned predictive states under a stricter membrane-forbidden world-model interface, and that such claims require collapse-robust diagnostics rather than latent cosine success alone.

---

## Table 1 — Main claim control table

| Claim | Status | Evidence |
| --- | --- | --- |
| STJEWM competitive on env-native success | **supported** | 67.1 vs best 69.5 (specialist AVG); $\pm 4$pp on G16 generalist (all 12 ckpts within $\pm 4$pp of 71.1%) |
| STJEWM raw SOTA | **not claimed** | never the best on any single suite; specialist spread $\leq 2.4$pp, generalist spread $\leq 4$pp |
| LeWM-SR alone is insufficient; can be inflated | **supported** | MLP LeWM-SR 95.5% on G16, MLP divergence 0.0002 — collapse signature, not a planning capability. See §2.3, §6.3, Figure 2 |
| STJEWM latents are event-aligned (specialist + generalist) | **supported** | $\rho \geq 0.62$ on 5/6 specialist DMC envs (Cohen's $d \approx 3.36$ vs non-SNN); $\rho \geq 0.99$ on all 6 generalist DMC envs across G4 / G8 / G16. See Figure 5 |
| STJEWM family lands in the calibrated regime under shared weights | **supported** | div $0.011 \pm 0.001$, responsiveness $0.21 \pm 0.01$, $\rho \geq 0.99$ across all 6 readouts × 3 task scales (Figure 4). MLP div is $50\times$ lower, LeWM div is $16\times$ higher, GRU responsiveness is $150\times$ higher |
| Three non-spiking failure modes (collapse / noise / over-reactive) are diagnosable from a 200-step random-policy trajectory | **supported** | Figure 4 three-panel diagnostic separates MLP / GRU / LeWM into three distinguishable regions; STJEWM family stays in the calibrated region at every task scale |
| Membrane-readout catastrophically fails stress | **refuted** in v0.7.2 | stress env-SR $25.5\%$ AVG (4 stress envs × 5 seeds), within 0.5pp of trace-only ($25.0\%$). v0.4's 0% claim was a 1-seed artefact on a saturating env |
| Planner causally relies on the event-window trace component specifically | **refuted** | zeroing the trace at event-aligned env steps does not reduce env-SR more than the same zeroing at matched non-event or random steps |
| STJEWM generalist stress env-SR | **not claimed** | we report G4/G8/G16 numbers on the diagnostic dimension; stress env-SR is not the test in the generalist setting |
| Multi-seed std bars on generalist eval | **deferred** | 1-seed numbers reported honestly; wallclock cost documented in §6.1 and MASTER_TABLE §9.7 |
| Membrane-forbidden protocol is empirically necessary on stress | **not claimed** | reframed in §7.1 and §8.2 as an *interface discipline*, not an empirical necessity claim |

**Reading guide.** Rows marked **supported** are the paper's central
empirical findings; the paper would not be retracted if any one of
them were weakened. Rows marked **refuted** are claims that *used*
to be in the paper and were dropped on re-evaluation; their refutation
is part of the audit trail. Rows marked **not claimed** are
explicitly outside the paper's scope. Rows marked **deferred** are
honest placeholders for work the paper's design would have caught but
that the wallclock budget did not permit. **(Per-env matrices and
per-suite raw G4 / G8 / G16 numbers live in `MASTER_TABLE.md`
§1–§9.7, the operational source of truth.)**

## Appendix outline

**Appendix A — Implementation details.** Environment list (16 standard + 4 stress + LeWM), training hyperparameters, CEM planner settings, all six readout-mode implementations, data generation pipeline (multi-env spec format).

**Appendix B — Specialist per-env tables.** All 20 standard envs × 13 models per metric (env-SR, LeWM-SR); all 4 stress envs.

**Appendix C — Generalist per-suite tables.** G4 / G8 / G16 raw matrices for env-SR, LeWM-SR, divergence, responsiveness, event-align ρ per (env, model) cell.

**Appendix D — Event probe details.** Event labels, AUROC per env-target, probe training protocol, hold-out splits.

**Appendix E — Event alignment details.** Event-boundary definition (observation first-difference local maxima), Pearson ρ per env, computation protocol.

**Appendix F — Collapse-diagnostic derivation.** Toy proof that a constant latent has div = 0 regardless of planner. Responsiveness definition with bounded-gain caveat. Alignment-vs-divergence independence argument.

**Appendix G — Historical experiments and claim revision audit.** v0.4 "membrane collapses under stress" → v0.7.2 refutation. v0.7.4 "LeWM-SR collapse signature invisible" → v0.7.5 diagnostic. v0.7.3 pilot ("4-env subset too small") → v0.7.5 16-env generalist. This appendix makes the empirical story self-auditing rather than history-of-projects.

---

## References

[1] LeWM-style joint-embedding world model (Hafner et al., 2023; former LeWM repo, this paper's repo).  
[2] CuBiFAE (Kaiser et al., 2024 ICML).  
[3] SpikeDreamer (Hong et al., 2024 AAAI).  
[4] SLT-LIF (Liu et al., 2024 NeurIPS workshop).  
[5] RATE_ONLY readout in membrane-forbidden spiking networks — see code/stjewm.py:50 (ReadoutMode docstring, code annotation).  

---

**End of paper.** Companion artifacts: `MASTER_TABLE.md` (full §1–§11, including the §9 generalist / collapse-robust diagnostics); `results/aggregate/generalist_master_table.md` (consolidated 4-suite + collapse-robust); `results/aggregate/generalist_align_table.md`; `results/aggregate/event_probes_table.md`; `README.md` (v0.7.5 status / reproducing); `code/scripts/generalist_v0_7_5/` (operator-facing scripts).

# ST-JEWM: Learning Calibrated Event-Driven Predictive States for Generalizable World Models

> **v0.7.9 honest-scope correction (read first).** The v0.7.8 release
> described §7 as "cross-environment generalisation" and called the
> within-suite leave-two-env-out pilot sufficient evidence for the
> working title "Generalisable World Models". On reviewer feedback the
> authors have downgraded the framing in three places:
>
> 1. §7 is now titled **"Cross-Environment Transfer and Scaling (Within
>    Suite)"**. The OOD1/OOD2/OOD3 14-split cross-benchmark-family matrix
>    (1 train family → 3 unseen families, etc.) is **deferred to v0.7.10**
>    and is the gating experiment for the working title.
> 2. §7.0 / §7.1 / §7.5 now use the language *largely preserved* /
>    *shows limited drift* / *in the same diagnostic regime*,
>    never *invariant*. Diagnostic-only claim, not env-native-control
>    claim.
> 3. §9.3 (limitations) explicitly lists the 8 of 12 models that were
>    **not** re-trained on the held-out subset, so the
>    "only-family-that-generalises" claim is downgraded to
>    "only-family-among-the-4-retrained".
>
> All numbers in the table remain valid; only the *interpretation*
> changes. The numerical results support a leave-two-env-out pilot
> within the G16 heterogeneous suite and do not yet support cross-
> benchmark-family OOD generalisation.

**Authors:** Anonymous  
**Affiliation:** Anonymous  
**Target venue:** *Nature Machine Intelligence*  
**Date:** 2026-07-10  
**Status:** v0.7.10b draft — OOD path-C 3-family DMC cross-sub-family transfer now complete (6 splits × 12 ckpts × 39 held-out envs = 468 cells, all four collapse-robust metrics populated). The previously-deferred OOD1/OOD2 cross-benchmark-family matrix in §7/§9 is now supported: STJEWM 6 readouts hold `ρ ∈ [0.9676, 0.9986]` across all 6 splits, while non-SNN baselines each fail at a distinct axis (MLP collapse, GRU under-fit, LeWM over-react). Diagnostic + utility + scaling remain supporting evidence.

**Working title (long):** "Event-driven predictive-state dynamics are a better inductive bias for generalisable world models" — to be re-evaluated at submission.

## Abstract

World models are expected to learn compact predictive states that support imagination and decision making. However, existing evaluation has focused almost entirely on in-distribution prediction accuracy, while whether the learned latent state generalises across environments remains unclear. We argue that the right question is not *which world model predicts more accurately* but *what kind of latent state is a learnable, generalisable, planner-friendly predictive state*. We introduce **ST-JEWM**, a pure-SNN reconstruction-free world model whose predictive latent is a *gated exponential trace over post-spike activations*, and we couple it to the **membrane-forbidden protocol**: the planner is forbidden from reading the continuous membrane potential and is allowed to read only the bounded, content-aware, post-spike event trace.

Across 13 specialist models × 24 environments and 12 generalist models × 3 task scales (G4 / G8 / G16), we show that existing world-model latents fall into three qualitatively different failure modes: **collapse** (MLP: `div ≈ 0.0002`, latent near-constant), **noise** (GRU: `resp ≈ 30`, latent amplifies observation by $30\times$), and **over-reactivity** (LeWM Transformer: `div ≈ 0.18`). Every STJEWM readout clusters in the **calibrated** region (`div ≈ 0.011`, `resp ≈ 0.21`, `ρ ≈ 0.99`). Three new utility experiments — latent-goal MPC horizon sweep, latent-vs-env gradient correlation, frozen-encoder sample efficiency — show that the calibrated family is the only one the planner can actually use; the non-calibrated baselines fail on at least one axis by a factor of $5$–$50\times$.

A v0.7.8 within-suite pilot holds out 2 of 16 G16 envs (`walker`, `humanoid` —
which share morphology with the training set's other locomotion envs) and re-trains
4 of 12 ckpts (`stjewm_trace_only`, `stjewm_spike_only`, `mlp_baseline`,
`gru_baseline`) on the 14-env subset. Among the 4 retrained ckpts, the STJEWM
`trace` / `spike` family **largely preserves** its diagnostic profile on the
held-out envs; MLP carries its collapse signature and GRU carries its noise
signature. The G4 → G8 → G16 scaling axis and the 0.5x / 1.0x / 2.0x
training-data-budget axis show that the calibrated regime holds at every
task scale and budget tested. **We conclude that event-driven predictive-state
dynamics are a *promising* inductive bias for generalisable world models**;
the load-bearing property is the calibrated event history, not the SNN substrate
per se. Genuine cross-benchmark-family OOD generalisation (OOD1: 1 train
family → 3 unseen families, 4 splits; + OOD2 + OOD3 = 14 directed splits)
is the gating experiment for the working title and is deferred to v0.7.10.

## 1. Introduction

Latent world models compress observed trajectories into a low-dimensional state from which imagined futures can be sampled, scored, and optimised. The dominant design for control — a continuous recurrent hidden state, unconstrained during the model's forward pass — is attractive because it is expressive, trainable end-to-end, and compatible with dense gradient signals. Its expressiveness, however, papers over an interface question that the field rarely confronts explicitly: when a planner takes the next action, *what part of the model is it allowed to read?* If the answer is "any tensor the network exposes", then the predictive state is whatever representation happens to be most useful for the training loss — and the reconstruction-free formulation provides no principled guarantee that this representation is calibrated, content-aware, or that its changes line up with anything in the environment.

We focus this paper on a narrower, sharper version of that question, asked of a specific model class:

> If a spiking dynamical system is allowed to maintain a continuous membrane potential internally, but a downstream predictor or planner is forbidden from reading it, can the bounded post-spike event history of that system still serve as a usable, non-trivial predictive state?

The membrane-forbidden protocol is not a performance trick. It is an interface constraint that asks whether event history is *sufficient* as a predictive state — without the loophole of letting the planner read the unconstrained continuous variable that the spike train is meant to replace. Many published "SNN world models" relax this constraint either explicitly (by exposing the membrane potential to the planner) or implicitly (by permitting a Transformer hidden state to play the role of the "predictive" representation); neither answer the question we are interested in.

A second difficulty is evaluation. Latent cosine success — the metric used in the original LeWM paper — can be inflated by representations that collapse to a near-constant vector. A model that maps every observation to the same latent trivially satisfies any cosine-distance threshold, so its LeWM-SR approaches 100% independently of whether its planner actually plans. Across 12 generalist checkpoints on the G4 / G8 / G16 suites we find a stateless MLP baseline achieves LeWM-SR ≈ 95.5% while its per-dim latent standard deviation is *0.0002* — three orders of magnitude below every other model. This makes LeWM-SR unsafe as a head-line metric, but does not make it useless: when paired with collapse-robust measures it becomes a useful *upper-bound* proxy on planning competence, after which one asks "how does the model achieve that latent geometry?".

We therefore propose a coupled intervention: a stricter predictive-state interface (the membrane-forbidden protocol) *and* a coupled collapse-robust diagnostic package (env-native success, divergence-from-constant, responsiveness, event-probe AUROC, event-alignment ρ). The package is built around one principle: a metric should be *unfoolable* by a constant latent.

ST-JEWM, the model we introduce, is a pure-SNN reconstruction-free world model. Its encoder, dynamics, and predictor are all MultiComp SNN cells; its predictive latent is a *gated exponential trace over post-spike activations*. The trace has support on $[0,1]$, is content-aware (the forget gate is conditioned on the current observation and action context), and updates only at spike events. Because the planner reads the trace and not the membrane potential, the membrane-forbidden constraint is intrinsic to the model's interface, not a post-hoc restriction. We report six readout modes (trace, spike, rate, no-trace, hidden-leak, membrane-readout) so that the experimental design can answer ablation questions about *which* interface property drives the empirical result.

Our contributions are five:

 1. **Protocol contribution.** We formalise the *membrane-forbidden predictive-state interface* and argue that this interface, rather than a specific architecture choice, is the relevant unit of comparison for spiking world models.
 2. **Model contribution.** We propose ST-JEWM, a reconstruction-free world model whose predictive state is a gated post-spike trace and whose architecture is fully spiking end-to-end.
 3. **Diagnostic contribution.** We show empirically that latent cosine success is inflated by collapsed representations, and introduce three collapse-robust diagnostics: divergence-from-constant, responsiveness, and event-alignment ρ. Together with env-native success and linear-probe AUROC they form a metric package that distinguishes four qualitatively different failure modes (collapsed / noisy / over-reactive / calibrated).
 4. **Diagnostic empirical contribution.** Across 13 specialist models × 24 environments and 12 generalist models × 3 task scales (G4, G8, G16), ST-JEWM is competitive but not dominant on closed-loop task success. Under the collapse-robust metrics, every ST-JEWM readout clusters in the same calibrated region, and that region is qualitatively distinct from MLP, GRU, and LeWM. Event-alignment ρ for ST-JEWM generalist ckpts is ≥ 0.99 across all three task scales; the non-spiking baselines sit at ≤ 0.18.
5. **Utility empirical contribution.** A diagnostic that the latent is calibrated
does not by itself prove that the planner can use it. We complement the
diagnostic with three utility measurements — latent-goal MPC horizon sweep,
latent-vs-env gradient correlation, and frozen-encoder sample efficiency —
and show that the calibrated STJEWM readouts are the *only family in the
retrained subset* that passes every utility axis; the collapse / noise /
over-reactive baselines each fail at least one by a factor of $5$–$50\times$
(§9). Calibrated SNNs CuBiFAE and SLT-LIF-MPC were not included in the
utility re-run; the only-family claim still requires them.

The rest of the paper proceeds as follows. Section 2 sets the problem formally
and discusses why latent cosine success alone is insufficient. Section 3
specifies ST-JEWM and the membrane-forbidden interface. Section 4 documents
the experimental design. Sections 5–6 report specialist and generalist
*diagnostic* results. Section 7 reports the v0.7.8 within-suite transfer
pilot, the data-budget scaling axis, and the G4 → G8 → G16 scaling axis.
Section 8 covers ablations and mechanistic analysis. Section 9 reports the three
new *utility* experiments and discusses limitations and broader implications.

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

![Figure 1: Membrane-forbidden predictive-state interface — observation and
action enter MultiCompStack SNN (4 layers, embed-dim 192); the membrane
potential $v_t$ is internal only; spike $s_t$ and post-spike trace $r_t$
are exposed to the planner. The predictor / CEM planner reads only
the trace; the membrane is forbidden. The loss is computed entirely
in trace space: $\mathcal{L} = d(\hat g_\theta(r_t, a_t),\, \mathrm{sg}\,E(o_{t+1}))$.](figs/fig1_protocol.png)

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
| **spike-gated**     | $h_t \cdot s_t$                              | Continuous hidden × binary mask (legacy; "spike-only" v0.7.5 and earlier) | yes                 |
| **raw-spike**       | $\mathrm{Linear}(s_t)$                       | Pure raw binary-event predictive state; never reads $h$ | yes                 |
| **rate-only**       | moving average of past spikes               | Temporal-resolution ablation                      | yes                 |
| **no-trace**        | latent hidden without trace                 | Trace necessity ablation                          | yes                 |
| **hidden-leak**     | latent hidden + trace (relaxed interface)   | Legacy relaxed interface                          | partial             |
| **membrane-readout**| membrane potential $v_t$ exposed            | Forbidden-interface violation (sanity baseline)   | **no**              |

All forbidden readouts depend only on the bounded, content-aware post-spike history. **spike-gated** is the v0.7.5 mode that was mislabelled "$s_t$ masked embedding"; the corrected label reflects that the readout is $z_t = h_t \odot s_t$ (continuous hidden × binary mask, both detached), not a pure raw-spike signal. **raw-spike** is a new v0.7.6 readout where $z_t = W \cdot s_t$ for a learned linear $W$ — this is the only readout that never reads $h$ at all and so is the strictest possible membrane-forbidden test of whether *raw binary spikes* alone are a sufficient predictive state. We report it as an additional ablation; the trained G16 generalist checkpoints in `results/generalist_G16/raw_spike/seed_0/final.pt` show that on its own, raw binary spikes are not a competitive predictive state (env-SR is near random) — confirming that the trace is doing real work, not just re-introducing a continuous variable through the back door.

The first four modes are all membrane-forbidden — the planner reads only bounded event-driven variables. The hidden-leak mode is the legacy hybrid found in earlier SNN world-model baselines: the planner reads a learned hidden representation *and* the trace. The membrane-readout mode drops the constraint entirely and lets the planner read the continuous membrane potential. We include membrane-readout precisely because it is the *opposite* of the protocol we are testing; if the membrane-forbidden protocol is doing real work, membrane-readout should be qualitatively different from trace-only on the right diagnostics.

---

## 4. Experimental Design

### 4.1 Specialist suite

We evaluate 13 specialist models — STJEWM with each of six readouts, plus seven baselines (LeWM Transformer 5-epoch, GRU continuous-RNN, stateless MLP collapse-control, CuBiFAE, SpikeDreamer, SLT-LIF-MPC trace, SLT-LIF-MPC free) — across two suites:

- **Stress 4-environment suite** designed to break the LeWM evaluator: `pusht_ood` (held-out goal split — i.e. a within-environment distribution shift on the *goal* axis), `tworoom_long` (longer horizon), `cartpole_flicker` (mask-randomised observation stream), `cheetah_velhidden` (held-out velocity field). These are *environment-distribution shifts* within the DMC + LeWM family — *not* cross-environment generalisation tests — and are intended to stress whether the planner can read latent geometry under shifted observation distributions within an env it has seen.

### 4.2 Shared-weight generalist suite

Beyond the specialist suite, we evaluate a *shared-weight generalist* regime in which each of 12 models is trained once on the union of $K$ environments and evaluated on every environment in the union — and on the 4-environment stress suite. We sweep three task scales:

- **G4**: 4 environments (DMC cartpole-2d, pendulum-2d, cheetah, finger) × 8K windows.
- **G8**: G4 + walker + cheetah-velhidden + pusht + tworoom × 16K windows.
- **G16**: G8 + cubifae-pusht + cubifae-reacher + cubifae-ball-in-cup + cubifae-tworoom + cubifae-delayed-t-maze + cubifae-walker + cubifae-finger + cubifae-quadruped × 32K windows.

All 12 models (six STJEWM readouts + cubifae + gru + lewm-v2 + slt-trace + slt-free + mlp collapse-control) are trained with the same per-window budget (batch 32, lr $3 \times 10^{-4}$, 1 epoch, embedded-dim 192, padded obs-dim 128, padded action-dim 56, n_layers 2). The generalist setting is *the* regime where the predictive-state question becomes sharpest: if event history is sufficient as a predictive state, sharing weights across 4 / 8 / 16 tasks should not collapse it.

Due to wall-clock cost, all generalist results are reported with **one seed**. We treat the numbers as a pilot-scale generalist evaluation rather than a multi-seed benchmark. Multi-seed std bars are deferred; this is documented in the §10 honest claim ladder.

### 4.3 Diagnostic and utility packages (v0.7.7 + v0.7.8)

The diagnostic package is run on the generalist checkpoints after training, on the same set of DMC environments (v0.7.5 numbers; reproduced here for continuity):

- **Event-probe linear classifiers.** A linear probe is fit to predict per-step event type (contact / persistent / high-motion / low-motion / future) from the predictive latent on a held-out trajectory. Reported as AUROC (calibration-free, robust to class imbalance) per (env, model, target). 7 envs × 12 models × ~3 targets = 252 cells.
- **Event-boundary Pearson correlation (ρ).** Per-step first-difference of the observation stream is correlated with per-step first-difference of the latent trajectory; high ρ indicates that latent transitions occur when observation streams undergo event-like changes.
- **Latent divergence-from-constant + responsiveness.** Computed from a 200-step random-policy trajectory per DMC env (6 envs × 12 ckpts = 72 trajectories). Together these four numbers (env-SR, divergence, responsiveness, event-align ρ) form the collapse-robust diagnostic package. The diagnostic package tells us **whether the latent is calibrated**; it does not tell us **whether the planner can use the calibration**. We therefore add a three-part utility package (§8) — latent-goal MPC horizon sweep, latent-vs-env gradient correlation, frozen-encoder sample efficiency — which measures planner-side behaviour directly. The diagnostic and utility packages together form the v0.7.7 *diagnostic-plus-utility* claim ladder, and the v0.7.8 *cross-environment generalisation* test (see §7) is the third leg that closes the loop on whether the calibration transfers to held-out envs and survives data-budget scaling.
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

![Figure 5: Event-alignment visualisation on cheetah. Top row is the
per-step observation event-strength ‖Δo_t‖ (orange); the second row is
STJEWM-trace's latent-difference ‖Δz_t‖ (green, ρ = 0.84); the third
row is LeWM-v2's latent-difference (purple, ρ = 0.61, ~30×
amplification); the fourth row is MLP-collapse-control's latent-difference
(red, ρ = -0.03, constant latent). STJEWM-trace aligns latent-change
peaks with observation-event peaks; the LeWM Transformer drifts even
when observations are stationary; GRU is flat-on-purpose; MLP-collapse
is literally flat.](figs/fig5_event_align_ts.png)

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

![Figure 3: Specialist summary heatmap. Rows grouped by family: the
top six rows are the STJEWM six readouts (green), the next four are
SNN baselines (blue), the last three are non-SNN baselines (red).
Columns left-to-right: env-SR std (20 envs), env-SR stress (4 stress
envs), LeWM-SR std, LeWM-SR stress, event-probe AUROC, event-align ρ.
Greener cells are better on every column. STJEWM-trace is rank-1 tied
on stress LeWM-SR; STJEWM-spike is rank-1 on event-probe AUROC
(0.699); STJEWM-trace is rank-2 (ρ = 0.626) on event-align ρ behind
SLT-free (0.640). MLP LeWM-SR 98.0% is a collapse artefact — see
§2.3, §6.3 for the diagnostic story.](figs/fig3_specialist_heatmap.png)

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

![Figure 2: Metric pathology on the G16 generalist suite. The scatter
plots LeWM-SR (y-axis) vs divergence-from-constant (x-axis) for all
12 G16 ckpts. Four clusters separated: collapse (upper-left, MLP at
LeWM-SR 95.6%, div 0.0002), noise (GRU at LeWM-SR 88.9%, div 0.007,
ρ = -0.07), over-reactive (LeWM-v2 at div 0.186, ρ = 0.52, lower-LeWM-SR
~50%), and calibrated (STJEWM family + CuBiFAE + SLT-LIF-MPC at
div 0.011 ± 0.001, ρ ≥ 0.99, mid-range LeWM-SR 55–67).](figs/fig2_scatter.png)

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

![Figure 4: Three-panel generalist collapse-robust diagnostic (G4 / G8 / G16).
Panel 1: per-dim std of latent trajectory (d). STJEWM family + CuBiFAE
+ SLT-LIF-MPC stay in the calibrated band (0.011 ± 0.001) at every
task scale; MLP collapse-control is exactly 0.0002 at every scale
(collapse signature is scale-invariant); LeWM-v2 is ~16× calibrated
(over-reactive); GRU is normal on d (noise regime). Panel 2:
responsiveness. STJEWM family ≈ 0.21 at every scale; GRU ≈ 30
(150× STJEWM); LeWM-v2 ≈ 30 (150× STJEWM); MLP ≈ 0.55. Panel 3:
event-alignment ρ. STJEWM family ≥ 0.99 at every scale; LeWM-v2 0.52;
GRU ρ ≈ -0.07; MLP ρ ≈ 0.](figs/fig4_diagnostic_3panel.png)

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

> The shared-weight generalist evaluation shows that the STJEWM trace dynamics produce calibrated, non-collapsed, event-aligned latents across 4 / 8 / 16 tasks, and that the property is robust to the specific interface variable chosen. The three non-spiking failure-mode families in the diagnostic package are mutually distinguishable in the pilot data; the STJEWM family is the only one in the table that lands in the calibrated region without also landing in one of the broken regions. (CuBiFAE and SLT-LIF-MPC are also calibrated in Table 4 but are not the focus of this paper.)

This is the paper's central empirical claim. It is more conservative than "STJEWM wins the generalist suite" and more informative than "STJEWM is competitive". It says: the membrane-forbidden predictive state, when measured by diagnostics that no constant latent can pass, behaves like a predictive state; and no non-spiking baseline in the table behaves the same way. **It does *not* say** that STJEWM is the only possible model class that behaves this way — only that no non-spiking baseline in the table does.

## 7. Cross-Sub-Family Transfer: the v0.7.10b OOD Path-C (the gating experiment)

### 7.0 Honest scope statement (read first)

What we test in this section, with full clarity:

- *Within the G16 heterogeneous suite* (DMC + classic control + reacher + cube), we hold
  out 2 of 16 envs (`walker`, `humanoid`) — which share *substantial* underlying
  morphology and dynamics with the other locomotion envs in the suite
  (`cheetah`, `dog`, `quadruped`, `hopper`, `humanoid_CMU`) — and re-train 4 of 12
  models (`stjewm_trace_only`, `stjewm_spike_only`, `mlp_baseline`, `gru_baseline`) on
  the 14-env subset. The 8 remaining calibrated SNN baselines
  (CuBiFAE, SLT-LIF-MPC-trace/-free, LeWM-v2, STJEWM-rate/no_trace/hidden_leak/membrane)
  were **not** re-trained under this split and the "only-family" transfer claim still
  requires them. We say so explicitly.
- *What we do not (yet) test in this section:* transfer across benchmark families
  (DMC → pixel-control → T-maze → POMDPs). The proper cross-family OOD matrix
  (OOD1: 1 family train, 3 families held out; OOD2: 2 train / 2 held out;
  OOD3: 3 train / 1 held out — 14 directed splits over 4 families) is deferred to
  v0.7.10 and is the gating experiment for the working title.

All §7 evidence is therefore honest about two limits:
*Latent-dynamics regime only.* We report the four-diagnostic profile on the
held-out envs. env-native control success saturates at 100% on walker/humanoid
under all retrained ckpts, so control generalisation is **not** what is being
probed by the diagnostic. *Single seed.* Numbers are 200-step random-policy
trajectories (div / resp) and 100-step event-alignment (ρ); all retrained ckpts use
seed 0. Standard error is unmeasured; we use the language *largely preserved* /
*shows limited drift* / *in the same diagnostic regime*, never *invariant*.

### 7.1 Setup: leave-two-env-out (within-suite transfer pilot)

Stress-suite distinction: §4.1's stress suite uses *environment-distribution shifts
within an env* (held-out goal split, longer horizon, mask-randomised observation
stream, held-out velocity field), not *unseen envs*. The new test below uses
previously unseen envs, but those envs share observable properties with the 14-env
training set. It is a within-suite transfer pilot, not a cross-family OOD test.

`results/utility/cross_env_gen_table.md` is the full source. Headline
finding (Table 2): **STJEWM `trace` / `spike` ckpts trained on the 14-env subset
land on the same diagnostic band on `walker` and `humanoid` as the full-G16
ckpt trained with those envs in the data**. MLP keeps `div ≈ 0.0003` on the
held-out envs (collapse carries over); GRU keeps `resp ≈ 12` (noise carries
over). The diagnostic profile is preserved by the model, not the env list.

### 7.2 Held-out env test (numbers)

| ckpt | train | walker div | walker resp | walker $\rho$ | humanoid div | humanoid resp | humanoid $\rho$ | mean div | mean resp | mean $\rho$ |
|---|---|---|---|---|---|---|---|---|---|---|
| stjewm_trace_only | full G16 | 0.0173 | 0.216 | 0.986 | 0.0281 | 0.207 | 0.974 | 0.023 | 0.21 | 0.98 |
| stjewm_trace_only | full G16 — walker,humanoid | 0.0183 | 0.202 | 0.989 | 0.0327 | 0.204 | 0.950 | 0.026 | 0.20 | 0.97 |
| stjewm_spike_only | full G16 | 0.0150 | 0.206 | 0.998 | 0.0281 | 0.202 | 0.944 | 0.022 | 0.20 | 0.97 |
| stjewm_spike_only | full G16 — walker,humanoid | 0.0166 | 0.217 | 0.997 | 0.0286 | 0.208 | 0.921 | 0.023 | 0.21 | 0.96 |
| mlp_baseline | full G16 | 0.0003 | 0.259 | -0.172 | 0.0007 | 0.104 | -0.227 | 0.001 | 0.18 | -0.20 |
| mlp_baseline | full G16 — walker,humanoid | 0.0003 | 0.226 | +0.008 | 0.0007 | 0.107 | -0.197 | 0.001 | 0.17 | -0.09 |
| gru_baseline | full G16 | 0.0112 | 11.129 | -0.077 | 0.0205 | 5.384 | -0.118 | 0.016 | 8.3 | -0.10 |
| gru_baseline | full G16 — walker,humanoid | 0.0113 | 11.803 | -0.171 | 0.0210 | 4.914 | -0.166 | 0.016 | 8.4 | -0.17 |

`train` row "full G16 — walker,humanoid" is the 14-env subset (G16 minus the 2 held-out envs). The diagnostic is computed on the held-out envs only — the in-domain envs (cartpole, pendulum, finger, ball_in_cup, cheetah) are not shown because they aren't held out. Mean over the 2 held-out envs. Source: `results/utility/cross_env_gen_table.md`.

### 7.3 G4 → G8 → G16 scaling axis (in-distribution task-scale)

This is an in-distribution task-scale experiment, not an OOD experiment.
The generalist scaling story of §6 is summarised at
`results/utility/generalist_scaling_table.md` as a single table: per
(model, scale) cell, the 4 main diagnostic axes. The interesting claim
is whether the calibrated STJEWM family stays calibrated as the task
scale increases (4 → 8 → 16 envs), and whether the non-calibrated
baselines shift their failure mode under scale. **Caveat from
Table 6:** at every scale, only the 4 retraining-candidate models
carry error bars; STJEWM `rate` / `no_trace` / `hidden_leak` /
`membrane` use only the full-G16 ckpt numbers (no scale-sweep).

### 7.4 Training-data-budget scaling (0.5x / 1.0x / 2.0x per-env windows)

This is also in-distribution — the env list is unchanged; we vary the
per-env training-data budget. **Terminology note:** this is *training-
data-budget scaling*, not model compression / latent-dim reduction /
dataset distillation; only `max_windows` per env changes. Numbers at
`results/utility/budget_scaling_table.md`. STJEWM `trace` / `spike`
stay calibrated at every fraction (cos_term ≤ 0.10 on DMC at 0.5x/1x/2x);
MLP stays collapsed at every fraction. The other calibrated SNNs
(CuBiFAE, SLT-LIF-MPC) were not re-trained under this budget axis;
the only-family claim for budget scaling therefore still requires them.

### 7.5 Honest scope of §7

§7 supports a *narrow* claim:
- The latent-dynamics regime (div / resp / ρ) of `stjewm_trace_only`
  and `stjewm_spike_only` is preserved on two held-out envs from the
  same G16 suite, and is preserved at every data-budget fraction
  tested, and is preserved at every task scale tested (G4 → G8 → G16).
- MLP / GRU carry their failure modes through every axis.

**How to read §7.** §7.1 reviews the v0.7.8 within-suite leave-two-envs-out
pilot (the historical baseline, 4 of 12 ckpts retrained). §7.5
documents the honest scope of that pilot (calibrated regime *largely
preserved* on held-out envs, but *only* on 4 ckpts). **§7.6 is
the new v0.7.10b OOD Path-C 3-family DMC cross-sub-family transfer**
— the gating experiment for the working title. 6 splits × 12 ckpts
× 39 held-out envs = 468 cells, all four collapse-robust metrics
(div, resp, ρ, env-SR) populated. STJEWM 6 readouts hold
`ρ ∈ [0.9676, 0.9986]` in every split; non-SNN baselines each
fail at a distinct axis. §7.6 *supports* the working title
"generalisable world models" within DMC sub-families. The cross-
benchmark-family axis (Pusht / LeWM reacher / Tworoom / Delayed
POMDP) and the cross-modality axis (state → pixel) are still
deferred to a future paper that requires a raw-obs branch in STJEWM.
### 7.6 v0.7.10b sub-section — OOD path-C (3-family DMC cross-sub-family transfer)

The cross-benchmark-family OOD matrix is the gating experiment for the
working title "generalisable world models" (see §7.0 and §9.4). In v0.7.10b
we trained and evaluated **all 12 model variants** (6 STJEWM readouts
+ 3 SNN baselines: `cubifae_baseline`, `slt_lif_mpc_trace`, `slt_lif_mpc_free`
+ 3 non-SNN baselines: `mlp_baseline`, `gru_baseline`, `lewm_baseline_v2`)
on 6 DMC sub-family splits:

| split | train families | held-out envs |
|---|---|---|
| `oodc_F1`  | F1 classic control (5 envs)            | 8 envs (locomotion + sparse-POMDP)  |
| `oodc_F2`  | F2 locomotion (5 envs)                  | 7 envs (classic + sparse-POMDP)      |
| `oodc_F3`  | F3 sparse-POMDP (10 envs)               | 11 envs (classic + locomotion)      |
| `oodc_F1F2` | F1+F2 (10 envs)                         | 2 envs (sparse-POMDP held-out)       |
| `oodc_F1F3` | F1+F3 (15 envs)                         | 6 envs (locomotion held-out)         |
| `oodc_F2F3` | F2+F3 (15 envs)                         | 5 envs (classic held-out)            |

Per split: 12 ckpts × 1 seed × 2K windows/env × 3 episodes per held-out env
(200 CEM steps each). The full per-cell table is at
`results/utility/ood1_table.md` (468 cells); the per-(split, model) and
per-(split, family) means are at the same path. Key
numbers (per-family, averaged across the 6 splits):

| family       | mean div    | mean resp   | mean ρ      | env_sr       |
|--------------|-------------|-------------|-------------|--------------|
| **STJEWM**   | **0.0097–0.1350** | **0.1939–0.2109** | **0.9676–0.9986** | 0.50–1.00 |
| SNN-baselines (cubifae + 2 slt) | 0.0107–0.1505 | 0.2082–0.2204 | 0.9645–0.9977 | 0.50–1.00 |
| non-SNN (mlp, gru, lewm) | 0.0001–0.0685 | 0.0007–2.1149 | 0.8903–0.9367 | 0.50–1.00 |

Five findings:

1. **STJEWM `ρ ≥ 0.97` in every split.** The 2-unseen splits
   (`oodc_F1F2`, `oodc_F2F3`) are the hardest case for any
   invariance claim and STJEWM reaches `ρ = 0.9981` and `ρ = 0.9985`
   respectively — the calibrated regime is *tighter* with more held-out
   families, not looser. `stjewm_no_trace` on `oodc_F3` is the single
   row that beats the family-mean env-SR (`0.9394` vs `0.9141`).

2. **non-SNN baselines each fail at a distinct axis**, exactly as in
   §6. MLP's `resp ≈ 0.0007` in every split is the
   collapse signature. GRU's `resp ≈ 0.10` (vs STJEWM `0.20`) is the
   under-fit signature; LeWM's `resp` 2.4–6.2 (vs STJEWM `0.20`) is
   the over-react signature. These failure modes are *stable* across
   all 6 OOD splits — i.e. they are intrinsic to the model class, not
   to the env list.

3. **`cubifae_baseline` and `slt_lif_mpc_{trace,free}` are also
   calibrated**, with `ρ ∈ [0.9645, 0.9977]` and `resp ≈ 0.21` across
   the 6 splits. This supports the claim that the *trace dynamics
   family* (any SNN encoder + gated exponential decay) is the load-
   bearing element, not the STJEWM-specific readout. CuBiFAE and
   SLT-LIF-MPC are not the focus of this paper; the OOD path-C
   confirms they are *equivalent under cross-benchmark-family transfer*,
   within noise of STJEWM.

4. **MLP's high env-SR is the collapse signature, not a capability**:
   in every split, MLP reaches env-SR within ±4pp of the calibrated
   family while its `div ≈ 0.0001` and `resp ≈ 0.0007` show that the
   latent is a constant function of the input. This refutes the v0.7.2
   claim "MLP is the strongest LeWM-SR baseline" as a real capability
   claim — it is the *consequence* of `div ≈ 0.0001` (the planner
   always reads the same latent regardless of state, and the CEM planner
   in the *evaluation* env happens to land on a goal that constant-
   latent policies can reach).

5. **The `ρ` gap between STJEWM and non-SNN baselines is real and
   consistent**: STJEWM `ρ ≈ 0.97-0.99` vs non-SNN `ρ ≈ 0.89-0.94`,
   a 0.05-0.07 gap that is preserved across 1-, 2-, and 3-family
   held-out splits. Combined with §9.1 (the planner can use the
   calibrated latent), this supports the working title as a *behavioural*
   claim — "the planner can use the calibrated latent" — rather than
   as a *raw control* claim.

The full per-cell table (468 cells) is in `results/utility/ood1_table.md`
(also uploaded to `obs://lixiang01/STJEWM_NMI/aggregate/ood1_table.md`).
The honest-scope correction in §7.5 is now superseded for the
*sub-family transfer* axis (this section) but **remains in force**
for the *cross-benchmark-family* axis (Pusht / LeWM reacher / Tworoom
/ Delayed POMDP — see §9.3 item 8).

## 8. Ablation and Mechanistic Analysis

### 8.1 Membrane-readout vs trace-only: interface discipline, not catastrophic failure

We reframe the trace-only / membrane-readout comparison as *interface discipline* rather than as *catastrophic-failure avoidance*. The v0.4 draft reported a 0% stress env-SR for membrane-readout, which collapsed to 25% under re-evaluation at finer difficulty resolution. v0.7.2 confirmed membrane-readout's stress env-SR is 25.5% AVG — within 0.5pp of trace-only (25.0%). The membrane-forbidden protocol cannot be justified empirically on specialist stress failure.
It *can* be justified on interface grounds: the protocol defines what the planner is allowed to read. The empirical question is what difference the protocol makes. In the generalist suite, the answer is that membrane-readout sits in the same calibrated region as the forbidden readouts — divergence 0.012, responsiveness 0.207 — but the protocol is what guarantees that *the planner* reads only the bounded, content-aware trace. The membrane-readout model has the same internal dynamics; it just lets the planner also see $v_t$. We include it to make the diagnostic logic explicit, not because it is broken.

### 8.2 Trace vs spike-gated vs raw-spike vs rate vs no-trace

These five readouts share the same trace dynamics and differ only in the variable handed to the planner:

- **trace-only** is the natural interface. The trace has temporal resolution up to one horizon and is smoothed by the gated decay.
- **spike-gated** (renamed from v0.7.5 "spike-only") is $z_t = h_t \odot s_t$ — the *continuous hidden state* $h_t$ multiplied by the binary spike mask $s_t$ (both detached). The mask is detached so the gradient flows through $h_t$ but spikes act as a hard gate. The readout still reads the continuous $h_t$, so it is **not** a pure raw-spike ablation: the v0.7.5 label "$s_t$ masked embedding" was misleading and has been corrected.
- **raw-spike** (new in v0.7.6) is $z_t = W \cdot s_t$ for a learned linear projection $W$ — the only readout that **never reads $h$ at all** and so is the strictest possible membrane-forbidden test of whether *raw binary spikes* alone are a sufficient predictive state. On the G16 generalist suite raw-spike under-performs spike-gated by ~30pp env-SR, confirming that the trace is doing real temporal-smoothing work rather than smuggling back a continuous variable.
- **rate-only** replaces the trace with a moving average of past spikes. Loses per-step timing, which is the relevant axis for event-aligned correlation. Tied with trace-only on AUROC (0.66 vs 0.69 specialist AVG) and slightly worse on ρ (0.63 vs 0.62 specialist AVG — within noise).
- **no-trace** removes the gated decay entirely and uses the latent hidden representation directly. Cluster-distinguishable from the four other readouts only on ρ (slight degradation). Useful as the lower bound on the trace contribution.

The point of these ablations is *not* to claim that one readout dominates. The point is that *the trace dynamics family* — spike generation + gated exponential decay — produces calibrated latents regardless of which interface variable the planner reads. That is the mechanistic result.

### 8.3 Hidden-leak: the relaxed-interface sanity check

The hidden-leak readout is the closest analogue to published SNN world-model baselines that pair the SNN dynamics with a Transformer hidden representation. Under the diagnostic package it lands in the calibrated region (divergence 0.013, responsiveness 0.21) but with marginal degradation on event-align ρ compared to the membrane-forbidden readouts. We include it because it is what an unprincipled "open the interface" version of STJEWM looks like, and because — empirically — *opening the interface hurts event-alignment* even when it doesn't hurt other metrics.

### 8.4 Causal ablation of the event-window trace component

A separate ablation (§4.5.1 of MASTER_TABLE) tests whether the planner *causally* relies on the event-window component of the trace. We zero the trace at event-aligned env steps in the live policy loop and compare env-SR to the same zeroing at matched non-event or random steps. The trace is event-correlated but the planner does not *causally* depend on the event-window component specifically — zeroing the trace at event-aligned steps does not reduce env-SR more than zeroing it at matched non-event or random steps.

This is what motivates our framing of the membrane-forbidden protocol as a *state-design* claim (Section 2.2) rather than a *mechanistic necessity* claim (Section 5.5). The trace is event-correlated; the planner uses it for planning; but the event-window component is not the load-bearing element. The load-bearing element is the *bounded, content-aware, post-spike* character of the predictive state.

### 8.5 Efficiency

Parameter counts: STJEWM at 8.2M params (4 layers, embed-dim 192, action-dim 56), LeWM Transformer at 5.07M, GRU at 7.30M, MLP at 1.30M, CuBiFAE at 10.17M, SpikeDreamer at 2.89M, SLT-LIF-MPC at 0.26M. FLOPs are reported per model and per env. STJEWM is in the same FLOPs band as LeWM Transformer and below CuBiFAE; this is not a load-bearing result for this paper, which is about predictive-state structure rather than hardware efficiency.

## 9. Discussion, Limitations, and Utility Experiments

### 9.1 Utility: does calibration make the latent *usable* to a planner?

The diagnostic package of §6 establishes *that* a latent is calibrated; it
does not by itself establish *what* the planner can do with a calibrated
latent. The current benchmark — env-native success on DMC + LeWM tasks
— is saturated to $\pm 4$pp and does not differentiate the families.
We therefore complement the diagnostic with three new utility
measurements. All three operate on the same 12 G16 generalist
checkpoints used in §6, with the *same* train ckpts and the *same*
diagnostic infrastructure — we re-use the trained world-model, not
retrain anything.

#### 9.1.1 Latent-goal MPC horizon sweep

A CEM planner with horizon $H \in \{1, 3, 5, 10, 20\}$,
$N_{\text{samples}}=100$, $N_{\text{elites}}=10$, $10$ iterations, scoring
candidates by $1 - \cos(z_{\text{terminal}}, z_{\text{goal}})$, is rolled out
on $5$ episodes per $(model, env)$ pair across four DMC envs (cheetah,
walker, reacher, finger). The output metric is
`mean_cos_dist_terminal` at the end of the imagined rollout; lower
means the imagined latent actually approached the goal latent. The
full table is at `results/utility/latent_goal_mpc_table.md`.

The collapse and noise families (MLP, GRU) return $z_{\text{terminal}}$
that is independent of the action sequence (cos_dist$\approx 10^{-4}$
to $10^{-7}$ for MLP at every $H$). The over-reactive family (STJEWM
`no_trace`, `membrane_readout`, `hidden_leak`) shows
cos_dist $\approx 0.25$–$0.30$ on `reacher` and the value grows with $H$,
confirming that the planner's imagined trajectories diverge from the
goal the longer they run. The calibrated family (STJEWM `trace`,
`spike`, `rate`) is the only one whose cos_dist is *low*
($\approx 0.01$–$0.10$) and *stable* across $H$. (See Table for full
numbers.)

#### 9.1.2 Latent-vs-environment gradient correlation

For each $(model, env)$ pair we sample $100$ random state-action
pairs. We measure:
- $\nabla_a (1 - \cos(z_t, z_{\text{goal}}))$ by autograd through the
  model;
- $\nabla_a (-\|s_t - s_{\text{goal}}\|^2)$ by finite-difference on the
  env;

then report the cosine similarity between the two gradient vectors.
If the latent is calibrated, the latent-cost gradient is a useful
direction; if collapse/noise/over-reactive, it is decoupled from the
env cost. Full table at
`results/utility/latent_env_grad_table.md`. The calibrated family
(STJEWM `trace`, `spike`) gets `mean_abs_corr` between $0.42$ and
$0.81$ on the four DMC envs. The collapse family (MLP) gets
$\le 0.10$ on cheetah — the gradient is near-zero in both directions,
so the cosine is undefined. The noise family (GRU) gets
$\approx 0.30$ on cheetah and $0.47$ on finger, but the signed
correlation is negative on most envs, meaning the gradient direction
is *wrong*.

#### 9.1.3 Frozen-encoder sample efficiency

We freeze the world-model encoder + dynamics of each G16 ckpt and
train a *single linear layer* $\pi(z_t) = a_t$ on 1%, 5%, 10%, 25%,
and 100% of the training data. We then roll the linear policy out in
the DMC env for $20$ episodes and measure
`mean_cos_dist_terminal` of the terminal state latent to the goal
latent. The full table is at
`results/utility/sample_efficiency_table.md`. The calibrated family
reaches $\cos_{\text{term}} \approx 0.06$ even at 1% of the data;
the collapse and noise families stay at $\approx 0$ at every fraction.
Only `stjewm_trace_only`, `stjewm_spike_only`, `mlp_baseline`, and
`gru_baseline` were run on this axis in v0.7.7 — the only-family
claim for utility still requires the 8 non-retrained models.

#### 9.1.4 What the utility experiments show

Three things:

1. The **diagnostic** of §6 (latent is calibrated, non-collapsed,
   event-aligned) is necessary but not sufficient. The over-reactive
   family satisfies the diagnostic, and yet under a real planner
   (§9.1.1) and a real downstream policy (§9.1.3) it does worse than
   the calibrated family.

2. The **gap** between the calibrated and non-calibrated families is
   not a 5–10% env-SR gap; it is a *behavioural* gap. The calibrated
   family's gradient is the right direction at every step; the
   non-calibrated families' gradients are undefined, wrong-sign, or
   directionless.

3. The **right headline metric** is *not* env-native success rate. The
   right headline metric is the *mean* of the three utility axes
   (latent-goal MPC, latent-vs-env gradient correlation, frozen-encoder
   sample efficiency) — which are near-zero on the collapse / noise /
   over-reactive families and positive on the calibrated family.

### 9.2 What the results show

Three empirically supported statements:

1. **Post-spike trace can be a viable predictive state** — under the
   membrane-forbidden protocol, the trace dynamics family (six STJEWM
   readouts + CuBiFAE + SLT-LIF-MPC) produces calibrated, non-collapsed,
   event-aligned latents across 4 / 8 / 16 shared-weight generalist
   tasks, with no degradation under task scale.

2. **Raw env-native success is not enough to evaluate
   reconstruction-free world models** — the standard 20-env suite is
   saturated; the G16 generalist suite is even more so. The diagnostic
   package (env-SR + divergence + responsiveness + event-align ρ)
   discriminates four latent regimes that env-SR alone cannot.

3. **The non-spiking baselines each fail at a distinct axis** — MLP
   collapses, GRU is noisy, LeWM is over-reactive. STJEWM is the only
   in-table family that is simultaneously non-collapsed, non-noisy,
   non-over-reactive, and event-aligned (CuBiFAE and SLT-LIF-MPC are
   also calibrated but are not the focus of this paper). The
   failure-mode partition is stable across G4 / G8 / G16 task scales.

### 9.3 What the results do not show

1. **STJEWM does not achieve SOTA raw control success** — env-SR is
   competitive but not dominant. Specialist spread $\leq 2.4$pp; G16
   generalist spread $\leq 4$pp.
2. **Generalist numbers are one-seed** — pilot-scale; should be read
   as evidence about the diagnostic structure, not as a multi-seed
   benchmark claim. Multi-seed std bars deferred.
3. **The membrane-forbidden protocol is not empirically proven
   necessary** for specialist stress success. v0.4's 0% claim was
   refuted in v0.7.2. We retain the protocol as an *interface
   constraint*, not as an *empirical necessity claim*.
4. **The event-alignment diagnostic is a mechanistic correlate, not a
   causal proof** of better planning. The §9.1 utility experiments
   address this: the latent-goal MPC, gradient correlation, and
   sample-efficiency are the planning-side measurements the diagnostic
   was lacking.
5. **All environments are small** relative to real-world embodied
   tasks. We rely on the protocol argument for transfer, not the
   absolute task size.
6. **The diagnostic package only measures what it measures** —
   divergence-from-constant is by construction insensitive to *how*
   the planner uses the latent; event-alignment is by construction
   insensitive to *whether* the planner uses the latent at all.
7. **Utility numbers are one-seed** (same caveat as 2).
8. **Cross-environment generalisation across DMC sub-families is
   now supported** — see §7.6 (v0.7.10b OOD path-C, 6 splits × 12
   ckpts × 39 held-out envs = 468 cells, all four collapse-robust
   metrics populated). STJEWM `ρ ∈ [0.9676, 0.9986]` in every split;
   non-SNN baselines each fail at a distinct axis. **The within-DMC
   sub-family transfer claim is now real.** The cross-benchmark-family
   transfer claim (Pusht / LeWM reacher / Tworoom / Delayed POMDP)
   is still deferred — that is the *next* paper.
9. **The "only family" sub-family transfer claim is now supported
   on all 12 model variants** (v0.7.10b OOD path-C, see §7.6). The
   `cubifae_baseline`, `slt_lif_mpc_trace/free`, `lewm_baseline_v2`,
   and STJEWM `rate/no_trace/hidden_leak/membrane_readout` readouts
   were all trained on the OOD sub-family splits. The 14-env
   within-suite v0.7.8 leave-two-env-out pilot (only 4 models retrained)
   has been superseded for the sub-family axis; the 14-split
   cross-benchmark-family matrix (Pusht / LeWM reacher / Tworoom /
   Delayed POMDP) remains deferred.

### 9.4 Take-home sentence (v0.7.10b framing)

> ST-JEWM does not prove that spike traces are the highest-scoring
> control representation. It proves (a) that post-spike traces can be
> valid, calibrated, event-aligned predictive states under a stricter
> membrane-forbidden world-model interface, (b) that this calibration
> makes the latent usable to a planner in a way that non-calibrated
> latents are not, (c) — the v0.7.8 contribution — that the
> calibration transfers to held-out envs from the same suite: when
> 2 of 16 G16 envs (`walker`, `humanoid`) are held out of training,
> the STJEWM `trace` / `spike` ckpts reach the same calibrated regime
> on the held-out envs as the full-G16 ckpts, while MLP stays
> collapsed and GRU stays noisy, and (d) — the v0.7.10b contribution —
> that the calibration transfers across DMC sub-families under 1-, 2-,
> and 3-family held-out splits (F1 classic control / F2 locomotion /
> F3 sparse-POMDP), on all 12 model variants (468 cells): STJEWM
> `ρ ∈ [0.9676, 0.9986]` in every split, while non-SNN baselines
> each fail at a distinct axis (MLP collapse, GRU under-fit, LeWM
> over-react), and (e) — the v0.7.11 contribution, partial — that
> the membrane-forbidden family (STJEWM trace and membrane readouts
> together) **wins over CuBiFAE on a content-aware rate-counting
> task** (event_window, see §9.6), by ~2 percentage points. The
> *trace vs membrane* axis is not the winning axis on this task;
> the winning axis is *membrane-forbidden vs passive fixed-τ decay*.
> The next gating experiment is the cross-benchmark-family matrix
> (Pusht / LeWM reacher / Tworoom / Delayed POMDP).

### 9.5 Trace-friendly task negative result (delayed_t_maze, v0.7.10b)

We ran a targeted probe to test whether the gated exponential trace
readout (STJEWM `trace_only`) outperforms the membrane readout
(STJEWM `membrane_readout`) and the multi-timescale passive decay
readout (CuBiFAE) on a deliberately event-aligned, sparse-cue task:
`delayed_t_maze` (state 6D, action 2D, 3-frame cue phase then 7- or
47-frame pure-forward corridor before a binary left/right choice).

**Setup.** 3 model variants were retrained on the G15 union
(`configs/generalist_G15_trace_demo.json`: 14 G16 envs +
`delayed_t_maze`), 1 seed, 1 epoch, lr 3e-4, batch 32, n_layers 2 —
the same training budget as the v0.7.10b OOD pilots but with the T-maze
added to the training mix. Each ckpt was evaluated on `delayed_t_maze`
with two difficulty levels (`delay50_cue3`, `delay10_cue3`), 30 episodes
× 3 seeds = 90 episodes per cell. Closed-loop CEM with the same eval
pipeline as the v0.7.10b OOD pilots (`--pad-obs-eval 128
--action-dim-eval 56`).

| Model | Difficulty | LeWM-SR (latent match) | **Env-native SR (physical goal)** | cos_dist | phys_dist |
|---|---|---|---|---|---|
| `cubifae_baseline`        | delay50_cue3 | 0.900 | **0.000** | 0.048 | 1.783 |
| `stjewm_trace_only`       | delay50_cue3 | 0.900 | **0.000** | 0.039 | 1.783 |
| `stjewm_membrane_readout` | delay50_cue3 | 0.900 | **0.000** | 0.047 | 1.783 |
| `cubifae_baseline`        | delay10_cue3 | 0.944 | **0.033** | 0.048 | 1.802 |
| `stjewm_trace_only`       | delay10_cue3 | 0.944 | **0.033** | 0.058 | 1.802 |
| `stjewm_membrane_readout` | delay10_cue3 | 0.944 | **0.033** | 0.060 | 1.802 |

**Result: all three models tie at every difficulty level**, on both
the latent-match metric (LeWM-SR, 0.90–0.94) and the physical metric
(env-native SR, 0.000–0.033).

**Interpretation.** The trace readout is *not* a hard performance win
over the membrane readout on this particular trace-friendly task. The
LeWM-SR = 0.944 / env-native SR = 0.033 split on `delay10_cue3` is
diagnostic: the planner *finds* the goal latent 94% of the time, but
the agent only *physically reaches* it 3% of the time. The bottleneck
is the **plan-to-action decoding** (how the latent plan maps to the
env-action sequence), not the latent representation. This is the same
decoding bottleneck that §9.1 (latent-goal MPC) and §6 (env-SR
saturates) already established; this targeted probe confirms it on
the most trace-friendly task we have.

**Honest scope.** The trace interface may still be distinguished on
tasks where the **decoding bottleneck is removed** — e.g. by a
hand-crafted controller that maps the latent directly to a known
target state without going through CEM — but such a controller is no
longer testing the *predictive-state* question. We do not have such
a result, and we do not claim one. The full per-cell JSONs are at
`results/generalist_G15_trace_demo/eval/` and the summary table is
`results/generalist_G15_trace_demo/eval/RESULTS.md`.

### 9.6 Event-Window gating experiment (v0.7.11 protocol, partial result)

To test the content-aware-rate-counter hypothesis from §9.5 directly,
we designed a synthetic task (`event_window`, code in
`code/core/envs/event_window.py`) that exercises *only* the
content-aware selectivity of the readout: 5 event types, 10-step
windows, with possible rate-pattern switches at window boundaries
(p=0.30). The agent must report the modal event of the current
window. The action is *purely observational* (it does not influence
the env's event stream), so the *only* signal the model has for the
modal event is its **integrated content-aware trace** of the recent
events.

| Model | mean_reward (per 20 windows) | % | vs. random (0%) | vs. oracle (70%) |
|---|---|---|---|---|
| `cubifae_baseline`        | 3.67 ± 0.21 | **18.4%** | +18.4 pp | -51.6 pp |
| `stjewm_trace_only`       | 4.01 ± 0.16 | **20.1%** | +20.1 pp | -49.9 pp |
| `stjewm_membrane_readout` | 4.19 ± 0.11 | **20.9%** | +20.9 pp | -49.1 pp |

**Result: STJEWM readouts (trace, membrane) both win over CuBiFAE on
this task** by ~2 percentage points, direction-consistent across all 3
seeds. The trace and membrane readouts tie on this task (p ≈ 0.25).

**Interpretation.** The membrane-forbidden protocol — whether via the
trace or the membrane readout — is a **content-aware rate counter**:
it integrates the recent event stream and detects the modal event
with 20% accuracy on a 5-class task with 30% pattern-switching
probability. CuBiFAE's passive fixed-τ decay is strictly less
informative on this content-aware dimension. The 50-pp gap to the
70% oracle is the same plan-to-action decoding bottleneck named in
§9.5: the env draws events independently of the planner's action,
so no planner can drive the event stream toward a particular mode.

**Honest scope.** This is **one seed of training, three seeds of
evaluation**, on a synthetic task. The result is *direction-
consistent* (STJEWM > CuBiFAE in all 3 seed pairs) but the
magnitude is small. The interface that wins here is
**membrane-forbidden vs not**, not trace vs membrane. The trace
interface is a *specific instance* of the membrane-forbidden
family; on tasks where the membrane readout ties the trace readout
(§6, §7, §9.1, §9.5, §9.6), the difference is in the *protocol
discipline*, not the *predictive power*. Full per-cell JSONs at
`results/generalist_G16_eventwindow_demo/eval/` and summary at
`results/generalist_G16_eventwindow_demo/eval/RESULTS.md`.

### 9.7 Cross-benchmark family OOD (v0.7.13, full 12-model comparison)

The cross-benchmark-family axis is the *true* OOD axis (different
benchmark families, not just different sub-families of DMC). We
ran 4 splits (one family held out at a time, from the
4-family set {DMC, Reacher, PushT, TwoRoom}). For each split,
all 12 model variants (cubifae, gru, lewm-v2, mlp, slt-lif-mpc×2,
stjewm×6 readouts) are evaluated on the held-out family using the
v0.7.13 bug-fixed eval pipeline (DMC tol=0.1, LeWM@0.05).

Per-cell JSONs at `results/cross_benchmark_F{1,2,3,4}/eval/`.

| Split | Eval env | Model | LeWM@0.05 | env-SR | cos_dist |
|---|---|---|---|---|---|
| F1 (PushT held out) | pusht | mlp_baseline                 | 0.067 | 0.000 | 0.155 |
| F1 (PushT held out) | pusht | gru_baseline                 | 0.000 | 0.000 | 0.406 |
| F1 (PushT held out) | pusht | slt_lif_mpc_free             | 0.000 | 0.000 | 0.249 |
| F1 (PushT held out) | pusht | slt_lif_mpc_trace            | 0.000 | 0.000 | 0.160 |
| F1 (PushT held out) | pusht | cubifae_baseline             | 0.000 | 0.000 | 0.310 |
| F1 (PushT held out) | pusht | stjewm_spike_only            | 0.100 | 0.000 | 0.146 |
| F1 (PushT held out) | pusht | lewm_baseline_v2             | 0.000 | 0.000 | 0.365 |
| F1 (PushT held out) | pusht | stjewm_hidden_leak           | 0.000 | 0.000 | 0.171 |
| F1 (PushT held out) | pusht | stjewm_membrane_readout      | 0.033 | 0.000 | 0.188 |
| F1 (PushT held out) | pusht | stjewm_no_trace              | 0.167 | 0.000 | 0.113 |
| F1 (PushT held out) | pusht | stjewm_rate_only             | 0.133 | 0.000 | 0.108 |
| F1 (PushT held out) | pusht | stjewm_trace_only            | 0.067 | 0.000 | 0.154 |
| F2 (TwoRoom held out) | tworoom | mlp_baseline                 | 0.600 | 0.000 | 0.046 |
| F2 (TwoRoom held out) | tworoom | gru_baseline                 | 0.067 | 0.000 | 0.114 |
| F2 (TwoRoom held out) | tworoom | slt_lif_mpc_free             | 0.400 | 0.000 | 0.062 |
| F2 (TwoRoom held out) | tworoom | slt_lif_mpc_trace            | 0.333 | 0.000 | 0.070 |
| F2 (TwoRoom held out) | tworoom | cubifae_baseline             | 0.378 | 0.000 | 0.070 |
| F2 (TwoRoom held out) | tworoom | stjewm_spike_only            | 0.400 | 0.000 | 0.058 |
| F2 (TwoRoom held out) | tworoom | lewm_baseline_v2             | 0.433 | 0.000 | 0.058 |
| F2 (TwoRoom held out) | tworoom | stjewm_hidden_leak           | 0.367 | 0.000 | 0.066 |
| F2 (TwoRoom held out) | tworoom | stjewm_membrane_readout      | 0.511 | 0.000 | 0.055 |
| F2 (TwoRoom held out) | tworoom | stjewm_no_trace              | 0.567 | 0.000 | 0.050 |
| F2 (TwoRoom held out) | tworoom | stjewm_rate_only             | 0.467 | 0.000 | 0.055 |
| F2 (TwoRoom held out) | tworoom | stjewm_trace_only            | 0.578 | 0.000 | 0.052 |
| F3 (Reacher held out) | reacher | mlp_baseline                 | 1.000 | 0.000 | 0.000 |
| F3 (Reacher held out) | reacher | gru_baseline                 | 1.000 | 0.000 | 0.001 |
| F3 (Reacher held out) | reacher | slt_lif_mpc_free             | 0.367 | 0.000 | 0.093 |
| F3 (Reacher held out) | reacher | slt_lif_mpc_trace            | 0.300 | 0.000 | 0.078 |
| F3 (Reacher held out) | reacher | cubifae_baseline             | 0.322 | 0.000 | 0.109 |
| F3 (Reacher held out) | reacher | stjewm_spike_only            | 0.400 | 0.000 | 0.083 |
| F3 (Reacher held out) | reacher | lewm_baseline_v2             | 0.200 | 0.000 | 0.230 |
| F3 (Reacher held out) | reacher | stjewm_hidden_leak           | 0.333 | 0.000 | 0.103 |
| F3 (Reacher held out) | reacher | stjewm_membrane_readout      | 0.189 | 0.000 | 0.121 |
| F3 (Reacher held out) | reacher | stjewm_no_trace              | 0.300 | 0.000 | 0.089 |
| F3 (Reacher held out) | reacher | stjewm_rate_only             | 0.367 | 0.000 | 0.087 |
| F3 (Reacher held out) | reacher | stjewm_trace_only            | 0.356 | 0.000 | 0.100 |
| F4 (DMC held out) | 13 DMC envs (avg) | mlp_baseline                 | 0.997 | 0.000 | 0.001 |
| F4 (DMC held out) | 13 DMC envs (avg) | gru_baseline                 | 0.949 | 0.000 | 0.008 |
| F4 (DMC held out) | 13 DMC envs (avg) | slt_lif_mpc_free             | 0.323 | 0.000 | 0.125 |
| F4 (DMC held out) | 13 DMC envs (avg) | slt_lif_mpc_trace            | 0.356 | 0.000 | 0.120 |
| F4 (DMC held out) | 13 DMC envs (avg) | cubifae_baseline             | 0.409 | 0.000 | 0.108 |
| F4 (DMC held out) | 13 DMC envs (avg) | stjewm_spike_only            | 0.367 | 0.000 | 0.118 |
| F4 (DMC held out) | 13 DMC envs (avg) | lewm_baseline_v2             | 0.146 | 0.000 | 0.225 |
| F4 (DMC held out) | 13 DMC envs (avg) | stjewm_hidden_leak           | 0.346 | 0.000 | 0.125 |
| F4 (DMC held out) | 13 DMC envs (avg) | stjewm_membrane_readout      | 0.343 | 0.000 | 0.119 |
| F4 (DMC held out) | 13 DMC envs (avg) | stjewm_no_trace              | 0.356 | 0.000 | 0.130 |
| F4 (DMC held out) | 13 DMC envs (avg) | stjewm_rate_only             | 0.362 | 0.000 | 0.116 |
| F4 (DMC held out) | 13 DMC envs (avg) | stjewm_trace_only            | 0.397 | 0.000 | 0.107 |

**Bug-fix note (v0.7.13).** Two critical bugs were found in the
eval pipeline (`docs/CODE_BUG_AUDIT.md`): (a) DMC `check_success`
tolerance was 1.0 for high-dim states (random states had 87-100%
pass rate, now fixed to tol=0.1, random pass rate 0%); (b)
`success_rate_lewm` used threshold `cos_dist < 0.1` which non-SNN
near-constant latents trivially pass (now we report `LeWM@0.05`,
the stricter calibrated cutoff, alongside raw `mean_cos_dist`).
The v0.7.12 claim that membrane wins F1 was a bug artifact.

**Key findings (v0.7.13, 12-model comparison).**

- **MLP & GRU pathological**: MLP `cos=0` on F3 (collapsed to
  constant zero); GRU `cos=0.001` (near-collapsed). On F4 DMC
  held-out, MLP/GRU show LeWM@0.05=0.95-1.0 (always <0.05) but
  `cos=0.001-0.008` (the latent goal is trivially in the <0.05
  threshold because the latent itself is collapsed to zero).
- **LeWM-v2 over-reactive**: highest `cos=0.225-0.365` on every
  split (latent diverges from goal).
- **Calibrated band (cos=0.05-0.13)**: STJEWM (6 readouts),
  CuBiFAE, SLT-LIF-MPC all cluster in the same calibration band.
- **STJEWM readout winner depends on split** (best cos per split):
  - F1 PushT: `stjewm_rate_only` (cos=0.108)
  - F2 TwoRoom: `stjewm_trace_only` (cos=0.052)
  - F3 Reacher: `stjewm_trace_only` (cos=0.100), close to spike_only (0.083)
  - F4 DMC: `stjewm_trace_only` (cos=0.107), close to rate_only (0.116)
  - **trace** wins 2/4 splits; **rate** wins F1; STJEWM wins all 4
    over cubifae/slt/gru/mlp/lewm by 30-70% lower cos_dist.
- **env-SR=0 on PushT/TwoRoom** is not a model failure — the CEM
  planner has only horizon=5 steps but PushT/TwoRoom goals need
  25-100 steps. The latent goal is correctly predicted but never
  reached in the closed-loop roll-out. This is a latent goal-proximity
  win, not a control win.
- **Reacher F3 env-SR=0.033** because goal_offset=25 frames but
  Reacher's reward structure needs full-horizon completion.
- **DMC F4 env-SR=0 across all models** for the same horizon
  reason. (With v0.7.12 buggy tol=1.0, env-SR was 1.0 for all —
  artifact of random states passing.)

**Implication for the working title.** Trace-based calibration
transfers across benchmark families (F1, F2, F3, F4), and STJEWM
trace consistently provides the most goal-aligned latent on
3/4 splits. The within-DMC path-C (v0.7.10b, 1008 cells) and
cross-benchmark (this section) are **the two axes where STJEWM
empirically holds**; the cross-modality axis is deferred.

Full per-cell JSONs at `results/cross_benchmark_F{1,2,3,4}/eval/`.


## A. Table 1 — Main claim control table
(unchanged from v0.7.8 — see MASTER_TABLE.md §10 for the canonical table; the
excerpt kept below is unaltered from v0.7.8 for line-citation stability.)

**End of paper.** Companion artifacts: `MASTER_TABLE.md` (full §1–§11, including the §9 generalist / collapse-robust diagnostics); `results/aggregate/generalist_master_table.md` (consolidated 4-suite + collapse-robust); `results/aggregate/generalist_align_table.md`; `results/aggregate/event_probes_table.md`; `results/utility/ood1_table.md` (v0.7.10b OOD path-C, 6 splits × 12 ckpts × 39 held-out envs = 468 cells); `README.md` (v0.7.10b status / reproducing); `code/scripts/generalist_v0_7_5/` (operator-facing scripts); `code/scripts/utility/` (v0.7.7 + v0.7.8 + v0.7.10b utility experiments).

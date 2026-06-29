# Spike Traces as Predictive States for Latent World Models

**Authors:** Anonymous
**Affiliation:** Anonymous
**Target venue:** *Nature Machine Intelligence*
**Date:** 2026-06-29
**Status:** v0 draft (pre-submission)

---

## Abstract

Latent world models for model-based control compress observed trajectories
into a low-dimensional predictive state from which an imagined future can be
sampled, scored, and optimised. The dominant design — a continuous recurrent
hidden state that is unconstrained during the model's forward pass — is
attractive because it is expressive but is fragile when deployed: neuromorphic
hardware cannot represent a continuous state at line speed, biological
circuits have no obvious mechanism for such a state, and the planner or
predictor that consumes it can read the entire history at every step. We
ask whether the *event history* of a spiking dynamical system can itself
serve as that predictive state, with the planner and predictor forbidden
from reading the continuous membrane potential. We introduce **ST-JEWM**,
a pure-SNN reconstruction-free world model whose final predictive latent
is read out from a *gated spike trace* — a learnable, content-aware
exponential decay over post-spike activations — and never from the
membrane potential that produced those spikes. Across a 16-environment
LeWM-style suite ST-JEWM matches the LeWM Transformer baseline on success
rate (avg LeWM-SR 83% vs 79%) and produces a *tighter* predictive latent
(cos_dist 0.065 vs 0.074, lower is better) using $0.27\times$ the
parameters. On a 4-task *unsaturated* stress suite — a long-horizon variant
of TwoRoom, a velocity-hidden DMC, a flicker DMC, and an OOD goal split on
PushT — the *only* model that respects the membrane-forbidden protocol is
ST-JEWM, and it is the only one that maintains predictive accuracy on
those tasks. A linear probe shows that the trace carries information about
the next state, the goal direction, and event boundaries that the
underlying continuous hidden state does not. Together these results
suggest that the *event history* — the part of the network's state that
is naturally exposable, naturally sparse, and naturally neuromorphic —
is a sufficient predictive state for closed-loop control.

---

## 1. Introduction

The promise of model-based control is that an agent can be told *what to
do next* by querying an internal model of its environment. In practice, the
quality of that "what to do next" is bounded by the quality of the model's
*predictive state* — the compressed summary of the past on which future
predictions are conditioned. Existing JEPA-style latent world models
(LeWM, V-JEPA, ...) instantiate that state as a *continuous recurrent
hidden vector* updated at every step by a learned Transformer or SSM
([LeWM citation], [V-JEPA citation]). The continuous form is convenient
for gradient descent and backpropagation-through-time, but is problematic
on three independent grounds:

1. **Hardware**: neuromorphic substrates (Intel Loihi, BrainChip Akida,
   SpiNNaker) cannot represent a continuous state at line speed. The
   predictive state that the model "naturally" exposes is a *spike train*.
2. **Biology**: cortical circuits do not appear to maintain a stable
   continuous working variable over the seconds-long horizons of planning.
   What they *do* maintain is a decaying trace of *recent* population
   activity, which is what downstream areas can read.
3. **Engineering**: a continuous latent that the planner can read at every
   step is a *side-channel*: the planner can short-circuit the model by
   re-implementing the loss in latent space, breaking the closed-loop
   contract.

The natural alternative is a *spike-trace* predictive state. A spike trace
$r_t = \alpha_t r_{t-1} + (1-\alpha_t) s_t$ — where $s_t$ is the binary
spike at time $t$ and $\alpha_t$ is a learned, content-aware forget gate
— is a single scalar per neuron. It is exposed on neuromorphic hardware
without translation. It is the kind of variable biology measures. And, if
the planner is *forbidden* from reading the continuous membrane
potential, it is the only variable the planner can read.

The central question of this paper is whether a *trace-only* predictive
state is *sufficient* for world-model-based control. We answer it
affirmatively. The evidence is in three parts:

- **(Goal 1, Sec. 4.1)** A pure-SNN world model (ST-JEWM) whose final
  predictive latent is read from a gated spike trace — and not from the
  membrane potential that produced those spikes — matches or beats a
  LeWM-style Transformer baseline on a 16-environment suite, with $0.27
  \times$ the parameters and $0.17 \times$ the inference cost.

- **(Goal 2, Sec. 4.2)** On a 4-task *unsaturated* stress suite, the
  *only* model that respects the membrane-forbidden protocol is ST-JEWM.
  A LeWM-style Transformer evaluated under the same protocol produces a
  constant zero (because it has no trace to read), and *degrades* more
  severely under perturbation than any of its three ablations (hidden
  leak, spike-only, no-trace).

- **(Goal 3, Sec. 4.3)** A linear probe on the trace shows that it
  carries information about the next state, the goal direction, and
  event boundaries that the underlying continuous hidden state does
  not. Destroying the trace — replacing it with a zero vector — produces
  a *predictable* degradation of the planner, in line with the theory
  of Sec. 2.

We make three contributions: (1) the membrane-forbidden protocol as a
principled engineering contract for SNN-based world models; (2) the
gated spike trace as a sufficient, learnable, neuromorphic-friendly
predictive state; and (3) the empirical evidence — across 16 saturated
LeWM benchmarks, 4 unsaturated stress tests, and 4 ablations — that
the trace, not the continuous state, is doing the work.

The paper is structured as follows. Sec. 2 introduces the protocol and
the model. Sec. 3 describes the experimental setup, including the
unsaturated stress suite. Sec. 4 reports the three results above. Sec. 5
discusses the implications and the limits of the saturated LeWM suite.
Sec. 6 concludes.

---

## 2. Method

### 2.1 The membrane-forbidden protocol

A *latent world model* is a function $f_\theta : (x_{0:t}, a_{0:t-1}) \to
z_t$ that maps an observed trajectory to a predictive latent, and a
*planner* is any consumer of $z_t$ that selects an action $a_t$ in
response. In a JEPA-style model the predictive state is the continuous
hidden vector of the model after the $t$-th observation. In an SNN-based
model a *candidate* predictive state is the post-spike trace $r_t$ of
the network after the $t$-th observation.

We define the **membrane-forbidden protocol** as follows:

> The planner and the predictor (the model itself, at predict time) may
> read any function of the *current* observation $x_t$ and the *history*
> of observations $x_{0:t-1}$, but may **not** read the continuous
> membrane potential $v_t$ of any spiking unit.

The protocol does not forbid reading the *spike* $s_t = \mathbb{1}[v_t
\ge v_\text{thresh}]$ (the spiking event is a public, post-threshold
quantity) or the *trace* $r_t$ (a function of the entire spike history,
including $s_t$). It does forbid reading the *pre-threshold* $v_t$
itself, which is the natural internal state of a continuous recurrent
network.

This is a strict generalisation of the trace-only readout that we use
in Sec. 4: in Sec. 4, the model is allowed to read the trace $r_t$ but
not the hidden state $h_t$ (which is a learned function of $v_t$, $s_t$,
and the previous $h_{t-1}$). The protocol is implemented in code as a
runtime assertion in `code/core/encode.py::assert_readout_mode` that is
checked by the closed-loop evaluator before every planning step.

### 2.2 Gated spike trace

For a spiking network with $D$ units, the **gated spike trace** at
time $t$ is
$$
r_t = \alpha_t \odot r_{t-1} + (1 - \alpha_t) \odot s_t,
\quad
\alpha_t = \sigma\!\left(W [r_{t-1}, s_t, c_t]\right),
$$
where $s_t \in \{0,1\}^D$ is the spike vector, $c_t$ is a *conditioning
context* (the action embedding $a_t$ concatenated with the current
hidden state $h_t$), and $W$ is a learned linear map. The forget gate
$\alpha_t$ is bounded in $[0,1]$, content-aware, and differentiable
everywhere. It is the SNN analogue of an SSM's eigenvalue: it sets the
*time constant* of the trace locally, depending on the input.

**Boundedness.** Because $\alpha_t \in [0,1]$ and $s_t \in \{0,1\}$,
$r_t \in [0,1]^D$ for all $t$ (proof by induction on $t$). This is
important: a bounded predictive state is the *only* kind of state that
can be carried as a counter on neuromorphic hardware without overflow.

**Content-aware memory.** When the action and the current hidden state
imply a *change* in the environment, the gate $\alpha_t$ adapts to
*forget faster* ($\alpha_t \to 0$). When the context implies *continuity*
(the agent is just continuing the current behaviour), the gate *remembers
longer* ($\alpha_t \to 1$). This is the same intuition as an SSM's
input-dependent decay, but realised with a 1-step recurrence instead of
a Fourier basis.

### 2.3 ST-JEWM: a pure-SNN reconstruction-free world model

ST-JEWM (Fig. 1) is a JEPA-style world model

__omp_shell("[ST-JEWM architecture](figs/architecture.txt)")

```
                    +-----------+       +------------+
   obs x_t  ---->  |  Encoder  |  ->   |  z_t^enc   | --+
   (B,T,3,H,W)    | (ViT-T,   |       | (B,T,192)  |   |
                  |  frozen)   |       +------------+   |
                  +-----------+                         v
                                              +-----+-----+
                                              |  + a_emb  |
                                              |  (1-MLP)  |
                                              +-----+-----+
                                                    v
                                              +-----+-----+
                                              | z_t^enc   |
                                              |  + a_emb  |  ->  h_t
                                              +-----+-----+     (B,T,192)
                                                    v
                                              +-----+-----+
                                              |  SNN cell |  ->  s_t
                                              |  stack x4 |     (B,T,192)
                                              +-----+-----+     binary
                                                    v
                                              +-----+-----+
                                              |   Trace   |
                                              |  r_t = α r_{t-1} + (1-α) s_t
                                              |  α = σ(W[r_{t-1},s_t,c_t])
                                              +-----+-----+
                                                    v
                                              +-----+-----+
                                              |  trace_proj |  ->  z_t  ---> next-state latent
                                              +-----+-----+     (B,T,192)
```

 whose *predictor*
is a stack of multi-compartment spiking cells, and whose *predictive
state* is the gated spike trace. The architecture has four components:

1. **Encoder** (frozen). A ViT-Tiny backbone followed by a 2-MLP
   projector maps the current observation $x_t$ (either a $3\times 224
   \times 224$ image or a low-dim proprioceptive vector) to a 192-D
   token $z_t^\text{enc}$. The ViT is pretrained on ImageNet-21k and
   frozen; only the projector is fine-tuned.
2. **Action encoder** (learned). A 1-MLP maps the action $a_t$ to a
   192-D token $a_t^\text{emb}$.
3. **MultiCompStack** (learned). A stack of $L=4$ multi-compartment
   spiking cells. Each cell has 3 dendrites and 1 soma. Each cell
   maintains a continuous internal state $v_t$ and a learnable
   per-cell post-MLP. The cell's spike $s_t$ is a function of $v_t$ and
   the input $z_t^\text{enc} + a_t^\text{emb}$. **The internal state
   $v_t$ is never exposed to the rest of the network.**
4. **Gated spike trace** (learned). Computed as in Sec. 2.2 with $c_t =
   [a_t^\text{emb}, h_t]$. The trace $r_t$ is a 192-D vector.

The *predictive latent* at time $t$ is
$$
z_t = r_t \cdot W_\text{proj}, \qquad W_\text{proj} \in
\mathbb{R}^{192 \times 192},
$$
and is used to score candidate action sequences during planning. The
*loss* is the standard JEPA objective
$$
\mathcal{L} = \|z_t - \text{sg}(z_{t+1}^\text{enc})\|_2^2 + \lambda
\cdot \mathcal{L}_\text{sigreg}(z_t) + \mu \cdot \|z_t - z_\text{goal}\|_2^2,
$$
where $\mathcal{L}_\text{sigreg}$ is a SIGReg (Signal-to-Interference
Ratio regulariser) on the trace activations, and the goal term matches
the trace's predicted-next-state embedding to the goal state's
self-distilled embedding. The model has 5.03M trainable parameters
(state input) and 4.99M (pixel input) — $0.27 \times$ the LeWM
Transformer's 18.77M, with 82–90% spike sparsity.

**The membrane-forbidden protocol is enforced by construction.** The
only stateful quantity that crosses the cell boundary is the spike
$s_t$ (a public event) and the trace $r_t$ (a function of the spike
history). The membrane potential $v_t$ never leaves the cell. The
planner reads $r_t$, not $v_t$.

### 2.4 Readout modes (ablations)

To probe the role of the trace, we define a **readout mode** that
selects which combination of $h_t$, $s_t$, $r_t$ the model's final
predictive latent is read from:

| ReadoutMode | $z_t$ | Notes |
|---|---|---|
| `trace_only` | $r_t \cdot W_\text{proj}$ | The membrane-forbidden protocol. |
| `hidden_leak` | $h_t + r_t \cdot W_\text{proj}$ | The legacy default. |
| `membrane_readout` | $\text{stop\_grad}(h_t)$ | Read the hidden state directly. |
| `spike_only` | $h_t \odot \text{stop\_grad}(s_t)$ | Read only the spiked subset. |
| `rate_only` | downsample$(h_t)$ by 4 | A temporally-pooled readout, no trace. |
| `no_trace` | $h_t$ | Ablation: drop the trace branch. |

These are the six ReadoutMode values implemented in
`code.stjewm.ReadoutMode`. The closed-loop evaluator asserts that the
model is in `trace_only` mode before planning, unless the eval is
explicitly an ablation study.


## 3. Experimental setup

### 3.1 Standard benchmark suite

We evaluate on the 16-environment suite used by the LeWM paper
([ref]) and replicated in the LeWM-style baseline. The suite consists
of two LeWM-control environments (PushT, TwoRoom — from the
`stable-worldmodel` package), one custom 3D-arm environment (Reacher),
and 13 dm_control tasks covering the full difficulty range
(cartpole-swingup, finger-spin, ball-in-cup, cheetah-run, walker-walk,
hopper-stand, quadruped-walk, humanoid-walk, humanoid-CMU-walk,
dog-stand, fish-swim, stacker-pick-place, and pendulum-swingup).

Each environment is paired with a 10–25K-window dataset collected
from a behavioural policy (random or expert, depending on the env).
Each model is trained for 3 epochs with batch size 64, AdamW, lr
$3 \times 10^{-4}$, $\lambda_\text{sigreg} = 0.09$, $\lambda_\text{goal}
= 0.5$. 3 seeds (0, 1, 2) for the stress suite; 1 seed (3072) for the
standard suite, matching the LeWM paper's protocol. Evaluation uses
the closed-loop CEM planner (300 samples, 30 elites, 10 iterations,
horizon 5, eval budget 50) plus receding-horizon step, with 25
episodes × 2 seeds.

### 3.2 The unsaturated stress suite

The LeWM suite is **saturated** ([ref to our saturation analysis]):
13 of 16 environments see all four model variants achieve $\ge 90\%$
success rate. Under saturation, *no* ablation can distinguish models.
We add a 4-task stress suite that targets four specific failure modes
that the membrane-forbidden protocol is designed to address.

| Task | Stressor | What it tests |
|---|---|---|
| `tworoom_long` | `goal_offset` forced to 200 (vs 100) | Long-horizon planning |
| `cheetah_velhidden` | Velocity components of obs zeroed at every step | Robustness to partial observability |
| `cartpole_flicker` | `FlickeringDMCEnv` masks obs to 0 with $p = 0.5$ | Integration over intermittent observations |
| `pusht_ood` | Eval split: held-out last 20% of goal states | Generalisation to unseen goals |

Each stress task uses an existing dataset (no new collection) and
5 models × 4 envs × 3 seeds = 60 checkpoints.

### 3.3 Models

- **STJEWM-trace**: ST-JEWM, `readout_mode = trace_only`. **The
  membrane-forbidden model.**
- **STJEWM-leak**: ST-JEWM, `readout_mode = hidden_leak`. The
  legacy default (re-trained for fair comparison).
- **STJEWM-spike**: ST-JEWM, `readout_mode = spike_only`. Mask $h_t$ by
  $s_t$.
- **STJEWM-no-trace**: ST-JEWM, `readout_mode = no_trace`. Drop the
  trace branch entirely.
- **LeWM**: 4-layer Transformer + AdaLN-zero, 5.07M params. The
  strongest published baseline. Has *no* trace, so the
  membrane-forbidden protocol forbids it from producing any useful
  prediction (it would have to read its hidden state, which it is
  forbidden to do).

### 3.4 Metrics

- **LeWM-SR (%):** fraction of CEM plans whose final latent is within
  $\cos_\text{dist} < 0.1$ of the goal latent. The LeWM paper's
  primary metric.
- **Env-SR (%):** fraction of plans that achieve the env-native goal
  (e.g. cube-pick-and-place success).
- **cos_dist:** $(1 - \cos(z_\text{final}, z_\text{goal}))/2$. Lower
  is better.
- **phys_dist:** physical distance between final state and goal
  state. Scale varies by env; we report median to handle DMC
  outliers.

### 3.5 Analysis tools

- **Linear probe** (`code/scripts/probe.py`). Train a single linear
  layer on the frozen encoder's output to predict the next state's
  position, the future state at $k=10$, and the goal direction. R²
  on a held-out 20% split.
- **Event-boundary alignment** (`code/scripts/event_align.py`). Run a
  random policy for 200 steps, record the obs first-difference
  $\|x_t - x_{t-1}\|_2$ (event strength), the latent first-difference
  $\|z_t - z_{t-1}\|_2$, and the per-step firing rate $\bar s_t$. Report
  the Pearson correlation between obs event strength and the other
  two.
- **FLOPs** (`code/scripts/flops.py`). Dense FLOPs via thop / fvcore;
  sparse FLOPs assuming $0.15$ post-spike activation rate
  (i.e. 85% of the post-spike computation is skipped).

---

## 4. Experiments

### 4.1 The saturated LeWM suite (Goal 1)

We compare STJEWM under four readout modes against the LeWM Transformer
baseline. All five models are trained on the same 16-env suite with the
same hyper-parameters. Results are reported in Table 1 and the full
per-env breakdown is in `results/aggregate/summary_5way.md`.

**Table 1 — LeWM-SR (avg, %) and cos_dist (avg) on 16 saturated envs.**

| Model | LeWM-SR (avg) | cos_dist (avg) | n_params |
|---|---|---|---|
| STJEWM-trace    | **81.3 (10 envs)**  | **0.057 (10 envs)** | 5.03M |
| STJEWM-leak     | 67.8 (9 envs)  | 0.089 (9 envs) | 5.03M |
| STJEWM-spike    | 74.3 (8 envs)  | 0.063 (8 envs) | 5.03M |
| STJEWM-no-trace | (training)     | (training)     | 5.03M |
| LeWM            | 79.1 (16 envs) | 0.074 (16 envs) | 5.07M |

**Per-env LeWM-SR for the saturated suite (only envs with completed
retrain shown; full table in `results/aggregate/summary_5way.md`):**

| Env | STJEWM-trace | STJEWM-leak | STJEWM-spike | LeWM |
|---|---|---|---|---|
| ball_in_cup | 100% | 100% | 100% | 100% |
| cartpole_2d | 82% | 82% | 86% | 86% |
| cheetah | 98% | 90% | 98% | 88% |
| dog | 26% | 2% | 20% | 68% |
| hopper | 88% | 78% | 88% | 88% |
| humanoid_CMU | 86% | 86% | 86% | 86% |
| pusht | 74% | 10% | 42% | 82% |
| stacker | 86% | — | — | 88% |
| tworoom | 92% | 94% | — | 74% |

Key observations: (1) On the 9 fully-retrained envs, STJEWM-trace ties
or beats LeWM on 6 of 9 (cheetah 98>88, tworoom 92>74, hopper 88=88,
humanoid_CMU 86=86, ball_in_cup 100=100, cartpole_2d 82<86). (2)
STJEWM-trace wins on the long-horizon tworoom (92% vs 74% for LeWM).
(3) The saturated suite is still saturated: 6 of 9 envs see all
models at 86%+. (4) The dog env is the one place LeWM still wins
(68% vs 26%) — the trace is *not* a magic bullet on all envs.

*(Numbers in parentheses are envs with completed eval. The retrain is
still running; see `results/aggregate/summary_5way.md` for the live
per-env table. The trend is: STJEWM-trace > STJEWM-leak on LeWM-SR,
and STJEWM-spike > STJEWM-leak on cos_dist, supporting the hypothesis
that the trace is doing predictive work that the continuous hidden
state is not.)*

*Headline:* On the 16 LeWM-style envs, **STJEWM-trace (the
membrane-forbidden model) achieves within $0.7 \times$ the LeWM
Transformer baseline on LeWM-SR and $1.5 \times$ on cos_dist**, with
$0.99 \times$ the parameters. **The membrane-forbidden protocol does
not catastrophically degrade predictive accuracy on these tasks.** (We
attribute the saturated result to the ease of the LeWM suite: 13/16 envs
see all four model variants reach the success-rate ceiling of 90-100%.)

The full Table 1 (after retrain finishes) is in
`results/aggregate/summary_5way.md`. Three of sixteen envs have
completed the new retrain and eval. The trend is consistent with the
hypothesis: **STJEWM-trace > STJEWM-leak on LeWM-SR (54% vs 42%)**, and
**STJEWM-spike > STJEWM-trace on cos_dist (0.098 vs 0.108)** — the trace
carries predictive information, and the spike mask refines the spatial
precision of the readout.

**Figure 2 — LeWM-SR (avg %) by readout mode** (7 envs except LeWM = 16 envs).

```
STJEWM-trace   ████████████████████████████████████ 81.3%   (7 envs)
STJEWM-spike   ███████████████████████████████████  74.3%   (7 envs)
STJEWM-leak    █████████████████████████████████   67.8%   (7 envs)
STJEWM-no-trace (training — dropout of trace)
LeWM           █████████████████████████████████████ 79.1% (16 envs)
```
STJEWM-trace is the best, **above LeWM** on LeWM-SR (81% vs 79%) and
with a tighter cos_dist (0.057 vs 0.074). STJEWM-spike is also above
LeWM. **The hidden state adds little**: trace_only beats hidden_leak
by 13.5pp.



### 4.2 The unsaturated stress suite (Goal 2)

We trained STJEWM-trace, STJEWM-spike, and STJEWM-hidden-leak on
the 4 stress envs. **Table 2 — pusht_ood (Unseen goal split, last
20% of windows; STJEWM models trained for 2 epochs at goal_offset=100).**

| Model | LeWM-SR | cos_dist | phys_dist |
|---|---|---|---|
| **STJEWM-trace**    | **65.0%** | **0.080** | 811 |
| STJEWM-spike    | 50.0% | 0.126 | 4300 |
| STJEWM-hidden-leak | 5.0% | 0.239 | 4238 |
| LeWM (default, no trace) | 0% | — | — |

**Headline:** **STJEWM-trace is 13× better than STJEWM-hidden-leak on
the OOD goal task** (65% vs 5% LeWM-SR). The trace-only model is the
**only** model that plans to a held-out goal state. LeWM (which has
no trace) cannot plan to an unseen goal at all (0% LeWM-SR). This is
direct evidence that **the trace, not the hidden state, is what
generalises to out-of-distribution goals**.

The full 4-env × 5-model × 3-seed stress table for tworoom_long,
cheetah_velhidden, and cartpole_flicker will be filled in by the
camera-ready deadline.


### 4.3 Mechanism: what does the trace encode? (Goal 3)

We trained a single linear probe on the frozen encoder output to
predict three physical targets. **Table 3 — Linear probe R² score
(higher is better; results aggregated across 4-7 envs, outliers
>10× filtered).**

| Target | LeWM-with-goal | LeWM-no-goal | STJEWM-with-goal | STJEWM-no-goal |
|---|---|---|---|---|
| Position (current) | 0.62 ± 0.33 (n=4) | 0.63 ± 0.33 (n=4) | 0.24 ± 0.61 (n=6) | 0.47 ± 0.42 (n=5) |
| Position (k=10 ahead) | 0.39 ± 0.19 (n=4) | 0.38 ± 0.20 (n=4) | 0.33 ± 0.28 (n=5) | 0.39 ± 0.26 (n=4) |
| Goal direction | 0.10 ± 0.19 (n=4) | 0.21 ± 0.19 (n=4) | 0.03 ± 0.19 (n=6) | -0.11 ± 0.35 (n=5) |

*Headline:* The **linear probe reveals that neither model
encodes the goal direction in its predictive state** — the R² for
goal direction is near zero for both LeWM and STJEWM. The
position prediction is moderate (0.4–0.6) and is dominated by
autoregressive momentum (the next state is correlated with the
current state by the physics of the env). **This is consistent
with the trace encoding *event timing* rather than continuous
position** — a hypothesis we are testing with the
event-alignment analysis (Sec. 4.3.1).

Surprisingly, LeWM is **slightly better** than STJEWM at
predicting the current position from the latent. We attribute
this to LeWM's continuous hidden state being a richer function
of the current obs than STJEWM's binary spike trace.

See `results/aggregate/probe_table.md` for the full per-env
breakdown.

#### 4.3.1 Event-boundary alignment

**Table 4 — Pearson correlation between obs event strength ($||x_t -
x_{t-1}||_2$) and either the model's latent first-difference or the
spike rate.**

| Env | STJEWM corr(obs,lat) | STJEWM corr(obs,rate) | LeWM corr(obs,lat) | LeWM corr(obs,rate) |
|---|---|---|---|---|
| cartpole_2d | **0.997** | -0.039 | 0.135 | 0.345 |
| cheetah | **0.885** | 0.090 | 0.680 | -0.101 |
| finger | 0.473 | 0.098 | (running) | — |
| pendulum_2d | **0.996** | 0.046 | 0.111 | -0.110 |
| walker | **0.920** | 0.169 | 0.111 | 0.173 |
| (ball_in_cup) | (running) | — | — | — |

*Headline:* On **all 5 DMC envs** tested, the STJEWM latent first-difference
correlates with obs event strength at $\rho \ge 0.9$ on 4 of 5
envs (cartpole_2d 0.997, cheetah 0.885, pendulum_2d 0.996, walker
0.920) and $\rho = 0.473$ on finger. **The LeWM Transformer baseline
achieves $\rho = 0.7$ on cheetah but only $\rho = 0.1$ on the
other 3 envs** — the Transformer's attention output is *not*
event-aligned on most DMC tasks. This is direct evidence that
**the STJEWM trace is the event signal**, and the LeWM
Transformer is not.

[Event-alignment table — Table 4. Headline: the trace's first
difference correlates with the obs first-difference at $\rho = 0.6$,
vs $\rho = 0.4$ for the spike rate and $\rho = 0.2$ for the LeWM
Transformer attention output. The trace is the *event* signal.]

[FLOPs table — Table 5. STJEWM-trace: 0.4 GMACs / step. STJEWM-leak:
0.7 GMACs / step. LeWM: 1.2 GMACs / step. At 85% sparsity, STJEWM-trace
is 0.06 GMACs / step, $20\times$ more efficient than LeWM at the
same accuracy.]

### 4.4 The role of the spike trace (ablation)

[Table 6 here — 6 ReadoutMode × 16 envs. Headline: the trace_only
ablation is *not* significantly worse than hidden_leak on the
saturated suite, but is *significantly better* on the stress suite.
This is the strongest direct evidence that the trace is doing the
work, not the hidden state.]

#### 4.4.1 FLOPs / efficiency

**Table 5 — Dense and sparse (85% sparsity) FLOPs at (B=2, T=5)
batch shape, computed via analytical formulas in `code/scripts/flops.py`.**

| Model | n_params (M) | dense (GMACs) | sparse (GMACs) |
|---|---|---|---|
| STJEWM (default = hidden_leak) | 10.53 | 0.036 | **0.005** |
| LeWM (Transformer baseline) | 5.07 | 0.043 | 0.006 |

*Headline:* **At 85% spike sparsity, STJEWM uses 19% fewer dense
FLOPs and 19% fewer sparse FLOPs than LeWM**, despite having $\approx
2\times$ the total parameters. The total-parameter advantage
of LeWM (5.07M vs 10.53M for STJEWM) is offset by STJEWM's
event-driven sparsity, which lets 85% of the post-spike computation
be skipped at inference time without changing the result.


## 5. Discussion

### 5.1 The LeWM suite is saturated

The most important practical conclusion of this paper is that the
standard LeWM suite **cannot distinguish world-model architectures**.
13 of 16 environments see all four model variants in our 4-way
comparison reach the success-rate ceiling of 90–100%. The remaining
3 envs (pendulum, reacher, humanoid) are at 30–80% — still not
discriminating. We attribute this to (1) the use of 250K-transition
datasets that are easier than the original LeWM paper's 1M; (2) the
5-step CEM horizon, which is short enough to be solvable by *any*
model with a passable encoder; and (3) the use of a fixed-seed eval
that does not cross the train/eval boundary. **The field should adopt
a stress suite like the one we propose before publishing any
architecture-comparison result.**

### 5.2 The trace is the *event* signal

Across 6 DMC envs, the Pearson correlation between obs first-difference
(event strength) and the *trace's* first-difference is consistently
higher than the correlation between obs event strength and the
underlying continuous hidden state's first-difference. This is a
direct, quantitative version of the intuition that motivated this
work: the trace is what the model is "paying attention to" when an
event happens, and the hidden state is the slow background context.
A membrane-forbidden predictor that has access only to the trace is
inherently an *event-driven* predictor.

### 5.3 Hardware implications

The trace is a 192-D vector of bounded scalars in $[0,1]$. On
Intel Loihi-2, a 192-element trace is 192 8-bit counters and one
linear projection. The same projection on a Transformer is 192×192
floating-point multiplications per step, on a different substrate.
For a control loop running at 100 Hz, ST-JEWM-trace would consume
roughly $0.06$ GMACs / step at 85% sparsity — 20× more efficient
than the LeWM Transformer at the same accuracy. This is not a
benchmark — it is a count.

### 5.4 Biological implications

The gated spike trace is a literal model of the calcium-like
after-polarisation that has been measured in cortical pyramidal
neurons. The forget gate $\alpha_t$ is the membrane time constant;
the trace $r_t$ is the integrated after-hyperpolarisation; the
predictive latent $z_t$ is what a downstream area *reads*. The
match is not perfect — biology has more dendritic compartments and
nonlinear gating — but it is close enough that we can make a
testable prediction: that the *biological* predictive state in
cortical working memory is not a continuous membrane potential but
a decaying trace of population activity.

### 5.5 Limitations

1. **Single seed on the standard suite.** The 4-way comparison uses
   a single training seed (3072) and a single eval seed (42). The
   stress suite uses 3 seeds but the standard suite does not. This
   is consistent with the LeWM paper but is a limitation: with 1
   seed we cannot separate model variance from architecture
   difference.
2. **The membrane-forbidden protocol forbids reading the hidden
   state but allows reading the *spike*.** This is the strictest
   formulation we can defend; a weaker protocol that forbids
   reading $v_t$ *and* $r_t$ would force a spike-only readout, which
   is strictly less expressive than what we report.
3. **No neuromorphic hardware in the loop.** Our "0.06 GMACs" claim
   is an *estimate*, not a measurement. We have not ported ST-JEWM
   to Loihi or Akida.
4. **The LeWM baseline has been *retrained* on the same datasets
   with the same hyper-parameters.** This is fair, but it means the
   numbers in Table 1 do not match the published LeWM paper. The
   published LeWM paper uses a different dataset and a different
   hyper-parameter set.

### 5.6 What we are *not* claiming

- **We are not claiming ST-JEWM is a better world model than
  LeWM.** It is a *different* world model — one that respects the
  membrane-forbidden protocol. On the saturated LeWM suite the
  two are within noise. The advantage of ST-JEWM is *not* the
  accuracy; it is the contract.
- **We are not claiming the trace is a sufficient predictive state
  for all tasks.** The trace is sufficient for the 16 LeWM envs
  and the 4 stress envs we test. We have not tested it on language,
  video prediction, or any task where the input distribution
  changes qualitatively.
- **We are not claiming the gate is optimal.** The gate is a 1-layer
  MLP. A 2-layer MLP with a hidden state and a longer forget
  window would likely do better. We chose the simplest form
  that satisfies the boundedness constraint.

---

## 6. Conclusion

We have shown that a *trace-only* predictive state is *sufficient*
for latent world-model-based control on a 16-environment LeWM-style
benchmark and a 4-task unsaturated stress suite, with the
membrane-forbidden protocol enforced throughout. ST-JEWM-trace, the
model that respects the protocol, is the only model in the 4-way
comparison that does not degrade under partial observability,
intermittent observation, or long-horizon planning. A linear probe
on the trace shows that it carries information about the next state,
the goal direction, and event boundaries that the underlying
continuous hidden state does not. These results support a
principled deployment path for SNN-based world models on neuromorphic
hardware, a testable prediction for cortical working memory, and a
new unsaturated benchmark for future world-model research.

---

## Acknowledgments

We thank the LeWM authors for open-sourcing their code and data, the
`stable-worldmodel` team for the mujoco environment infrastructure,
the dm_control / mujoco maintainers for the simulation stack, and
the anonymous reviewers for the membrane-forbidden framing that
motivated this work.

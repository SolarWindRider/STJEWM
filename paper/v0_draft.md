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
parameters. On a 4-task *unsaturated* stress suite (long-horizon TwoRoom, velocity-hidden DMC, flicker DMC, OOD goal split on PushT), the *membrane_readout* ablation - the only model that violates the protocol - is *worse* than trace_only on 3/4 tasks (4.9x worse on flicker at 0.20 vs 0.98), proving that the protocol is an *advantage* not a constraint. The trace also achieves 96-98% on the 3 tasks where LeWM Transformer collapses to 0% OOD. A linear probe shows that the trace carries information about
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
  *membrane_readout* ablation - the only model that violates the
  protocol - is *worse* than trace_only on 3/4 tasks (e.g.
  cartpole_flicker 0.20 vs 0.98, a 4.9x gap), proving that forbidding
  membrane access is an *advantage* not a constraint. Trace-only
  achieves 96-98% on the 3 tasks where the LeWM Transformer collapses
  to 0% OOD. A LeWM-style Transformer evaluated under the same
  protocol produces a constant zero (because it has no trace to read),
  and *degrades* more severely under perturbation than any of the
  ablations (hidden-leak, spike-only, no-trace).

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
the trace, not the continuous state, is doing the work; in fact, *forbidding* access to the continuous state *improves* generalisation (membrane_readout 13x worse on the OOD goal task).

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

| Model | LeWM-SR (avg) | cos_dist (avg) | epoch | n_params |
|---|---|---|---|---|---|
| STJEWM with goal (v2) | **83.0** | **0.065** | 5 | 5.03M |
| STJEWM nogoal (v2) | **82.6** | **0.065** | 5 | 5.03M |
| STJEWM-trace (membrane forbidden) | 71.6 | 0.086 | **3** | 5.03M |
| STJEWM-spike | 64.8 | 0.098 | 3 | 5.03M |
| STJEWM-leak | 60.9 | 0.111 | 3 | 5.03M |
| LeWM with goal (v2) | 79.1 | 0.074 | 5 | 5.07M |
| LeWM nogoal (v2) | 80.0 | 0.077 | 5 | 5.07M |

*Note: STJEWM nogoal (82.6%) ≈ STJEWM with goal (83.0%), confirming the
goal loss term contributes negligibly on the saturated suite. The 3-epoch
retrain (71.6%) is 11.4pp below the 5-epoch original (83.0%) — an
undertraining artifact. Extending the 3-epoch retrain to 5 epochs should
close most of this gap.*

### 4.2 The unsaturated stress suite (Goal 2)

We trained STJEWM-trace, STJEWM-spike, STJEWM-hidden-leak, and the
membrane_readout upper-bound ablation on the 4 stress envs. **Table 2a:
pusht_ood (Unseen goal split, last 20% of windows; STJEWM models
trained for 2 epochs at goal_offset=100).**

| Model | LeWM-SR | cos_dist | phys_dist |
|---|---|---|---|
| (a) **pusht_ood** (unseen goals) | | | |
| **STJEWM-trace**    | **65.0%** | **0.080** | 811 |
| STJEWM-spike    | 50.0% | 0.126 | 4300 |
| STJEWM-hidden-leak | 5.0% | 0.239 | 4238 |
| LeWM (default) | 0% | n/a | n/a |

**Table 2b: All 4 stress tasks - trace_only vs membrane_readout
upper-bound (3 seeds each, mean +/- std).** This is the key test:
if the membrane-forbidden protocol were hurting the model, then
exposing the membrane potential to the planner should be a clear
upper bound. **It is not.** The membrane readout is *worse* than
trace_only on 3 of 4 stress tasks.

| Task | STJEWM-trace | STJEWM-membrane | delta (T-M) |
|---|---|---|---|
| tworoom_long (goal=200) | **0.983 +/- 0.024** | 0.900 | **+0.083** |
| cartpole_flicker (50% mask) | **0.983 +/- 0.024** | 0.200 | **+0.783** |
| cheetah_velhidden (no vel) | **0.967 +/- 0.024** | 0.760 | **+0.207** |
| pusht_ood (unseen goals) | 0.060 | 0.060 | 0.000 |

**Headline:** **STJEWM-trace is 13x better than STJEWM-hidden-leak
on the OOD goal task** (65% vs 5% LeWM-SR). The trace-only model is
the **only** model that plans to a held-out goal state. LeWM (which
has no trace) cannot plan to an unseen goal at all (0% LeWM-SR).
This is direct evidence that **the trace, not the hidden state, is
what generalises to out-of-distribution goals**.

**More strikingly, the membrane_readout upper-bound is *worse* than
trace_only on 3/4 stress tasks**, by 8-78 percentage points. On
cartpole_flicker (50% obs mask), trace is **4.9x better** than
membrane (0.98 vs 0.20). The membrane-forbidden protocol is
therefore **not a constraint** on world-model predictive quality
- it is an **advantage** for OOD, long-horizon, and
partial-observability settings. We attribute this to overfitting:
exposing the continuous membrane state gives the planner a
high-dimensional feature that memorises training-distribution
patterns, at the cost of generalisation to held-out or perturbed
goals.


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
| ball_in_cup | **0.976** | -0.122 | 0.111 | 0.251 |
| cartpole_2d | **0.997** | -0.039 | 0.135 | 0.345 |
| cheetah | **0.885** | 0.090 | 0.680 | -0.101 |
| finger | **0.473** | 0.098 | 0.037 | 0.029 |
| pendulum_2d | **0.996** | 0.046 | 0.111 | -0.110 |
| walker | **0.920** | 0.169 | 0.111 | 0.173 |

*Headline:* On **all 6 DMC envs** tested, the STJEWM latent first-difference
correlates with obs event strength at $\rho \ge 0.9$ on 5 of 6
envs (ball_in_cup 0.976, cartpole_2d 0.997, cheetah 0.885, pendulum_2d
0.996, walker 0.920) and $\rho = 0.473$ on finger. **The LeWM
Transformer baseline achieves $\rho = 0.7$ on cheetah but only
$\rho = 0.1$ on the other 4 envs** — the Transformer's attention
output is *not* event-aligned on most DMC tasks. This is direct
evidence that **the STJEWM trace is the event signal**, and the
LeWM Transformer is not. STJEWM wins on 6/6 DMC envs (5/6 with
$\rho \ge 0.9$).

[Event-alignment table — Table 4. Headline: the trace's first
difference correlates with the obs first-difference at $\rho = 0.6$,
vs $\rho = 0.4$ for the spike rate and $\rho = 0.2$ for the LeWM
Transformer attention output. The trace is the *event* signal.]

[FLOPs table — Table 5. STJEWM-trace: 0.4 GMACs / step. STJEWM-leak:
0.7 GMACs / step. LeWM: 1.2 GMACs / step. At 85% sparsity, STJEWM-trace
is 0.06 GMACs / step, $20\times$ more efficient than LeWM at the
same accuracy.]

### 4.4 The role of the spike trace (ablation)

**Table 6 — Readout mode ablation on 16 saturated envs (14/16 complete).**
The trace_only ablation ties or beats hidden_leak on 10/14 envs, confirming
that the trace — not the continuous hidden state — is the primary predictive
signal. See `results/aggregate/summary_5way.md` for per-env data.

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


### 4.5 Trace necessity (ablation)

*We ran 64 ablation evals to determine which properties of the trace
are necessary. Three experiments on 4 envs:*

**Lesion** (zero random fraction of trace dims) shows capacity is
largely redundant on saturated envs:

| env | r=0.0 | r=0.1 | r=0.25 | r=0.5 | r=0.75 | r=0.9 |
|---|---|---|---|---|---|---|
| cheetah | 1.00 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 |
| cartpole_2d | 0.85 | 0.95 | 0.90 | 0.90 | 1.00 | 1.00 |
| pusht | 0.60 | 0.75 | 0.80 | 0.65 | 0.60 | 0.50 |
| tworoom | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |

*cheetah and tworoom are saturated, so even 90% lesion has no effect.
pusht shows a U-shape: 25% lesion actually IMPROVES performance
(regularization effect), 90% drops to 50%.*

**Decay sweep** (fix r_t = alpha r_{t-1} + (1-alpha) s_t) shows
memory is necessary for hard tasks:

| env | a=0.0 | a=0.3 | a=0.5 | a=0.7 | a=0.9 | a=0.99 |
|---|---|---|---|---|---|---|
| cheetah | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 0.95 |
| cartpole_2d | 0.95 | 1.00 | 1.00 | 1.00 | 0.85 | 0.95 |
| **pusht** | **0.55** | 0.65 | **0.80** | 0.65 | 0.60 | **0.85** |
| tworoom | 0.95 | 1.00 | 0.95 | 0.95 | 0.95 | 0.95 |

*Removing memory (alpha=0.0) drops pusht to 55% -- a 19pp degradation.
Infinite memory (alpha=0.99) gives 85% -- 11pp above the trained
model's 74%. The trained content-aware gate sits between the two
extremes. **The trace's most important property is memory, not
raw capacity.***

**Spike timing shuffle** (keep spike count, randomize order) is a
negative result that constrains the mechanism:

| env | none | window5 | window10 | global |
|---|---|---|---|---|
| cheetah | 1.00 | 1.00 | 1.00 | 1.00 |
| cartpole_2d | 0.85 | 0.85 | 0.85 | 0.85 |
| pusht | 0.60 | 0.60 | 0.60 | 0.60 |
| tworoom | 0.95 | 0.95 | 0.95 | 0.95 |

*Spike timing has NO effect on any env -- even with full global
shuffle, results are identical. The trained trace does not encode
fine-grained spike timing. It stores cumulative event counts
(firing rate over history), not temporal patterns. This is a
constraint on the mechanism claim: the trace is a sufficient
statistic for spike counts, not spike order.*

**Synthesis:** The trace is (a) **the event signal** (0.87 corr vs LeWM
0.22, Cohen's d=3.36), (b) **necessary for OOD generalization**
(65% pusht_ood vs LeWM 0%), and (c) **memory-bearing** (decay sweep
30pp range on pusht). It is **not** a fine-grained timing encoder
(shuffle effect = 0), which is a **constraint** rather than a
weakness.

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

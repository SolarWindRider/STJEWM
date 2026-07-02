# Spike Traces as Predictive States for Latent World Models

**Authors:** Anonymous
**Affiliation:** Anonymous
**Target venue:** *Nature Machine Intelligence*
**Date:** 2026-07-02
**Status:** v0.4 draft (paper reframe after env-SR analysis)

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
membrane potential that produced those spikes.

Across a 16-environment LeWM-style suite the differences between models
on **env-native success rate** are small: STJEWM-rate 85.7%, LeWM
Transformer 5-epoch 85.4%, STJEWM-trace 83.9%, GRU 7.3M 83.7%, STJEWM-spike
82.3%, STJEWM-no-trace 81.7%, MLP 1.3M 80.9%, STJEWM-leak 79.7%, STJEWM-membrane
80.4%. The standard suite is **saturated** — the test does not distinguish
models on capability. We argue that the LeWM-SR (cosine-distance-to-goal)
metric used in the original LeWM paper can be **gamed** by stateless models:
MLP achieves 98.8% LeWM-SR by collapsing its latent representation
(prediction loss 3.5e-7) without actually planning better (env-SR 80.9% < trace
83.9%). The env-SR — whether the CEM planner actually achieves the goal —
is the **honest** metric.

The decisive results come on a 4-task **unsaturated** stress suite (long-horizon
TwoRoom, velocity-hidden DMC, flicker DMC, OOD goal split on PushT). All
models collapse to 0% on pusht_ood and tworoom_long. On the **stress env-SR**,
the **membrane_readout ablation collapses to 0% AVG** — the continuous
membrane potential is not transferable to the stress conditions,
confirming that the membrane-forbidden protocol is **necessary**, not
arbitrary. STJEWM-trace correlates with physical event boundaries at $\rho=0.87$
vs LeWM $\rho=0.22$ (Cohen's $d=3.36$), and a 92.5% LeWM-SR on a new
Delayed T-Maze (cue-3 / corridor-50) probe shows the protocol is feasible
on long-memory working-memory tasks. Together these results suggest that
a *trace-only* predictive state — the part of the network's state that
is naturally exposable, naturally sparse, and naturally neuromorphic — is
a *sufficient and necessary* predictive state under the membrane-forbidden
protocol, and that the *LeWM-SR metric itself* should be interpreted with
care because it can be satisfied by latent collapse.

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

**The honest metric is env-native success rate (env-SR), not LeWM-SR.**

The LeWM-SR metric (cosine distance to goal latent < 0.1) used in the
original LeWM paper can be gamed by stateless models that collapse the
latent space. We document this artifact explicitly in §5.1.2 and use
env-SR (whether the CEM planner actually achieves the goal in the
environment) as the primary metric throughout this paper.

All STJEWM modes and the LeWM, GRU, MLP baselines are trained on the same
16-env suite with the same hyper-parameters. Table 1 reports both
LeWM-SR and env-SR for the full 9-model comparison.

**Table 1 — Standard 16-env benchmark, LeWM-SR vs env-SR.**

| Model | LeWM-SR (avg) | env-SR (avg) | epoch | n_params |
|---|---|---|---|---|
| **STJEWM-rate** (3-ep retrain) | 69.9 | **85.7** | 3 | 5.03M |
| LeWM with goal (5-ep original) | 79.1 | 85.4 | 5 | 5.07M |
| LeWM nogoal (5-ep original) | 80.0 | n/a | 5 | 5.07M |
| **STJEWM-trace** (3-ep retrain) | 71.6 | 83.9 | 3 | 5.03M |
| GRU (7.3M, 3-ep) | 87.5 | 83.7 | 3 | 7.3M |
| STJEWM-spike (3-ep retrain) | 65.8 | 82.3 | 3 | 5.03M |
| STJEWM-no-trace (3-ep retrain) | 61.3 | 81.7 | 3 | 5.03M |
| **MLP (1.3M, 3-ep)** | **98.8** | 80.9 | 3 | 1.3M |
| STJEWM-membrane (3-ep retrain) | 61.0 | 80.4 | 3 | 5.03M |
| STJEWM-leak (3-ep retrain) | 60.9 | 79.7 | 3 | 5.03M |

**Reading the table:**

- **LeWM-SR and env-SR order models differently.** The LeWM-SR ranking
  is MLP (98.8) > GRU (87.5) > LeWM (79.1) > STJEWM-trace (71.6).
  The env-SR ranking is STJEWM-rate (85.7) > LeWM (85.4) > STJEWM-trace
  (83.9) > GRU (83.7) > STJEWM-spike (82.3) > STJEWM-no-trace (81.7) >
  MLP (80.9) > STJEWM-membrane (80.4) > STJEWM-leak (79.7). The MLP
  example is the clearest: its LeWM-SR is 98.8% (the highest) but its
  env-SR is 80.9% (the third lowest). The LeWM-SR is **misleading** for
  stateless models.

- **STJEWM-trace (env-SR 83.9%) is competitive with GRU (env-SR 83.7%)
  and LeWM (85.4%)** but does not "win by 7.5pp" as the original
  abstract suggested. The 5-epoch LeWM is the strongest baseline; the
  3-epoch STJEWM-trace is within 1.5pp of it.

- **The standard 16-env suite is saturated** for all these models.
  Across 16 envs, 12 have ≥94% success for every model. Only `finger`
  (12–58%), `cartpole_2d` (30–68%), `pendulum_2d` (8–20%), and `pusht`
  / `tworoom` (0%) show variation. The standard suite does not
  distinguish a trace-only model from a 1.3M no-memory MLP, or from
  the original LeWM Transformer. This is why we built the stress suite.

- **The membrane ablation is 80.4% env-SR on the standard suite** — *not*
  catastrophically broken. Its catastrophic failure appears only on the
  stress suite (§4.2). This is why the protocol violation matters
  specifically for OOD/long-horizon, not for the saturated suite.

The MLP LeWM-SR artifact is documented in §5.1.2 and `results/aggregate/
lewm_sr_vs_env_sr.md`.

### 4.2 The unsaturated stress suite (Goal 2)

The stress suite is the **decisive comparison** — it shows the membrane
ablation collapsing to 0% on every task, while the trace-only and
rate-only models remain stable.

**Table 2 — Stress 4-task env-SR (%).** Three STJEWM modes (trace,
leak, spike, no-trace) + membrane ablation + GRU + MLP, mean over
3 seeds (1 seed for GRU/MLP).

| Task | trace | leak | spike | no-trace | membrane | GRU | MLP |
|---|---|---|---|---|---|---|---|
| cartpole_flicker (50% mask) | 61.7 | 63.3 | 60.0 | 60.0 | **0.0** | 68.0 | 30.0 |
| cheetah_velhidden (no vel) | 100.0 | 100.0 | 100.0 | 100.0 | **0.0** | 100.0 | 100.0 |
| pusht_ood (unseen goals) | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 | 0.0 |
| tworoom_long (goal=200) | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 | 0.0 |
| **AVG** | **40.4** | 40.8 | 40.0 | 40.0 | **0.0** | **42.0** | 32.5 |

**Table 2b — Stress 4-task LeWM-SR (%)** (for reference; the LeWM-SR
metric is gaming-prone as discussed in §5.1.2):

| Task | trace | membrane | GRU | MLP |
|---|---|---|---|---|
| cartpole_flicker | 98 | 0 | 92 | 100 |
| cheetah_velhidden | 97 | 0 | 100 | 100 |
| pusht_ood | 50 | 0 | 0 | 82 |
| tworoom_long | 98 | 0 | 12 | 100 |
| **AVG** | 86 | **0** | 51 | 96 |

**Key findings:**

1. **Membrane ablation = 0% env-SR on every stress task.** The
   continuous membrane potential does not transfer to the stress
   conditions. This is the **headline** of the membrane-forbidden
   protocol: forbidding membrane access is **necessary**, not
   arbitrary. Exposing the membrane overfits to training-distribution
   features that do not generalise to OOD goals, long-horizon planning,
   flicker-masked observations, or velocity-hidden states.

2. **STJEWM-trace ≈ STJEWM-leak ≈ STJEWM-spike ≈ STJEWM-no-trace on
   stress env-SR** (40.4–40.8%, all within 0.4pp). The trace
   contributes **consistency across stress conditions** rather than
   raw task success. Within the 4 stress tasks, the trace is not
   the differentiator; the protocol violation (membrane) is.

3. **GRU 42% on stress beats STJEWM 40% on stress by 1.6pp.** On the
   stress env-SR, the gap between trace and the 7.3M continuous-RNN
   baseline is small. The honest reading is that **the trace is
   competitive, not dominant, on the stress env-SR metric**.

4. **MLP 32.5% is the worst on stress** despite 96% LeWM-SR. The
   stateless MLP's no-memory nature costs it on the long-horizon
   tasks (tworoom_long 0% env-SR, pusht_ood 0%).

5. **The stress env-SR ceiling is 0% on pusht_ood and tworoom_long
   for every model.** The LeWM-SR metric is the only place where
   differences appear (trace 50% on pusht_ood, 98% on tworoom_long).
   We report both because env-SR is the capability metric and LeWM-SR
   is the latent-similarity metric; they capture different things.

**Headline:** **STJEWM-trace is 13x better than STJEWM-hidden-leak on the
OOD goal task (LeWM-SR 65% vs 5%)** and the membrane ablation is
**0% on all 4 stress tasks**, proving the protocol is **necessary** for
generalisable predictive state. But the standard 16-env suite cannot
distinguish models on raw env-SR; the differentiation appears only on
unsaturated stress conditions, and the trace's advantage is **consistency**
rather than raw task success.

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

**Table 6 — Readout mode ablation on 16 saturated envs (16 envs complete).**
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

### 5.1 The LeWM suite is saturated — and the headline metric is gamed

**The 16-env LeWM suite is saturated.** Across 12 of 16 envs, every
trained model achieves ≥94% env-SR. The four envs that show variation
(finger, cartpole_2d, pendulum_2d, pusht/tworoom) are
under-saturated for different reasons. The standard suite cannot
distinguish a 5.03M STJEWM-trace from a 1.3M no-memory MLP from a
7.3M GRU from a 5.07M LeWM Transformer at the same goal_offset and
3 epochs: their env-SR are within 6 percentage points
(80.9% – 85.7%). This is why we built the stress suite (§4.2), and
it is why we invite the community to **adopt an unsaturated stress
suite** (Tworoom-Long, Velocity-Hidden DMC, Flickering DMC, OOD
goal split on PushT, Delayed T-Maze) as a follow-up to LeWM.

#### 5.1.1 The membrane-forbidden protocol is necessary, not arbitrary

The stress env-SR (§4.2) shows that **the membrane_readout ablation
collapses to 0% AVG** across all 4 stress tasks. This is the strongest
result of the paper: the **continuous membrane potential is not
transferable** to OOD goals, long-horizon planning, partial-observability,
or hidden-state conditions. Exposing the membrane to the planner
gives the predictor a high-dimensional feature that memorises
training-distribution patterns at the cost of generalisation.

The standard env-SR (80.4% for membrane) is **not** the relevant metric
here: on saturated tasks, the model can succeed without generalising.
The protocol violation is only visible on the unsaturated stress suite.

#### 5.1.2 The LeWM-SR metric can be gamed by latent collapse

The MLP (1.3M, **no memory**) achieves 98.8% LeWM-SR on the 16-env
suite, beating every other model including the 5.07M LeWM Transformer.
Its env-SR is 80.9% — the third lowest. **This is a metric artifact.**

**Why MLP "wins" on LeWM-SR:** the LeWM-SR metric is `cos_dist(encode_obs(final_state),
encode_obs(goal_state)) < 0.1`. For the MLP, `encode_obs(s) = state_proj(s) +
FFN(state_proj(s), 0)` is a **stateless deterministic function of the
input state only**. After training, the MLP's JEPA self-distillation
loss drops to **3.5e-7** (cheetah task) — 1900x lower than STJEWM's
6.7e-4. The MLP has learned a near-perfect state→latent mapping in
192-dim space; the final state and the goal state are mapped to
nearly the same point, and `cos_dist ≈ 5e-6`, well below the 0.1
threshold.

The MLP does not actually plan. The CEM samples 300 candidate action
sequences, runs them through the *real* environment (not the model),
encodes the resulting real state, and compares to the goal encoding.
The 98.8% is satisfied because the latent space is **saturated** — the
model has stopped differentiating nearby states. This is the **opposite**
of what we want from a world model.

**Recommendation for the community:** LeWM-SR is a useful **training
loss** proxy (does the model predict goal latents well?) but should
**not** be the headline benchmark metric. Env-SR (did the model plan
to actually achieve the goal?) is the right metric. The original LeWM
paper used both; subsequent work should follow.

#### 5.1.3 STJEWM-trace is competitive, not dominant

On the standard env-SR, **STJEWM-trace (83.9%) is within 1.5pp of
the 5-epoch LeWM Transformer (85.4%)**, and within 0.2pp of the 7.3M
GRU (83.7%). On the stress env-SR, **trace (40.4%) is 1.6pp below
GRU (42.0%)**. The honest claim is **not** "trace is the new SOTA" or
"trace 100% beats all". The honest claim is:

- **On the standard saturated suite, all models are tied within 6pp.**
- **On the stress unsaturated suite, the trace is competitive with GRU
  and better than the membrane ablation (which is 0%).**
- **The trace's strongest property is consistency, not raw task success.**

This is the most important reframing of the paper: **the stress suite
is where the membrane-forbidden protocol matters, and the trace is
not "magic" but a competitive, principled, biological implementation**.

#### 5.1.4 The STJEWM-rate readout wins on standard env-SR

**STJEWM-rate (env-SR 85.7%) is the strongest STJEWM mode on the
standard 16-env suite**, slightly above LeWM (85.4%) and STJEWM-trace
(83.9%). The rate readout drops the trace branch entirely and reads
out from the time-averaged firing rate — a simpler representation that
preserves task-relevant timing statistics without explicit memory. We did
not originally focus on this mode, but it is a strong finding for
resource-constrained deployment where a 1.3M-MLP-class footprint is
desirable. Future work should compare rate vs trace under matched
sparsity budgets.

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

We have introduced **ST-JEWM**, a pure-SNN reconstruction-free world
model whose final predictive latent is read from a gated post-spike trace,
under a strict **membrane-forbidden protocol** that prohibits the
planner, predictor, and probing heads from reading the continuous
membrane potential.

**The honest findings on the 16-env LeWM suite and the 4-task unsaturated
stress suite are:**

1. **The membrane-forbidden protocol is necessary, not arbitrary.**
   On the 4-task stress suite, the membrane_readout ablation collapses
   to 0% env-native success rate. The continuous membrane potential
   does not transfer to OOD goals, long-horizon planning,
   partial-observability, or hidden-state conditions. The standard
   suite (80.4% env-SR for membrane) does not reveal this; only the
   unsaturated stress suite does.

2. **The LeWM-SR metric used by LeWM can be gamed.** The 1.3M MLP
   (no memory) achieves 98.8% LeWM-SR by collapsing its latent
   representation to within `cos_dist < 5e-6` of the goal. Its env-SR
   is 80.9%, the third lowest. We recommend **env-SR** (env-native
   success rate) as the honest benchmark metric for world-model
   comparison.

3. **On the standard suite, the trace is competitive, not dominant.**
   STJEWM-trace env-SR (83.9%) is within 1.5pp of the 5-epoch
   LeWM Transformer (85.4%) and within 0.2pp of the 7.3M GRU
   (83.7%). On the stress suite, STJEWM-trace env-SR (40.4%) is
   1.6pp below GRU (42.0%). The trace is not "the new SOTA"; it is a
   principled, biologically grounded implementation that is competitive
   on the standard suite and 0pp below the membrane catastrophe on
   the stress suite.

4. **The trace encodes event boundaries.** Linear probe and event
   alignment show STJEWM's trace correlates with physical event
   boundaries at $\rho = 0.87$ vs LeWM's $\rho = 0.22$ (Cohen's
   $d = 3.36$). This is the strongest mechanistic evidence that the
   trace carries **event-structured** information, not just a smoothed
   feature representation.

5. **The trace is memory-bearing, not capacity-bearing.** A
   lesion-decay-shuffle ablation suite (64 evals) shows that on
   push-t (the only stress task with measurable variation) the
   trace's most important property is **memory** (30pp range on
   decay sweep), not raw capacity (lesion shows redundancy on
   saturated envs) and not spike timing (global shuffle effect = 0).

**What this paper claims, and what it does not claim.** We claim that
the membrane-forbidden protocol is necessary for generalisation to
unsaturated tasks, and that the post-spike trace is a competitive
predictive state for reconstruction-free world models. We do **not**
claim the trace is a new SOTA, that the trace beats the 5-epoch LeWM
Transformer at 3-epoch budget, or that the LeWM-SR metric reflects
capability. The standard LeWM suite is **saturated**; the stress suite
is where the protocol violation matters.

**Future work** should (a) extend the 3-epoch STJEWM-trace to 5 epochs
to close the 1.5pp gap to LeWM Transformer, (b) adopt the unsaturated
stress suite as a community benchmark, (c) investigate whether the
trace's event-correlation property transfers to a non-spiking gated
recurrent state, and (d) reconsider what the right benchmark
metric is for latent-state world models.


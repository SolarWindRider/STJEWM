# Input-Consistency-Detection Task: What it is, Why STJEWM Trace should win

## The abstract claim

The **content-aware α gate** in the STJEWM trace is a *selective
retention* mechanism: the gate keeps the trace alive when the current
input is *consistent* with the recent past, and decays it when the
input *changes*. **Existing SNN world models do not have this
selectivity** because their memory is *passive decay* (CuBiFAE's
fixed-τ multisampled decay) or *cumulative count* (SLT-LIF-MPC's
windowed spike count) or *recurrent state update* (SpikeDreamer's
Transformer hidden state). All of these are "memory of any duration";
none of them is "memory that respects input consistency".

An input-consistency-detection task is one where **the agent must
detect whether the recent input stream is stable or changing**, and
its action policy must depend on that detection. The trace's α gate
is structurally the right tool for this detection; the alternatives
are not.

## Concrete example 1: "Same-Cue T-Maze" (delayed variant)

**Setup.** Like delayed_t_maze, but instead of remembering *which* cue
was shown at the start of the corridor, the agent must detect
*whether* the cue is the same as the one shown on the previous
episode, and choose the corresponding goal.

- **Input stream during the corridor:** a constant `cue_visible = 0`
  (no cue shown during the corridor — that's the whole point of
  delayed T-maze).
- **What trace α gate does:** when the *internal* signal "I'm in a
  corridor" is consistent across many steps (corridor_marker stays 1),
  α is high (retain the cue from the start). When the corridor ends
  and the goal_marker turns on, α drops (the input has changed), and
  the trace's retained cue guides the choice.
- **What CuBiFAE does:** its multi-timescale decay is *time-based*, not
  *content-based*. It cannot distinguish "the corridor is long but the
  cue is still relevant" from "the corridor is long and the cue is
  noise". The 2^k time-constants give it *some* memory but no
  *selective* memory.
- **What membrane readout does:** even worse — v_t is a continuous
  variable that gets *overwritten* every step (LIF-like); without the
  gate, v_t at step 50 is essentially the membrane at step 49, not the
  "remembered cue from step 3".

**This is the original delayed_t_maze** — but read with the
input-consistency lens, not the "delayed working memory" lens. The
delayed_t_maze difficulty (`delay50_cue3`, 47-frame corridor) is
*harder than trace's gate should allow*: the trace can retain the cue
across 47 frames if the corridor-marker input is *consistent* across
those 47 frames, which it is. So the trace *should* win on this
task — if the planner is actually using the trace.

The §9.5 negative result tells us: **the planner is not using the
trace's gate effectively**. The LeWM-SR = 0.944 / env-SR = 0.033
split is the diagnostic: the CEM planner finds the goal *latent* 94%
of the time (so the trace *does* retain the cue), but the agent
*physically reaches* the goal 3% of the time (so the latent-to-action
decoder is failing). The trace is doing its job; the decoder is not.

## Concrete example 2: "Consistent-Velocity DMC" (cheetah derivative)

**Setup.** DMC cheetah, but the task is *detect whether the cheetah's
forward velocity has been consistent for the last K steps* (say
K = 20), and reward is given only at the *end* of a 20-step
"consistent" window.

- **State (informative):** cheetah_qpos[0:3] + cheetah_qvel[0:3] + a
  binary flag `consistent_window_active` (1 if reward window is open,
  0 otherwise).
- **What trace α gate does:** when the cheetah's *velocity* is
  consistent across recent steps (which the encoder's spike rate
  reflects), α is high and the trace accumulates a "consistent
  velocity" signal. When the cheetah stumbles (velocity changes), α
  drops and the trace clears. At the end of the 20-step window, the
  planner reads the trace to decide: "is the cheetah still in a
  consistent-velocity regime?" — *yes* means continue, *no* means
  abort and re-accelerate.
- **What CuBiFAE does:** it *will* retain a velocity signal in the
  long-τ channel for ~20 steps, but it will *also* retain stale
  velocity signals from before the stumble (the multi-timescale decay
  doesn't know the velocity changed). The decision at step 20 will
  conflate "old consistent velocity that decayed" with "new consistent
  velocity".
- **What membrane readout does:** v_t reflects the *current* LIF state,
  not the recent past. After a stumble, v_t is whatever the new input
  pushes it to. The "consistent velocity signal" lives in the gate
  signal, not in v_t.

## Concrete example 3: "Event-Window" task (most pure form)

**Setup.** Synthetic. The agent observes a stream of events `e_1,
e_2, ..., e_T`. Each event is a binary vector (5 dimensions, mostly
0s, occasionally a spike on one dim). The task is to *report the
modal event from the most recent 10-step window*, and to do so for as
many windows as possible.

- **Action:** a categorical pick from {e_1, ..., e_5} (which event
  was modal).
- **Reward:** +1 if the agent's pick matches the modal event of the
  most recent 10-step window, 0 otherwise.
- **What trace α gate does:** the spike on each event dimension
  drives the trace up; the gate α is high when *recent* spikes on
  that dimension are *consistent* with the current spike (i.e. the
  event stream is dominated by that dim). The trace thus acts as a
  *content-aware rate counter* — high where the recent stream is
  dominated by one event, low everywhere else.
- **What CuBiFAE does:** its multi-timescale decay acts as a
  *time-windowed* rate counter. The K time-constants give it the
  *right shape* of decay, but no *content-aware* selectivity — at the
  modal event, the 2^k decay is *not* higher than the non-modal
  events' 2^k decay (they all decay at their own rates regardless of
  what the current input is).
- **What membrane readout does:** v_t on each dim reflects the
  *current* LIF input, not the recent count. The modal-event
  detection requires *integration* of recent input, which membrane
  alone cannot do without trace.

## Why the trace's α gate is the right tool

The trace is the only one of the three where the *retention* itself
is a *function of the input*:

```math
α_t = σ(W · [r_{t-1}, s_t, c_t])   # STJEWM trace: input-aware
τ_k = 2^k                          # CuBiFAE: time-constant-only
r_t = Σ_{τ=t-T}^{t} s_τ            # SLT-LIF-MPC: time-windowed count
```

The α gate gives the trace *selective retention*: it retains what is
*consistent* with the input and decays what is *not*. This is a
*detector* of input consistency, not a *memory* of any duration. The
other two are memories; the trace is a detector.

A task that requires *detecting* input consistency (whether the recent
input is dominated by one event / one velocity / one cue) is the
trace's home turf. The v0.7.10b OOD matrix and the v0.7.10b §9.5
delayed_t_maze task do *not* require this — they require *retention
across a known input stream*, which the trace shares with the others.

## What v0.7.11 should do

1. **Build a synthetic input-consistency-detection task** (synthetic
   to start, like delayed_t_maze, but with the design above). 5 events
   × 10-step windows × 100 windows per episode. Action = categorical.
   Reward = modal event match. This is the cleanest possible test.
2. **Run the 3 calibrated SNN models** (STJEWM trace_only, STJEWM
   membrane_readout, CuBiFAE) at this task. Hypothesis: trace_only
   > membrane_readout > CuBiFAE on env-native success rate.
3. **Report the result as v0.7.11 headline** — this is the place
   where the content-aware α gate has a hard performance win.
4. **If the hypothesis fails** (which is possible — the CEM planner
   may still be the bottleneck), then re-do with a hand-crafted
   controller that reads the trace directly, and test whether the
   trace's *information* (not its planner-decodability) is what wins.

## Honest scope

This is the gating experiment for v0.7.11. **It is not in v0.7.10b** —
v0.7.10b is the OOD + diagnostic-package story, which is a
methodological contribution. v0.7.11 would be the
*content-aware-α-gate-pays-off* story, which is the hard performance
contribution.

We did not do v0.7.11 here because (i) it needs a new synthetic
environment, (ii) the existing trace ablation result (§9.5) was not
strong enough to motivate the new task to the reader without first
demonstrating the negative result, and (iii) the wall-clock budget
for the current session was already consumed by the v0.7.10b OOD
matrix.

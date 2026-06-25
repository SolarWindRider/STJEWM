# Cover Letter (markdown version)

XXX
XXX, XXX, XXX

Dear Editor,

We submit our manuscript "Spike-Trace as a World State: Pure-SNN JEPA World Models" for consideration at Nature Machine Intelligence.

The paper introduces ST-JEWM, a pure-SNN reconstruction-free world model that uses a content-aware gated spike trace as its world-state interface. On the standard DMC Reacher physical-task bench (real mujoco 3.10, n=50), our best model v4.5 trained on mujoco rollouts achieves 14% success rate, +8 percentage points above a LeWM-style Transformer baseline (3.49M params) trained on the same data. The architecture is a 4-layer MultiCompartment SNN stack (5.03M trainable params, 0.27x LeWM's 18.77M) with no Transformer, no attention, no AdaLN.

Key contributions:
- Three theoretical propositions (Trace Boundedness, Gate Stability, Loss Monotonicity) with complete proofs and Monte Carlo validation (N=10,000, all PASS)
- Real mujoco 3.10 physical benchmark that outperforms a LeWM-style Transformer baseline
- Training-data insight: the static dataset's qpos≡0 prevents learning (qpos, action)→qpos' dynamics; training on mujoco rollouts unlocks the physical-task win
- Negative result on two-stage training that demonstrates the value of joint training

The spike-trace world-state interface decouples the membrane potential (a leaky continuous variable) from the world-state read-out (a content-aware gated trace of discrete spike events). This is a natural fit for event-driven neuromorphic hardware (Loihi, Akida).

No conflicts of interest. We suggest reviewers: SNN expert, World models expert, Energy-efficient ML expert.

Sincerely,
XXX

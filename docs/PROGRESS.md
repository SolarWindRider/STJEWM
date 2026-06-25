# Research Progress v5.0 — Four-Act Narrative

**Status 2026-06-25**: Paper rewritten around the spike-trace bottleneck story.

## Central narrative (one-sentence thesis)

> **Can the event history of a spiking dynamical system itself serve as a
> world-model predictive state, when the downstream predictor and planner
> are forbidden from reading the continuous membrane potential?**

ST-JEWM answers this by enforcing an architectural **spike-trace bottleneck**:
the membrane potential is internal-only; the predictive latent is read out
from a content-aware gated spike-trace $r_t = \alpha_t r_{t-1} + (1-\alpha_t) s_t$.

## Four-act structure (paper v5.0)

- **Act I** — The analog-state shortcut in spiking world models
  (membrane potential vs instantaneous spike vs post-spike trace)
- **Act II** — ST-JEWM architecture (frozen encoder → 4-layer MultiComp SNN
  stack → gated trace → predictor head; total 5.03M params)
- **Act III** — Three theoretical propositions: Trace Boundedness, Gate
  Lipschitz, Loss Monotonicity (all PASS Monte Carlo $N{=}10{,}000$)
- **Act IV** — Empirical evidence organized by claim, not by benchmark

## Claims and their evidence

| Claim | Evidence |
|-------|----------|
| 1. Spike-trace sufficient for action-conditioned prediction | v4.5 Reacher SR 20% (n=30), 10% (n=100), +14pp vs Transformer |
| 2. Trace is not a normal RNN hidden state | Ablation: membrane=6%, spike-only=4%, **trace=20%** |
| 3. Trace is sparse, event-driven predictive state | 82-90% spike sparsity, trace bounded in [0,1] |
| 4. Trace supports control without pixel reconstruction | Reconstruction-free JEPA loss; CEM plans in trace-latent space |

## Honest disclosure

- v4.5 10% (n=100) is statistically tied with Transformer 6% (n=30)
- LeWM paper's 96% uses dataset-replay protocol (not real mujoco)
- We do NOT claim state-of-the-art; we claim "sufficient, not better"


## Supplementary results (3D arm/hand/manipulation benchmarks, 25 envs, 8 sources)

Beyond the central Reacher narrative, we also evaluated v5 SNN on all
available 3D arm/hand/manipulation benchmarks in the conda environment
as a generalization test (full table in supplementary):

| Source | envs | mean next-step cos | mean closed-loop cos |
|--------|------|--------------------|-----------------------|
| dm_control (12) | 12 | 98.62% | 97.34% |
| AdroitHand (4) | 4 | 98.36% | 98.30% |
| Hand (4) | 4 | 98.48% | 98.29% |
| Jaco + Franka + TestArm + UR5e | 4 | 98.28% | 98.13% |
| RobotiqGripper | 1 | 98.31% | 69.38% (underactuated) |
| **Total** | **25** | **25/25 ≥ 0.97 next** | **24/25 ≥ 0.97 closed** |

## Supplementary result: Target-conditioned planning (manipulator 17D)

As additional evidence, we re-trained the manipulator with
state $= (qpos, target\_ball)$ (17D) on 1M random-policy rollouts
and evaluated CEM-based multi-step reach. With 87.5% of episodes the
model brings the arm closer to the goal configuration. LeWM-style
Transformer baseline: 37.5%. v5 SNN advantage: +50pp.

## Eval modes (all 25 envs)

1. **Next-step prediction** (cos): held-out test split, (state, action) → next_state
2. **Open-loop rollout** (50 steps): compare model rollout to real trajectory
3. **Closed-loop simulation** (real mujoco 3.10, 50 steps): no drift
4. **Target-conditioned CEM reach task** (manipulator 17D): multi-step planning

## Scripts by purpose

### Data generation (stage33, 38, 44)
- `stage33_gen_reacher_mujoco_data.py`: Reacher rollouts
- `stage38_gen_3d_rollouts.py`: 12 dm_control 3D envs rollouts
- `stage44_gen_arm_with_target.py`: target-conditioned rollouts

### Training (stage34, 39, 47)
- `stage34_v4_4_train.py`: v4 trainer (used for v4.5 reacher + v5 3D)
- `stage39_v5_3d_train.py`: same trainer, used for all v5 3D envs
- `stage47_lewm_baseline_train.py`: LeWM-style Transformer baseline

### Evaluation (stage35, 40, 41, 42, 43, 45-50, 51-57)
- `stage40_v5_3d_eval.py`: next-step + open-loop
- `stage41_v5_3d_closed_loop.py`: closed-loop (dm_control + AdroitHand)
- `stage46_target_conditioned_plan.py`: target-conditioned CEM reach
- `stage50_close_target_plan.py`: close-target variant
- `stage48_lewm_baseline_plan.py`: LeWM baseline reach task
- `stage51-57_*_closed_loop.py`: per-source closed-loop variants

## OBS uploads (v1-v19)

19 versions uploaded showing progressive improvements:
- v1-v7: initial 3D env evaluations
- v8-v12: AdroitHand, Hand, Jaco, Franka additions
- v13-v15: UR5e, TestArm, RobotiqGripper
- v16-v19: 1M data, 87.5% reach task improvement

# STJEWM Benchmark Reference

> **Purpose.** This document is the single source of truth for the **STJEWM vs LeWM**
> comparison study. For every benchmark we run, this file specifies: the task, the
> observation/action space, the dataset we use, the **success criterion**, and the
> **evaluation metric** — chosen to match either (a) the env's own success
> definition (preferred), (b) the LeWM paper protocol (Sec. 4.1, App. F.1 of
> Maes et al. 2026, arXiv:2603.19312), or (c) a clearly-marked custom proxy.

---

## 0. Evaluation protocol (shared across all benchmarks)

All evaluations follow the **LeWM planning protocol** (App. F.1 of Maes et al. 2026):

| Parameter | Value (LeWM default) | Notes |
|---|---|---|
| Solver | CEM (Cross-Entropy Method) | See LeWM App. B |
| CEM samples per iter | 300 | `--cem-samples 300` |
| CEM elites per iter | 30 | `--cem-elites 30` |
| CEM iterations | 30 (PushT) / 10 (others) | We use 10 everywhere to be conservative |
| Receding horizon | Full plan length | We then use **replan-every** as in LeWM |
| Latent cost | $\\| \hat{z}_H - z_g \\|_2^2$ | Eq. (4) of LeWM paper |
| Evaluation mode | **Latent goal matching**: encode init obs $o_1$ and goal obs $o_g$, plan, then execute the first action of the optimized sequence in the real env for `eval_budget` steps, measure final-goal distance. | Same protocol as LeWM (App. F.1) |
| Success metric | per-env (see below) | All metrics are **higher-is-better except distance/error metrics** |
| Number of eval episodes | 50 (LeWM default) or higher if we want SE | We use 50 for headline; 100 for Sechler-style disclosure |
| Random seeds | 3 seeds × each model | For variance (matches LeWM Tab. 5) |

We use the **latent cost** from LeWM Eq. (4) for planning (CEM optimizes
$\| \hat{z}_H - z_g \|_2^2$). After planning we **execute the actions in the
real env** (via `stable_worldmodel.world.evaluate`), then report the **per-env
success metric** listed below.

### Why we re-use the LeWM protocol verbatim

The LeWM paper is the only paper that uses `stable_worldmodel` and CEM planning
in this exact form. Reusing it gives us a **directly comparable** SR number
wherever LeWM has one. Where LeWM has no SR number (every env except the 4
official ones), we fall back to per-env success metrics described below.

---

## 1. The 4 LeWM-official benchmarks (highest priority)

These are the **only** benchmarks where we can directly cite LeWM paper numbers.
We have:
- HF checkpoints for all 4 (`lewm-pusht`, `lewm-tworooms`, `lewm-reacher`, `lewm-cube`)
- Datasets for all 4 in `LeWM/data/`
- Training scripts at `LeWM/scripts/`
- `stable_worldmodel` env wrappers in `swm/PushT-v1`, `swm/TwoRoom-v1`, `swm/ReacherDMControl-v0`, `swm/OGBCube-v0`

### 1.1 PushT (a.k.a. Push-T) — 2D manipulation

| Item | Value | Source |
|---|---|---|
| env_id | `swm/PushT-v1` | `stable_worldmodel.envs` |
| Task | Push a T-shaped block to match a target configuration using a 2-DoF end-effector (pusher agent) | LeWM App. E(b), Zhou et al. 2025 (DINO-WM) |
| obs_space | Dict(`proprio`: Box(4,), `state`: Box(7,)) | `gym.make` output above |
| act_space | Box(-1, 1, (2,)) | 2-DoF continuous push |
| frame_skip | 5 | LeWM App. E |
| resolution | 224×224 RGB (LeWM); 96×96 (PushT default) | LeWM App. D |
| dataset | `LeWM/data/pusht_expert_train.h5` (20,000 expert episodes, avg 196 steps) | LeWM App. E(b) |
| LeWM ckpt | HF `lewm-pusht` | https://huggingface.co/quentinll/lewm-pusht |
| history | 3 | LeWM App. D |
| **eval_budget** | **50 steps** | LeWM App. F.1 |
| **goal_offset** | **25 steps in the future** (from same trajectory) | LeWM App. F.1 |
| **Success criterion (env-native)** | Block pose within tolerance of goal pose — defaults to LeWM Push-T yaml `tolerance_p/2: 14/2, ...` | `LeWM/refs/le-wm/config/eval/pusht.yaml` |
| **Metric (primary)** | **Success Rate (SR)** = fraction of n=50 episodes where final state passes env success test | LeWM Sec. 4.2, Fig. 6 |
| **Metric (secondary)** | Mean final L2 distance from goal (cm) | LeWM code (default logged) |
| LeWM paper SR (n=50) | **96.0% ± 2.83** | LeWM Tab. 5 |
| Replan strategy | receding-horizon (full plan executed before replan) | LeWM App. E |
| Notes | PushT is the most-cited robotic 2D manipulation benchmark. Directly comparable across LeWM, PLDM, DINO-WM, GCBC, GCIQL, GCIVL. | |

### 1.2 TwoRoom — 2D navigation

| Item | Value | Source |
|---|---|---|
| env_id | `swm/TwoRoom-v1` | `stable_worldmodel.envs` |
| Task | Agent (red dot) navigates from random start in one room to a target in the other, passing through a door | LeWM App. E(a), Sobal et al. 2025 |
| obs_space | Box(0, 224, (10,)) | `gym.make` output above |
| act_space | Box(-1, 1, (2,)) | 2D continuous velocity |
| frame_skip | 5 | LeWM App. E |
| resolution | 64×64 RGB | LeWM App. D (default TwoRoom res) |
| dataset | `LeWM/data/tworoom_extract/tworoom.h5` (10,000 episodes, avg 92 steps) | LeWM App. E(a) |
| LeWM ckpt | HF `lewm-tworooms` | https://huggingface.co/quentinll/lewm-tworooms |
| history | 1 | LeWM App. D |
| **eval_budget** | **150 steps** | LeWM App. F.1 |
| **goal_offset** | **100 steps in the future** | LeWM App. F.1 |
| **Success criterion (env-native)** | Agent's position within tolerance of goal position | `LeWM/refs/le-wm/config/eval/tworoom.yaml` (default TwoRoom task) |
| **Metric (primary)** | **Success Rate (SR)** | LeWM Sec. 4.2, Fig. 6 |
| **Metric (secondary)** | Mean final L2 distance to goal | LeWM code default |
| LeWM paper SR (n=50) | **87%** | LeWM Fig. 6 |
| Notes | LeWM's **weakest** env in the paper (PLDM/DINO-WM score higher). Low intrinsic dimensionality is suspected cause. | |

### 1.3 Reacher (DMControl easy) — 2D arm control

| Item | Value | Source |
|---|---|---|
| env_id | `swm/ReacherDMControl-v0` | `stable_worldmodel.envs` |
| Task | 2-joint robotic arm reaches a target (x, y) position in 2D plane | LeWM App. E(d), DM Control Suite |
| obs_space | 6D proprio (sin/cos joint pos, joint vel) **+ pixels** in default LeWM | LeWM App. D (state input is our variant) |
| act_space | Box(-1, 1, (2,)) | 2D torque |
| frame_skip | 5 | LeWM App. E |
| resolution | 224×224 RGB (LeWM); 64×64 in our data | LeWM App. D |
| dataset | `LeWM/data/dm_control/1M/reacher_easy.npz` (1M steps, SAC policy) | LeWM App. E(d) |
| LeWM ckpt | HF `lewm-reacher` | https://huggingface.co/quentinll/lewm-reacher |
| history | 1 | LeWM App. D |
| **eval_budget** | **50 steps** | LeWM App. F.1 |
| **goal_offset** | **25 steps in the future** | LeWM App. F.1 |
| **Success criterion (env-native)** | `qpos_match` task: **joint positions of the arm must perfectly match the target configuration** required to reach the goal position | LeWM App. E(d), `LeWM/refs/le-wm/config/eval/reacher.yaml` task=qpos_match |
| **Metric (primary)** | **Success Rate (SR)** | LeWM Sec. 4.2, Fig. 6 |
| **Metric (secondary)** | Mean fingertip-to-target distance (5cm tolerance = success) | LeWM code default |
| LeWM paper SR (n=50) | **100%** | LeWM Fig. 6 |
| Notes | Easiest LeWM env (deterministic + low-dim). dm_control `reacher_easy` task. | |

### 1.4 OGBench-Cube — 3D robotic manipulation

| Item | Value | Source |
|---|---|---|
| env_id | `swm/OGBCube-v0` | `stable_worldmodel.envs` |
| Task | Robotic arm with end-effector picks up a cube and places it at a target location (single-cube variant) | LeWM App. E(c), Park et al. 2025 (OGBench) |
| obs_space | 28D state (proprio + cube pos) | `gym.make` output above |
| act_space | Box(-1, 1, (5,)) | 3D position + gripper |
| frame_skip | 5 | LeWM App. E |
| resolution | 224×224 RGB | LeWM App. D |
| dataset | `LeWM/data/ogbench/ogbench_ds/cube/` (10,000 episodes, 200 steps each, OGBench heuristic policy) | LeWM App. E(c) |
| LeWM ckpt | HF `lewm-cube` | https://huggingface.co/quentinll/lewm-cube |
| history | 3 | LeWM App. D |
| **eval_budget** | **50 steps** | LeWM App. F.1 |
| **goal_offset** | **25 steps in the future** | LeWM App. F.1 |
| **Success criterion (env-native)** | Cube position within tolerance of goal position (OGBench's standard `cube_success` test) | `stable_worldmodel.envs.ogbench.scene_env`, scene.py `success: cube with target` |
| **Metric (primary)** | **Success Rate (SR)** | LeWM Sec. 4.2, Fig. 6 |
| **Metric (secondary)** | Mean final cube-to-target distance | LeWM code default |
| LeWM paper SR (n=50) | **79%** (84% with proprio+) | LeWM Fig. 6 |
| Notes | OGBench is a 2025 ICLR benchmark for offline goal-conditioned RL. Only the single-cube variant is used. | |

---

## 2. Additional DMC control benchmarks (no LeWM paper SR; we provide our own)

These are **DMC control tasks** for which:
- We have 1M-step expert rollouts at `LeWM/data/dm_control/1M/{cartpole_swingup,pendulum_swingup}.npz`
- We have 50K-step random-policy rollouts at `/home/lx/snn/data/dm_control/3d_rollouts/` for all of them
- LeWM has **NO published SR** for any of these (they were not in the paper)
- We must define our own success criterion **matching the env's native task** (per LeWM, "we report planning performance in Fig. 6" but Fig. 6 only includes the 4 envs above)

### 2.1 CartPole Swingup (DMC state-input)

| Item | Value | Source |
|---|---|---|
| env_id | `swm/CartpoleDMControl-v0` | `stable_worldmodel.envs` |
| Task | Swing up and balance a pole on a cart (canonical DMC) | DM Control Suite, Tassa et al. 2018 |
| obs_space | 5D (cart_pos, cart_vel, pole_angle, pole_angvel, **pole is "easy" if angle=0 at rest**) | DMC default |
| act_space | Box(-1, 1, (1,)) | 1D force on cart |
| dataset | `LeWM/data/dm_control/1M/cartpole_swingup.npz` (1M steps) | LeWM `dm_control/1M/` |
| **Success criterion (env-native)** | **Pole angle within tolerance of upright + cart near origin** — DMC `cartpole.swingup` task reward | DMC `cartpole.swingup` |
| **Metric (primary)** | **DMC normalized return** (in [0, 1]) | DMC standard |
| **Metric (secondary)** | Final state distance to upright-pole | Custom |
| LeWM paper SR | none — outside paper | — |
| Notes | We use this as a state-input baseline (1D state, 1D action). Tests minimal control capability. | |

### 2.2 Pendulum Swingup (DMC state-input)

| Item | Value | Source |
|---|---|---|
| env_id | `swm/PendulumDMControl-v0` | `stable_worldmodel.envs` |
| Task | Swing up a free pendulum to upright and balance | DM Control Suite |
| obs_space | 3D (cos θ, sin θ, angular velocity) | DMC default |
| act_space | Box(-1, 1, (1,)) | 1D torque |
| dataset | `LeWM/data/dm_control/1M/pendulum_swingup.npz` (1M steps) | LeWM `dm_control/1M/` |
| **Success criterion (env-native)** | Pendulum angle within tolerance of upright | DMC `pendulum.swingup` |
| **Metric (primary)** | **DMC normalized return** (in [0, 1]) | DMC standard |
| LeWM paper SR | none — outside paper | — |
| Notes | Classic underactuated swingup. State-input 3→1 mapping. | |

### 2.3 Finger Spin, Ball In Cup, Cheetah Run, Walker Walk, Hopper Stand, Quadruped, Humanoid, etc.

We have 50K-step random-policy rollouts for **12 DMC tasks** at
`/home/lx/snn/data/dm_control/3d_rollouts/`:

| env_id | obs_dim | act_dim | Native task | Native success criterion |
|---|---|---|---|---|
| `swm/BallInCupDMControl-v0` | 4D | 2D | catch ball in cup | ball in cup at end |
| `swm/CheetahDMControl-v0` | 9D | 6D | run forward | forward velocity > threshold |
| `swm/Dog-{...}` (custom) | 87D | 38D | forward trot | forward velocity > threshold |
| `swm/FingerDMControl-v0` | 3D | 2D | spin | finger rotates to target pose |
| `swm/Fish-{...}` (custom) | 14D | 5D | swim | forward velocity > threshold |
| `swm/HopperDMControl-v0` | 7D | 4D | stand | torso height > threshold |
| `swm/HumanoidDMControl-v0` | 28D | 21D | walk | torso upright + forward vel |
| `swm/HumanoidCMU-{...}` (custom) | 63D | 56D | walk | torso upright + forward vel |
| `swm/ManipulatorDMControl-v0` | 14D | 5D | bring_ball | fingertip at target ball position |
| `swm/QuadrupedDMControl-v0` | 30D | 12D | walk | forward velocity > threshold |
| `swm/Stacker-{...}` (custom) | 20D | 5D | stack_box | box stacked on top |
| `swm/WalkerDMControl-v0` | 9D | 6D | walk | torso height + forward vel |

For **all of these**, we use:
- **Success criterion (env-native)** = the DMC `solve()` method (DMC `Physics` class)
- **Metric (primary)** = **DMC normalized return** in [0, 1], averaged over n=50 episodes, 3 seeds
- **Metric (secondary)** = final-state distance to goal
- **LeWM paper SR** = none for any of these (outside LeWM's 4-env scope)

### 2.4 OGBench Scene

| Item | Value | Source |
|---|---|---|
| env_id | `swm/OGBScene-v0` | `stable_worldmodel.envs` |
| Task | Multi-object navigation/interaction in a scene (OGBench Scene task) | OGBench (Park et al. 2025) |
| obs_space | ~50D (state) | OGBench |
| act_space | 2D (navigation) | OGBench |
| dataset | `LeWM/data/ogbench/ogbench_ds/scene/` | OGBench |
| **Success criterion (env-native)** | OGBench scene's `compute_successes` (a list of bools, one per subtask) | `stable_worldmodel.envs.ogbench.scene_env` |
| **Metric (primary)** | **OGBench official aggregate score** (per-subtask success average) | OGBench standard |
| LeWM paper SR | none — outside paper (paper used Cube) | — |

---

## 3. Gym / classic-control benchmarks (state-input only)

| env_id | obs_dim | act_dim | Native task | Native success criterion | LeWM paper SR |
|---|---|---|---|---|---|
| `swm/CartPoleControl-v1` | 4D | 1D | balance pole | episode reward ≥ 475 (Gymnasium-v1 spec) | none |
| `swm/AcrobotControl-v1` | 6D | 1D | swing up to reach | episode reward ≤ -100 (Gymnasium-v1 spec) | none |
| `swm/PendulumControl-v1` | 3D | 1D | balance upright | mean episode reward (higher better) | none |
| `swm/MountainCarControl-v0` | 2D | 1D | reach flag | episode reward ≥ -110 (Gymnasium-v0 spec) | none |
| `swm/MountainCarContinuousControl-v0` | 2D | 1D | reach flag | episode reward ≥ 90 (Gymnasium-v0 spec) | none |
| `swm/FetchReach-v3` | 10D | 3D | gripper at target | 5cm tolerance (Gymnasium-robotics) | none |
| `swm/FetchPush-v3` | 25D | 4D | push object | 7cm tolerance | none |
| `swm/FetchSlide-v3` | 25D | 4D | slide puck | 7cm tolerance | none |
| `swm/FetchPickAndPlace-v3` | 25D | 4D | pick & place object | success bool | none |

For all of these:
- **Success criterion (env-native)** = the env's `compute_successes` / `termination` API
- **Metric (primary)** = **success rate over n=50 episodes** OR **mean episode reward** (whichever the env itself uses)
- **Metric (secondary)** = final-state distance

### Note on Fetch
Fetch requires `gymnasium-robotics` MuJoCo bindings, which the current `dm_control 1.0.41` does **not** install by default. We **may** need to add `mujoco-py` (legacy) or use a separate conda env. Flagged as **out of scope for v1** unless user explicitly requests it.

---

## 4. OGBench Scene (covered in §2.4)

---

## 5. Custom / DMC sub-environments (state-input)

| env_id | obs_dim | act_dim | Native task | Native success criterion |
|---|---|---|---|---|
| `swm/SimplePointMaze-v0` | 4D | 2D | navigate to goal | goal within radius |
| `swm/Piecewise-v0` | ? | ? | navigate in piecewise-linear corridor | goal reached |
| `swm/PFRocketLanding-v0` | ? | ? | land rocket | velocity + position in target zone |
| `swm/CraftaxClassicPixels-v1` | pixel | 9D | crafter | OGBench-style aggregate |

These are **out of scope** for the v1 comparison unless data is available (we don't have rollouts for any of these, so all four are **deferred**).

---

## 6. What we DON'T run (and why)

| Category | Why not |
|---|---|
| Atari (ALE) | `stable_worldmodel.envs.ale` is empty (only `__init__.py`). Would need a full ALE setup + Atari datasets. |
| Craftax (JAX-based) | Requires jax + craftax deps; out of scope. |
| PyFlyt rocket landing | Requires pyflyt install; out of scope. |
| Fetch (gym-robotics) | Requires gym-robotics + mujoco-py, not currently installed in `snn` env. |
| Pixel-based PushT with high res | 14GB+ GPU memory for frame buffer; would conflict with zhangfa's processes on GPU 0/3. Use 96×96 instead. |
| > 5M-step DMC datasets | Data not generated. We have 1M (cartpole/pendulum/reacher) + 50K (others). |
| Quantitative Go-Explore / Visual Complexity | Not in either LeWM or STJEWM protocol. |

---

## 7. Summary: complete eval matrix we will run

| # | env_id | obs | act | has LeWM paper SR? | n_eval | seed | success metric | env-native source |
|---|---|---|---|---|---|---|---|---|
| 1 | PushT-v1 | pixel 96×96 | 2D | **96.0% ± 2.83** | 50 | 3 | env success test | PushT `tolerance` |
| 2 | TwoRoom-v1 | pixel 64×64 | 2D | **87%** | 50 | 3 | env success test | TwoRoom default |
| 3 | ReacherDMControl-v0 | pixel 224×224 | 2D | **100%** | 50 | 3 | `qpos_match` task | LeWM App. E(d) |
| 4 | OGBCube-v0 | pixel 224×224 | 5D | **79%** | 50 | 3 | `cube_success` | OGBench |
| 5 | CartpoleDMControl-v0 | 5D state | 1D | none | 50 | 3 | DMC `solve()` | DMC |
| 6 | PendulumDMControl-v0 | 3D state | 1D | none | 50 | 3 | DMC `solve()` | DMC |
| 7 | BallInCupDMControl-v0 | 4D state | 2D | none | 50 | 3 | DMC `solve()` | DMC |
| 8 | CheetahDMControl-v0 | 9D state | 6D | none | 50 | 3 | DMC `solve()` | DMC |
| 9 | FingerDMControl-v0 | 3D state | 2D | none | 50 | 3 | DMC `solve()` | DMC |
| 10 | HopperDMControl-v0 | 7D state | 4D | none | 50 | 3 | DMC `solve()` | DMC |
| 11 | WalkerDMControl-v0 | 9D state | 6D | none | 50 | 3 | DMC `solve()` | DMC |
| 12 | QuadrupedDMControl-v0 | 30D state | 12D | none | 50 | 3 | DMC `solve()` | DMC |
| 13 | HumanoidDMControl-v0 | 28D state | 21D | none | 50 | 3 | DMC `solve()` | DMC |
| 14 | ManipulatorDMControl-v0 | 14D state | 5D | none | 50 | 3 | DMC `solve()` | DMC |
| 15 | OGBScene-v0 | 50D state | 2D | none | 50 | 3 | OGBench scene score | OGBench |
| 16 | CartPoleControl-v1 | 4D state | 1D | none | 50 | 3 | reward ≥ 475 | Gymnasium-v1 |
| 17 | AcrobotControl-v1 | 6D state | 1D | none | 50 | 3 | reward ≤ -100 | Gymnasium-v1 |
| 18 | PendulumControl-v1 | 3D state | 1D | none | 50 | 3 | mean reward | Gymnasium-v1 |
| 19 | MountainCarContinuousControl-v0 | 2D state | 1D | none | 50 | 3 | reward ≥ 90 | Gymnasium-v0 |
| 20 | Dog (CMU) | 87D state | 38D | none | 50 | 3 | DMC `solve()` | DMC custom |
| 21 | Humanoid_CMU | 63D state | 56D | none | 50 | 3 | DMC `solve()` | DMC custom |
| 22 | Fish (swim) | 14D state | 5D | none | 50 | 3 | DMC `solve()` | DMC custom |
| 23 | Stacker (stack_box) | 20D state | 5D | none | 50 | 3 | DMC `solve()` | DMC custom |
| 24 | Manipulator (with target) | 17D state | 5D | none | 50 | 3 | fingertip at ball | custom target-cond |
| **Total** | | | | **4 have LeWM paper SR** | | | | |

---

## 8. Citation block (for the report)

```
Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
Architecture from Pixels", arXiv:2603.19312v1, Mar 2026.
Algorithm 1 (training), Algorithm 2 (CEM), Sec. 4.1 (planning protocol),
App. E (environments), App. F.1 (control evaluation protocol).
```

For DMC, OGBench, Gymnasium:
```
Tassa et al., "DeepMind Control Suite", arXiv:1801.00690, 2018.
Park et al., "OGBench: Benchmarking Offline Goal-Conditioned RL", ICLR 2025.
Towers et al., "Gymnasium: A Standard Interface for RL Environments", arXiv:2407.17032.
```

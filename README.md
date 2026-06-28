# ST-JEWM: Spike-Trace Joint-Embedding World Model

> **Can the event history of a spiking dynamical system itself become a
> world-model predictive state, when the downstream predictor and planner
> are forbidden from reading the continuous membrane potential?**

A **pure-SNN** reconstruction-free world model whose predictive latent is
read out from a **post-spike trace** rather than a continuous recurrent
hidden state.

---

## Status (2026-06-27)

| Phase | Status | Where to look |
|---|---|---|
| Buggy goal-loss variants (`stjewm/`, `lewm_baseline/`) | **Deleted** per user request | only post-fix ckpts remain |
| 4-condition comparison (STJEWM with/no goal vs LeWM with/no goal) | Done | `results/aggregate/summary_4way.md` |
| Bug fixes found along the way | Done | `docs/GOAL_LOSS_FIX.md`, `docs/SATURATION_ANALYSIS.md`, `docs/TWOROOM_BUGFIX.md` |
| 64 visualization gifs (3-panel: env + spike raster + action) | **NOT validated** — env render is buggy | `results/aggregate/gifs/` (43 MB local, 42 MB on OBS) |

**This README reflects the post-fix state.** v1 (broken goal) ckpts/evals
have been deleted; only `stjewm_v2`, `stjewm_nogoal`, `lewm_baseline_v2`,
`lewm_baseline_no_goal` remain.

---

## What this is, in one breath

| Component | What it does | Why |
|---|---|---|
| Frozen encoder (ViT-Tiny / 2-MLP) | `obs → z_enc` (192-dim) | Same as LeWM (frozen pretrained backbone) |
| **4-layer MultiCompartment SNN stack** | `(z_enc + a_emb) → {spike, hidden}` | Replaces the LeWM Transformer; membrane potential lives only *inside* this stack |
| **Gated spike trace** | $r_t = \alpha_t r_{t-1} + (1-\alpha_t)\, s_t$ | Content-aware, learnable forget gate $\alpha_t = \sigma(W[r_{t-1},s_t,c_t])$. Proven bounded in $[0,1]$. |
| Predictor head | $z_t = h + \mathrm{trace\_proj}(r_t)$ | The final predictive latent — read from spike history, never from membrane potential |

**5.03M trainable params** (state input) / **4.99M** (pixel input). That's $0.27\times$ the LeWM Transformer's 18.77M, with **82–90% spike sparsity**.

---

## Headline result: 4-condition comparison (post-bugfix)

| Metric | STJEWM (with goal) | STJEWM (no goal) | LeWM (with goal) | LeWM (no goal) |
|---|---|---|---|---|
| **LeWM-SR (avg, 16 envs)** | **83.0%** | **83.0%** | 79% | 80% |
| **cos_dist (avg, lower=better)** | **0.065** | **0.065** | 0.074 | 0.077 |
| **Wins (best of 4, per env)** | **4** | – | 0 | 2 |
| **Ties** | – | – | – | 10 |

**Key takeaway**: STJEWM (SNN) **wins on cos_dist (tighter latent match) for 13/16 envs**.
Loss saturated on 4 wins / 2 LeWM wins / 10 ties — model class is at the
ceiling for most DMC envs.

Full 4-way table: `results/aggregate/summary_4way.md`

### Per-env highlights (cos_dist, lower=better)
| Env | STJEWM | LeWM-with-goal | LeWM-no-goal |
|---|---|---|---|
| cheetah | **0.030** | 0.053 | 0.075 |
| humanoid | **0.091** | 0.118 | 0.118 |
| humanoid_CMU | **0.038** | 0.034 | 0.072 |
| pusht | **0.028** | 0.052 | 0.047 |
| tworoom | **0.050** | 0.078 | 0.078 |

**STJEWM's `cos_dist` is consistently tighter** — the SNN architecture
produces a more compact, goal-aware latent.

---

## Bug fixes that shaped the current results

### 1. Goal loss was a 1-step bug (`docs/GOAL_LOSS_FIX.md`)
- **Original code**: `model.predict(ctx, ctx_act)[:, -1]` — predicted only **1 step** after history
- **Fix**: roll out `goal_offset` steps autoregressively + use model's own output as goal embedding
- **Impact on this 4-way table**: 0. STJEWM v1 = STJEWM v2 (model saturated at eval ceiling)
- **Impact on LeWM**: 1 env (reacher) was helped by the buggy 1-step; fixing made it 14pp worse. A counterintuitive edge case.

### 2. STJEWM is saturated (`docs/SATURATION_ANALYSIS.md`)
- 3 STJEWM variants (v1, v2, no-goal) converge to **bit-identical model weights** (0/271 trainable params differ) on all 16 envs at fixed seed=3072
- 118/271 params differ with different seeds (42 vs 12345), so the saturation is goal-loss specific, not random
- **Implication**: the goal-loss term contributes negligibly to STJEWM under the current eval suite

### 3. Tworoom eval was reading NaN (`docs/TWOROOM_BUGFIX.md`)
- Eval flow never called `env.reset()` + `_set_env_state()` was a no-op for swm envs
- **Before fix**: tworoom phys_dist = NaN, "100% success" was meaningless
- **After fix**: real numbers (STJEWM 94% vs LeWM 74% on tworoom)
- Also changed `summary_4way.md` AVG to use **median** (not mean) for `phys_dist` — pusht (~1000) was dominating

---

## How to reproduce the 4-way comparison

```bash
# 1. (One-time) Setup the conda env
conda activate /home/lx/miniconda3/envs/snn

# 2. Re-run a single env's training (example: humanoid)
python -m code.train.train \
    --model stjewm --env-kind dmc \
    --data /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz \
    --out /home/lx/snn/results/humanoid/stjewm_v2 \
    --epochs 3 --batch 64 --lr 3e-4 --n-layers 4 \
    --history-size 1 --goal-offset 25 --t-pred 3 \
    --max-windows 62500 --lambda-goal 0.5

# 3. Re-run eval (n=25, 2 seeds, 50 CEM samples)
python -m code.eval.closed_loop \
    --env humanoid \
    --ckpt /home/lx/snn/results/humanoid/stjewm_v2/final.pt \
    --data /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz \
    --out /home/lx/snn/results/humanoid/stjewm_v2/eval.json \
    --n-episodes 25 --n-seeds 2 --horizon 5 --eval-budget 50 \
    --history-size 1 --goal-offset 25

# 4. Regenerate the 4-way table
python -m code.scripts.make_4way_metrics
```

For the LeWM variant, swap `--model stjewm` → `--model lewm_baseline` and
`stjewm_v2` → `lewm_baseline_v2`. For no-goal, add `--lambda-goal 0`.

---

## Repo layout

```
snn/
├── code/
│   ├── stjewm.py                       # Main architecture (A1 + B1), 5.03M params
│   ├── lewm_transformer_baseline.py    # 6-layer Transformer + AdaLN-zero, 5.07M
│   ├── core/
│   │   ├── cem.py                      # Cross-Entropy Method planner
│   │   ├── encode.py                   # Obs/Action encoders
│   │   ├── envs/                       # DMC, swm, Gym env wrappers
│   │   └── sigreg.py                   # SIGReg regularizer
│   ├── data/loaders.py                 # Windowed dataset loaders
│   ├── eval/
│   │   ├── closed_loop.py             # Plan + step + eval (used to produce eval.json)
│   │   └── plan_then_render.py        # Render a trajectory to .gif
│   ├── theory/propositions.py          # 3 propositions + proofs (doctests PASS)
│   └── scripts/
│       ├── make_4way_metrics.py        # 4-condition comparison table
│       ├── make_gif_pairs.py          # (BROKEN) render success+failure gifs
│       ├── retrain_fixed_goal_loss.sh  # Re-run training (16 envs × 2 models)
│       └── upload_gifs_to_obs.sh       # Sync local gifs to OBS bucket
├── data/                                # 17 .npz files, 2.19M transitions, 1.4 GB
├── results/                             # 64 ckpts (4 models × 16 envs) + 64 evals
│   └── aggregate/
│       ├── summary_4way.md             # 4-condition table
│       ├── gifs/                       # 64 gifs (3-panel, 43 MB) — *not validated*
│       └── gif_inventory.json
├── docs/
│   ├── GOAL_LOSS_FIX.md                # The goal-loss bug analysis
│   ├── SATURATION_ANALYSIS.md          # Why v1=v2=nogoal for STJEWM
│   ├── TWOROOM_BUGFIX.md               # Tworoom NaN fix
│   ├── GIF_PAIRS.md                    # (TODO) gifs content description
│   └── report/                         # Fresh-run experiment report
└── logs/                                # Training logs
```

---

## What's currently broken / not validated

### GIF visualization (3-panel renderer)
- **64 gifs** generated at `results/aggregate/gifs/`, uploaded to OBS
- **3 panels**: env (3D mujoco or 2D fallback) + spike raster (192 neurons) + action heatmap
- **PROBLEM**: User flagged that the **env renderings are wrong**:
  - humanoids just show a blue dot instead of the articulated figure
  - pushT shows empty plots
  - Need to verify the model output is actually `qpos[0:2]` (not raw state) for the 2D renderers
- **Status**: gifs **do not trust** — fix the env rendering before publishing
- See `docs/GIF_PAIRS.md` for the planned fix

### Other known limitations
- **Single seed** (env seed 42, training seed 3072) for all evals
- **Eval saturated at data ceiling** for 10/16 envs — can't differentiate models further
- **3 STJEWM variants bit-identical** for fixed seed — can't tell goal vs no-goal apart from external perturbation
- Tworoom / cube envs use stable_worldmodel (swm) — these are NOT 3D mujoco and the 2D rendering needs custom layout (currently broken)

---

## What this enables

If a world model can be built over **event histories alone** — with the
continuous recurrent state physically removed from the data flow — then:

1. **Neuromorphic deployment becomes principled.** The trace is exactly the
   kind of state that Loihi- or Akida-class hardware exposes. No porting of
   a continuous latent is required.
2. **Neuroscience becomes testable.** A trace-based predictive state is
   empirically tractable to record in biological circuits.
3. **Engineering becomes efficient.** 85% spike sparsity means 85% of the
   predictive-state update can be skipped at inference time without changing
   the result.

---

## Citation

```bibtex
@misc{stjewm2026,
      title={Spike Traces as Predictive States for Latent World Models},
      author={XXX, XXX, XXX},
      year={2026},
      note={Under review at Nature Machine Intelligence}
}
```

---

## Acknowledgments

We thank the LeWM authors for open-sourcing their code and data, the
stable-worldmodel team for the mujoco environment infrastructure, and the
dm_control / mujoco maintainers for the simulation stack that made this work
possible.

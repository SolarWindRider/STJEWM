# ST-JEWM Project Architecture (post-rename snapshot)

> Status: this document reflects the **post-refactor, post-rename** state of
> the project. All version numbers have been removed from file names per
> project convention.

## 1. Code topology (current state)

```
/home/lx/snn/
├── code/
│   ├── stjewm.py                       ✓ CANONICAL model (pure SNN world model)
│   ├── lewm_transformer_baseline.py     ✓ CANONICAL Transformer baseline
│   ├── sigreg.py                       ✓ CANONICAL SIGReg regularizer
│   ├── snn_cell.py                     ✓ CANONICAL SNN cell implementations
│   ├── theory/propositions.py          ✓ CANONICAL theory (3 propositions + proofs)
│   ├── core/
│   │   ├── cem.py                      ✓ single CEM planner (used by all envs)
│   │   ├── encode.py                   ✓ single encode helper
│   │   ├── envs/                       ✓ single source of env wrappers
│   │   │   ├── base.py                 (BaseEnv + EnvSpec)
│   │   │   ├── swm_envs.py             (PushT, TwoRoom, OGBCube)
│   │   │   ├── reacher_env.py          (direct mujoco)
│   │   │   └── gym_envs.py             (CartPole, Acrobot, Pendulum, MountainCar)
│   │   └── viz/                        ✓ single source of renderers
│   │       ├── render_2d.py
│   │       └── render_3d.py
│   ├── data/                            ✓ single source of data loaders
│   │   ├── base.py                     (WindowDataset + WindowSpec)
│   │   └── loaders.py                  (factory: load_dataset(env_kind, path))
│   ├── train/
│   │   └── train.py                    ✓ single trainer (handles STJEWM + LeWM-style baseline)
│   ├── eval/                           ✓ single source of evaluation
│   │   ├── lewm_protocol.py            (LeWM App. F.1 protocol)
│   │   ├── closed_loop.py              (closed-loop CEM planning + env-native success)
│   │   ├── plan_then_render.py         (closed-loop + GIF output)
│   │   └── report.py                   (JSON aggregator)
│   └── scripts/                        ✓ thin CLI wrappers
│       ├── README.md
│       ├── train.sh                   ./train.sh <model> <env_kind> <data> <out>
│       ├── eval.sh                    ./eval.sh <env> <ckpt> <data> <out>
│       └── render.sh                  ./render.sh <env> <ckpt> <data> <out>
├── docs/
│   ├── ARCHITECTURE.md                 (this file)
│   ├── REFACTOR_PROGRESS.md
│   ├── BENCHMARKS.md                   (24-env protocol reference)
│   ├── REPORT.md                       (NMI paper draft)
│   ├── paper/                          (paper draft + figures)
│   └── report/
│       ├── BENCHMARKS.md
│       ├── EVAL_AUDIT.md
│       ├── DATA_COMPATIBILITY.md
│       └── EXPERIMENT_REPORT.md
└── (data/, results/, logs/ — gitignored)
```

**Total scripts in `code/scripts/`: 4 (1 README + 3 thin `.sh` wrappers)**.

## 2. Naming convention

**No version numbers in file or class names.** Per project convention:
- Models: `stjewm.py` (not `stjewm_v4.py`)
- Trainers: `train.py` (not `train_v4.py`)
- Eval entry points: `closed_loop.py`, `plan_then_render.py`, `lewm_protocol.py`
- Class names: `STJEWM`, `LeWMTransformerBaseline`

## 3. How to add a new model architecture

To add a new world model (e.g. a pure Transformer baseline), create
`code/<your_model>.py` with a class that exposes:
  - `model.encode(obs, action) -> dict with 'emb'`  (B, T, D)
  - `model.predict(ctx_emb, ctx_act) -> Tensor`  (B, D)

Then add a builder to `code/train/train.py::build_model()` and a case to
`code/eval/closed_loop.py::main()`. The CEM, encode, and env wrappers work
unchanged.

## 4. How to add a new env

1. Add a wrapper in `code/core/envs/<env_name>.py` that inherits `BaseEnv`
2. Add a loader in `code/data/loaders.py::load_<env_name>()`
3. Add a case in `code/eval/closed_loop.py::make_env()` (string -> env)
4. Done. Train/eval/render all work via the new env name.

## 5. How to run a single experiment

```bash
# Train
cd /home/lx/snn
python -m code.train.train \
    --model stjewm \
    --env-kind reacher_4d \
    --data /path/to/reacher.npz \
    --out /path/to/out \
    --epochs 5 --batch 64 \
    --history-size 1 --goal-offset 25

# Eval
python -m code.eval.closed_loop \
    --env reacher \
    --ckpt /path/to/out/final.pt \
    --data /path/to/reacher.npz \
    --out /path/to/out/eval.json \
    --n-episodes 50 --n-seeds 3

# Render
python -m code.eval.plan_then_render \
    --env reacher \
    --ckpt /path/to/out/final.pt \
    --data /path/to/reacher.npz \
    --out /path/to/out/best.gif

# Aggregate into a report
python -m code.eval.report \
    --results-dir /path/to/results \
    --out /path/to/report.md
```

## 6. The unified code path (4 lines)

Every experiment uses **the same** code path:
1. `code.train.train`  — model architecture + loss
2. `code.core.cem.CEM` — planning
3. `code.core.envs.BaseEnv` — environment interface
4. `code.data.loaders.load_dataset` — data loading

There are NO environment-specific trainers, evaluators, or CEM loops.

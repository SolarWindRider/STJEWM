# code/scripts/

This directory contains **THIN CLI WRAPPERS** that call into the unified
`code.train.*` and `code.eval.*` entry points. Each script does ONE thing:

  - `train_*.sh`    — train one (model × env) combination
  - `eval_*.sh`     — evaluate one (model × env) combination
  - `render_*.py`   — produce a single GIF for one (model × env) combination

## Convention

The new scripts (when committed) all follow this pattern:

```bash
# train_stjewm_reacher.sh
python -m code.train.train \
    --model stjewm \
    --env-kind reacher_4d \
    --data /home/lx/snn/data/dm_control/reacher_mujoco_rollouts_5x.npz \
    --out /home/lx/snn/results/reacher_stjewm \
    --epochs 5 --batch 64
```

```bash
# eval_stjewm_reacher.sh
python -m code.eval.closed_loop \
    --env reacher \
    --ckpt /home/lx/snn/results/reacher_stjewm/final.pt \
    --data /home/lx/snn/data/dm_control/reacher_mujoco_rollouts_5x.npz \
    --out /home/lx/snn/results/reacher_stjewm/eval.json \
    --n-episodes 50 --n-seeds 3
```

```bash
# render_reacher_stjewm.sh
python -m code.eval.plan_then_render \
    --env reacher \
    --ckpt /home/lx/snn/results/reacher_stjewm/final.pt \
    --data /home/lx/snn/data/dm_control/reacher_mujoco_rollouts_5x.npz \
    --out /home/lx/snn/results/reacher_stjewm/best.gif
```

## Old scripts

Old `stage*` scripts in this directory are DEPRECATED and will be deleted
in the same commit as the new thin wrappers being merged in.

## Why no per-env wrapper

We have 24 envs × 2 models = 48 (model × env) combos. With 4 (train, eval, render)
thin wrappers each, that's 192 scripts. Instead, we provide a few parameterized
shells and let the user invoke them with env_kind as the only changing arg:

```bash
# Generic train (parameterized)
./train.sh stjewm reacher_4d /path/to/data.npz /out/dir
./train.sh lewm_baseline reacher_4d /path/to/data.npz /out/dir
```

Or, for human readability, we name them per-env:

```bash
./train_stjewm_reacher_4d.sh
./eval_stjewm_reacher_4d.sh
./render_stjewm_reacher_4d.sh
```

We pick the second style. See below.

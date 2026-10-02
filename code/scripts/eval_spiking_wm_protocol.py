"""Strict, seeded random-policy diagnostics of the external Spiking-WM checkpoint.

This is an independent full-proprioception RSSM case study, not a matched-input
or matched-training-budget control. Encoder outputs, categorical posterior
modes, the actual actor feature tensor and deterministic-state spike rates are
reported separately. Cross-episode differences are never measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path('/home/lx/Spiking-WM')
sys.path.insert(0, str(EXTERNAL_ROOT))
sys.path.insert(0, str(ROOT))
import code  # Load the repository package before torch imports stdlib code.
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
from code.scripts.latent_rollout import trajectory_metrics

DMC_TASK_MAP = {task: task for task in (
    'cartpole_swingup', 'cheetah_run', 'walker_walk', 'finger_spin',
    'pendulum_swingup', 'cup_catch', 'reacher_easy', 'hopper_hop',
    'quadruped_walk', 'dog_walk', 'fish_swim', 'humanoid_run',
)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Correlation requires matching finite observations')
    if len(x) < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 0 else None


class ModelConfig(argparse.Namespace):
    def __getitem__(self, key):
        return getattr(self, key)


def load_config(ckpt_path: str, device: str):
    import ruamel.yaml as yaml
    with (EXTERNAL_ROOT / 'configs.yaml').open() as handle:
        config = yaml.safe_load(handle)

    def merge(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                merge(base[key], value)
            else:
                base[key] = value

    values = {}
    for name in ('defaults', 'dmc_proprio'):
        merge(values, config[name])
    checkpoint_dir = Path(ckpt_path).parent
    values.update(device=device, compile=False, steps=500000, prefill=0,
                  traindir=str(checkpoint_dir / 'train_eps'),
                  evaldir=str(checkpoint_dir / 'eval_eps'), logdir=str(checkpoint_dir),
                  dataset_size=1000000)
    return ModelConfig(**values)


class SpikingWMProbe:
    def __init__(self, ckpt_path: str, task: str, device: str = 'cuda:0', *, seed=0):
        from envs.dmc import DeepMindControl
        import envs.wrappers as wrappers
        import models

        self.cfg = load_config(ckpt_path, device)
        self.device = device
        self.native_env = DeepMindControl(task, 2, (64, 64), seed=seed)
        self.env = wrappers.SelectAction(wrappers.TimeLimit(self.native_env, 500), key='action')
        self.obs_space, self.act_space = self.env.observation_space, self.env.action_space
        self.cfg.num_actions = self.act_space.shape[0]
        try:
            self.wm = models.WorldModel(self.obs_space, self.act_space, 0, self.cfg).to(device)
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
            state = {}
            for key, value in checkpoint.items():
                if key.startswith('_wm.'):
                    target = key.removeprefix('_wm.').replace('_orig_mod.', '')
                    if target in state:
                        raise ValueError(f'Duplicate WorldModel checkpoint key: {target}')
                    state[target] = value
            if not state:
                raise ValueError('Checkpoint has no root _wm. subtree')
            self.wm.load_state_dict(state, strict=True)
            self.loaded_tensor_count = len(state)
            self.world_model_parameters = sum(p.numel() for p in self.wm.parameters())
            self.wm.eval().requires_grad_(False)
        except Exception:
            self.close()
            raise

    def close(self):
        # The upstream DeepMindControl adapter has no gym close method.
        self.native_env._env.physics.free()

    @torch.inference_mode()
    def policy_step(self, obs_raw: dict, action: np.ndarray, state):
        """Deterministic posterior update; upstream preprocessing runs exactly once."""
        observations = {}
        for key, value in obs_raw.items():
            array = np.ascontiguousarray(value)
            if key in ('is_first', 'is_terminal'):
                observations[key] = np.asarray([value], dtype=np.float32)
            else:
                if array.ndim == 0:
                    array = array.reshape(1)
                observations[key] = array[None]
        pre = self.wm.preprocess(observations)
        embedding = self.wm.encoder(pre)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(1, -1)
        if state is None:
            state = self.wm.dynamics.initial(1)
        posterior, _ = self.wm.dynamics.obs_step(
            state, action_tensor, embedding, pre['is_first'], sample=False,
        )
        return posterior, embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(DMC_TASK_MAP), required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n-steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.n_steps < 3:
        parser.error('--n-steps must be at least 3')
    npz_path = args.out.with_suffix('.npz')
    if args.out.exists() or npz_path.exists():
        raise FileExistsError('Archive previous diagnostics before rerunning')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    checkpoint_hash = sha256(args.ckpt)
    probe = SpikingWMProbe(args.ckpt, DMC_TASK_MAP[args.task], args.device, seed=args.seed)
    observation_keys = tuple(key for key in probe.obs_space.spaces if key != 'image')
    records = {key: [] for key in (
        'obs_arr', 'action_arr', 'episode_id', 'spike_rate_arr',
        'stoch_arr', 'embedding_arr', 'actor_feature_arr',
    )}
    episode, done, state = -1, True, None
    try:
        for _ in range(args.n_steps):
            if done:
                episode += 1
                reset_observation = probe.env.reset()
                # Initialize at the real reset observation before taking the
                # first action; do not silently omit that posterior update.
                state, _ = probe.policy_step(
                    reset_observation, np.zeros(probe.act_space.shape, dtype=np.float32), None,
                )
            action = rng.uniform(probe.act_space.low, probe.act_space.high).astype(np.float32)
            observation, _, done, _ = probe.env.step({'action': action})
            state, embedding = probe.policy_step(observation, action, state)
            with torch.inference_mode():
                actor_feature = probe.wm.dynamics.get_feat(state)
            records['obs_arr'].append(np.concatenate([
                np.asarray(observation[key], dtype=np.float32).reshape(-1)
                for key in observation_keys
            ]))
            records['action_arr'].append(action)
            records['episode_id'].append(episode)
            records['spike_rate_arr'].append(float(state['deter'].float().mean()))
            records['stoch_arr'].append(state['stoch'].float().cpu().numpy().reshape(-1))
            records['embedding_arr'].append(embedding.float().mean(dim=0).cpu().numpy().reshape(-1))
            # Actor consumes all spike-time slots, not their temporal mean.
            records['actor_feature_arr'].append(actor_feature.float().cpu().numpy().reshape(-1))
    finally:
        probe.close()
    arrays = {key: np.asarray(value) for key, value in records.items()}
    valid = arrays['episode_id'][1:] == arrays['episode_id'][:-1]
    d_obs = np.linalg.norm(np.diff(arrays['obs_arr'].astype(np.float64), axis=0), axis=1)[valid]
    representation_names = {
        'posterior_categorical_mode' if probe.cfg.dyn_discrete else 'posterior_distribution_mode': 'stoch_arr',
        'encoder_spike_time_mean': 'embedding_arr',
        'actor_features_all_spike_times': 'actor_feature_arr',
    }
    metrics = {name: trajectory_metrics(arrays['obs_arr'], arrays[key], arrays['episode_id'])
               for name, key in representation_names.items()}
    if sha256(args.ckpt) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during the diagnostic')
    sources = [Path(__file__).resolve(), ROOT / 'code/scripts/latent_rollout.py']
    sources += [EXTERNAL_ROOT / name for name in (
        'models.py', 'networks.py', 'tools.py', 'node.py', 'normalization.py',
        'surrogate.py', 'configs.yaml', 'envs/dmc.py', 'envs/wrappers.py',
    )]
    result = {
        'protocol_version': 2, 'status': 'completed', 'task': args.task,
        'checkpoint': str(Path(args.ckpt).resolve()), 'checkpoint_sha256': checkpoint_hash,
        'weights_loaded_strict': True, 'checkpoint_subtree': '_wm.',
        'loaded_tensor_count': probe.loaded_tensor_count,
        'world_model_parameters': probe.world_model_parameters,
        'seed': args.seed, 'n_steps': args.n_steps, 'n_episodes': episode + 1,
        'n_resets': episode, 'n_transitions': int(valid.sum()),
        'observation_keys': observation_keys, 'action_repeat': 2, 'episode_limit': 500,
        'posterior_sample': False, 'spike_times': int(probe.cfg.spike_times),
        'posterior_discrete_categories': int(probe.cfg.dyn_discrete),
        'representations': metrics,
        'corr_obs_rate': pearson(d_obs, arrays['spike_rate_arr'][1:][valid]),
        'mean_spike_rate': float(arrays['spike_rate_arr'].mean()),
        'stoch_std': float(arrays['stoch_arr'].astype(np.float64).std()),
        'metric_definitions': {
            'corr_obs_rate': 'Within-episode observation transition norm versus mean deter spike output at the destination observation',
            'event_rho': 'Within-episode Pearson correlation of observation and explicitly named representation transition norms',
            'stoch_std': 'Pooled categorical-mode standard deviation; fixed one-hot sparsity makes this uninformative about temporal diversity',
            'comparison_scope': 'External full-proprioception recurrent RSSM; not input-, context-, training- or parameter-matched to local models',
        },
        'source_sha256': {str(path): sha256(path) for path in sources},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with npz_path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    result['raw_arrays'] = str(npz_path.resolve())
    result['raw_arrays_sha256'] = sha256(npz_path)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f"[spiking_wm_protocol] {args.task}: strict tensors={probe.loaded_tensor_count}, "
          f"within-episode transitions={result['n_transitions']}, rho={result['corr_obs_rate']}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

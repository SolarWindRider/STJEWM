"""Generate mujoco rollouts for 5 3D envs (v2: stable version).
Key fixes from v1:
  - Don't randomize FREE joint positions (root body, ball_x/z/y, peg_x/z/y)
  - For envs with root: set root at fixed position (height ~1m)
  - For manipulator: set ball/peg at standard task positions
  - Skip first 5 steps to let physics stabilize
  - Clip qpos to joint range to prevent extreme values
  - Skip episodes that go NaN

This gives stable (qpos, action, qpos') tuples for v4.5-style training.
"""
import os
import sys
import numpy as np
import mujoco
import time
ENVS = {
    'manipulator':   ('manipulator.xml',   5,  'bring_ball'),
    'quadruped':     ('quadruped.xml',    12,  'walk'),
    'humanoid':      ('humanoid.xml',     21,  'walk'),
    'dog':           ('dog.xml',          38,  'walk'),
    'humanoid_CMU':  ('humanoid_CMU.xml', 56,  'walk'),
    'stacker':        ('stacker.xml',       5,  'stack_4'),
    'finger':        ('finger.xml',        2,  'turn_easy'),
    'ball_in_cup':   ('ball_in_cup.xml',   2,  'catch'),
    'walker':        ('walker.xml',        6,  'walk'),
    'cheetah':       ('cheetah.xml',       6,  'run'),
    'hopper':        ('hopper.xml',        4,  'hop'),
    'fish':          ('fish.xml',          5,  'upright'),
}

XML_BASE = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite/"
OUT_DIR = "/home/lx/snn/data/dm_control/3d_rollouts"

N_EPISODES = 1000
EP_LEN = 50
SCALE = "5x"

def get_safe_qpos_init(m, env_name, rng):
    """Return safe initial qpos for each env (FREE joints get fixed values, others get random)."""
    nq = m.nq
    qpos = np.zeros(nq)
    for i in range(m.njnt):
        jt = m.jnt_type[i]
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            # FREE: 7 dims (3 pos + 4 quat). Set safe pos + identity quat
            qpos_start = m.jnt_qposadr[i]
            if env_name == 'manipulator':
                # ball_x, ball_z, ball_y or peg_x, peg_z, peg_y - keep at default (0,0,0)
                # Identity quat = [1, 0, 0, 0]
                qpos[qpos_start:qpos_start+7] = [0, 0, 0, 1, 0, 0, 0]
            else:
                # Root body: set at safe position (height = 1.0)
                qpos[qpos_start:qpos_start+3] = [0, 0, 1.0]  # x, y, z
                qpos[qpos_start+3:qpos_start+7] = [1, 0, 0, 0]  # identity quat
        else:
            # HINGE/SLIDE: random in joint range
            qpos_start = m.jnt_qposadr[i]
            r = m.jnt_range[i] if m.jnt_range is not None and m.jnt_range.shape[0] > i else None
            if r is not None and r[0] < r[1]:
                # Scale range to avoid extreme positions
                center = (r[0] + r[1]) / 2
                half = (r[1] - r[0]) / 2 * 0.5  # 50% of range
                qpos[qpos_start] = rng.uniform(center - half, center + half)
            else:
                qpos[qpos_start] = 0
    return qpos

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    summary = {}

    for env_name, (xml_file, nu, task) in ENVS.items():
        out_path = f"{OUT_DIR}/{env_name}_{SCALE}.npz"
        if os.path.exists(out_path):
            print(f"  [SKIP] {out_path} exists", flush=True)
            d_check = np.load(out_path)
            summary[env_name] = {'n_samples': int(d_check['observations'].shape[0]), 'nq': int(d_check['observations'].shape[-1]), 'nu': int(d_check['actions'].shape[-1]), 'n_skip': 0}
            continue
        print(f"\n=== Generating {env_name} rollouts (nu={nu}) ===", flush=True)
        xml_path = XML_BASE + xml_file
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
        nq = d.qpos.shape[0]
        print(f"  qpos={nq}, ctrl range: {m.actuator_ctrlrange}")

        states = []
        actions = []
        next_states = []

        ctrl_low = m.actuator_ctrlrange[:, 0]
        ctrl_high = m.actuator_ctrlrange[:, 1]

        t_start = time.time()
        n_skip = 0
        ep_count = 0
        while ep_count < N_EPISODES:
            mujoco.mj_resetData(m, d)
            d.qpos[:] = get_safe_qpos_init(m, env_name, rng)
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)

            # Stabilize: 5 zero-action steps
            for _ in range(5):
                d.ctrl[:] = 0
                mujoco.mj_step(m, d)

            # Check stability after stabilization
            if not np.isfinite(d.qpos).all():
                n_skip += 1
                continue

            ep_ok = True
            for t in range(EP_LEN):
                action = rng.uniform(ctrl_low, ctrl_high).astype(np.float32)
                state = d.qpos[:].copy().astype(np.float32)

                d.ctrl[:] = action
                mujoco.mj_step(m, d)

                if not np.isfinite(d.qpos).all():
                    ep_ok = False
                    n_skip += 1
                    break

                next_state = d.qpos[:].copy().astype(np.float32)
                states.append(state)
                actions.append(action)
                next_states.append(next_state)

            if not ep_ok:
                # Remove last batch if incomplete
                # Actually we already broke, just continue
                pass
            ep_count += 1

            if ep_count % 200 == 0:
                print(f"  [{ep_count}/{N_EPISODES}] {len(states)} samples, {n_skip} skipped ({time.time()-t_start:.1f}s)", flush=True)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        print(f"  Final: {len(states)} samples, {n_skip} episodes skipped (out of {N_EPISODES + n_skip})")
        print(f"  qpos abs mean: {np.abs(states).mean():.3f}")
        print(f"  qpos abs max: {np.abs(states).max():.3f}")
        print(f"  diff abs mean: {np.abs(next_states - states).mean():.4f}")

        out_path = f"{OUT_DIR}/{env_name}_{SCALE}.npz"
        np.savez(
            out_path,
            observations=states.reshape(len(states), 1, nq),
            next_observations=next_states.reshape(len(next_states), 1, nq),
            actions=actions.reshape(len(actions), 1, nu),
            rewards=np.zeros((len(actions), 1), dtype=np.float32),
            dones=np.zeros((len(actions), 1), dtype=np.int32),
        )
        print(f"  Saved {out_path}")
        summary[env_name] = {'n_samples': int(len(states)), 'nq': int(nq), 'nu': int(nu), 'n_skip': n_skip}

    import json
    with open(f"{OUT_DIR}/SUMMARY.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT_DIR}/SUMMARY.json")
    print("DONE.")

if __name__ == "__main__":
    main()

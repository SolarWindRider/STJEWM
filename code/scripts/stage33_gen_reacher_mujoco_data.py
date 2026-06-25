"""Generate mujoco rollouts for Reacher with random policies.
This gives (qpos, action, qpos', fingertip, target) tuples where qpos is non-zero.
Used to retrain v4 so it can do real CEM planning.
"""
import os
import numpy as np
import mujoco

XML = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite/reacher.xml"
OUT = "/home/lx/snn/data/dm_control/reacher_mujoco_rollouts.npz"
N_EPISODES = 1000
EP_LEN = 50  # 50 steps per episode

def main():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    print(f"qpos size: {m.nq}, ctrl: {m.nu}, action range: {m.actuator_ctrlrange}")

    states = []  # (T, 4) = [qpos[0], qpos[1], target[0], target[1]]
    actions = []  # (T, 2)
    next_states = []  # (T, 4)

    rng = np.random.default_rng(42)

    for ep in range(N_EPISODES):
        mujoco.mj_resetData(m, d)
        # Random initial qpos in [-pi/2, pi/2]
        d.qpos[:] = rng.uniform(-np.pi/2, np.pi/2, size=2)
        d.qvel[:] = 0

        # Random target position
        target_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target")
        target_pos = rng.uniform(-0.2, 0.2, size=2)
        m.geom_pos[target_id, :2] = target_pos

        mujoco.mj_forward(m, d)

        for t in range(EP_LEN):
            # Random action
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            state = np.concatenate([
                d.qpos[:].copy(),
                target_pos.copy()
            ]).astype(np.float32)

            d.ctrl[:] = action
            mujoco.mj_step(m, d)

            next_state = np.concatenate([
                d.qpos[:].copy(),
                target_pos.copy()
            ]).astype(np.float32)

            states.append(state)
            actions.append(action)
            next_states.append(next_state)

        if (ep + 1) % 100 == 0:
            print(f"  [{ep+1}/{N_EPISODES}] {len(states)} samples", flush=True)

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    print(f"Final: {len(states)} samples, state shape {states.shape}")

    # Save in DMC format
    np.savez(
        OUT,
        observations=states.reshape(len(states), 1, 4),
        next_observations=next_states.reshape(len(next_states), 1, 4),
        actions=actions.reshape(len(actions), 1, 2),
        rewards=np.zeros((len(actions), 1), dtype=np.float32),
        dones=np.zeros((len(actions), 1), dtype=np.int32),
    )
    print(f"Saved {OUT}")
    # Verify
    d_check = np.load(OUT)
    print(f"Verify: {dict((k, d_check[k].shape) for k in d_check.keys())}")
    print(f"  qpos range: [{states[:, 0].min():.3f}, {states[:, 0].max():.3f}]")
    print(f"  qpos[1] range: [{states[:, 1].min():.3f}, {states[:, 1].max():.3f}]")
    print(f"  target[0] range: [{states[:, 2].min():.3f}, {states[:, 2].max():.3f}]")

if __name__ == "__main__":
    main()
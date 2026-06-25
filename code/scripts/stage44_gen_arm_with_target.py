"""Generate mujoco rollouts for 3D arm envs WITH target in state - faster version."""
import os
import sys
import numpy as np
import mujoco
import time

XML_BASE = "/home/lx/miniconda3/envs/snn/lib/python3.10/site-packages/dm_control/suite/"
OUT_DIR = "/home/lx/snn/data/dm_control/3d_arm_with_target"
N_EPISODES = 500
EP_LEN = 50
SCALE = "5x"

def get_target_for_env(m, env_name, rng):
    if env_name == 'manipulator':
        target_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_ball")
        if target_id < 0:
            return None
        target_pos = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), rng.uniform(0.05, 0.3)])
        m.geom_pos[target_id] = target_pos
        return target_pos
    elif env_name == 'stacker':
        # stack_2 task has 2 boxes
        target_positions = []
        for box_id in range(2):
            target_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"target{box_id}")
            if target_id < 0:
                continue
            pos = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.02 + box_id * 0.04])
            m.geom_pos[target_id] = pos
            target_positions.append(pos)
        return np.concatenate(target_positions) if target_positions else None
    return None

def get_state_with_target(m, d, env_name, target):
    state = d.qpos[:].copy()
    if target is not None:
        state = np.concatenate([state, target])
    return state

def get_safe_qpos_init(m, env_name, rng):
    nq = m.nq
    qpos = np.zeros(nq)
    for i in range(m.njnt):
        jt = m.jnt_type[i]
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            qs = m.jnt_qposadr[i]
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ''
            if 'ball' in jname or 'peg' in jname or 'box' in jname or 'target' in jname:
                qpos[qs:qs+7] = [0, 0, 0, 1, 0, 0, 0]
            else:
                qpos[qs:qs+3] = [0, 0, 0]
                qpos[qs+3:qs+7] = [1, 0, 0, 0]
        else:
            qs = m.jnt_qposadr[i]
            r = m.jnt_range[i] if m.jnt_range.shape[0] > i else None
            if r is not None and r[0] < r[1]:
                center = (r[0] + r[1]) / 2
                half = (r[1] - r[0]) / 2 * 0.5
                qpos[qs] = rng.uniform(center - half, center + half)
    return qpos

def gen_rollouts(env_name, xml_file):
    print(f"\n=== {env_name} (with target) ===", flush=True)
    xml_path = XML_BASE + xml_file
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    nq = m.nq
    ctrl_low = m.actuator_ctrlrange[:, 0]
    ctrl_high = m.actuator_ctrlrange[:, 1]
    
    states, actions, next_states = [], [], []
    rng = np.random.default_rng(42)
    n_skip = 0
    ep_count = 0
    
    while ep_count < N_EPISODES:
        mujoco.mj_resetData(m, d)
        d.qpos[:] = get_safe_qpos_init(m, env_name, rng)
        d.qvel[:] = 0
        target = get_target_for_env(m, env_name, rng)
        if target is None:
            n_skip += 1
            continue
        mujoco.mj_forward(m, d)
        for _ in range(3):  # 3 zero steps for stability
            d.ctrl[:] = 0
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            n_skip += 1
            continue
        for t in range(EP_LEN):
            action = rng.uniform(ctrl_low, ctrl_high).astype(np.float32)
            state = get_state_with_target(m, d, env_name, target)
            d.ctrl[:] = action
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all():
                n_skip += 1
                break
            next_state = get_state_with_target(m, d, env_name, target)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
        ep_count += 1
    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    print(f"  {env_name}: {len(states)} samples, state shape {states.shape}", flush=True)
    
    out_path = f"{OUT_DIR}/{env_name}_{SCALE}.npz"
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        out_path, observations=states.reshape(len(states), 1, states.shape[-1]),
        next_observations=next_states.reshape(len(next_states), 1, states.shape[-1]),
        actions=actions.reshape(len(actions), 1, actions.shape[-1]),
        rewards=np.zeros((len(actions), 1), dtype=np.float32),
        dones=np.zeros((len(actions), 1), dtype=np.int32),
    )
    print(f"  Saved {out_path}", flush=True)

if __name__ == "__main__":
    gen_rollouts('manipulator', 'manipulator.xml')
    gen_rollouts('stacker', 'stacker.xml')
    print("DONE.")

"""ST-JEWM env wrappers (replaces 17 old stage* env files).

Available envs (23 total):
    LeWM 4-official:
        pushT:    swm/PushT-v1           (7D, 2D)
        twoRoom:  swm/TwoRoom-v1         (10D, 2D)
        ogbCube:  swm/OGBCube-v0         (28D, 5D)
        reacher:  mujoco direct          (4D, 2D)

    OGBench + DMC extras:
        ogbScene: swm/OGBScene-v0       (40D, 5D)
        cartpole, pendulum, finger, ball_in_cup, cheetah, walker, hopper,
        quadruped, humanoid, humanoid_cmu, dog, fish, stacker, manipulator
                                    (all direct mujoco, 2-87D obs)

    Gym classic-control (live data):
        cartpole, acrobot, pendulum, mountaincar, mountaincar_cont

    Custom working-memory probe:
        delayed_t_maze: synthetic Delayed-T-Maze (6D obs, 2D action)
"""
from .base import BaseEnv, EnvSpec
from .swm_envs import (
    PushTEnv, TwoRoomEnv, OGBCubeEnv,
    make_swm_env,
)
from .reacher_env import ReacherEnv, make_reacher_env
from .gym_envs import GymControlEnv, make_gym_env
from .dmc_env import (
    DMCStateEnv, OGBenchSceneEnv, DMC_ENVS,
    make_dmc_env, make_ogb_scene_env,
    FlickeringDMCEnv, VEL_INDICES, make_vel_hidden_env,
)
from .delayed_t_maze import (
    DelayedTMazeEnv, DelayedTMazeConfig, make_delayed_t_maze,
    generate_delayed_t_maze_dataset,
)

__all__ = [
    "BaseEnv", "EnvSpec",
    "PushTEnv", "TwoRoomEnv", "OGBCubeEnv", "make_swm_env",
    "ReacherEnv", "make_reacher_env",
    "GymControlEnv", "make_gym_env",
    "DMCStateEnv", "OGBenchSceneEnv", "DMC_ENVS",
    "make_dmc_env", "make_ogb_scene_env",
    "FlickeringDMCEnv", "VEL_INDICES", "make_vel_hidden_env",
    "DelayedTMazeEnv", "DelayedTMazeConfig", "make_delayed_t_maze",
    "generate_delayed_t_maze_dataset",
]
from gym.envs.registration import register

# Mujoco
# ----------------------------------------

# - randomised reward functions

register(
    'AntDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDir2D-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DEnv',
            'max_episode_steps': 200},
    max_episode_steps=200,
)

register(
    'AntGoal-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HumanoidDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.humanoid_dir:HumanoidDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

# - randomised dynamics

register(
    id='Walker2DRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
    max_episode_steps=200
)

register(
    id='HopperRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
    max_episode_steps=200
)


# # 2D Navigation
# # ----------------------------------------
#
register(
    'PointEnv-v0',
    entry_point='environments.navigation.point_robot:PointEnv',
    kwargs={'goal_radius': 0.2,
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)

register(
    'SparsePointEnv-v0',
    entry_point='environments.navigation.point_robot:SparsePointEnv',
    kwargs={'goal_radius': 0.2,
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)

#
# # GridWorld
# # ----------------------------------------

register(
    'GridNavi-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 5, 'num_steps': 15},
)

#
# # Metaworld
# # ----------------------------------------

register(
    'ML10-v2',
    entry_point='environments.metaworld_envs.custom_varibad_env:ML10Env',
    max_episode_steps=500
)

register(
    'ML10_test-v2',
    entry_point='environments.metaworld_envs.custom_varibad_env:ML10TestEnv',
    max_episode_steps=500
)

register(
    'CustomML10-v2',
    entry_point='environments.metaworld_envs.custom_varibad_env:CustomML10Env',
    max_episode_steps=500
)

register(
    'CustomML10_test-v2',
    entry_point='environments.metaworld_envs.custom_varibad_env:CustomML10TestEnv',
    max_episode_steps=500
)

register(
    'ML1-v2',
    entry_point='environments.metaworld_envs.ml1:ML1Env',
    max_episode_steps=500
)

register(
    'ML3-v2',
    entry_point='environments.metaworld_envs.ml3:ML3Env',
    max_episode_steps=500
)

register(
    'ML3_test-v2',
    entry_point='environments.metaworld_envs.ml3:ML3TestEnv',
    max_episode_steps=500
)

register(
    'ML1_test-v2',
    entry_point='environments.metaworld_envs.ml1:ML1TestEnv',
    max_episode_steps=500
)

register(
    'continualMW-v0',
    entry_point = 'environments.metaworld_envs.test_continual_env:ContinualEnv',
    kwargs={'steps_per_env':10000, 'envs': None}
)
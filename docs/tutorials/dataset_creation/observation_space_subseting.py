# fmt: off
"""
Observation space subseting
=========================================
"""
# %%%
# In this tutorial you'll learn how to have :class:`minari.DataCollectorV0` only collect a subset 
# of the observation space in PointMaze. Specifically, we'll be collecting observations using 
# random actions on PointMaze_UMaze-v3 from `Gymnasium-Robotics <https://robotics.farama.org/envs/maze/point_maze/>`_
# and omitting ``achieved_goal`` from the observation space of PointMaze.
# This might be useful if you don't plan on training goal oriented learning agents 
# on the Minari dataset generated.
#
# Please note that while only subseting the observation space of PointMaze is touched on in 
# this tutorial, the outlined procedure can be extended to both action/observation spaces 
# of any other environment.
# 
# Let's get started by importing the required modules:

# %%
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import minari
from minari import DataCollectorV0
from minari.data_collector.callbacks import StepDataCallback

# %%
# We'll first need to initialize our PointMaze_UMaze environment and find 
# its the observation space shape

env = gym.make("PointMaze_UMaze-v3")

print(f'Observation space: {env.observation_space}')
# %%
# Which should output:
#
# .. code:: py
#
#  Observation space: Dict('achieved_goal': Box(-inf, inf, (2,), float64), 'desired_goal': Box(-inf, inf, (2,), float64), 'observation': Box(-inf, inf, (4,), float64))
#
# We now need to replicate the observation space with ``achieved_goal`` omitted
# using Gymnasium spaces. We'll also need to define a :class:`minari.StepDataCallback` in order
# to modify ``step_data`` to delete ``achieved_goal``

observation_space_subset = spaces.Dict(
    {
        # "achieved_goal": spaces.Box(low=float('-inf'), high=float('inf'), shape=(2,), dtype=np.float64),
        "desired_goal": spaces.Box(low=float('-inf'), high=float('inf'), shape=(2,), dtype=np.float64),
        "observation": spaces.Box(low=float('-inf'), high=float('inf'), shape=(4,), dtype=np.float64),
    }
)


class CustomSubsetStepDataCallback(StepDataCallback):
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        del step_data["observations"]["achieved_goal"]
        return step_data


# %%
# Finally we'll record 10 episodes with our observation space subset and 
# callback passed to :class:`minari.DataCollectorV0`.

dataset_id = "point-maze-subseted-v3"

# delete the test dataset if it already exists
local_datasets = minari.list_local_datasets()
if dataset_id in local_datasets:
    minari.delete_dataset(dataset_id)

env = DataCollectorV0(
    env,
    observation_space=observation_space_subset,
    # action_space=action_space_subset,
    step_data_callback=CustomSubsetStepDataCallback,
)
num_episodes = 10

env.reset(seed=42)

for episode in range(num_episodes):
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()  # Choose random actions
        _, _, terminated, truncated, _ = env.step(action)
    env.reset()

# Create Minari dataset and store locally
dataset = minari.create_dataset_from_collector_env(
    dataset_id=dataset_id,
    collector_env=env,
    algorithm_name="random_policy",
)

print(dataset.sample_episodes(1)[0].observations.keys())

# %%
# The output from the final line above, should be
#
# .. code:: py
#
#   dict_keys(['desired_goal', 'observation'])
#
# Showing that we have successfully omitted ``achieved_goal`` from the observations.

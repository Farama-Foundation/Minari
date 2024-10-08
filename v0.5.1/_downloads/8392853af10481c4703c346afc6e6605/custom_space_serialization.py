# fmt: off
"""
Serializing a custom space
=========================================
"""
# %%%
# In this tutorial you'll learn how to serialize a custom Gymnasium observation space and use that
# to create a Minari dataset with :class:`minari.DataCollector`. We'll use the
# `MiniGrid Empty <https://minigrid.farama.org/environments/minigrid/EmptyEnv/>`_ environment and
# show how to serialize its unique observation space.
#
# Serializing a custom space can be applied to both observation and action spaces.
#
# Let's start by installing the minigrid library:
#
# ``pip install minigrid``

# %% [markdown]
# Then we can import the required modules:

# %%
import json
from typing import Dict, Union

import gymnasium as gym
from minigrid.core.mission import MissionSpace

import minari
from minari import DataCollector
from minari.serialization import deserialize_space, serialize_space


# %% [markdown]
# First we'll initialize the MiniGrid Empty environment and take a look at its observation and action space.

# %%
env = gym.make("MiniGrid-Empty-16x16-v0")

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# %% [markdown]
# We can see the output from above looks like:
#
#   Observation space: Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function EmptyEnv._gen_mission at 0x12253a940>, None))
#
#   Action space: Discrete(7)
#
# If we take a look at Minari's `serialization functions <https://github.com/Farama-Foundation/Minari/blob/main/minari/serialization.py#L13>`_
# we can see that ``Dict``, ``Discrete``, and ``Box`` are all supported. However ``MissionSpace`` is not
# supported and if try to serialize it with:
#
#  serialize_space(env.observation_space['mission'])
#
# Then we will encounter a ``NotImplementedError`` error:
#
#   NotImplementedError: No serialization method available for MissionSpace(<function EmptyEnv._gen_mission at 0x12253a940>, None)
#
# But what is ``MissonSpace``? If we look at the `source code <https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/mission.py#L14>`_
# we can see that it is simply a wrapper around a ``Callable`` that returns a randomly generated
# string representing the environment's mission. Let's sample from the mission space to see an example:

# %%
env.observation_space['mission'].sample()

# %% [markdown]
# This will print out:
#
#   'get to the green goal square'
#
# For this particular environment we don't have to worry about the mission string varying from sample to sample.
#
# Now that we have a custom observation space we need to define functions that
# properly serialize and deserialize it.
#
# When creating a Minari dataset, the space data gets `serialized <https://minari.farama.org/content/dataset_standards/#space-serialization>`_
# to a JSON format before saving to disk. The `serialize_space <https://github.com/Farama-Foundation/Minari/blob/main/minari/serialization.py#L13C5-L13C20>`_
# function takes care of this conversion for various supported Gymnasium spaces. To enable serialization
# for a custom space we can register 2 new functions that will serialize the space into a JSON object
# and also deserialize it back into the custom space.

# %%


@serialize_space.register(MissionSpace)
def serialize_custom_space(space: MissionSpace, to_string=True) -> Union[Dict, str]:
    result = {}
    result["type"] = "MissionSpace"
    result["mission_func"] = space.mission_func()

    if to_string:
        result = json.dumps(result)
    return result


@deserialize_space.register("MissionSpace")
def deserialize_custom_space(space_dict: Dict) -> MissionSpace:
    assert space_dict["type"] == "MissionSpace"
    mission_func = lambda: space_dict["mission_func"]  # noqa: E731

    return MissionSpace(
        mission_func=mission_func
    )

# %% [markdown]
# Now that we have serialization functions for ``MissionSpace`` we can collect some episode data.

# %%


dataset_id = "minigrid/custom-space-v0"

env = DataCollector(env)
num_episodes = 10

env.reset(seed=42)

for episode in range(num_episodes):
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()  # Choose random actions
        _, _, terminated, truncated, _ = env.step(action)
    env.reset()

# %% [markdown]
# Finally we can create a Minari dataset.

# %%
dataset = env.create_dataset(
    dataset_id=dataset_id,
    algorithm_name="random_policy",
    author="Farama",
    author_email="contact@farama.org",
    code_permalink="https://github.com/Farama-Foundation/Minari/blob/main/docs/tutorials/dataset_creation/custom_space_serialization.py"
)

# %% [markdown]
# To show that the custom space was properly serialized we
# can load the dataset we just created and take a look at
# the observation space.

# %%
del dataset
dataset = minari.load_dataset(dataset_id)

print(dataset.spec.observation_space)

# %% [markdown]
# The output should show the original observation space from earlier
# except with a different ``MissionSpace`` function name since
# we created it inside ``deserialize_custom_space``:
#
#  Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function deserialize_custom_space.<locals>.<lambda> at 0x11f2608b0>, None))
#
# Finally to clean things up, we'll delete the dataset we created earlier:

# %%
minari.delete_dataset(dataset_id)

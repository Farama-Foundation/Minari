# fmt: off
"""
Serializing a custom space
=========================================
"""
# %%%
# In this tutorial you'll learn how to serialize a custom Gym observation space and use that
# to create a Minari dataset with :class:`minari.DataCollectorV0`. We'll use the
# simple `CartPole-v1 <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ environment,
# modify the observation space, show how to serialize it, and create a Minari dataset.
# 
# Serializing a custom space can be applied to both observation and action spaces.
# 
# Let's get started by importing the required modules:

# %%
import gymnasium as gym
from typing import Any, Sequence, Union, Dict
import json
import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray

import minari
from minari import DataCollectorV0
from minari.serialization import serialize_space, deserialize_space
from minari.data_collector.callbacks import StepDataCallback

# %% [markdown]
# First we'll initialize the CartPole environment and take a look at its observation and action space.

# %%
env = gym.make("CartPole-v1")

print(f"Observation space: {env.observation_space}")
print(f"Observation space: {env.action_space}")

# %% [markdown]
# Now we can create a new observation space that inherits from the `gym.Space <https://gymnasium.farama.org/api/spaces/#the-base-class>`_ class.

# %%
class CartPoleObservationSpace(Space):
    def __init__(
        self,
        low: NDArray[Any],
        high: NDArray[Any],
        shape: Sequence[int],
        dtype: type[np.floating[Any]] = np.float32,
    ):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        super().__init__(shape, dtype)

    def sample(self) -> NDArray[Any]:
        """Sample a random observation according to low/high boundaries"""
        sample = np.empty(self.shape)
        sample = self.np_random.uniform(low=self.low, high=self.high, size=self.shape)
        return sample.astype(self.dtype)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space"""
        if not isinstance(x, np.ndarray):
            try:
                x = np.asarray(x, dtype=self.dtype)
            except (ValueError, TypeError):
                return False

        return bool(
            np.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

# %% [markdown]
# Now that we have a custom observation space we need to define functions that properly serialize it.
# 
# When creating a Minari dataset, the space data gets `serialized <https://minari.farama.org/content/dataset_standards/#space-serialization>`_ to a JSON format when saving to disk. The `serialize_space <https://github.com/Farama-Foundation/Minari/blob/main/minari/serialization.py#L13C5-L13C20>`_ function takes care of this conversion for various supported Gym spaces. To enable serialization for a custom space we can register 2 new functions that will serialize the space into a JSON object and also deserialize it into our custom space object.

# %%
@serialize_space.register(CartPoleObservationSpace)
def serialize_custom_space(space: CartPoleObservationSpace, to_string=True) -> Union[Dict, str]:
    result = {}
    result["type"] = "CartPoleObservationSpace"
    result["dtype"] = str(space.dtype)
    result["shape"] = space.shape
    result["low"] = space.low.tolist()
    result["high"] = space.high.tolist()

    if to_string:
        result = json.dumps(result)
    return result

@deserialize_space.register("CartPoleObservationSpace")
def deserialize_custom_space(space_dict: Dict) -> CartPoleObservationSpace:
    assert space_dict["type"] == "CartPoleObservationSpace"
    dtype = space_dict["dtype"]
    shape = space_dict["shape"]
    low = np.array(space_dict["low"])
    high = np.array(space_dict["high"])
    return CartPoleObservationSpace(
        low=low,
        high=high,
        shape=shape,
        dtype=dtype
    )

# %% [markdown]
# Now we can initialize the custom observation space for our environment and collect some episode data.
# 
# The x-position of CartPole's observation space can take values between -4.8 and +4.8.
# For this tutorial we'll use our new class to create an observation space where the
# cart's x-position can only take on values between 0 and +4.8.
# 
# We also need to define a :class:`minari.StepDataCallback` object to manipulate the x-position to be
# above 0.
#
# First element in array is x-position. The rest of elements are CartPole's default values

# %%
custom_observation_space = CartPoleObservationSpace(
    low=np.array([0, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38], dtype=np.float32),
    high=np.array([4.8, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38], dtype=np.float32),
    shape=(4,),
    dtype=np.float32
)

class CustomSpaceStepDataCallback(StepDataCallback):
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        step_data["observations"][0] = max(step_data["observations"][0], 0)
        return step_data

# %%
dataset_id = "cartpole-custom-space-v1"

# delete the test dataset if it already exists
local_datasets = minari.list_local_datasets()
if dataset_id in local_datasets:
    minari.delete_dataset(dataset_id)

env = DataCollectorV0(
    env,
    observation_space=custom_observation_space,
    step_data_callback=CustomSpaceStepDataCallback
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

# %% [markdown]
# Now that we have collected some episode data we can create a Minari dataset and take a look at a sample episode.

# %%
dataset = minari.create_dataset_from_collector_env(
    dataset_id=dataset_id,
    collector_env=env,
    algorithm_name="random_policy",
    author="Farama",
    author_email="contact@farama.org",
    code_permalink="https://github.com/Farama-Foundation/Minari/blob/main/docs/tutorials/dataset_creation/custom_space_serialization.py"
)

ep = dataset.sample_episodes(1)
print(f"Min x-position: {ep[0].observations[:, 0].min():.2f}")

# %%
# The output from the above section will take a sampled episode and show the minimum x-position for a sampled episode being greater than or equal to 0.
#
#   Min x-position: 0.00
#
# To get an idea of what the serialization is doing under the hood we can directly call
# the `serialize_custom_space` function we defined earlier and see the JSON string it returns.

# %%
serialize_custom_space(custom_observation_space)
# %%
# The output should show our custom observation space object as a string:
#
#  '{"type": "CartPoleObservationSpace", "dtype": "float32", "shape": [4], "low": [0.0, -3.4028234663852886e+38, -0.41887903213500977, -3.4028234663852886e+38], "high": [4.800000190734863, 3.4028234663852886e+38, 0.41887903213500977, 3.4028234663852886e+38]}'

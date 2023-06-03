import pytest
import os
import h5py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from minari.dataset.minari_storage import MinariStorage


class DummyEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Dict(
            {
                "component_1": spaces.Box(low=-1, high=1, dtype=np.float32),
                "component_2": spaces.Dict(
                    {
                        "subcomponent_1": spaces.Box(low=2, high=3, dtype=np.float32),
                        "subcomponent_2": spaces.Box(low=4, high=5, dtype=np.float32),
                    }
                ),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "component_1": spaces.Box(low=-1, high=1, dtype=np.float32),
                "component_2": spaces.Dict(
                    {
                        "subcomponent_1": spaces.Box(low=2, high=3, dtype=np.float32),
                        "subcomponent_2": spaces.Box(low=4, high=5, dtype=np.float32),
                    }
                ),
            }
        )

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return self.observation_space.sample(), 0, terminated, False, {}

    def reset(self, seed=0, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}


register(
    id="DummyEnv-v0",
    entry_point="test_dataset_creation:DummyEnv",
    max_episode_steps=5,
)

def create_dummy_dataset(file_path):

    os.makedirs(file_path, exist_ok=True)
    
    with h5py.File(os.path.join(file_path,'dummy-test-v0.hdf5'), 'w') as f:

        f.attrs['flatten_observation'] = False
        f.attrs['flatten_action'] = False
        f.attrs['env_spec'] = r"""{"id": "DummyEnv-v0", "entry_point": "test_dataset_creation:DummyEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 300, "order_enforce": true, "autoreset": false, "disable_env_checker": false, "apply_api_compatibility": false, "kwargs": {"reward_type": "dense"}, "additional_wrappers": [], "vector_entry_point": null}"""
        f.attrs['total_episodes'] = 100
        f.attrs['total_steps'] = 1000
        f.attrs['dataset_id'] = 'dummy-test-v0'


def test_MinariStorage():
    
    file_path = os.path.join(os.path.expanduser("~"), ".minari", "datasets")
    
    create_dummy_dataset(file_path)
    
    with pytest.raises(ModuleNotFoundError, match='Install test_dataset_creation for loading DummyEnv-v0 data'):
        MinariStorage(os.path.join(file_path, "dummy-test-v0.hdf5"))

    os.remove(os.path.join(file_path,'dummy-test-v0.hdf5'))


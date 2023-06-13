from typing import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import data_equivalence

import minari
from minari import MinariDataset
from minari.dataset.minari_storage import MinariStorage


class DummyBoxEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=4, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=4, shape=(3,), dtype=np.float32
        )

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return self.observation_space.sample(), 0, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}


class DummyMultiDimensionalBoxEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(
            low=-1, high=4, shape=(2, 2, 2), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1, high=4, shape=(3, 3, 3), dtype=np.float32
        )

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return self.observation_space.sample(), 0, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}


class DummyTupleDisceteBoxEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(1),
                spaces.Discrete(5),
            )
        )
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=-1, high=4, dtype=np.float32),
                spaces.Discrete(5),
            )
        )

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return self.observation_space.sample(), 0, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}


class DummyDictEnv(gym.Env):
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

    def reset(self, seed=None, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}


class DummyTupleEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple(
            (
                spaces.Box(low=2, high=3, dtype=np.float32),
                spaces.Box(low=4, high=5, dtype=np.float32),
            )
        )

        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=2, high=3, dtype=np.float32),
                spaces.Tuple(
                    (
                        spaces.Box(low=2, high=3, dtype=np.float32),
                        spaces.Box(low=4, high=5, dtype=np.float32),
                    )
                ),
            )
        )

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return self.observation_space.sample(), 0, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}


class DummyComboEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple(
            (
                spaces.Box(low=2, high=3, dtype=np.float32),
                spaces.Box(low=4, high=5, dtype=np.float32),
            )
        )

        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=2, high=3, dtype=np.float32),
                spaces.Tuple(
                    (
                        spaces.Box(low=2, high=3, dtype=np.float32),
                        spaces.Dict(
                            {
                                "component_1": spaces.Box(
                                    low=-1, high=1, dtype=np.float32
                                ),
                                "component_2": spaces.Dict(
                                    {
                                        "subcomponent_1": spaces.Box(
                                            low=2, high=3, dtype=np.float32
                                        ),
                                        "subcomponent_2": spaces.Tuple(
                                            (
                                                spaces.Box(
                                                    low=4, high=5, dtype=np.float32
                                                ),
                                                spaces.Discrete(10),
                                            )
                                        ),
                                    }
                                ),
                            }
                        ),
                    )
                ),
            )
        )

    def step(self, action):
        terminated = self.timestep > 5
        self.timestep += 1

        return self.observation_space.sample(), 0, terminated, False, {}

    def reset(self, seed=0, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}


def register_dummy_envs():

    register(
        id="DummyBoxEnv-v0",
        entry_point="tests.common:DummyBoxEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyMultiDimensionalBoxEnv-v0",
        entry_point="tests.common:DummyMultiDimensionalBoxEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyTupleDisceteBoxEnv-v0",
        entry_point="tests.common:DummyTupleDisceteBoxEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyDictEnv-v0",
        entry_point="tests.common:DummyDictEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyTupleEnv-v0",
        entry_point="tests.common:DummyTupleEnv",
        max_episode_steps=5,
    )

    register(
        id="DummyComboEnv-v0",
        entry_point="tests.common:DummyComboEnv",
        max_episode_steps=5,
    )


test_spaces = [
    gym.spaces.Box(low=-1, high=4, shape=(2,), dtype=np.float32),
    gym.spaces.Box(low=-1, high=4, shape=(3,), dtype=np.float32),
    gym.spaces.Box(low=-1, high=4, shape=(2, 2, 2), dtype=np.float32),
    gym.spaces.Box(low=-1, high=4, shape=(3, 3, 3), dtype=np.float32),
    gym.spaces.Tuple(
        (
            gym.spaces.Discrete(1),
            gym.spaces.Discrete(5),
        )
    ),
    gym.spaces.Tuple(
        (
            gym.spaces.Box(low=-1, high=4, dtype=np.float32),
            gym.spaces.Discrete(5),
        )
    ),
    gym.spaces.Dict(
        {
            "component_1": gym.spaces.Box(low=-1, high=1, dtype=np.float32),
            "component_2": gym.spaces.Dict(
                {
                    "subcomponent_1": gym.spaces.Box(low=2, high=3, dtype=np.float32),
                    "subcomponent_2": gym.spaces.Box(low=4, high=5, dtype=np.float32),
                }
            ),
        }
    ),
    gym.spaces.Tuple(
        (
            gym.spaces.Box(low=2, high=3, dtype=np.float32),
            gym.spaces.Box(low=4, high=5, dtype=np.float32),
        )
    ),
    gym.spaces.Tuple(
        (
            gym.spaces.Box(low=2, high=3, dtype=np.float32),
            gym.spaces.Tuple(
                (
                    gym.spaces.Box(low=2, high=3, dtype=np.float32),
                    gym.spaces.Box(low=4, high=5, dtype=np.float32),
                )
            ),
        )
    ),
    gym.spaces.Tuple(
        (
            gym.spaces.Box(low=2, high=3, dtype=np.float32),
            gym.spaces.Tuple(
                (
                    gym.spaces.Box(low=2, high=3, dtype=np.float32),
                    gym.spaces.Dict(
                        {
                            "component_1": gym.spaces.Box(
                                low=-1, high=1, dtype=np.float32
                            ),
                            "component_2": gym.spaces.Dict(
                                {
                                    "subcomponent_1": gym.spaces.Box(
                                        low=2, high=3, dtype=np.float32
                                    ),
                                    "subcomponent_2": gym.spaces.Tuple(
                                        (
                                            gym.spaces.Box(
                                                low=4, high=5, dtype=np.float32
                                            ),
                                            gym.spaces.Discrete(10),
                                        )
                                    ),
                                }
                            ),
                        }
                    ),
                )
            ),
        )
    ),
]


def check_env_recovery_with_subset_spaces(
    gymnasium_environment: gym.Env,
    dataset: MinariDataset,
    action_space_subset: gym.spaces.Space,
    observation_space_subset: gym.spaces.Space,
):
    """Test that the recovered environment from MinariDataset is the same as the one used to generate the dataset.

    Args:
        gymnasium_environment (gym.Env): original Gymnasium environment
        dataset (MinariDataset): Minari dataset created with gymnasium_environment
        action_space_subset (gym.spaces.Space): desired subset action space
        observation_space_subset (gym.spaces.Space): desired subset observation space

    """
    recovered_env = dataset.recover_environment()

    # Check that environment spec is the same
    assert recovered_env.spec == gymnasium_environment.spec

    # Check that action/observation spaces are the same
    assert data_equivalence(
        recovered_env.observation_space, gymnasium_environment.observation_space
    )
    assert data_equivalence(dataset.spec.observation_space, observation_space_subset)
    assert data_equivalence(
        recovered_env.action_space, gymnasium_environment.action_space
    )
    assert data_equivalence(dataset.spec.action_space, action_space_subset)


def check_env_recovery(gymnasium_environment: gym.Env, dataset: MinariDataset):
    """Test that the recovered environment from MinariDataset is the same as the one used to generate the dataset.

    Args:
        gymnasium_environment (gym.Env): original Gymnasium environment
        dataset (MinariDataset): Minari dataset created with gymnasium_environment
    """
    recovered_env = dataset.recover_environment()

    # Check that environment spec is the same
    assert recovered_env.spec == gymnasium_environment.spec

    # Check that action/observation spaces are the same
    assert data_equivalence(
        recovered_env.observation_space, gymnasium_environment.observation_space
    )
    assert data_equivalence(
        dataset.spec.observation_space, gymnasium_environment.observation_space
    )
    assert data_equivalence(
        recovered_env.action_space, gymnasium_environment.action_space
    )
    assert data_equivalence(
        dataset.spec.action_space, gymnasium_environment.action_space
    )


def check_data_integrity(data: MinariStorage, episode_indices: Iterable[int]):
    """Checks to see if a MinariStorage episode has consistent data and has episodes at the expected indices.

    Args:
        data (MinariStorage): a MinariStorage instance
        episode_indices (Iterable[int]): the list of episode indices expected
    """
    episodes = data.get_episodes(episode_indices)
    # verify we have the right number of episodes, available at the right indices
    assert data.total_episodes == len(episodes)
    # verify the actions and observations are in the appropriate action space and observation space, and that the episode lengths are correct
    for episode in episodes:
        assert episode["total_timesteps"] + 1 == len(episode["observations"])
        assert episode["total_timesteps"] == len(episode["actions"])
        assert episode["total_timesteps"] == len(episode["rewards"])
        assert episode["total_timesteps"] == len(episode["terminations"])
        assert episode["total_timesteps"] == len(episode["truncations"])

        for observation in episode["observations"]:
            assert observation in data.observation_space
        for action in episode["actions"]:
            assert action in data.action_space


def check_load_and_delete_dataset(dataset_id: str):
    """Test loading and deletion of local Minari datasets.

    Args:
        dataset_id (str): name of Minari dataset to test
    """
    # check dataset name is present in local database
    local_datasets = minari.list_local_datasets()
    assert dataset_id in local_datasets

    # load dataset
    loaded_dataset = minari.load_dataset(dataset_id)
    assert isinstance(loaded_dataset, MinariDataset)
    assert dataset_id == loaded_dataset.spec.dataset_id

    # delete dataset and check that it's no longer present in local database
    minari.delete_dataset(dataset_id)
    local_datasets = minari.list_local_datasets()
    assert dataset_id not in local_datasets

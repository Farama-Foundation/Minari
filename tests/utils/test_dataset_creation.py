from typing import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import data_equivalence

import minari
from minari import DataCollectorV0, MinariDataset
from minari.dataset.minari_storage import MinariStorage


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

    def reset(self, seed=0, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}


register(
    id="DummyDictEnv-v0",
    entry_point="test_dataset_creation:DummyDictEnv",
    max_episode_steps=5,
)


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

    def reset(self, seed=0, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}


register(
    id="DummyTupleEnv-v0",
    entry_point="test_dataset_creation:DummyTupleEnv",
    max_episode_steps=5,
)


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
                                        "subcomponent_2": spaces.Box(
                                            low=4, high=5, dtype=np.float32
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


register(
    id="DummyComboEnv-v0",
    entry_point="test_dataset_creation:DummyComboEnv",
    max_episode_steps=5,
)


def _check_env_recovery(gymnasium_environment: gym.Env, dataset: MinariDataset):
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


def _check_data_integrity(data: MinariStorage, episode_indices: Iterable[int]):
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
        print(type(episode))
        # assert episode["total_timesteps"] + 1 == len(episode["observations"])
        # assert episode["total_timesteps"] == len(episode["actions"])
        assert episode["total_timesteps"] == len(episode["rewards"])
        assert episode["total_timesteps"] == len(episode["terminations"])
        assert episode["total_timesteps"] == len(episode["truncations"])

        for observation in episode["observations"]:
            assert observation in data.observation_space
        for action in episode["actions"]:
            assert action in data.action_space


def _check_load_and_delete_dataset(dataset_id: str):
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


def test_generate_combo_dataset_with_collector_env():
    dataset_id = "dummy-combo-test-v0"

    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make("DummyComboEnv-v0")

    env = DataCollectorV0(env)
    num_episodes = 10

    # Step the environment, DataCollectorV0 wrapper will do the data collection job
    env.reset(seed=42)

    for episode in range(num_episodes):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            _, _, terminated, truncated, _ = env.step(action)

        env.reset()

    # Create Minari dataset and store locally
    dataset = minari.create_dataset_from_collector_env(
        dataset_id=dataset_id,
        collector_env=env,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="John U. Balis",
        author_email="balisujohn@gmail.com",
    )

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    _check_data_integrity(dataset._data, dataset.episode_indices)

    # check that the environment can be recovered from the dataset
    _check_env_recovery(env.env, dataset)

    env.close()

    # check load and delete local dataset
    _check_load_and_delete_dataset(dataset_id)


def test_generate_tuple_dataset_with_collector_env():
    dataset_id = "dummy-tuple-test-v0"

    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make("DummyTupleEnv-v0")

    env = DataCollectorV0(env)
    num_episodes = 10

    # Step the environment, DataCollectorV0 wrapper will do the data collection job
    env.reset(seed=42)

    for episode in range(num_episodes):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            _, _, terminated, truncated, _ = env.step(action)

        env.reset()

    # Create Minari dataset and store locally
    dataset = minari.create_dataset_from_collector_env(
        dataset_id=dataset_id,
        collector_env=env,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="John U. Balis",
        author_email="balisujohn@gmail.com",
    )

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    _check_data_integrity(dataset._data, dataset.episode_indices)

    # check that the environment can be recovered from the dataset
    _check_env_recovery(env.env, dataset)

    env.close()

    # check load and delete local dataset
    _check_load_and_delete_dataset(dataset_id)


def test_generate_dict_dataset_with_collector_env():
    dataset_id = "dummy-dict-test-v0"

    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make("DummyDictEnv-v0")

    env = DataCollectorV0(env)
    num_episodes = 10

    # Step the environment, DataCollectorV0 wrapper will do the data collection job
    env.reset(seed=42)

    for episode in range(num_episodes):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            _, _, terminated, truncated, _ = env.step(action)

        env.reset()

    # Create Minari dataset and store locally
    dataset = minari.create_dataset_from_collector_env(
        dataset_id=dataset_id,
        collector_env=env,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="John U. Balis",
        author_email="balisujohn@gmail.com",
    )

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    _check_data_integrity(dataset._data, dataset.episode_indices)

    # check that the environment can be recovered from the dataset
    _check_env_recovery(env.env, dataset)

    env.close()

    # check load and delete local dataset
    _check_load_and_delete_dataset(dataset_id)


def test_generate_dataset_with_collector_env():
    """Test DataCollectorV0 wrapper and Minari dataset creation."""
    dataset_id = "cartpole-test-v0"
    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make("CartPole-v1")

    env = DataCollectorV0(env)
    num_episodes = 10

    # Step the environment, DataCollectorV0 wrapper will do the data collection job
    env.reset(seed=42)

    for episode in range(num_episodes):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            _, _, terminated, truncated, _ = env.step(action)

        env.reset()

    # Create Minari dataset and store locally
    dataset = minari.create_dataset_from_collector_env(
        dataset_id=dataset_id,
        collector_env=env,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
    )

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    _check_data_integrity(dataset._data, dataset.episode_indices)

    # check that the environment can be recovered from the dataset
    _check_env_recovery(env.env, dataset)

    env.close()

    # check load and delete local dataset
    _check_load_and_delete_dataset(dataset_id)


def test_generate_dataset_with_external_buffer():
    """Test create dataset from external buffers without using DataCollectorV0."""
    buffer = []
    dataset_id = "cartpole-test-v0"

    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make("CartPole-v1")

    observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []

    num_episodes = 10

    observation, info = env.reset(seed=42)

    # Step the environment, DataCollectorV0 wrapper will do the data collection job
    observation, _ = env.reset()
    observations.append(observation)
    for episode in range(num_episodes):
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            observation, reward, terminated, truncated, _ = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)

        episode_buffer = {
            "observations": np.asarray(observations),
            "actions": np.asarray(actions),
            "rewards": np.asarray(rewards),
            "terminations": np.asarray(terminations),
            "truncations": np.asarray(truncations),
        }
        buffer.append(episode_buffer)

        observations.clear()
        actions.clear()
        rewards.clear()
        terminations.clear()
        truncations.clear()

        observation, _ = env.reset()
        observations.append(observation)

    # Create Minari dataset and store locally
    dataset = minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        env=env,
        buffer=buffer,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
    )

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    _check_data_integrity(dataset._data, dataset.episode_indices)
    _check_env_recovery(env, dataset)

    env.close()

    _check_load_and_delete_dataset(dataset_id)

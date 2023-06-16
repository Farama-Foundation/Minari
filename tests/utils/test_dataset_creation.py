import copy
from collections import OrderedDict
from typing import Dict

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import minari
from minari import DataCollectorV0, MinariDataset
from tests.common import (
    check_data_integrity,
    check_env_recovery,
    check_env_recovery_with_subset_spaces,
    check_load_and_delete_dataset,
    register_dummy_envs,
)


register_dummy_envs()


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("cartpole-test-v0", "CartPole-v1"),
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-box-test-v0", "DummyBoxEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDisceteBoxEnv-v0"),
    ],
)
def test_generate_dataset_with_collector_env(dataset_id, env_id):
    """Test DataCollectorV0 wrapper and Minari dataset creation."""
    # dataset_id = "cartpole-test-v0"
    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

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
            if terminated or truncated:
                assert not env._buffer[-1]
            else:
                assert env._buffer[-1]

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

    check_data_integrity(dataset._data, dataset.episode_indices)

    # check that the environment can be recovered from the dataset
    check_env_recovery(env.env, dataset)

    env.close()
    # check load and delete local dataset
    check_load_and_delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("cartpole-test-v0", "CartPole-v1"),
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDisceteBoxEnv-v0"),
    ],
)
def test_generate_dataset_with_external_buffer(dataset_id, env_id):
    """Test create dataset from external buffers without using DataCollectorV0."""
    buffer = []
    # dataset_id = "cartpole-test-v0"

    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

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
            "observations": copy.deepcopy(observations),
            "actions": copy.deepcopy(actions),
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

    check_data_integrity(dataset._data, dataset.episode_indices)
    check_env_recovery(env, dataset)

    env.close()

    check_load_and_delete_dataset(dataset_id)


def _space_subset_helper(entry: Dict):

    return OrderedDict(
        {
            "component_2": OrderedDict(
                {"subcomponent_2": entry["component_2"]["subcomponent_2"]}
            )
        }
    )


def test_generate_dataset_with_space_subset_external_buffer():
    """Test create dataset from external buffers without using DataCollectorV0."""
    buffer = []
    dataset_id = "dummy-dict-test-v0"

    # delete the test dataset if it already exists

    action_space_subset = spaces.Dict(
        {
            "component_2": spaces.Dict(
                {
                    "subcomponent_2": spaces.Box(low=4, high=5, dtype=np.float32),
                }
            ),
        }
    )
    observation_space_subset = spaces.Dict(
        {
            "component_2": spaces.Dict(
                {
                    "subcomponent_2": spaces.Box(low=4, high=5, dtype=np.float32),
                }
            ),
        }
    )

    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make("DummyDictEnv-v0")

    observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []

    num_episodes = 10

    observation, info = env.reset(seed=42)

    # Step the environment, DataCollectorV0 wrapper will do the data collection job
    observation, _ = env.reset()
    observations.append(_space_subset_helper(observation))
    for episode in range(num_episodes):
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            observation, reward, terminated, truncated, _ = env.step(action)
            observations.append(_space_subset_helper(observation))
            actions.append(_space_subset_helper(action))
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)

        episode_buffer = {
            "observations": copy.deepcopy(observations),
            "actions": copy.deepcopy(actions),
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
        observations.append(_space_subset_helper(observation))

    # Create Minari dataset and store locally
    dataset = minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        env=env,
        buffer=buffer,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
        action_space=action_space_subset,
        observation_space=observation_space_subset,
    )

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    check_data_integrity(dataset._data, dataset.episode_indices)
    check_env_recovery_with_subset_spaces(
        env, dataset, action_space_subset, observation_space_subset
    )

    env.close()

    check_load_and_delete_dataset(dataset_id)

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import data_equivalence

import minari
from minari import DataCollectorV0, MinariDataset


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
        dataset.observation_space, gymnasium_environment.observation_space
    )
    assert data_equivalence(
        recovered_env.action_space, gymnasium_environment.action_space
    )
    assert data_equivalence(dataset.action_space, gymnasium_environment.action_space)


def _check_load_and_delete_dataset(dataset_name: str):
    """Test loading and deletion of local Minari datasets.

    Args:
        dataset_name (str): name of Minari dataset to test
    """
    # check dataset name is present in local database
    local_datasets = minari.list_local_datasets()
    assert dataset_name in local_datasets

    # load dataset
    loaded_dataset = minari.load_dataset(dataset_name)
    assert isinstance(loaded_dataset, MinariDataset)
    assert dataset_name == loaded_dataset.name

    # delete dataset and check that it's no longer present in local database
    minari.delete_dataset(dataset_name)
    local_datasets = minari.list_local_datasets()
    assert dataset_name not in local_datasets


def test_generate_dataset_with_collector_env():
    """Test DataCollectorV0 wrapper and Minari dataset creation."""
    dataset_name = "CartPole-v1_test-dataset"
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
        dataset_name=dataset_name,
        collector_env=env,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
    )

    assert isinstance(dataset, MinariDataset)

    # check that the environment can be recovered from the dataset
    _check_env_recovery(env.env, dataset)

    env.close()

    # check load and delete local dataset
    _check_load_and_delete_dataset(dataset_name)


def test_generate_dataset_with_external_buffer():
    """Test create dataset from external buffers without using DataCollectorV0."""
    buffer = []
    dataset_name = "CartPole-v1_test-dataset"
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
        dataset_name=dataset_name,
        env=env,
        buffer=buffer,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
    )

    assert isinstance(dataset, MinariDataset)
    _check_env_recovery(env, dataset)

    env.close()

    _check_load_and_delete_dataset(dataset_name)

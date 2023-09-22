import copy
import os

import gymnasium as gym
import h5py
import numpy as np
import pytest

import minari
from minari import DataCollectorV0, __version__
from minari.dataset.minari_storage import MinariStorage, get_dataset_size
from minari.utils import get_dataset_path
from tests.common import (
    check_data_integrity,
    check_load_and_delete_dataset,
    register_dummy_envs,
)


register_dummy_envs()

file_path = os.path.join(os.path.expanduser("~"), ".minari", "datasets")


def _create_dummy_dataset(file_path):

    os.makedirs(file_path, exist_ok=True)

    with h5py.File(os.path.join(file_path, "dummy-test-v0.hdf5"), "w") as f:

        f.attrs["flatten_observation"] = False
        f.attrs["flatten_action"] = False
        f.attrs[
            "env_spec"
        ] = r"""{"id": "DummyEnv-v0", "entry_point": "dummymodule:dummyenv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 300, "order_enforce": true, "disable_env_checker": false, "apply_api_compatibility": false, "additional_wrappers": []}"""
        f.attrs["total_episodes"] = 100
        f.attrs["total_steps"] = 1000
        f.attrs["dataset_id"] = "dummy-test-v0"
        f.attrs["minari_version"] = f"=={__version__}"


def test_minari_storage_missing_env_module():

    file_path = os.path.join(os.path.expanduser("~"), ".minari", "datasets")

    _create_dummy_dataset(file_path)

    with pytest.raises(
        ModuleNotFoundError, match="Install dummymodule for loading DummyEnv-v0 data"
    ):
        MinariStorage(os.path.join(file_path, "dummy-test-v0.hdf5"))

    os.remove(os.path.join(file_path, "dummy-test-v0.hdf5"))


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("cartpole-test-v0", "CartPole-v1"),
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-box-test-v0", "DummyBoxEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-text-test-v0", "DummyTextEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDiscreteBoxEnv-v0"),
    ],
)
def test_minari_get_dataset_size_from_collector_env(dataset_id, env_id):
    """Test get_dataset_size method for dataset made using create_dataset_from_collector_env method."""
    # dataset_id = "cartpole-test-v0"
    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    env = DataCollectorV0(env)
    num_episodes = 100

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

    file_path = get_dataset_path(dataset_id)
    data_path = os.path.join(file_path, "data", "main_data.hdf5")
    original_dataset_size = os.path.getsize(data_path)
    original_dataset_size = np.round(original_dataset_size / 1000000, 1)

    assert get_dataset_size(dataset_id) == original_dataset_size

    check_data_integrity(dataset._data, dataset.episode_indices)

    env.close()

    check_load_and_delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("cartpole-test-v0", "CartPole-v1"),
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-box-test-v0", "DummyBoxEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-text-test-v0", "DummyTextEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDiscreteBoxEnv-v0"),
    ],
)
def test_minari_get_dataset_size_from_buffer(dataset_id, env_id):
    """Test get_dataset_size method for dataset made using create_dataset_from_buffers method."""
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

    file_path = get_dataset_path(dataset_id)
    data_path = os.path.join(file_path, "data", "main_data.hdf5")
    original_dataset_size = os.path.getsize(data_path)
    original_dataset_size = np.round(original_dataset_size / 1000000, 1)

    assert get_dataset_size(dataset_id) == original_dataset_size

    check_data_integrity(dataset._data, dataset.episode_indices)

    env.close()

    check_load_and_delete_dataset(dataset_id)

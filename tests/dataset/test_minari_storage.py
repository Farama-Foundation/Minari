import copy
import os

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import minari
from minari import DataCollectorV0
from minari.dataset.minari_storage import MinariStorage
from tests.common import (
    check_data_integrity,
    check_load_and_delete_dataset,
    register_dummy_envs,
)


register_dummy_envs()

file_path = os.path.join(os.path.expanduser("~"), ".minari", "datasets")


def _generate_episode_dict(
    observation_space: spaces.Space, action_space: spaces.Space, length=25
):
    terminations = np.zeros(length, dtype=np.bool_)
    truncations = np.zeros(length, dtype=np.bool_)
    terminated = np.random.randint(2, dtype=np.bool_)
    terminations[-1] = terminated
    truncations[-1] = not terminated

    return {
        "observations": [observation_space.sample() for _ in range(length + 1)],
        "actions": [action_space.sample() for _ in range(length)],
        "rewards": np.random.randn(length),
        "terminations": terminations,
        "truncations": truncations,
    }


def test_non_existing_data(tmp_dataset_dir):
    with pytest.raises(ValueError, match="The data path foo doesn't exist"):
        MinariStorage("foo")

    with pytest.raises(ValueError, match="No data found in data path"):
        MinariStorage(tmp_dataset_dir)


def test_metadata(tmp_dataset_dir):
    action_space = spaces.Box(-1, 1)
    observation_space = spaces.Box(-1, 1)
    storage = MinariStorage.new(
        data_path=tmp_dataset_dir,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert storage.data_path == tmp_dataset_dir

    extra_metadata = {"float": 3.2, "string": "test-value", "int": 2, "bool": True}
    storage.update_metadata(extra_metadata)

    storage_metadata = storage.metadata
    assert storage_metadata.keys() == {
        "action_space",
        "bool",
        "float",
        "int",
        "observation_space",
        "string",
        "total_episodes",
        "total_steps",
    }

    for key, value in extra_metadata.items():
        assert storage_metadata[key] == value

    storage2 = MinariStorage(tmp_dataset_dir)
    assert storage_metadata == storage2.metadata


def test_add_episodes(tmp_dataset_dir):
    action_space = spaces.Box(-1, 1, shape=(10,))
    observation_space = spaces.Text(max_length=5)
    n_episodes = 10
    steps_per_episode = 25
    episodes = [
        _generate_episode_dict(
            observation_space, action_space, length=steps_per_episode
        )
        for _ in range(n_episodes)
    ]
    storage = MinariStorage.new(
        data_path=tmp_dataset_dir,
        observation_space=observation_space,
        action_space=action_space,
    )
    storage.update_episodes(episodes)
    del storage
    storage = MinariStorage(tmp_dataset_dir)

    assert storage.total_episodes == n_episodes
    assert storage.total_steps == n_episodes * steps_per_episode

    for i, ep in enumerate(episodes):
        storage_ep = storage.get_episodes([i])[0]

        assert np.all(ep["observations"] == storage_ep["observations"])
        assert np.all(ep["actions"] == storage_ep["actions"])
        assert np.all(ep["rewards"] == storage_ep["rewards"])
        assert np.all(ep["terminations"] == storage_ep["terminations"])
        assert np.all(ep["truncations"] == storage_ep["truncations"])


def test_append_episode_chunks(tmp_dataset_dir):
    action_space = spaces.Discrete(10)
    observation_space = spaces.Text(max_length=5)
    lens = [10, 7, 15]
    chunk1 = _generate_episode_dict(observation_space, action_space, length=lens[0])
    chunk2 = _generate_episode_dict(observation_space, action_space, length=lens[1])
    chunk3 = _generate_episode_dict(observation_space, action_space, length=lens[2])
    chunk1["terminations"][-1] = False
    chunk1["truncations"][-1] = False
    chunk2["terminations"][-1] = False
    chunk2["truncations"][-1] = False
    chunk2["observations"] = chunk2["observations"][:-1]
    chunk3["observations"] = chunk3["observations"][:-1]

    storage = MinariStorage.new(tmp_dataset_dir, observation_space, action_space)
    storage.update_episodes([chunk1])
    assert storage.total_episodes == 1
    assert storage.total_steps == lens[0]

    chunk2["id"] = 0
    chunk3["id"] = 0
    storage.update_episodes([chunk2, chunk3])
    assert storage.total_episodes == 1
    assert storage.total_steps == sum(lens)


def test_apply(tmp_dataset_dir):
    action_space = spaces.Box(-1, 1, shape=(10,))
    observation_space = spaces.Text(max_length=5)
    n_episodes = 10
    episodes = [
        _generate_episode_dict(observation_space, action_space)
        for _ in range(n_episodes)
    ]
    storage = MinariStorage.new(
        data_path=tmp_dataset_dir,
        observation_space=observation_space,
        action_space=action_space,
    )
    storage.update_episodes(episodes)

    def f(ep):
        return ep["actions"].sum()

    episode_indices = [1, 3, 5]
    outs = storage.apply(f, episode_indices=episode_indices)
    assert len(episode_indices) == len(list(outs))
    for i, result in zip(episode_indices, outs):
        assert np.array(episodes[i]["actions"]).sum() == result


def test_episode_metadata(tmp_dataset_dir):
    action_space = spaces.Box(-1, 1, shape=(10,))
    observation_space = spaces.Text(max_length=5)
    n_episodes = 10
    episodes = [
        _generate_episode_dict(observation_space, action_space)
        for _ in range(n_episodes)
    ]
    storage = MinariStorage.new(
        data_path=tmp_dataset_dir,
        observation_space=observation_space,
        action_space=action_space,
    )
    storage.update_episodes(episodes)

    ep_metadatas = [
        {"foo1-1": True, "foo1-2": 7},
        {"foo2-1": 3.14},
        {"foo3-1": "foo", "foo3-2": 42, "foo3-3": "test"},
    ]

    ep_indices = [1, 4, 5]
    storage.update_episode_metadata(ep_metadatas, episode_indices=ep_indices)


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("cartpole-test-v0", "CartPole-v1"),
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-box-test-v0", "DummyBoxEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDiscreteBoxEnv-v0"),
    ],
)
def test_minari_get_dataset_size_from_collector_env(dataset_id, env_id):
    """Test get_dataset_size method for dataset made using create_dataset_from_collector_env method."""
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
        done = False
        while not done:
            action = env.action_space.sample()  # User-defined policy function
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.reset()

    # Create Minari dataset and store locally
    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
    )

    assert dataset.storage.metadata['dataset_size'] == dataset.storage.get_size()

    check_data_integrity(dataset.storage, dataset.episode_indices)

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

    assert dataset.storage.metadata['dataset_size'] == dataset.storage.get_size()

    check_data_integrity(dataset.storage, dataset.episode_indices)

    env.close()

    check_load_and_delete_dataset(dataset_id)

import copy
import json
import os
import re
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

import minari
from minari import DataCollector, MinariDataset
from minari.dataset.minari_dataset import EpisodeData
from minari.dataset.minari_storage import METADATA_FILE_NAME
from tests.common import (
    check_data_integrity,
    check_env_recovery,
    check_episode_data_integrity,
    check_load_and_delete_dataset,
    create_dummy_dataset_with_collecter_env_helper,
    register_dummy_envs,
    test_spaces,
)


register_dummy_envs()


@pytest.mark.parametrize("space", test_spaces)
def test_episode_data(space: gym.Space):
    id = np.random.randint(1024)
    seed = np.random.randint(1024)
    total_step = 100
    rewards = np.random.randn(total_step)
    choices = np.array([True, False])
    terminations = np.random.choice(choices, size=(total_step,))
    truncations = np.random.choice(choices, size=(total_step,))
    episode_data = EpisodeData(
        id=id,
        seed=seed,
        total_steps=total_step,
        observations=space.sample(),
        actions=space.sample(),
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
        infos={},
    )

    pattern = r"EpisodeData\("
    pattern += r"id=\d+, "
    pattern += r"seed=\d+, "
    pattern += r"total_steps=100, "
    pattern += r"observations=.+, "
    pattern += r"actions=.+, "
    pattern += r"rewards=.+, "
    pattern += r"terminations=.+, "
    pattern += r"truncations=.+"
    pattern += r"infos=.+"
    pattern += r"\)"
    assert re.fullmatch(pattern, repr(episode_data))


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
def test_update_dataset_from_collector_env(dataset_id, env_id):
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    env = DataCollector(env)
    num_episodes = 10

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

    # Step the environment, DataCollector wrapper will do the data collection job
    env.reset(seed=42)

    for episode in range(num_episodes):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            _, _, terminated, truncated, _ = env.step(action)

        env.reset()

    env.add_to_dataset(dataset)

    assert isinstance(dataset, MinariDataset)
    assert isinstance(dataset.total_steps, int)
    assert dataset.total_episodes == num_episodes * 2
    assert dataset.spec.total_episodes == num_episodes * 2
    assert len(dataset.episode_indices) == num_episodes * 2

    check_data_integrity(dataset.storage, dataset.episode_indices)
    check_env_recovery(env.env, dataset)

    env.close()

    check_load_and_delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-box-test-v0", "DummyBoxEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDiscreteBoxEnv-v0"),
    ],
)
def test_filter_episodes_and_subsequent_updates(dataset_id, env_id):
    """Tests to make sure that episodes are filtered filtered correctly.

    Additionally ensures indices are correctly updated when adding more episodes to a filtered dataset.
    """
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    env = DataCollector(env)
    num_episodes = 10

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

    def filter_by_index(episode: Any):
        return int(episode.id) <= 6

    filtered_dataset = dataset.filter_episodes(filter_by_index)

    assert isinstance(filtered_dataset, MinariDataset)
    assert filtered_dataset.total_episodes == 7
    assert filtered_dataset.spec.total_episodes == 7
    assert len(filtered_dataset.episode_indices) == 7

    check_data_integrity(
        filtered_dataset.storage, dataset.episode_indices
    )  # checks that the underlying episodes are still present in the `MinariStorage` object
    check_env_recovery(env.env, filtered_dataset)

    # Step the environment, DataCollector wrapper will do the data collection job
    env.reset(seed=42)

    for episode in range(num_episodes):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            _, _, terminated, truncated, _ = env.step(action)

        env.reset()

    env.add_to_dataset(filtered_dataset)

    assert isinstance(filtered_dataset, MinariDataset)
    assert isinstance(filtered_dataset.spec.total_steps, int)
    assert filtered_dataset.total_episodes == 17
    assert filtered_dataset.spec.total_episodes == 17
    assert filtered_dataset.spec.total_steps == 17 * 5
    assert tuple(filtered_dataset.episode_indices) == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
    )
    assert filtered_dataset.storage.total_episodes == 20
    assert dataset.storage.total_episodes == 20
    check_env_recovery(env.env, filtered_dataset)

    env.close()

    env = gym.make(env_id)
    buffer = []

    observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []

    num_episodes = 10

    observation, info = env.reset(seed=42)

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

    filtered_dataset.update_dataset_from_buffer(buffer)

    assert isinstance(filtered_dataset, MinariDataset)
    assert isinstance(filtered_dataset.spec.total_steps, int)
    assert filtered_dataset.total_episodes == 27
    assert filtered_dataset.spec.total_episodes == 27
    assert filtered_dataset.spec.total_steps == 27 * 5

    assert tuple(filtered_dataset.episode_indices) == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
    )
    assert filtered_dataset.storage.total_episodes == 30
    assert dataset.storage.total_episodes == 30
    check_env_recovery(env, filtered_dataset)

    check_load_and_delete_dataset(dataset_id)
    env.close()


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
def test_sample_episodes(dataset_id, env_id):
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    env = DataCollector(env)
    num_episodes = 10

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

    def filter_by_index(episode: Any):
        return int(episode.id) >= 3

    filtered_dataset = dataset.filter_episodes(filter_by_index)
    for i in [1, 7]:
        episodes = list(filtered_dataset.sample_episodes(i))
        assert len(episodes) == i
        check_episode_data_integrity(
            episodes,
            filtered_dataset.spec.observation_space,
            filtered_dataset.spec.action_space,
        )
    with pytest.raises(ValueError):
        episodes = filtered_dataset.sample_episodes(8)

    env.close()


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
def test_iterate_episodes(dataset_id, env_id):
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    env = DataCollector(env)
    num_episodes = 10

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )
    env.close()

    episodes = list(dataset.iterate_episodes([1, 3, 5]))

    assert {1, 3, 5} == {episode.id for episode in episodes}

    assert len(episodes) == 3
    check_episode_data_integrity(
        episodes, dataset.spec.observation_space, dataset.spec.action_space
    )

    all_episodes = list(dataset.iterate_episodes())
    assert len(all_episodes) == 10
    assert {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} == {episode.id for episode in all_episodes}

    length = 0
    for i, ep in enumerate(dataset):
        assert dataset[i].id == i
        assert ep.id == i
        length += 1
    assert length == 10
    assert len(dataset) == 10


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
def test_update_dataset_from_buffer(dataset_id, env_id):
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    collector_env = DataCollector(env)
    num_episodes = 10

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, collector_env, num_episodes=num_episodes
    )

    buffer = []

    observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []

    num_episodes = 10

    observation, info = env.reset(seed=42)

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

    dataset.update_dataset_from_buffer(buffer)

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes * 2
    assert dataset.spec.total_episodes == num_episodes * 2
    assert len(dataset.episode_indices) == num_episodes * 2

    check_data_integrity(dataset.storage, dataset.episode_indices)
    check_env_recovery(env, dataset)

    collector_env.close()
    check_load_and_delete_dataset(dataset_id)


def test_missing_env_module():
    dataset_id = "dummy-test-v0"

    env = gym.make("CartPole-v1")
    env = DataCollector(env)
    num_episodes = 10

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

    path = os.path.join(dataset.storage.data_path, METADATA_FILE_NAME)
    with open(path) as file:
        metadata = json.load(file)
    metadata[
        "env_spec"
    ] = r"""{
        "id": "DummyEnv-v0",
        "entry_point": "dummymodule:dummyenv",
        "reward_threshold": null,
        "nondeterministic": false,
        "max_episode_steps": 300,
        "order_enforce": true,
        "disable_env_checker": false,
        "apply_api_compatibility": false,
        "additional_wrappers": []
    }"""
    with open(path, "w") as file:
        json.dump(metadata, file)

    dataset = minari.load_dataset(dataset_id)
    with pytest.raises(ModuleNotFoundError, match="No module named 'dummymodule'"):
        dataset.recover_environment()

    minari.delete_dataset(dataset_id)

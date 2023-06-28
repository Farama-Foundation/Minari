import copy
import re
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

import minari
from minari import DataCollectorV0, MinariDataset
from minari.dataset.minari_dataset import EpisodeData
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
    total_timestep = 100
    rewards = np.random.randn(total_timestep)
    terminations = np.random.choice([True, False], size=(total_timestep,))
    truncations = np.random.choice([True, False], size=(total_timestep,))
    episode_data = EpisodeData(
        id=id,
        seed=seed,
        total_timesteps=total_timestep,
        observations=space.sample(),
        actions=space.sample(),
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
    )

    pattern = r"EpisodeData\("
    pattern += r"id=\d+, "
    pattern += r"seed=\d+, "
    pattern += r"total_timesteps=100, "
    pattern += r"observations=.+, "
    pattern += r"actions=.+, "
    pattern += r"rewards=.+, "
    pattern += r"terminations=.+, "
    pattern += r"truncations=.+"
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
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDisceteBoxEnv-v0"),
    ],
)
def test_update_dataset_from_collector_env(dataset_id, env_id):

    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    env = DataCollectorV0(env)
    num_episodes = 10

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

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

    dataset.update_dataset_from_collector_env(env)

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes * 2
    assert dataset.spec.total_episodes == num_episodes * 2
    assert len(dataset.episode_indices) == num_episodes * 2

    check_data_integrity(dataset._data, dataset.episode_indices)
    check_env_recovery(env.env, dataset)

    env.close()

    check_load_and_delete_dataset(dataset_id)


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
def test_filter_episodes(dataset_id, env_id):
    """Tests to make sure that episodes are filtered and sampled correctly. These tests are combined because filter_episodes creates a filtered dataset
    and it is necessary to create a filtered dataset to see if"""
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    env = DataCollectorV0(env)
    num_episodes = 10

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

    def filter_by_index(episode: Any):
        return int(episode.attrs.get("id")) >= 3

    filtered_dataset = dataset.filter_episodes(filter_by_index)

    assert isinstance(filtered_dataset, MinariDataset)
    assert filtered_dataset.total_episodes == 7
    assert filtered_dataset.spec.total_episodes == 10
    assert len(filtered_dataset.episode_indices) == 7

    check_data_integrity(
        filtered_dataset._data, dataset.episode_indices
    )  # checks that the underlying episodes are still present in the `MinariStorage` object
    check_env_recovery(env.env, filtered_dataset)

    env.close()

    check_load_and_delete_dataset(dataset_id)


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
def test_sample_episodes(dataset_id, env_id):
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    env = DataCollectorV0(env)
    num_episodes = 10

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

    def filter_by_index(episode: Any):
        return int(episode.attrs.get("id")) >= 3

    filtered_dataset = dataset.filter_episodes(filter_by_index)
    for i in [1, 7]:
        episodes = filtered_dataset.sample_episodes(i)
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
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDisceteBoxEnv-v0"),
    ],
)
def test_iterate_episodes(dataset_id, env_id):
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    env = DataCollectorV0(env)
    num_episodes = 10

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

    episodes = list(dataset.iterate_episodes([1, 3, 5]))

    assert {1, 3, 5} == {episode.id for episode in episodes}

    assert len(episodes) == 3
    check_episode_data_integrity(
        episodes, dataset.spec.observation_space, dataset.spec.action_space
    )

    all_episodes = list(dataset.iterate_episodes())
    assert len(all_episodes) == 10
    assert {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} == {episode.id for episode in all_episodes}

    env.close()


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
def test_update_dataset_from_buffer(dataset_id, env_id):

    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    collector_env = DataCollectorV0(env)
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

    check_data_integrity(dataset._data, dataset.episode_indices)
    check_env_recovery(env, dataset)

    env.close()

    check_load_and_delete_dataset(dataset_id)
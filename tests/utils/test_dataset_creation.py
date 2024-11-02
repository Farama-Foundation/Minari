import dataclasses

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import minari
from minari import DataCollector, MinariDataset, StepData
from minari.data_collector import EpisodeBuffer
from minari.dataset._storages import get_storage_keys
from tests.common import (
    cartpole_test_dataset,
    check_data_integrity,
    check_env_recovery,
    check_env_recovery_with_subset_spaces,
    check_episode_data_integrity,
    check_load_and_delete_dataset,
    dummy_box_dataset,
    dummy_test_datasets,
    dummy_text_dataset,
    get_sample_buffer_for_dataset_from_env,
)


CODELINK = "https://github.com/Farama-Foundation/Minari/blob/main/tests/utils/test_dataset_creation.py"


@pytest.mark.parametrize(
    "dataset_id,env_id",
    cartpole_test_dataset + dummy_test_datasets + dummy_text_dataset,
)
def test_generate_dataset_with_collector_env(dataset_id, env_id, register_dummy_envs):
    """Test DataCollector wrapper and Minari dataset creation."""
    env = gym.make(env_id)
    env = DataCollector(env, record_infos=True)
    num_episodes = 10
    env.reset(seed=42)
    for episode in range(num_episodes):
        done = False
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.reset()

    eval_env_spec = gym.spec(env_id)
    eval_env_spec.max_episode_steps = 123
    eval_env = gym.make(eval_env_spec)

    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    dataset = env.create_dataset(
        dataset_id=dataset_id,
        eval_env=eval_env,
        algorithm_name="random_policy",
        code_permalink=CODELINK,
        author="Farama",
        author_email="farama@farama.org",
        description="Test dataset",
    )

    metadata = dataset.storage.metadata
    assert metadata["algorithm_name"] == "random_policy"
    assert metadata["code_permalink"] == CODELINK
    assert metadata["author"] == {"Farama"}
    assert metadata["author_email"] == {"farama@farama.org"}

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    check_data_integrity(dataset, list(dataset.episode_indices))
    check_episode_data_integrity(
        dataset, dataset.spec.observation_space, dataset.spec.action_space
    )

    check_env_recovery(env.env, dataset, eval_env)

    env.close()
    eval_env.close()
    check_load_and_delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "info_override",
    [
        None,
        {},
        {"foo": np.ones((10, 10), dtype=np.float32)},
        {"int": 1},
        {"bool": False},
        {
            "value1": True,
            "value2": 5,
            "value3": {"nested1": False, "nested2": np.empty(10)},
        },
    ],
)
def test_record_infos_collector_env(info_override, register_dummy_envs):
    """Test DataCollector wrapper and Minari dataset creation including infos."""
    dataset_id = "dummy-mutable-info-box/test-v0"
    env = gym.make("DummyInfoEnv-v0", info=info_override)

    env = DataCollector(env, record_infos=True)
    num_episodes = 10

    _, info_sample = env.reset(seed=42)

    for episode in range(num_episodes):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)

        env.reset()

    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="random_policy",
        code_permalink=CODELINK,
        author={"Farama", "Contributors"},
        author_email={"farama@farama.org"},
        description="Test dataset",
    )

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    check_data_integrity(dataset, list(dataset.episode_indices))
    check_episode_data_integrity(
        dataset,
        dataset.spec.observation_space,
        dataset.spec.action_space,
        info_sample=info_sample,
    )

    env.close()

    check_load_and_delete_dataset(dataset_id)


@pytest.mark.parametrize("data_format", get_storage_keys())
@pytest.mark.parametrize(
    "dataset_id,env_id",
    cartpole_test_dataset
    + [x for x in dummy_test_datasets if x not in dummy_box_dataset]
    + dummy_text_dataset,
)
def test_generate_dataset_with_external_buffer(
    dataset_id, env_id, data_format, register_dummy_envs
):
    """Test create dataset from external buffers without using DataCollector."""
    buffer = []

    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    num_episodes = 10
    seed = 42
    observation, _ = env.reset(seed=seed)
    episode_buffer = EpisodeBuffer(observations=observation, seed=seed)

    for episode in range(num_episodes):
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)
            step_data: StepData = {
                "observation": observation,
                "action": action,
                "reward": reward,
                "termination": terminated,
                "truncation": truncated,
                "info": {},
            }
            episode_buffer = episode_buffer.add_step_data(step_data)

        buffer.append(episode_buffer)
        observation, _ = env.reset()
        episode_buffer = EpisodeBuffer(observations=observation)

    gym.registry[f"eval/{env_id}"] = dataclasses.replace(
        gym.spec(env_id), max_episode_steps=123, id=f"eval/{env_id}"
    )

    eval_env = gym.make(f"eval/{env_id}")
    for env_dataset_id, eval_env_dataset_id in zip(
        [env, env.spec, env_id], [eval_env, eval_env.spec, f"eval/{env_id}"]
    ):
        # Create Minari dataset and store locally
        dataset = minari.create_dataset_from_buffers(
            dataset_id=dataset_id,
            buffer=buffer,
            env=env_dataset_id,
            eval_env=eval_env_dataset_id,
            algorithm_name="random_policy",
            code_permalink=CODELINK,
            author="Farama",
            author_email="farama@farama.org",
            description="Test dataset",
            data_format=data_format,
        )

        assert isinstance(dataset, MinariDataset)
        assert dataset.total_episodes == num_episodes
        assert dataset.spec.total_episodes == num_episodes
        assert len(dataset.episode_indices) == num_episodes

        check_data_integrity(dataset, list(dataset.episode_indices))
        check_episode_data_integrity(
            dataset, dataset.spec.observation_space, dataset.spec.action_space
        )
        check_env_recovery(env, dataset, eval_env)

        check_load_and_delete_dataset(dataset_id)

    env.close()
    eval_env.close()


@pytest.mark.parametrize("is_env_needed", [True, False])
@pytest.mark.parametrize("data_format", get_storage_keys())
def test_generate_dataset_with_space_subset_external_buffer(
    is_env_needed, data_format, register_dummy_envs
):
    """Test create dataset from external buffers without using DataCollector or environment."""
    dataset_id = "dummy-dict/test-v0"

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

    env = gym.make("DummyDictEnv-v0")
    num_episodes = 10
    buffer = get_sample_buffer_for_dataset_from_env(env, num_episodes)
    sub_buffer = []
    for episode_buffer in buffer:
        observations = {
            "component_2": {
                "subcomponent_2": episode_buffer.observations["component_2"][
                    "subcomponent_2"
                ]
            }
        }
        actions = {
            "component_2": {
                "subcomponent_2": episode_buffer.actions["component_2"][
                    "subcomponent_2"
                ]
            }
        }
        sub_buffer.append(
            EpisodeBuffer(
                observations=observations,
                actions=actions,
                rewards=episode_buffer.rewards,
                terminations=episode_buffer.terminations,
                truncations=episode_buffer.truncations,
                infos=episode_buffer.infos,
            )
        )

    # Create Minari dataset and store locally
    env_to_pass = env if is_env_needed else None
    dataset = minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        buffer=sub_buffer,
        env=env_to_pass,
        algorithm_name="random_policy",
        code_permalink=CODELINK,
        author={"Farama", "Contributors"},
        author_email={"farama@farama.org", "contributors@farama.org"},
        description="Test dataset",
        action_space=action_space_subset,
        observation_space=observation_space_subset,
        data_format=data_format,
    )

    metadata = dataset.storage.metadata
    assert metadata["algorithm_name"] == "random_policy"
    assert metadata["code_permalink"] == CODELINK
    assert metadata["author"] == {"Farama", "Contributors"}
    assert metadata["author_email"] == {"farama@farama.org", "contributors@farama.org"}

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    check_data_integrity(dataset, list(dataset.episode_indices))
    check_episode_data_integrity(
        dataset, dataset.spec.observation_space, dataset.spec.action_space
    )
    if is_env_needed:
        check_env_recovery_with_subset_spaces(
            env, dataset, action_space_subset, observation_space_subset
        )

    env.close()

    check_load_and_delete_dataset(dataset_id)


@pytest.mark.parametrize("data_format", get_storage_keys())
def test_generate_big_episode(data_format, register_dummy_envs):
    """Test generate a long episode and create a dataset."""
    dataset_id = "test/big-episode-v0"
    episode_length = 1024
    observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    action_space = spaces.Discrete(10)
    info_generator = spaces.Dict(
        {
            "info1": spaces.Discrete(10),
            "info2": spaces.Box(low=0.0, high=1.0, shape=(512,), dtype=np.uint8),
        }
    )

    buffer = EpisodeBuffer(
        observations=[observation_space.sample()], infos=info_generator.sample()
    )
    for step_id in range(episode_length):
        buffer = buffer.add_step_data(
            {
                "observation": observation_space.sample(),
                "action": action_space.sample(),
                "reward": 0.0,
                "termination": False,
                "truncation": step_id == episode_length - 1,
                "info": info_generator.sample(),
            }
        )

    dataset = minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        buffer=[buffer],
        observation_space=observation_space,
        action_space=action_space,
        algorithm_name="random_policy",
        code_permalink=CODELINK,
        author="Farama",
        author_email="farama@farama.org",
        description="Test dataset",
        data_format=data_format,
    )

    assert np.all(dataset[0].observations == buffer.observations)
    assert np.all(dataset[0].actions == buffer.actions)
    assert np.all(dataset[0].rewards == buffer.rewards)
    assert np.all(dataset[0].terminations == buffer.terminations)
    assert np.all(dataset[0].truncations == buffer.truncations)
    buffer_infos = buffer.infos
    assert buffer_infos is not None
    assert np.all(dataset[0].infos["info1"] == buffer_infos["info1"])
    assert np.all(dataset[0].infos["info2"] == buffer_infos["info2"])

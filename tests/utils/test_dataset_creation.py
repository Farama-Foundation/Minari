import dataclasses

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import minari
from minari import DataCollector, MinariDataset, StepData
from minari.data_collector import EpisodeBuffer
from minari.dataset._storages import registry as storage_registry
from tests.common import (
    check_data_integrity,
    check_env_recovery,
    check_env_recovery_with_subset_spaces,
    check_episode_data_integrity,
    check_load_and_delete_dataset,
    get_sample_buffer_for_dataset_from_env,
)


CODELINK = "https://github.com/Farama-Foundation/Minari/blob/main/tests/utils/test_dataset_creation.py"


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
def test_generate_dataset_with_collector_env(dataset_id, env_id, register_dummy_envs):
    """Test DataCollector wrapper and Minari dataset creation."""
    env = gym.make(env_id)

    env = DataCollector(env, record_infos=True)
    num_episodes = 10

    # Step the environment, DataCollector wrapper will do the data collection job
    env.reset(seed=42)

    for episode in range(num_episodes):
        done = False
        while not done:
            action = env.action_space.sample()  # User-defined policy function
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.reset()

    # Save a different environment spec for evaluation (different max_episode_steps)
    eval_env_spec = gym.spec(env_id)
    eval_env_spec.max_episode_steps = 123
    eval_env = gym.make(eval_env_spec)
    # dataset_id = "cartpole-test-v0"
    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)
    # Create Minari dataset and store locally
    dataset = env.create_dataset(
        dataset_id=dataset_id,
        eval_env=eval_env,
        algorithm_name="random_policy",
        code_permalink=CODELINK,
        author="WillDudley",
        author_email="wdudley@farama.org",
    )

    metadata = dataset.storage.metadata
    assert metadata["algorithm_name"] == "random_policy"
    assert metadata["code_permalink"] == CODELINK
    assert metadata["author"] == "WillDudley"
    assert metadata["author_email"] == "wdudley@farama.org"

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    check_data_integrity(dataset.storage, dataset.episode_indices)
    check_episode_data_integrity(
        dataset, dataset.spec.observation_space, dataset.spec.action_space
    )

    # check that the environment can be recovered from the dataset
    check_env_recovery(env.env, dataset, eval_env)

    env.close()
    eval_env.close()
    # check load and delete local dataset
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
    dataset_id = "dummy-mutable-info-box-test-v0"
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
        author="WillDudley",
        author_email="wdudley@farama.org",
    )

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    check_data_integrity(dataset.storage, dataset.episode_indices)
    check_episode_data_integrity(
        dataset,
        dataset.spec.observation_space,
        dataset.spec.action_space,
        info_sample=info_sample,
    )

    env.close()

    check_load_and_delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("cartpole-test-v0", "CartPole-v1"),
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-text-test-v0", "DummyTextEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDiscreteBoxEnv-v0"),
    ],
)
@pytest.mark.parametrize("data_format", storage_registry.keys())
def test_generate_dataset_with_external_buffer(
    dataset_id, env_id, data_format, register_dummy_envs
):
    """Test create dataset from external buffers without using DataCollector."""
    buffer = []

    # dataset_id = "cartpole-test-v0"
    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)
    print(f"{env.spec=}")

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

    # Save a different environment spec for evaluation (different max_episode_steps)
    gym.registry[f"eval/{env_id}"] = dataclasses.replace(
        gym.spec(env_id), max_episode_steps=123, id=f"eval/{env_id}"
    )

    eval_env = gym.make(f"eval/{env_id}")
    # Test for different types of env and eval_env (gym.Env, EnvSpec, and str id)
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
            author="WillDudley",
            author_email="wdudley@farama.org",
            data_format=data_format,
        )

        assert isinstance(dataset, MinariDataset)
        assert dataset.total_episodes == num_episodes
        assert dataset.spec.total_episodes == num_episodes
        assert len(dataset.episode_indices) == num_episodes

        check_data_integrity(dataset.storage, dataset.episode_indices)
        check_episode_data_integrity(
            dataset, dataset.spec.observation_space, dataset.spec.action_space
        )
        check_env_recovery(env, dataset, eval_env)

        check_load_and_delete_dataset(dataset_id)

    env.close()
    eval_env.close()


@pytest.mark.parametrize("is_env_needed", [True, False])
@pytest.mark.parametrize("data_format", storage_registry.keys())
def test_generate_dataset_with_space_subset_external_buffer(
    is_env_needed, data_format, register_dummy_envs
):
    """Test create dataset from external buffers without using DataCollector or environment."""
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
        author="WillDudley",
        author_email="wdudley@farama.org",
        action_space=action_space_subset,
        observation_space=observation_space_subset,
        data_format=data_format,
    )

    metadata = dataset.storage.metadata
    assert metadata["algorithm_name"] == "random_policy"
    assert metadata["code_permalink"] == CODELINK
    assert metadata["author"] == "WillDudley"
    assert metadata["author_email"] == "wdudley@farama.org"

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    check_data_integrity(dataset.storage, dataset.episode_indices)
    check_episode_data_integrity(
        dataset, dataset.spec.observation_space, dataset.spec.action_space
    )
    if is_env_needed:
        check_env_recovery_with_subset_spaces(
            env, dataset, action_space_subset, observation_space_subset
        )

    env.close()

    check_load_and_delete_dataset(dataset_id)

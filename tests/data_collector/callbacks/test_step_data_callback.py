import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from minari import DataCollector, MinariDataset
from minari.data_collector.callbacks import StepDataCallback
from tests.common import (
    check_data_integrity,
    check_env_recovery,
    check_env_recovery_with_subset_spaces,
    check_load_and_delete_dataset,
    register_dummy_envs,
)


register_dummy_envs()


class CustomSubsetStepDataCallback(StepDataCallback):
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        step_data["observations"] = {
            "component_2": {
                "subcomponent_2": step_data["observations"]["component_2"][
                    "subcomponent_2"
                ]
            }
        }
        if step_data["actions"] is not None:
            step_data["actions"] = {
                "component_2": {
                    "subcomponent_2": step_data["actions"]["component_2"][
                        "subcomponent_2"
                    ]
                }
            }
        return step_data


class CustomSubsetInfoPadStepDataCallback(StepDataCallback):
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        if step_data["infos"] == {}:
            step_data["infos"] = {"timestep": np.array([-1])}
        return step_data


def test_data_collector_step_data_callback():
    """Test DataCollector wrapper and Minari dataset creation."""
    dataset_id = "dummy-dict-test-v0"

    env = gym.make("DummyDictEnv-v0")

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

    env = DataCollector(
        env,
        observation_space=observation_space_subset,
        action_space=action_space_subset,
        step_data_callback=CustomSubsetStepDataCallback,
    )
    num_episodes = 10

    env.reset(seed=42)
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
        code_permalink=str(__file__),
        author="WillDudley",
        author_email="wdudley@farama.org",
    )

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    check_data_integrity(dataset.storage, dataset.episode_indices)

    check_env_recovery_with_subset_spaces(
        env.env, dataset, action_space_subset, observation_space_subset
    )

    env.close()
    check_load_and_delete_dataset(dataset_id)


def test_data_collector_step_data_callback_info_correction():
    """Test DataCollector wrapper and Minari dataset creation."""
    dataset_id = "dummy-inconsistent-info-v0"
    env = gym.make("DummyInconsistentInfoEnv-v0")

    env = DataCollector(
        env,
        record_infos=True,
        step_data_callback=CustomSubsetInfoPadStepDataCallback,
    )
    num_episodes = 10

    env.reset(seed=42)
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
        code_permalink=str(__file__),
        author="WillDudley",
        author_email="wdudley@farama.org",
    )

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    check_data_integrity(dataset.storage, dataset.episode_indices)

    check_env_recovery(env.env, dataset)

    env.close()
    check_load_and_delete_dataset(dataset_id)

    env = gym.make("DummyInconsistentInfoEnv-v0")

    env = DataCollector(
        env,
        record_infos=True,
    )
    # here we are checking to make sure that if we have an environment changing its info
    # structure across steps, it is caught by the data_collector
    with pytest.raises(
        ValueError,
        match=r"Info structure inconsistent with info structure returned by original reset."
    ):

        num_episodes = 10
        env.reset(seed=42)
        for _ in range(num_episodes):
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)

            env.reset()
    env.close()

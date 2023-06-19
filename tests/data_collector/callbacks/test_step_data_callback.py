import gymnasium as gym
import numpy as np
from gymnasium import spaces

import minari
from minari import DataCollectorV0, MinariDataset
from minari.data_collector.callbacks import StepDataCallback
from tests.common import (
    check_data_integrity,
    check_env_recovery_with_subset_spaces,
    check_load_and_delete_dataset,
    create_dummy_dataset_with_collecter_env_helper,
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


def test_data_collector_step_data_callback():
    """Test DataCollectorV0 wrapper and Minari dataset creation."""
    dataset_id = "dummy-dict-test-v0"
    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

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

    env = DataCollectorV0(
        env,
        observation_space=observation_space_subset,
        action_space=action_space_subset,
        step_data_callback=CustomSubsetStepDataCallback,
    )
    num_episodes = 10

    # Step the environment, DataCollectorV0 wrapper will do the data collection job
    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    check_data_integrity(dataset._data, dataset.episode_indices)

    # check that the environment can be recovered from the dataset
    check_env_recovery_with_subset_spaces(
        env.env, dataset, action_space_subset, observation_space_subset
    )

    env.close()
    # check load and delete local dataset
    check_load_and_delete_dataset(dataset_id)

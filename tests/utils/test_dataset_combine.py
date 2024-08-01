import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence

import minari
from minari import DataCollector, MinariDataset
from minari.utils import combine_datasets
from tests.common import create_dummy_dataset_with_collecter_env_helper


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
        dataset.spec.observation_space, gymnasium_environment.observation_space
    )
    assert data_equivalence(
        recovered_env.action_space, gymnasium_environment.action_space
    )
    assert data_equivalence(
        dataset.spec.action_space, gymnasium_environment.action_space
    )


def _check_load_and_delete_dataset(dataset_id: str):
    """Test loading and deletion of local Minari datasets.

    Args:
        dataset_id (str): name of Minari dataset to test
    """
    # check dataset name is present in local database
    local_datasets = minari.list_local_datasets()
    assert dataset_id in local_datasets

    # load dataset
    loaded_dataset = minari.load_dataset(dataset_id)
    assert isinstance(loaded_dataset, MinariDataset)
    assert dataset_id == loaded_dataset.spec.dataset_id

    # delete dataset and check that it's no longer present in local database
    minari.delete_dataset(dataset_id)
    local_datasets = minari.list_local_datasets()
    assert dataset_id not in local_datasets


def test_combine_datasets():
    num_datasets, num_episodes = 5, 10
    test_datasets_ids = [f"cartpole/test-{i}-v0" for i in range(num_datasets)]

    # generating multiple test datasets
    test_max_episode_steps = [5, 3, 7, 10, None]
    data_formats = ["hdf5", "arrow", None, "arrow", None]

    test_datasets = []
    for dataset_id, max_episode_steps, data_format in zip(
        test_datasets_ids, test_max_episode_steps, data_formats
    ):
        env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
        assert env.spec is not None
        env.spec.max_episode_steps = (
            max_episode_steps  # with None max_episode_steps=default
        )
        env = DataCollector(env, data_format=data_format)
        dataset = create_dummy_dataset_with_collecter_env_helper(
            dataset_id, env, num_episodes
        )
        test_datasets.append(dataset)

    combined_dataset = combine_datasets(
        test_datasets, new_dataset_id="cartpole/combined-test-v0"
    )

    assert test_datasets[1][0].id == 0
    assert isinstance(combined_dataset, MinariDataset)
    assert list(combined_dataset.spec.combined_datasets) == test_datasets_ids, list(
        combined_dataset.spec.combined_datasets
    )
    assert combined_dataset.spec.total_episodes == num_datasets * num_episodes
    assert isinstance(combined_dataset.spec.total_steps, int)
    assert combined_dataset.spec.total_steps == sum(
        d.spec.total_steps for d in test_datasets
    )
    assert combined_dataset.spec.env_spec is not None
    assert combined_dataset.spec.env_spec.max_episode_steps is None

    _check_load_and_delete_dataset("cartpole/combined-test-v0")

    # Check that we get max(max_episode_steps) when there is no max_episode_steps=None
    test_datasets.pop()
    test_max_episode_steps.pop()
    combined_dataset = combine_datasets(
        test_datasets, new_dataset_id="cartpole/combined-test-v0"
    )
    assert combined_dataset.spec.env_spec is not None
    assert combined_dataset.spec.env_spec.max_episode_steps == max(
        test_max_episode_steps
    )
    _check_env_recovery(
        gym.make("CartPole-v1", max_episode_steps=max(test_max_episode_steps)),
        combined_dataset,
    )
    _check_load_and_delete_dataset("cartpole/combined-test-v0")

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence

import minari
from minari import DataCollectorV0, MinariDataset
from minari.utils import combine_datasets


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


def _generate_dataset_with_collector_env(dataset_id: str, num_episodes: int = 10):
    """Helper function to create tmp dataset to combining.

    Args:
        dataset_id (str): name of the generated Minari dataset
        num_episodes (int): number of episodes in the generated dataset
    """
    env = gym.make("CartPole-v1")

    env = DataCollectorV0(env)
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
        dataset_id=dataset_id,
        collector_env=env,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
    )
    assert isinstance(dataset, MinariDataset)
    env.close()


def test_combine_datasets():
    num_datasets, num_episodes = 5, 10
    test_datasets_ids = [f"cartpole-test-{i}-v0" for i in range(num_datasets)]

    local_datasets = minari.list_local_datasets()
    # generating multiple test datasets
    for dataset_id in test_datasets_ids:
        if dataset_id in local_datasets:
            minari.delete_dataset(dataset_id)
        _generate_dataset_with_collector_env(dataset_id, num_episodes)

    test_datasets = [
        minari.load_dataset(dataset_id) for dataset_id in test_datasets_ids
    ]
    if "cartpole-combined-test-v0" in local_datasets:
        minari.delete_dataset("cartpole-combined-test-v0")

    # testing without creating a copy
    combined_dataset = combine_datasets(
        test_datasets, new_dataset_id="cartpole-combined-test-v0"
    )
    assert isinstance(combined_dataset, MinariDataset)
    assert list(combined_dataset.spec.combined_datasets) == test_datasets_ids
    assert combined_dataset.spec.total_episodes == num_datasets * num_episodes
    assert combined_dataset.spec.total_steps == sum(
        d.spec.total_steps for d in test_datasets
    )
    _check_env_recovery(gym.make("CartPole-v1"), combined_dataset)
    _check_load_and_delete_dataset("cartpole-combined-test-v0")

    # testing with copy
    combined_dataset = combine_datasets(
        test_datasets, new_dataset_id="cartpole-combined-test-v0", copy=True
    )
    assert isinstance(combined_dataset, MinariDataset)
    assert list(combined_dataset.spec.combined_datasets) == test_datasets_ids
    assert combined_dataset.spec.total_episodes == num_datasets * num_episodes
    assert combined_dataset.spec.total_steps == sum(
        d.spec.total_steps for d in test_datasets
    )
    _check_env_recovery(gym.make("CartPole-v1"), combined_dataset)

    # deleting test datasets
    for dataset_id in test_datasets_ids:
        minari.delete_dataset(dataset_id)

    # checking that we still can load combined dataset after deleting source datasets
    _check_load_and_delete_dataset("cartpole-combined-test-v0")

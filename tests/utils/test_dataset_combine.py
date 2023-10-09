from typing import Optional

import gymnasium as gym
import h5py
import pytest
from gymnasium.utils.env_checker import data_equivalence
from packaging.specifiers import SpecifierSet

import minari
from minari import DataCollectorV0, MinariDataset
from minari.utils import combine_datasets, combine_minari_version_specifiers


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


def _generate_dataset_with_collector_env(
    dataset_id: str, num_episodes: int = 10, max_episode_steps: Optional[int] = 500
):
    """Helper function to create tmp dataset to combining.

    Args:
        dataset_id (str): name of the generated Minari dataset
        num_episodes (int): number of episodes in the generated dataset
        max_episode_steps (int | None): max episodes per step of the environment
    """
    if max_episode_steps is None:
        # Force None max_episode_steps
        env_spec = gym.make("CartPole-v1").spec
        env_spec.max_episode_steps = None
        env = env_spec.make()
    else:
        env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)

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

    # testing re-calculation of env_spec.max_episode_steps: max(max_episode_steps) or None propagates.
    dataset_max_episode_steps = [5, 10, None]
    test_datasets_ids = [
        f"cartpole-test-{i}-v0" for i in range(len(dataset_max_episode_steps))
    ]

    local_datasets = minari.list_local_datasets()
    # generating multiple test datasets
    for dataset_id, max_episode_steps in zip(
        test_datasets_ids, dataset_max_episode_steps
    ):
        if dataset_id in local_datasets:
            minari.delete_dataset(dataset_id)
        _generate_dataset_with_collector_env(
            dataset_id, num_episodes, max_episode_steps
        )

    test_datasets = [
        minari.load_dataset(dataset_id) for dataset_id in test_datasets_ids
    ]

    # testing without creating a copy
    combined_dataset = combine_datasets(
        test_datasets, new_dataset_id="cartpole-combined-test-v0"
    )
    assert combined_dataset.spec.env_spec.max_episode_steps is None
    _check_load_and_delete_dataset("cartpole-combined-test-v0")

    # testing with copy
    combined_dataset = combine_datasets(
        test_datasets, new_dataset_id="cartpole-combined-test-v0", copy=True
    )
    assert combined_dataset.spec.env_spec.max_episode_steps is None
    _check_load_and_delete_dataset("cartpole-combined-test-v0")

    # Check that we get max(max_episode_steps) when there is no max_episode_steps=None
    test_datasets.pop()
    # testing without creating a copy
    combined_dataset = combine_datasets(
        test_datasets, new_dataset_id="cartpole-combined-test-v0"
    )
    assert combined_dataset.spec.env_spec.max_episode_steps == 10
    _check_load_and_delete_dataset("cartpole-combined-test-v0")

    # testing with copy
    combined_dataset = combine_datasets(
        test_datasets, new_dataset_id="cartpole-combined-test-v0", copy=True
    )
    assert combined_dataset.spec.env_spec.max_episode_steps == 10
    _check_load_and_delete_dataset("cartpole-combined-test-v0")

    # deleting test datasets
    for dataset_id in test_datasets_ids:
        minari.delete_dataset(dataset_id)


@pytest.mark.parametrize(
    "specifier_intersection,version_specifiers",
    [
        (
            SpecifierSet(">3.0.0, <=3.9.1"),
            SpecifierSet(">3.0.0") & SpecifierSet("<=3.9.1"),
        ),
        (
            SpecifierSet(">3.2, <=3.2.5"),
            SpecifierSet(">=3.0.0, <3.3.0") & SpecifierSet(">3.2, <=3.2.5"),
        ),
        (SpecifierSet(">=1.3.0, !=1.4.0"), SpecifierSet(">=1.3.0, !=1.4.0")),
        (SpecifierSet(">=1.3.0"), SpecifierSet(">=1.3.0, !=1.2.0")),
        (
            SpecifierSet(">=3.0.0, <=3.9.1"),
            SpecifierSet("~=3.0") & SpecifierSet("<=3.9.1"),
        ),
    ],
)
def test_combine_minari_version_specifiers(specifier_intersection, version_specifiers):
    intersection = combine_minari_version_specifiers(version_specifiers)

    assert specifier_intersection == intersection


# in the future, if the logic of save metadata of combined dataset changes, this should be changed as well
def test_combine_dataset_with_different_metadata():
    n_data = 2
    dataset_list = []
    for i in range(n_data):
        dataset_id = f"cartpole-test-{i}-v0"
        env = gym.make("CartPole-v1", max_episode_steps=500)
        env = DataCollectorV0(env)
        env.reset(seed=42)
        for episode in range(5):
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = env.action_space.sample()  # User-defined policy function
                _, _, terminated, truncated, _ = env.step(action)
            env.reset()

        # Create Minari dataset and store locally
        permalink = "https://github.com/Farama-Foundation/Minari/blob/main/tests/utils/test_dataset_combine.py"
        dataset = minari.create_dataset_from_collector_env(
            dataset_id=dataset_id,
            collector_env=env,
            algorithm_name="random_policy" + str(i),
            code_permalink=permalink + str(i),
            author="WillDudley" + str(i),
            author_email="wdudley@farama.org" + str(i),
        )
        assert isinstance(dataset, MinariDataset)
        env.close()
        dataset_list.append(dataset)

    combined_dataset = combine_datasets(
        dataset_list, new_dataset_id="cartpole-combined-test-v0"
    )
    permalink = "https://github.com/Farama-Foundation/Minari/blob/main/tests/utils/test_dataset_combine.py"
    with h5py.File(combined_dataset.spec.data_path) as dt_file:
        assert dt_file.attrs["algorithm_name"] == "random_policy" + str(n_data - 1)
        _final_code_link = permalink + str(n_data - 1)
        assert dt_file.attrs["code_permalink"] == _final_code_link
        assert dt_file.attrs["author"] == "WillDudley" + str(n_data - 1)
        assert dt_file.attrs["author_email"] == "wdudley@farama.org" + str(n_data - 1)

    for i in range(n_data):
        minari.delete_dataset(f"cartpole-test-{i}-v0")
    minari.delete_dataset("cartpole-combined-test-v0")
    return


import os
import warnings
from typing import Dict, List, Optional, Union

import gymnasium as gym
import h5py
import numpy as np
from minari.minari_dataset import MinariDataset
from minari.data_collector import DataCollectorV0

from minari.storage.datasets_root_dir import get_dataset_path


def clear_episode_buffer(episode_buffer: Dict, eps_group: h5py.Group) -> h5py.Group:
    """Save an episode dictionary buffer into an HDF5 episode group recursively.

    Args:
        episode_buffer (dict): episode buffer
        eps_group (h5py.Group): HDF5 group to store the episode datasets

    Returns:
        episode group: filled HDF5 episode group
    """
    for key, data in episode_buffer.items():
        if isinstance(data, dict):
            if key in eps_group:
                eps_group_to_clear = eps_group[key]
            else:
                eps_group_to_clear = eps_group.create_group(key)
            clear_episode_buffer(data, eps_group_to_clear)
        else:
            # assert data is numpy array
            assert np.all(np.logical_not(np.isnan(data)))
            # add seed to attributes
            eps_group.create_dataset(key, data=data, chunks=True)

    return eps_group


def combine_datasets(datasets_to_combine: List[MinariDataset], new_dataset_name: str):
    """Combine a group of MinariDataset in to a single dataset with its own name id.

    A new HDF5 metadata attribute will be added to the new dataset called `combined_datasets`. This will
    contain a list of strings with the dataset names that were combined to form this new Minari dataset.

    Args:
        datasets_to_combine (list[MinariDataset]): list of datasets to be combined
        new_dataset_name (str): name id for the newly created dataset
    """
    new_dataset_path = get_dataset_path(new_dataset_name)

    # Check if dataset already exists
    if not os.path.exists(new_dataset_path):
        new_dataset_path = os.path.join(new_dataset_path, "data")
        os.makedirs(new_dataset_path)
        new_data_path = os.path.join(new_dataset_path, "main_data.hdf5")
    else:
        raise ValueError(
            f"A Minari dataset with ID {new_dataset_name} already exists and it cannot be overridden. Please use a different dataset name or version."
        )

    with h5py.File(new_data_path, "a", track_order=True) as combined_data_file:
        combined_data_file.attrs["total_episodes"] = 0
        combined_data_file.attrs["total_steps"] = 0
        combined_data_file.attrs["dataset_name"] = new_dataset_name

        combined_data_file.attrs["combined_datasets"] = [
            dataset.name for dataset in datasets_to_combine
        ]

        for dataset in datasets_to_combine:
            if not isinstance(dataset, MinariDataset):
                raise ValueError(
                    f"The dataset {dataset} is not of type MinariDataset.")

            with h5py.File(dataset.data_path, "r", track_order=True) as data_file:
                group_paths = [group.name for group in data_file.values()]

                if combined_data_file.attrs.get("env_spec") is None:
                    combined_data_file.attrs["env_spec"] = data_file.attrs["env_spec"]
                else:
                    if (
                        combined_data_file.attrs["env_spec"]
                        != data_file.attrs["env_spec"]
                    ):
                        raise ValueError(
                            "The datasets to be combined have different values for `env_spec` attribute."
                        )

            if combined_data_file.attrs.get("flatten_action") is None:
                combined_data_file.attrs["flatten_action"] = dataset.flatten_actions
            else:
                if (
                    combined_data_file.attrs["flatten_action"]
                    != dataset.flatten_actions
                ):
                    raise ValueError(
                        "The datasets to be combined have different values for `flatten_action` attribute."
                    )

            if combined_data_file.attrs.get("flatten_observation") is None:
                combined_data_file.attrs[
                    "flatten_observation"
                ] = dataset.flatten_observations
            else:
                if (
                    combined_data_file.attrs["flatten_observation"]
                    != dataset.flatten_observations
                ):
                    raise ValueError(
                        "The datasets to be combined have different values for `flatten_observation` attribute."
                    )

            last_episode_id = combined_data_file.attrs["total_episodes"]

            for i, eps_group_path in enumerate(group_paths):
                combined_data_file[
                    f"episode_{last_episode_id + i}"
                ] = h5py.ExternalLink(dataset.data_path, eps_group_path)
                combined_data_file[f"episode_{last_episode_id + i}"].attrs.modify(
                    "id", last_episode_id + i
                )

            # Update metadata of minari dataset
            combined_data_file.attrs.modify(
                "total_episodes", last_episode_id + dataset.total_episodes
            )
            combined_data_file.attrs.modify(
                "total_steps",
                combined_data_file.attrs["total_steps"] + dataset.total_steps,
            )

    return MinariDataset(new_data_path)


def split_dataset(dataset: MinariDataset, sizes: List[int], seed: Optional[int] = None) -> List[MinariDataset]:
    """Split a MinariDataset in multiple datasets.

    Args:
        dataset (MinariDataset): the MinariDataset to split
        sizes (List[int]): sizes of the resulting datasets
        seed (Optiona[int]): random seed
    
    Returns:
        datasets (List[MinariDataset]): resulting list of datasets
    """
    if sum(sizes) > dataset.total_episodes:
        raise ValueError(
            "Incompatible arguments: the sum of sizes exceeds ",
            f"the number of episodes in the dataset ({dataset.total_episodes})"
        )
    generator = np.random.default_rng(seed=seed)
    indices = generator.permutation(dataset._episode_indices)
    out_datasets = []
    start_idx = 0
    for length in sizes:
        end_idx = start_idx + length
        slice_dataset = MinariDataset(
            dataset._data, indices[start_idx:end_idx])
        out_datasets.append(slice_dataset)
        start_idx = end_idx

    return out_datasets


def create_dataset_from_buffers(
    dataset_name: str,
    env: gym.Env,
    buffer: List[Dict[str, Union[list, Dict]]],
    algorithm_name: Optional[str] = None,
    author: Optional[str] = None,
    author_email: Optional[str] = None,
    code_permalink: Optional[str] = None,
):
    """Create Minari dataset from a list of episode dictionary buffers.

    Each episode dictionary buffer must have the following items:
        * `observations`: np.ndarray of step observations. shape = (total_episode_steps + 1, (observation_shape)). Should include initial and final observation
        * `actions`: np.ndarray of step action. shape = (total_episode_steps + 1, (action_shape)).
        * `rewards`: np.ndarray of step rewards. shape = (total_episode_steps + 1, 1).
        * `terminations`: np.ndarray of step terminations. shape = (total_episode_steps + 1, 1).
        * `truncations`: np.ndarray of step truncations. shape = (total_episode_steps + 1, 1).

    Other additional items can be added as long as the values are np.ndarray's or other nested dictionaries.

    Args:
        dataset_name (str): name id to identify Minari dataset
        env (gym.Env): Gymnasium environment used to collect the buffer data
        buffer (list[Dict[str, Union[list, Dict]]]): list of episode dictionaries with data
        algorithm_name (Optional[str], optional): name of the algorithm used to collect the data. Defaults to None.
        author (Optional[str], optional): author that generated the dataset. Defaults to None.
        author_email (Optional[str], optional): email of the author that generated the dataset. Defaults to None.
        code_permalink (Optional[str], optional): link to relevant code used to generate the dataset. Defaults to None.

    Returns:
        MinariDataset
    """
    # NoneType warnings
    if code_permalink is None:
        warnings.warn(
            "`code_permalink` is set to None. For reproducibility purposes it is highly recommended to link your dataset to versioned code.",
            UserWarning,
        )
    if author is None:
        warnings.warn(
            "`author` is set to None. For longevity purposes it is highly recommended to provide an author name.",
            UserWarning,
        )
    if author_email is None:
        warnings.warn(
            "`author_email` is set to None. For longevity purposes it is highly recommended to provide an author email, or some other obvious contact information.",
            UserWarning,
        )

    dataset_path = get_dataset_path(dataset_name)

    # Check if dataset already exists
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(dataset_path, "data")
        os.makedirs(dataset_path)
        data_path = os.path.join(dataset_path, "main_data.hdf5")

        total_steps = 0
        with h5py.File(data_path, "w", track_order=True) as file:
            for i, eps_buff in enumerate(buffer):
                # check episode terminated or truncated
                assert (
                    eps_buff["terminations"][-1] or eps_buff["truncations"][-1]
                ), "Each episode must be terminated or truncated before adding it to a Minari dataset"
                assert len(eps_buff["actions"]) + 1 == len(
                    eps_buff["observations"]
                ), f"Number of observations {len(eps_buff['observations'])} must have an additional \
                                                                                        element compared to the number of action steps {len(eps_buff['actions'])} \
                                                                                        The initial and final observation must be included"
                seed = eps_buff.pop("seed", None)
                eps_group = clear_episode_buffer(
                    eps_buff, file.create_group(f"episode_{i}")
                )

                eps_group.attrs["id"] = i
                total_steps = len(eps_buff["actions"])
                eps_group.attrs["total_steps"] = total_steps
                total_steps += total_steps

                if seed is None:
                    eps_group.attrs["seed"] = str(None)
                else:
                    assert isinstance(seed, int)
                    eps_group.attrs["seed"] = seed

                # TODO: save EpisodeMetadataCallback callback in MinariDataset and update new episode group metadata

            file.attrs["total_episodes"] = len(buffer)
            file.attrs["total_steps"] = total_steps

            # TODO: check if observation/action have been flatten and update
            file.attrs["flatten_observation"] = False
            file.attrs["flatten_action"] = False

            file.attrs[
                "env_spec"
            ] = env.spec.to_json()  # pyright: ignore [reportOptionalMemberAccess]
            file.attrs["dataset_name"] = dataset_name

        return MinariDataset(data_path)
    else:
        raise ValueError(
            f"A Minari dataset with ID {dataset_name} already exists and it cannot be overridden. Please use a different dataset name or version."
        )


def create_dataset_from_collector_env(
    dataset_name: str,
    collector_env: DataCollectorV0,
    algorithm_name: Optional[str] = None,
    author: Optional[str] = None,
    author_email: Optional[str] = None,
    code_permalink: Optional[str] = None,
):
    """Create a Minari dataset using the data collected from stepping with a Gymnasium environment wrapped with a `DataCollectorV0` Minari wrapper.

    Args:
        dataset_name (str): name id to identify Minari dataset
        collector_env (DataCollectorV0): Gymnasium environment used to collect the buffer data
        buffer (list[Dict[str, Union[list, Dict]]]): list of episode dictionaries with data
        algorithm_name (Optional[str], optional): name of the algorithm used to collect the data. Defaults to None.
        author (Optional[str], optional): author that generated the dataset. Defaults to None.
        author_email (Optional[str], optional): email of the author that generated the dataset. Defaults to None.
        code_permalink (Optional[str], optional): link to relevant code used to generate the dataset. Defaults to None.

    Returns:
        MinariDataset
    """
    # NoneType warnings
    if code_permalink is None:
        warnings.warn(
            "`code_permalink` is set to None. For reproducibility purposes it is highly recommended to link your dataset to versioned code.",
            UserWarning,
        )
    if author is None:
        warnings.warn(
            "`author` is set to None. For longevity purposes it is highly recommended to provide an author name.",
            UserWarning,
        )
    if author_email is None:
        warnings.warn(
            "`author_email` is set to None. For longevity purposes it is highly recommended to provide an author email, or some other obvious contact information.",
            UserWarning,
        )

    assert collector_env.datasets_path is not None
    dataset_path = os.path.join(collector_env.datasets_path, dataset_name)

    # Check if dataset already exists
    if not os.path.exists(dataset_path):
        dataset_patest_and_return_nameth = os.path.join(dataset_path, "data")
        os.makedirs(dataset_path)
        data_path = os.path.join(dataset_path, "main_data.hdf5")
        collector_env.save_to_disk(
            data_path,
            dataset_metadata={
                "dataset_name": str(dataset_name),
                "algorithm_name": str(algorithm_name),
                "author": str(author),
                "author_email": str(author_email),
                "code_permalink": str(code_permalink),
            },
        )
        return MinariDataset(data_path)
    else:
        raise ValueError(
            f"A Minari dataset with ID {dataset_name} already exists and it cannot be overridden. Please use a different dataset name or version."
        )

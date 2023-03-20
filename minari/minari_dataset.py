import os
import warnings
from typing import Callable, Dict, List, Optional, Union

import gymnasium as gym
import h5py
import numpy as np
from gymnasium.envs.registration import EnvSpec

from minari.storage.datasets_root_dir import get_dataset_path
from minari.utils.data_collector import DataCollectorV0


class MinariDataset:
    """Main Minari dataset class to sample data and get metadata information from a dataset.

    TODO: Currently sampling data is not implemented
    """

    def __init__(self, data_path: str):
        """Initialize properties of the Minari Dataset.

        Args:
            data_path (str): full path to the `main_data.hdf5` file of the dataset.
        """
        self._data_path = data_path
        self._extra_data_id = 0
        with h5py.File(self._data_path, "r") as f:
            self._flatten_observations = f.attrs["flatten_observation"]
            self._flatten_actions = f.attrs["flatten_action"]
            self._env_spec = EnvSpec.from_json(f.attrs["env_spec"])

            self._total_episodes: int = f.attrs["total_episodes"]
            self._total_steps: int = f.attrs["total_steps"]

            self._dataset_name = f.attrs["dataset_name"]
            self._combined_datasets = f.attrs.get("combined_datasets")

            env = gym.make(self._env_spec)

            self._observation_space = env.observation_space
            self._action_space = env.action_space

            self._author = f.attrs["author"]
            self._author_email = f.attrs["author_email"]

            env.close()

        self._episode_idx = list(range(self._total_episodes))

    def recover_environment(self):
        """Recover the Gymnasium environment used to create the dataset.

        Returns:
            environment: Gymnasium environment
        """
        return gym.make(self._env_spec)

    @property
    def flatten_observations(self) -> bool:
        """If the observations have been flatten when creating the dataset."""
        return self._flatten_observations

    @property
    def flatten_actions(self) -> bool:
        """If the actions have been flatten when creating the dataset."""
        return self._flatten_actions

    @property
    def observation_space(self):
        """Original observation space of the environment before flatteining (if this is the case)."""
        return self._observation_space

    @property
    def action_space(self):
        """Original action space of the environment before flatteining (if this is the case)."""
        return self._action_space

    @property
    def data_path(self):
        """Full path to the `main_data.hdf5` file of the dataset."""
        return self._data_path

    @property
    def total_steps(self) -> int:
        """Total steps recorded in the Minari dataset along all episodes."""
        return self._total_steps

    @property
    def total_episodes(self) -> int:
        """Total episodes recorded in the Minari dataset."""
        return self._total_episodes

    @property
    def combined_datasets(self):
        """If this Minari dataset is a combination of other subdatasets, return a list with the subdataset names."""
        if self._combined_datasets is None:
            return []
        else:
            return self._combined_datasets

    @property
    def name(self) -> str:
        """Name of the Minari dataset."""
        return self._dataset_name

    @property
    def author(self) -> str:
        return self._author

    @property
    def email(self) -> str:
        return self._author_email

    def _filter_episode_group(
        self, condition_func: Callable[[h5py.Group], bool], hdf5_file: h5py.File
    ) -> Callable[[int], bool]:
        """Decorator to filter episodes by group given episode id.

        Args:
            condition_func (Callable[[h5py.Group], bool]): HDF5 episode group
            hdf5_file (h5py.File): opened HDF5 dataset file

        Returns:
            Callable[[int], bool]: True if condition is met, False otherwise
        """

        def condition_from_episode_id(episode_id: int) -> bool:
            """Check condition by episode index.

            Args:
                episode_id (int): the index of the episode

            Returns:
                bool: True if condition is met, False otherwise
            """
            episode_group = hdf5_file[f"episode_{episode_id}"]
            assert isinstance(episode_group, h5py.Group)
            keep_episode = condition_func(episode_group)
            if not keep_episode:
                total_episode_steps = episode_group.attrs["total_steps"]
                assert isinstance(total_episode_steps, int)
                self._total_steps -= total_episode_steps

            return keep_episode

        return condition_from_episode_id

    def filter_episodes(self, condition: Callable[[h5py.Group], bool]):
        """Filter the dataset episodes with a condition.

        The condition must be a callable with  a single argument, the episode HDF5 group.
        The callable must return a `bool` True if the condition is met and False otherwise.
        i.e filtering for episodes that terminate:

        ```
        dataset.filter(condition=lambda x: x['terminations'][-1] )
        ```

        Args:
            condition (Callable[[h5py.Group], bool]): callable that accepts an episode group and returns True if certain condition is met.
        """
        with h5py.File(self._data_path, "r") as f:
            condition_func = self._filter_episode_group(condition, f)
            self._episode_idx = list(filter(condition_func, self._episode_idx))

        self._total_episodes = len(self._episode_idx)
        "TODO: return new updated Minari dataset object instead of updating the current object"

    def shuffle_episodes(self, seed: Optional[int] = None):
        """Suffle the episode iterator for sampling.

        Args:
            seed (Optional[int], optional): random seed to shuffle the episodes. Defaults to None.
        """
        pass

    def sample_episodes(
        self,
        n_episodes: Optional[int] = None,
        normalize_observation: bool = False,
        normalize_reward: bool = False,
    ):
        """Sample n number of episodes from the dataset.

        Args:
            n_episodes (Optional[int], optional): _description_. Defaults to None.
            normalize_observation (bool, optional): _description_. Defaults to False.
            normalize_reward (bool, optional): _description_. Defaults to False.
        """
        pass

    def reset_episodes(self):
        """Reset the dataset to its initial state.

        Re-fill index array with filtered episodes and sort them.
        """
        pass

    def update_dataset_from_collector_env(self, collector_env: DataCollectorV0):
        """Add extra data to Minari dataset from collector environment buffers (DataCollectorV0).

        This method can be used as a checkpoint when creating a dataset.
        A new HDF5 file will be created with the new dataset file in the same directory as `main_data.hdf5` called
        `additional_data_i.hdf5`. Both datasets are joined together by creating external links to each additional
        episode group: https://docs.h5py.org/en/stable/high/group.html#external-links

        Args:
            collector_env (DataCollectorV0): Collector environment
        """
        # check that collector env has the same characteristics as self._env_spec
        new_data_file_path = os.path.join(
            os.path.split(self.data_path)[0],
            f"additional_data_{self._extra_data_id}.hdf5",
        )

        collector_env.save_to_disk(path=new_data_file_path)

        with h5py.File(new_data_file_path, "r", track_order=True) as new_data_file:
            group_paths = [group.name for group in new_data_file.values()]
            new_data_total_episodes = new_data_file.attrs["total_episodes"]
            new_data_total_steps = new_data_file.attrs["total_steps"]

        with h5py.File(self.data_path, "a", track_order=True) as file:
            last_episode_id = file.attrs["total_episodes"]
            for i, eps_group_path in enumerate(group_paths):
                file[f"episode_{last_episode_id + i}"] = h5py.ExternalLink(
                    f"additional_data_{self._extra_data_id}.hdf5", eps_group_path
                )
                file[f"episode_{last_episode_id + i}"].attrs.modify(
                    "id", last_episode_id + i
                )

            # Update metadata of minari dataset
            file.attrs.modify(
                "total_episodes", last_episode_id + new_data_total_episodes
            )
            file.attrs.modify(
                "total_steps", file.attrs["total_steps"] + new_data_total_steps
            )
        self._extra_data_id += 1

    def update_dataset_from_buffer(self, buffer: List[dict]):
        """Additional data can be added to the Minari Dataset from a list of episode dictionary buffers.

        Each episode dictionary buffer must have the following items:
            * `observations`: np.ndarray of step observations. shape = (total_episode_steps + 1, (observation_shape)). Should include initial and final observation
            * `actions`: np.ndarray of step action. shape = (total_episode_steps + 1, (action_shape)).
            * `rewards`: np.ndarray of step rewards. shape = (total_episode_steps + 1, 1).
            * `terminations`: np.ndarray of step terminations. shape = (total_episode_steps + 1, 1).
            * `truncations`: np.ndarray of step truncations. shape = (total_episode_steps + 1, 1).

        Other additional items can be added as long as the values are np.ndarray's or other nested dictionaries.

        Args:
            buffer (list[dict]): list of episode dictionary buffers to add to dataset
        """
        additional_steps = 0
        with h5py.File(self.data_path, "a", track_order=True) as file:
            last_episode_id = file.attrs["total_episodes"]
            for i, eps_buff in enumerate(buffer):
                episode_id = last_episode_id + i
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
                    eps_buff, file.create_group(f"episode_{episode_id}")
                )

                eps_group.attrs["id"] = episode_id
                total_steps = len(eps_buff["actions"])
                eps_group.attrs["total_steps"] = total_steps
                additional_steps += total_steps

                if seed is None:
                    eps_group.attrs["seed"] = str(None)
                else:
                    assert isinstance(seed, int)
                    eps_group.attrs["seed"] = seed

                # TODO: save EpisodeMetadataCallback callback in MinariDataset and update new episode group metadata

            file.attrs.modify("total_episodes", last_episode_id + len(buffer))
            file.attrs.modify(
                "total_steps", file.attrs["total_steps"] + additional_steps
            )


def clear_episode_buffer(episode_buffer: Dict, eps_group: h5py.Group):
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

        current_env_spec = None

        for dataset in datasets_to_combine:
            if not isinstance(dataset, MinariDataset):
                raise ValueError(f"The dataset {dataset} is not of type MinariDataset.")

            with h5py.File(dataset.data_path, "r", track_order=True) as data_file:
                group_paths = [group.name for group in data_file.values()]
                dataset_env_spec = data_file.attrs["env_spec"]

                assert isinstance(dataset_env_spec, str)
                dataset_env_spec = EnvSpec.from_json(dataset_env_spec)
                # We have to check that all datasets can be merged by checking that they come from the same
                # environments. However, we override the time limit max_episode_steps with the max among all
                # the datasets to be combined. Then we check if the rest of the env_spec attributes are from
                # the same environment.
                if current_env_spec is None:
                    current_env_spec = dataset_env_spec
                elif dataset_env_spec.max_episode_steps is not None:
                    if current_env_spec.max_episode_steps is None:
                        current_env_spec.max_episode_steps = (
                            dataset_env_spec.max_episode_steps
                        )
                    else:
                        if (
                            current_env_spec.max_episode_steps
                            < dataset_env_spec.max_episode_steps
                        ):
                            current_env_spec.max_episode_steps = (
                                dataset_env_spec.max_episode_steps
                            )
                        else:
                            dataset_env_spec.max_episode_steps = (
                                current_env_spec.max_episode_steps
                            )

                if current_env_spec != dataset_env_spec:
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

            # TODO: list of authors, and emails
            combined_data_file.attrs["author"] = dataset.author
            combined_data_file.attrs["author_email"] = dataset.email

        assert current_env_spec is not None
        combined_data_file.attrs["env_spec"] = current_env_spec.to_json()

    return MinariDataset(new_data_path)


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
            file.attrs["author"] = str(author)
            file.attrs["author_email"] = str(author_email)
            file.attrs["code_permalink"] = str(code_permalink)

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
        dataset_path = os.path.join(dataset_path, "data")
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

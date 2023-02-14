import os
import warnings
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
from gymnasium.envs.registration import EnvSpec

from minari.storage.datasets_root_dir import get_dataset_path
from minari.utils.data_collector import DataCollectorV0


def clear_buffer(buffer: dict, eps_group):
    for key, data in buffer.items():
        if isinstance(data, dict):
            if key in eps_group:
                eps_group_to_clear = eps_group[key]
            else:
                eps_group_to_clear = eps_group.create_group(key)
            clear_buffer(data, eps_group_to_clear)
        else:
            # assert data is numpy array
            assert np.all(np.logical_not(np.isnan(data)))
            # add seed to attributes
            eps_group.create_dataset(key, data=data, chunks=True)

    return eps_group


class MinariDataset:
    def __init__(self, data_path: str):
        """The `id` parameter corresponds to the name of the dataset, with the syntax as follows:
        `(namespace)/(env_name)-v(version)` where `namespace` is optional.
        """

        self._data_path = data_path
        self._extra_data_id = 0
        with h5py.File(self._data_path, "r") as f:
            self._flatten_observations = f.attrs["flatten_observation"]
            self._flatten_actions = f.attrs["flatten_action"]
            self._env_spec = EnvSpec.from_json(f.attrs["env_spec"])

            self._total_episodes = f.attrs["total_episodes"]
            self._total_steps = f.attrs["total_steps"]

            self._dataset_name = f.attrs["dataset_name"]
            self._combined_datasets = f.attrs.get("combined_datasets")

            env = gym.make(self._env_spec)

            self._observation_space = env.observation_space
            self._action_space = env.action_space

            env.close()

    def recover_environment(self):
        return gym.make(self._env_spec)

    @property
    def flatten_observations(self) -> bool:
        return self._flatten_observations

    @property
    def flatten_actions(self) -> bool:
        return self._flatten_actions

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def data_path(self):
        return self._data_path

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def total_episodes(self):
        return self._total_episodes

    @property
    def combined_datasets(self):
        if self._combined_datasets is None:
            return []
        else:
            return self._combined_datasets

    @property
    def name(self):
        return self._dataset_name

    def update_dataset_from_collector_env(self, collector_env: DataCollectorV0):
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

    def update_dataset_from_buffer(self, buffer: list[dict]):
        """Additional data can be added to the Minari Dataset from a list of episode dictionary buffers.

        The episode dictionary buffer must have the following keys:
            * `observations`:
            * `actions`:
            * `rewards`:
            * `terminations`:
            * `truncations`:

        Other keys are optional as long as the data

        Args:
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
                eps_group = clear_buffer(
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


def combine_datasets(datasets_to_combine: list[MinariDataset], new_dataset_name: str):
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
                raise ValueError(f"The dataset {dataset} is not of type MinariDataset.")

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


def create_dataset_from_buffers(
    dataset_name: str,
    env,
    algorithm_name: str,
    environment,
    code_permalink,
    author,
    author_email,
    buffer,
):

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
                eps_group = clear_buffer(eps_buff, file.create_group(f"episode_{i}"))

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

            file.attrs["env_spec"] = env.spec.to_json()
            file.attrs["dataset_name"] = dataset_name

        return MinariDataset(data_path)
    else:
        raise ValueError(
            f"A Minari dataset with ID {dataset_name} already exists and it cannot be overridden. Please use a different dataset name or version."
        )


def create_dataset_from_collector_env(
    dataset_name,
    collector_env: DataCollectorV0,
    algorithm_name: Optional[str] = None,
    author: Optional[str] = None,
    author_email: Optional[str] = None,
    code_permalink: Optional[str] = None,
):

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

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union

import gymnasium as gym
import h5py
import numpy as np
from gymnasium import error
from gymnasium.envs.registration import EnvSpec

from minari.data_collector import DataCollectorV0
from minari.dataset.minari_storage import MinariStorage, PathLike


DATASET_ID_RE = re.compile(
    r"(?:(?P<environment>[\w]+?))?(?:-(?P<dataset>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)


def parse_dataset_id(dataset_id: str) -> tuple[str | None, str, int | None]:
    """Parse dataset ID string format - ``(env_name-)(dataset_name)(-v(version))``.

    Args:
        dataset_id: The dataset id to parse
    Returns:
        A tuple of environment name, dataset name and version number
    Raises:
        Error: If the dataset id is not valid dataset regex
    """
    match = DATASET_ID_RE.fullmatch(dataset_id)
    if not match:
        raise error.Error(
            f"Malformed dataset ID: {dataset_id}. (Currently all IDs must be of the form (env_name-)(dataset_name)-v(version). (namespace is optional))"
        )
    env_name, dataset_name, version = match.group("environment", "dataset", "version")
    if version is not None:
        version = int(version)

    return env_name, dataset_name, version


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


@dataclass(frozen=True)
class EpisodeData:
    """Contains the datasets data for a single episode.

    This is the object returned by :class:`minari.MinariDataset.sample_episodes`.
    """

    id: int
    seed: Optional[int]
    total_timesteps: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray

    def __repr__(self) -> str:
        return (
            "EpisodeData("
            f"id={repr(self.id)}, "
            f"seed={repr(self.seed)}, "
            f"total_timesteps={self.total_timesteps}, "
            f"observations={EpisodeData._repr_space_values(self.observations)}, "
            f"actions={EpisodeData._repr_space_values(self.actions)}, "
            f"rewards=ndarray of {len(self.rewards)} floats, "
            f"terminations=ndarray of {len(self.terminations)} bools, "
            f"truncations=ndarray of {len(self.truncations)} bools"
            ")"
        )

    @staticmethod
    def _repr_space_values(value):
        if isinstance(value, np.ndarray):
            return f"ndarray of shape {value.shape} and dtype {value.dtype}"
        elif isinstance(value, dict):
            reprs = [
                f"{k}: {EpisodeData._repr_space_values(v)}" for k, v in value.items()
            ]
            dict_repr = ", ".join(reprs)
            return "{" + dict_repr + "}"
        elif isinstance(value, tuple):
            reprs = [EpisodeData._repr_space_values(v) for v in value]
            values_repr = ", ".join(reprs)
            return "(" + values_repr + ")"
        else:
            return repr(value)


@dataclass
class MinariDatasetSpec:
    flatten_observations: bool
    flatten_actions: bool
    env_spec: EnvSpec
    total_episodes: int
    total_steps: int
    dataset_id: str
    combined_datasets: List[str]
    observation_space: gym.Space
    action_space: gym.Space
    data_path: str

    # post-init attributes
    env_name: str | None = field(init=False)
    dataset_name: str = field(init=False)
    version: int | None = field(init=False)

    def __post_init__(self):
        """Calls after the spec is created to extract the environment name, dataset name and version from the dataset id."""
        self.env_name, self.dataset_name, self.version = parse_dataset_id(
            self.dataset_id
        )


class MinariDataset:
    """Main Minari dataset class to sample data and get metadata information from a dataset."""

    def __init__(
        self,
        data: Union[MinariStorage, PathLike],
        episode_indices: Optional[np.ndarray] = None,
    ):
        """Initialize properties of the Minari Dataset.

        Args:
            data (Union[MinariStorage, _PathLike]): source of data.
            episode_indices (Optiona[np.ndarray]): slice of episode indices this dataset is pointing to.
        """
        if isinstance(data, MinariStorage):
            self._data = data
        elif (
            isinstance(data, str)
            or isinstance(data, os.PathLike)
            or isinstance(data, bytes)
        ):
            self._data = MinariStorage(data)
        else:
            raise ValueError(f"Unrecognized type {type(data)} for data")

        self._additional_data_id = 0
        if episode_indices is None:
            episode_indices = np.arange(self._data.total_episodes)
        self._episode_indices = episode_indices

        self.spec = MinariDatasetSpec(
            flatten_observations=self._data.flatten_observations,
            flatten_actions=self._data.flatten_actions,
            env_spec=self._data.env_spec,
            total_episodes=self._data.total_episodes,
            total_steps=self._data.total_steps,
            dataset_id=self._data.id,
            combined_datasets=self._data.combined_datasets,
            observation_space=self._data.observation_space,
            action_space=self._data.action_space,
            data_path=str(self._data.data_path),
        )
        self._total_steps = None
        self._generator = np.random.default_rng()

    @property
    def total_episodes(self):
        """Total episodes recorded in the Minari dataset."""
        assert self._episode_indices is not None
        return len(self._episode_indices)

    @property
    def total_steps(self):
        """Total episodes steps in the Minari dataset."""
        if self._total_steps is None:
            t_steps = self._data.apply(
                lambda episode: episode["total_steps"],
                episode_indices=self._episode_indices,
            )
            self._total_steps = sum(t_steps)
        return self._total_steps

    @property
    def episode_indices(self) -> np.ndarray:
        """Indices of the available episodes to sample within the Minari dataset."""
        return self._episode_indices

    def recover_environment(self):
        """Recover the Gymnasium environment used to create the dataset.

        Returns:
            environment: Gymnasium environment
        """
        return gym.make(self._data.env_spec)

    def set_seed(self, seed: int):
        """Set seed for random episode sampling generator."""
        self._generator = np.random.default_rng(seed)

    def filter_episodes(self, condition: Callable[[h5py.Group], bool]) -> MinariDataset:
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
        mask = self._data.apply(condition, episode_indices=self._episode_indices)
        assert self._episode_indices is not None
        return MinariDataset(self._data, episode_indices=self._episode_indices[mask])

    def sample_episodes(self, n_episodes: int) -> Iterable[EpisodeData]:
        """Sample n number of episodes from the dataset.

        Args:
            n_episodes (Optional[int], optional): number of episodes to sample.
        """
        indices = self._generator.choice(
            self._episode_indices, size=n_episodes, replace=False
        )
        episodes = self._data.get_episodes(indices)
        return list(map(lambda data: EpisodeData(**data), episodes))

    def iterate_episodes(
        self, episode_indices: Optional[List[int]] = None
    ) -> Iterator[EpisodeData]:
        """Iterate over episodes from the dataset.

        Args:
            episode_indices (Optional[List[int]], optional): episode indices to iterate over.
        """
        if episode_indices is None:
            assert self._episode_indices is not None
            assert self._episode_indices.ndim == 1
            episode_indices = self._episode_indices.tolist()

        assert episode_indices is not None

        for episode_index in episode_indices:
            data = self._data.get_episodes([episode_index])[0]
            yield EpisodeData(**data)

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
            os.path.split(self.spec.data_path)[0],
            f"additional_data_{self._additional_data_id}.hdf5",
        )

        collector_env.save_to_disk(path=new_data_file_path)

        with h5py.File(new_data_file_path, "r", track_order=True) as new_data_file:
            new_data_total_episodes = new_data_file.attrs["total_episodes"]
            new_data_total_steps = new_data_file.attrs["total_steps"]

        with h5py.File(self._data.data_path, "a", track_order=True) as file:
            last_episode_id = file.attrs["total_episodes"]
            for id in range(new_data_total_episodes):
                file[f"episode_{last_episode_id + id}"] = h5py.ExternalLink(
                    f"additional_data_{self._additional_data_id}.hdf5", f"/episode_{id}"
                )
                file[f"episode_{last_episode_id + id}"].attrs.modify(
                    "id", last_episode_id + id
                )

            # Update metadata of minari dataset
            file.attrs.modify(
                "total_episodes", last_episode_id + new_data_total_episodes
            )
            file.attrs.modify(
                "total_steps", file.attrs["total_steps"] + new_data_total_steps
            )
        self._additional_data_id += 1

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
        with h5py.File(self.spec.data_path, "a", track_order=True) as file:
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

    def __iter__(self):
        return self.iterate_episodes()

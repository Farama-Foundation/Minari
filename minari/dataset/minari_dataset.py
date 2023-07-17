from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import error
from gymnasium.envs.registration import EnvSpec

from minari.data_collector import DataCollectorV0
from minari.dataset.minari_storage import MinariStorage, PathLike


DATASET_ID_RE = re.compile(
    r"(?:(?P<environment>[\w]+?))?(?:-(?P<dataset>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)


def parse_dataset_id(dataset_id: str) -> tuple[str | None, str, int]:
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

    version = int(version)

    return env_name, dataset_name, version


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
    env_spec: EnvSpec
    total_episodes: int
    total_steps: int
    dataset_id: str
    combined_datasets: List[str]
    observation_space: gym.Space
    action_space: gym.Space
    data_path: str
    minari_version: str

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

        assert self._episode_indices is not None

        total_steps = sum(
            self._data.apply(
                lambda episode: episode["total_timesteps"],
                episode_indices=self._episode_indices,
            )
        )

        self.spec = MinariDatasetSpec(
            env_spec=self._data.env_spec,
            total_episodes=self._episode_indices.size,
            total_steps=total_steps,
            dataset_id=self._data.id,
            combined_datasets=self._data.combined_datasets,
            observation_space=self._data.observation_space,
            action_space=self._data.action_space,
            data_path=str(self._data.data_path),
            minari_version=str(self._data.minari_version),
        )
        self._total_steps = total_steps
        self._generator = np.random.default_rng()

    @property
    def total_episodes(self):
        """Total episodes recorded in the Minari dataset."""
        assert self._episode_indices is not None
        return self._episode_indices.size

    @property
    def total_steps(self):
        """Total episodes steps in the Minari dataset."""
        return self._total_steps

    @property
    def episode_indices(self) -> np.ndarray:
        """Indices of the available episodes to sample within the Minari dataset."""
        return self._episode_indices

    def recover_environment(self) -> gym.Env:
        """Recover the Gymnasium environment used to create the dataset.

        Returns:
            environment: Gymnasium environment
        """
        return gym.make(self._data.env_spec)

    def set_seed(self, seed: int):
        """Set seed for random episode sampling generator."""
        self._generator = np.random.default_rng(seed)

    def filter_episodes(
        self, condition: Callable[[EpisodeData], bool]
    ) -> MinariDataset:
        """Filter the dataset episodes with a condition.

        The condition must be a callable which takes an `EpisodeData` instance and retutrns a bool.
        The callable must return a `bool` True if the condition is met and False otherwise.
        i.e filtering for episodes that terminate:

        ```
        dataset.filter(condition=lambda x: x['terminations'][-1] )
        ```

        Args:
            condition (Callable[[EpisodeData], bool]): callable that accepts any type(For our current backend, an h5py episode group) and returns True if certain condition is met.
        """

        def dict_to_episode_data_condition(episode: dict) -> bool:
            return condition(EpisodeData(**episode))

        mask = self._data.apply(
            dict_to_episode_data_condition, episode_indices=self._episode_indices
        )
        assert self._episode_indices is not None
        return MinariDataset(self._data, episode_indices=self._episode_indices[mask])

    def sample_episodes(self, n_episodes: int) -> Iterable[EpisodeData]:
        """Sample n number of episodes from the dataset.

        Args:
            n_episodes (Optional[int], optional): number of episodes to sample.
        """
        indices = self._generator.choice(
            self.episode_indices, size=n_episodes, replace=False
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
            assert self.episode_indices is not None
            assert self.episode_indices.ndim == 1
            episode_indices = self.episode_indices.tolist()

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

        old_total_episodes = self._data.total_episodes

        self._data.update_from_collector_env(
            collector_env, new_data_file_path, self._additional_data_id
        )

        new_total_episodes = self._data._total_episodes

        self._additional_data_id += 1

        self._episode_indices = np.append(
            self._episode_indices, np.arange(old_total_episodes, new_total_episodes)
        )  # ~= np.append(self._episode_indices,np.arange(self._data.total_episodes))

        self.spec.total_episodes = self._episode_indices.size
        self.spec.total_steps = sum(
            self._data.apply(
                lambda episode: episode["total_timesteps"],
                episode_indices=self._episode_indices,
            )
        )

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
        old_total_episodes = self._data.total_episodes

        self._data.update_from_buffer(buffer, self.spec.data_path)

        new_total_episodes = self._data._total_episodes

        self._episode_indices = np.append(
            self._episode_indices, np.arange(old_total_episodes, new_total_episodes)
        )  # ~= np.append(self._episode_indices,np.arange(self._data.total_episodes))

        self.spec.total_episodes = self._episode_indices.size

        self.spec.total_steps = sum(
            self._data.apply(
                lambda episode: episode["total_timesteps"],
                episode_indices=self._episode_indices,
            )
        )

    def __iter__(self):
        return self.iterate_episodes()

    def __getitem__(self, idx: int) -> EpisodeData:
        episodes_data = self._data.get_episodes([self.episode_indices[idx]])
        assert len(episodes_data) == 1
        return EpisodeData(**episodes_data[0])

    def __len__(self) -> int:
        return self.total_episodes

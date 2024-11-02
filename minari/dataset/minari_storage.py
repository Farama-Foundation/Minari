from __future__ import annotations

import json
import os
import pathlib
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec

from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.serialization import deserialize_space, serialize_space


PathLike = Union[str, os.PathLike]
METADATA_FILE_NAME = "metadata.json"


class MinariStorage(ABC):
    """Class that handles disk access to the data."""

    FORMAT: str

    def __init__(
        self,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        self._data_path: pathlib.Path = data_path
        self._observation_space = observation_space
        self._action_space = action_space

    @classmethod
    def read_raw_metadata(cls, data_path: PathLike) -> Dict[str, Any]:
        """Read the raw metadata from a path.

        Args:
            data_path (str or Path): directory where the data is stored.

        Returns:
            metadata (dict): metadata of the dataset.

        Raises:
            ValueError: if the specified path doesn't exist or doesn't contain any data.
        """
        data_path = pathlib.Path(data_path)
        if not data_path.is_dir():
            raise ValueError(f"The data path {data_path} doesn't exist")
        metadata_file_path = data_path.joinpath(METADATA_FILE_NAME)
        if not metadata_file_path.exists():
            raise ValueError(f"No data found in data path {data_path}")
        with open(metadata_file_path) as file:
            metadata = json.load(file)
        return metadata

    @classmethod
    def read(cls, data_path: PathLike) -> MinariStorage:
        """Create a MinariStorage to read data from a path.

        Args:
            data_path (str or Path): directory where the data is stored.

        Returns:
            A new MinariStorage object to read the data.

        Raises:
            ValueError: if the specified path doesn't exist or doesn't contain any data.
        """
        metadata = MinariStorage.read_raw_metadata(data_path)

        observation_space = None
        action_space = None
        if "observation_space" in metadata.keys():
            serialized_space = metadata["observation_space"]
            assert isinstance(serialized_space, str)
            observation_space = deserialize_space(serialized_space)
        if "action_space" in metadata.keys():
            serialized_space = metadata["action_space"]
            assert isinstance(serialized_space, str)
            action_space = deserialize_space(serialized_space)

        if action_space is None or observation_space is None:
            env_spec_str = metadata.get("env_spec")
            assert isinstance(env_spec_str, str)
            env_spec_str = (  # for gymnasium 1.0.0 compatibility
                env_spec_str.replace('"order_enforce": true,', "")
                .replace('"apply_api_compatibility": false,', "")
                .replace('"autoreset": false, ', "")
            )
            env_spec = EnvSpec.from_json(env_spec_str)
            env = gym.make(env_spec)
            if observation_space is None:
                observation_space = env.observation_space
            if action_space is None:
                action_space = env.action_space

        from minari.dataset._storages import get_minari_storage  # avoid circular import

        return get_minari_storage(metadata["data_format"])(
            pathlib.Path(data_path),
            observation_space,
            action_space,
        )

    @classmethod
    def new(
        cls,
        data_path: PathLike,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        env_spec: Optional[EnvSpec] = None,
        data_format: str = "hdf5",
    ) -> MinariStorage:
        """Class method to create a new data storage.

        Args:
            data_path (str or Path): directory where the data will be stored.
            observation_space (gymnasium.Space, optional): Gymnasium observation space of the dataset.
            action_space (gymnasium.Space, optional): Gymnasium action space of the dataset.
            env_spec (EnvSpec, optional): Gymnasium EnvSpec of the environment that generates the dataset.
            data_format (str): Format of the data. Default value is "hdf5".

        Returns:
            A new MinariStorage object to write new data.

        Raises:
            ValueError: if the data format is incorrect or the data path already exists.
              Moreover, you need to specify the env_spec or both the observation_space and action_space.
        """
        if env_spec is None and (observation_space is None or action_space is None):
            raise ValueError(
                "Since env_spec is not specified, you need to specify both action space and observation space"
            )
        from minari.dataset._storages import (  # avoid circular import
            get_minari_storage,
            get_storage_keys,
        )

        if data_format not in get_storage_keys():
            raise ValueError(
                f"No storage implemented for {data_format}. Available formats: {get_storage_keys()}"
            )

        data_path = pathlib.Path(data_path)
        data_path.mkdir(exist_ok=True)
        if data_path.joinpath(METADATA_FILE_NAME).exists():
            raise ValueError(
                f"A dataset is already available at {data_path}. Delete it or specify another path"
            )

        if observation_space is None or action_space is None:
            assert env_spec is not None
            env = gym.make(env_spec)
            if observation_space is None:
                observation_space = env.observation_space
            if action_space is None:
                action_space = env.action_space

        metadata: Dict[str, Any] = {
            "total_episodes": 0,
            "total_steps": 0,
            "data_format": data_format,
            "observation_space": serialize_space(observation_space),
            "action_space": serialize_space(action_space),
        }
        if env_spec is not None:
            try:
                metadata["env_spec"] = env_spec.to_json()
            except TypeError as e:
                warnings.warn(
                    f"env_spec is not serializable as {str(e)}. "
                    "You will not be able to recover the environment from the dataset."
                )
        with open(data_path.joinpath(METADATA_FILE_NAME), "w") as f:
            json.dump(metadata, f)

        obj = get_minari_storage(data_format)._create(
            data_path, observation_space, action_space
        )
        return obj

    @classmethod
    @abstractmethod
    def _create(
        cls,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> MinariStorage: ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata of the dataset."""
        metadata = MinariStorage.read_raw_metadata(self.data_path)

        metadata["observation_space"] = self.observation_space
        metadata["action_space"] = self.action_space
        if "author" in metadata:
            metadata["author"] = set(metadata["author"])
        if "author_email" in metadata:
            metadata["author_email"] = set(metadata["author_email"])
        return metadata

    def update_metadata(self, metadata: Dict):
        """Update the metadata adding/modifying some keys.

        Args:
            metadata (dict): dictionary of keys-values to add to the metadata.
        """
        not_updatable_keys = {
            "dataset_id",
            "observation_space",
            "action_space",
            "env_spec",
            "data_format",
            "minari_version",
        }

        assert isinstance(metadata.get("dataset_id", ""), str)
        assert isinstance(metadata.get("total_episodes", 0), int)
        assert isinstance(metadata.get("total_steps", 0), int)
        assert isinstance(metadata.get("code_permalink", ""), str)
        assert isinstance(metadata.get("algorithm_name", ""), str)
        assert isinstance(metadata.get("author", set()), set)
        assert isinstance(metadata.get("author_email", set()), set)
        assert isinstance(metadata.get("minari_version", ""), str)

        saved_metadata = MinariStorage.read_raw_metadata(self.data_path)

        forbidden_keys = not_updatable_keys.intersection(metadata.keys())
        forbidden_keys = forbidden_keys.intersection(saved_metadata.keys())

        if forbidden_keys:
            raise ValueError(
                f"You are not allowed to update values for {', '.join(forbidden_keys)}"
            )

        saved_metadata.update(metadata)
        with open(self.data_path.joinpath(METADATA_FILE_NAME), "w") as file:
            json.dump(saved_metadata, file, default=_json_converter)

    @abstractmethod
    def update_episode_metadata(
        self, metadatas: Iterable[Dict], episode_indices: Optional[Iterable] = None
    ):
        """Update the metadata of episodes.

        Args:
            metadatas (Iterable[Dict]): metadatas, one for each episode.
            episode_indices (Iterable, optional): episode indices to update.
            If not specified, all the episodes are considered.
        """
        ...

    @abstractmethod
    def get_episode_metadata(self, episode_indices: Iterable[int]) -> Iterable[Dict]:
        """Get the metadata of episodes.

        Args:
            episode_indices (Iterable[int]): episodes id to return

        Returns:
            metadatas (Iterable[Dict]): episodes metadata
        """
        ...

    def apply(
        self,
        function: Callable[[dict], Any],
        episode_indices: Optional[Iterable] = None,
    ) -> Iterable[Any]:
        """Apply a function to a slice of the data.

        Args:
            function (Callable): function to apply to episodes
            episode_indices (Optional[Iterable]): episodes id to consider

        Returns:
            outs (Iterable): outputs returned by the function applied to episodes
        """
        if episode_indices is None:
            episode_indices = range(self.total_episodes)

        ep_dicts = self.get_episodes(episode_indices)
        return map(function, ep_dicts)

    @abstractmethod
    def get_episodes(self, episode_indices: Iterable[int]) -> Iterable[dict]:
        """Get a list of episodes.

        Args:
            episode_indices (Iterable[int]): episodes id to return

        Returns:
            episodes (Iterable[dict]): episodes data
        """
        ...

    @abstractmethod
    def update_episodes(self, episodes: Iterable[EpisodeBuffer]):
        """Update episodes in the storage from a list of episode buffer.

        Args:
            episodes (Iterable[EpisodeBuffer]): list of episodes buffer.
            They must contain the keys specified in EpsiodeData dataclass, except for `id` which is optional.
            If `id` is specified and exists, the new data is appended to the one in the storage.
        """
        ...

    def update_from_storage(self, storage: MinariStorage):
        """Update the dataset using another MinariStorage.

        Args:
            storage (MinariStorage): the other MinariStorage from which the data will be taken
        """
        for episode in storage.get_episodes(range(storage.total_episodes)):
            episode_buffer = EpisodeBuffer(
                id=None,
                seed=episode.get("seed"),
                observations=episode["observations"],
                actions=episode["actions"],
                rewards=episode["rewards"],
                terminations=episode["terminations"],
                truncations=episode["truncations"],
                infos=episode.get("infos"),
            )
            self.update_episodes([episode_buffer])

        author1 = self.metadata.get("author", set())
        author2 = storage.metadata.get("author", set())
        email1 = self.metadata.get("author_email", set())
        email2 = storage.metadata.get("author_email", set())

        self.update_metadata(
            {
                "author": author1.union(author2),
                "author_email": email1.union(email2),
                "dataset_size": self.get_size(),
            }
        )

    def get_size(self) -> float:
        """Returns the dataset size in MB.

        Returns:
            datasize (float): size of the dataset in MB
        """
        datasize = 0
        if self.data_path.exists():
            for filename in self.data_path.glob("**/*"):
                st_size = self.data_path.joinpath(filename).stat().st_size
                datasize += st_size / 1000000

        return np.round(datasize, 1)

    @property
    def data_path(self) -> pathlib.Path:
        """Full path to the dataset."""
        return self._data_path

    @property
    def total_episodes(self) -> int:
        """Total episodes in the dataset."""
        return self.metadata["total_episodes"]

    @property
    def total_steps(self) -> int:
        """Total steps in the dataset."""
        return self.metadata["total_steps"]

    @property
    def observation_space(self) -> gym.Space:
        """Observation Space of the dataset."""
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space of the dataset."""
        return self._action_space


def _json_converter(obj: Any):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

from __future__ import annotations

import json
import os
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec

from minari.serialization import deserialize_space, serialize_space


PathLike = Union[str, os.PathLike]
_METADATA_FILE_NAME = "metadata.json"


class MinariStorage(ABC):
    """Class that handles disk access to the data."""

    def __init__(
        self,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        """Initialize a MinariStorage with an existing data path.

        To create a new dataset, use the class method `new`.

        Args:
            data_path (str or Path): directory containing the data.

        Raises:
            ValueError: if the specified path doesn't exist or doesn't contain any data.

        """
        self._data_path: pathlib.Path = data_path
        self._observation_space = observation_space
        self._action_space = action_space

    @classmethod
    def read(cls, data_path: PathLike) -> MinariStorage:  # TODO: docs
        data_path = pathlib.Path(data_path)
        if not data_path.exists() or not data_path.is_dir():
            raise ValueError(f"The data path {data_path} doesn't exist")
        metadata_file_path = data_path.joinpath(_METADATA_FILE_NAME)
        if not metadata_file_path.exists():
            raise ValueError(f"No data found in data path {data_path}")
        with open(metadata_file_path) as file:
            metadata = json.load(file)

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
            env_spec = EnvSpec.from_json(env_spec_str)
            env = gym.make(env_spec)
            if observation_space is None:
                observation_space = env.observation_space
            if action_space is None:
                action_space = env.action_space

        from minari.dataset.storages import registry  # avoid circular import

        return registry[metadata["data_format"]](
            data_path, observation_space, action_space
        )

    @classmethod
    def new(
        cls,
        data_path: PathLike,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        env_spec: Optional[EnvSpec] = None,
        data_format: str = "hdf5",  # TODO: add to docs & check if value is correct with informative error
    ) -> MinariStorage:
        """Class method to create a new data storage.

        Args:
            data_path (str or Path): directory where the data will be stored.
            observation_space (gymnasium.Space, optional): Gymnasium observation space of the dataset.
            action_space (gymnasium.Space, optional): Gymnasium action space of the dataset.
            env_spec (EnvSpec, optional): Gymnasium EnvSpec of the environment that generates the dataset.

        Returns:
            A new MinariStorage object.

        Raises:
            ValueError: if you don't specify the env_spec, you need to specify both observation_space and action_space.
        """
        if env_spec is None and (observation_space is None or action_space is None):
            raise ValueError(
                "Since env_spec is not specified, you need to specify both action space and observation space"
            )
        data_path = pathlib.Path(data_path)
        data_path.mkdir(exist_ok=True)
        if data_path.joinpath("metadata.json").exists():
            raise ValueError(
                f"A dataset is already available at {data_path}. Delete it or specify another path"
            )
        metadata: Dict[str, Any] = {
            "total_episodes": 0,
            "total_steps": 0,
            "data_format": data_format,
        }
        if observation_space is not None:
            metadata["observation_space"] = serialize_space(observation_space)
        if action_space is not None:
            metadata["action_space"] = serialize_space(action_space)
        if env_spec is not None:
            metadata["env_spec"] = env_spec.to_json()
        with open(data_path.joinpath(_METADATA_FILE_NAME), "w") as f:
            json.dump(metadata, f)

        if observation_space is None or action_space is None:
            assert env_spec is not None
            env = gym.make(env_spec)
            if observation_space is None:
                observation_space = env.observation_space
            if action_space is None:
                action_space = env.action_space

        from minari.dataset.storages import registry  # avoid circular import

        obj = registry[data_format]._create(data_path, observation_space, action_space)
        return obj

    @classmethod
    @abstractmethod
    def _create(
        cls,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> MinariStorage:
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata of the dataset."""
        with open(self.data_path.joinpath(_METADATA_FILE_NAME)) as file:
            metadata = json.load(file)

        metadata["observation_space"] = self.observation_space
        metadata["action_space"] = self.action_space
        return metadata

    def update_metadata(self, metadata: Dict):
        """Update the metadata adding/modifying some keys.

        Args:
            metadata (dict): dictionary of keys-values to add to the metadata.
        """
        forbidden_keys = {"observation_space", "action_space", "env_spec"}.intersection(
            metadata.keys()
        )
        if forbidden_keys:
            raise ValueError(
                f"You are not allowed to update values for {', '.join(forbidden_keys)}"
            )

        with open(os.path.join(self.data_path, _METADATA_FILE_NAME)) as file:
            saved_metadata = json.load(file)

        saved_metadata.update(metadata)
        with open(os.path.join(self.data_path, _METADATA_FILE_NAME), "w") as file:
            file.write(json.dumps(saved_metadata))

    @abstractmethod
    def update_episode_metadata(
        self, metadatas: Iterable[Dict], episode_indices: Optional[Iterable] = None
    ):
        """Update the metadata of episodes.

        Args:
            metadatas (Iterable[Dict]): metadatas, one for each episode.
            episode_indices (Iterable, optional): episode indices to update.
            If not specified, all the episodes are considered.

        Warning:
            In case metadatas and episode_indices have different lengths, the longest is truncated silently.
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
    def get_episodes(self, episode_indices: Iterable[int]) -> List[dict]:
        """Get a list of episodes.

        Args:
            episode_indices (Iterable[int]): episodes id to return

        Returns:
            episodes (List[dict]): list of episodes data
        """
        ...

    @abstractmethod
    def update_episodes(self, episodes: Iterable[dict]):
        """Update epsiodes in the storage from a list of episode buffer.

        Args:
            episodes (Iterable[dict]): list of episodes buffer.
            They must contain the keys specified in EpsiodeData dataclass, except for `id` which is optional.
            If `id` is specified and exists, the new data is appended to the one in the storage.
        """
        ...

    @abstractmethod
    def update_from_storage(self, storage: MinariStorage):
        """Update the dataset using another MinariStorage.

        Args:
            storage (MinariStorage): the other MinariStorage from which the data will be taken
        """
        ...

    @property
    def data_path(self) -> pathlib.Path:
        """Full path to the dataset."""
        return self._data_path

    @property
    def total_episodes(self) -> np.int64:
        """Total episodes in the dataset."""
        return self.metadata["total_episodes"]

    @property
    def total_steps(self) -> np.int64:
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

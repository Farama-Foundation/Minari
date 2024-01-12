from __future__ import annotations

import importlib.metadata
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import error, logger
from gymnasium.envs.registration import EnvSpec
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from minari.dataset.episode_data import EpisodeData
from minari.dataset.minari_storage import MinariStorage, PathLike


# Use importlib due to circular import when: "from minari import __version__"
__version__ = importlib.metadata.version("minari")

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


@dataclass
class MinariDatasetSpec:
    env_spec: Optional[EnvSpec]
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
            data (Union[MinariStorage, PathLike]): source of data.
            episode_indices (Optional[np.ndarray]): slice of episode indices this dataset is pointing to.
        """
        if isinstance(data, MinariStorage):
            self._data = data
        elif isinstance(data, (str, os.PathLike)):
            self._data = MinariStorage(data)
        else:
            raise ValueError(f"Unrecognized type {type(data)} for data")

        if episode_indices is None:
            episode_indices = np.arange(self._data.total_episodes)
        self._episode_indices: np.ndarray = episode_indices
        self._total_steps = None

        metadata = self._data.metadata

        env_spec = metadata.get("env_spec")
        if env_spec is not None:
            assert isinstance(env_spec, str)
            env_spec = EnvSpec.from_json(env_spec)
        self._env_spec = env_spec

        eval_env_spec = metadata.get("eval_env_spec")
        if eval_env_spec is not None:
            assert isinstance(eval_env_spec, str)
            eval_env_spec = EnvSpec.from_json(eval_env_spec)
        self._eval_env_spec = eval_env_spec

        dataset_id = metadata["dataset_id"]
        assert isinstance(dataset_id, str)
        self._dataset_id = dataset_id

        minari_version = metadata["minari_version"]
        assert isinstance(minari_version, str)

        # Check that the dataset is compatible with the current version of Minari
        try:
            assert Version(__version__) in SpecifierSet(
                minari_version
            ), f"The installed Minari version {__version__} is not contained in the dataset version specifier {minari_version}."
            self._minari_version = minari_version
        except InvalidSpecifier:
            print(f"{minari_version} is not a version specifier.")

        self._combined_datasets = metadata.get("combined_datasets", [])

        self._observation_space = metadata["observation_space"]
        self._action_space = metadata["action_space"]
        assert isinstance(self._observation_space, gym.spaces.Space)
        assert isinstance(self._action_space, gym.spaces.Space)

        self._generator = np.random.default_rng()

    def recover_environment(self, eval_env: bool = False, **kwargs) -> gym.Env:
        """Recover the Gymnasium environment used to create the dataset.

        Args:
            eval_env (bool): if True the returned Gymnasium environment will be that intended to be used for evaluation. If no eval_env was specified when creating the dataset, the returned environment will be the same as the one used for creating the dataset. Default False.
            **kwargs: any other parameter that you want to pass to the `gym.make` function.

        Returns:
            environment: Gymnasium environment
        """
        if eval_env:
            if self._eval_env_spec is not None:
                return gym.make(self._eval_env_spec, **kwargs)
            logger.info(
                f"`eval_env` has been set to True but the dataset {self._dataset_id} doesn't provide an evaluation environment. Instead, the environment used for collecting the data will be returned: {self._env_spec}"
            )

        if self.env_spec is None:
            raise ValueError("Environment cannot be recovered when env_spec is None")

        return gym.make(self.env_spec, **kwargs)

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
            condition (Callable[[EpisodeData], bool]): function that gets in input an EpisodeData object and returns True if certain condition is met.
        """

        def dict_to_episode_data_condition(episode: dict) -> bool:
            return condition(EpisodeData(**episode))

        mask = self.storage.apply(
            dict_to_episode_data_condition, episode_indices=self.episode_indices
        )
        assert self.episode_indices is not None
        filtered_indices = self.episode_indices[list(mask)]
        return MinariDataset(self.storage, episode_indices=filtered_indices)

    def sample_episodes(self, n_episodes: int) -> Iterable[EpisodeData]:
        """Sample n number of episodes from the dataset.

        Args:
            n_episodes (Optional[int], optional): number of episodes to sample.
        """
        indices = self._generator.choice(
            self.episode_indices, size=n_episodes, replace=False
        )
        episodes = self.storage.get_episodes(indices)
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
            data = self.storage.get_episodes([episode_index])[0]
            yield EpisodeData(**data)

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
        first_id = self.storage.total_episodes
        self.storage.update_episodes(buffer)
        self.episode_indices = np.append(
            self.episode_indices, first_id + np.arange(len(buffer))
        )

    def __iter__(self):
        return self.iterate_episodes()

    def __getitem__(self, idx: int) -> EpisodeData:
        episodes_data = self.storage.get_episodes([self.episode_indices[idx]])
        assert len(episodes_data) == 1
        return EpisodeData(**episodes_data[0])

    def __len__(self) -> int:
        return self.total_episodes

    @property
    def total_episodes(self) -> int:
        return len(self.episode_indices)

    @property
    def total_steps(self) -> int:
        """Total episodes steps in the Minari dataset."""
        if self._total_steps is None:
            if self.episode_indices is None:
                self._total_steps = self.storage.total_steps
            else:
                self._total_steps = sum(
                    self.storage.apply(
                        lambda episode: episode["total_timesteps"],
                        episode_indices=self.episode_indices,
                    )
                )
        return int(self._total_steps)

    @property
    def episode_indices(self) -> np.ndarray:
        """Indices of the available episodes to sample within the Minari dataset."""
        return self._episode_indices

    @episode_indices.setter
    def episode_indices(self, new_value: np.ndarray):
        self._total_steps = None  # invalidate cache
        self._episode_indices = new_value

    @property
    def observation_space(self):
        """Original observation space of the environment before flatteining (if this is the case)."""
        return self._observation_space

    @property
    def action_space(self):
        """Original action space of the environment before flatteining (if this is the case)."""
        return self._action_space

    @property
    def env_spec(self):
        """Envspec of the environment that has generated the dataset."""
        return self._env_spec

    @property
    def combined_datasets(self) -> List[str]:
        """If this Minari dataset is a combination of other subdatasets, return a list with the subdataset names."""
        if self._combined_datasets is None:
            return []
        else:
            return self._combined_datasets

    @property
    def id(self) -> str:
        """Name of the Minari dataset."""
        return self._dataset_id

    @property
    def minari_version(self) -> str:
        """Version of Minari the dataset is compatible with."""
        return self._minari_version

    @property
    def storage(self) -> MinariStorage:
        """Minari storage managing access to disk."""
        return self._data

    @property
    def spec(self) -> MinariDatasetSpec:
        return MinariDatasetSpec(
            env_spec=self.env_spec,
            total_episodes=self._episode_indices.size,
            total_steps=self.total_steps,
            dataset_id=self.id,
            combined_datasets=self.combined_datasets,
            observation_space=self.observation_space,
            action_space=self.action_space,
            data_path=str(self.storage.data_path),
            minari_version=str(self.minari_version),
        )

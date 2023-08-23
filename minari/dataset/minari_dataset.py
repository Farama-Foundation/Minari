from __future__ import annotations

import importlib.metadata 
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import error
from gymnasium.envs.registration import EnvSpec
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from minari.dataset.minari_storage import MinariStorage, PathLike
from minari.dataset.episode_data import EpisodeData



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

        metadata = self._data.metadata

        env_spec = metadata["env_spec"]
        assert isinstance(env_spec, str)
        self._env_spec = EnvSpec.from_json(env_spec)

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

        # We will default to using the reconstructed observation and action spaces from the dataset
        # and fall back to the env spec env if the action and observation spaces are not both present
        # in the dataset.
        observation_space = metadata.get("observation_space")
        action_space = metadata.get("action_space")
        if observation_space is None or action_space is None:
            # Checking if the base library of the environment is present in the environment
            entry_point = json.loads(env_spec)["entry_point"]
            lib_full_path = entry_point.split(":")[0]
            base_lib = lib_full_path.split(".")[0]
            env_name = self._env_spec.id

            try:
                env = gym.make(self._env_spec)
                if observation_space is None:
                    observation_space = env.observation_space
                if action_space is None:
                    action_space = env.action_space
                env.close()
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Install {base_lib} for loading {env_name} data"
                ) from e
        assert isinstance(observation_space, gym.spaces.Space)
        assert isinstance(action_space, gym.spaces.Space)
        self._observation_space = observation_space
        self._action_space = action_space

        if episode_indices is None:
            total_episodes = metadata["total_episodes"]
            assert isinstance(total_episodes, np.ndarray)
            episode_indices = np.arange(total_episodes.item())
            total_steps = metadata["total_steps"]
            assert isinstance(total_steps, np.ndarray)
            total_steps = total_steps.item()
        else:
            total_steps = sum(
                self._data.apply(
                    lambda episode: episode["total_timesteps"],
                    episode_indices=episode_indices,
                )
            )
        
        assert isinstance(episode_indices, np.ndarray)
        self._episode_indices: np.ndarray = episode_indices
        assert isinstance(total_steps, int)
        self._total_steps = total_steps

        assert self._episode_indices is not None

        self.spec = MinariDatasetSpec(
            env_spec=self.env_spec,
            total_episodes=self._episode_indices.size,
            total_steps=total_steps,
            dataset_id=self.id,
            combined_datasets=self.combined_datasets,
            observation_space=self.observation_space,
            action_space=self.action_space,
            data_path=str(self._data.data_path),
            minari_version=str(self.minari_version),
        )
        self._generator = np.random.default_rng()

    def recover_environment(self) -> gym.Env:
        """Recover the Gymnasium environment used to create the dataset.

        Returns:
            environment: Gymnasium environment
        """
        return gym.make(self.env_spec)

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

    # def update_dataset_from_collector_env(self, collector_env: DataCollectorV0):
    #     """Add extra data to Minari dataset from collector environment buffers (DataCollectorV0).

    #     This method can be used as a checkpoint when creating a dataset.
    #     A new HDF5 file will be created with the new dataset file in the same directory as `main_data.hdf5` called
    #     `additional_data_i.hdf5`. Both datasets are joined together by creating external links to each additional
    #     episode group: https://docs.h5py.org/en/stable/high/group.html#external-links

    #     Args:
    #         collector_env (DataCollectorV0): Collector environment
    #     """
    #     # check that collector env has the same characteristics as self._env_spec
    #     new_data_file_path = os.path.join(
    #         os.path.split(self.spec.data_path)[0],
    #         f"additional_data_{self._additional_data_id}.hdf5",
    #     )

    #     old_total_episodes = self._data.total_episodes

    #     self._data.update_from_collector_env(
    #         collector_env, new_data_file_path, self._additional_data_id
    #     )

    #     new_total_episodes = self._data._total_episodes

    #     self._additional_data_id += 1

    #     self._episode_indices = np.append(
    #         self._episode_indices, np.arange(old_total_episodes, new_total_episodes)
    #     )  # ~= np.append(self._episode_indices,np.arange(self._data.total_episodes))

    #     self.spec.total_episodes = self._episode_indices.size
    #     self.spec.total_steps = sum(
    #         self._data.apply(
    #             lambda episode: episode["total_timesteps"],
    #             episode_indices=self._episode_indices,
    #         )
    #     )

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
        self._data.update_episodes(buffer)
        new_total_episodes = self._data.total_episodes

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
        return len(self.episode_indices)

    @property
    def total_steps(self):
        """Total episodes steps in the Minari dataset."""
        return self._total_steps

    @property
    def episode_indices(self) -> np.ndarray:
        """Indices of the available episodes to sample within the Minari dataset."""
        return self._episode_indices

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

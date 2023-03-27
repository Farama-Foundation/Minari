import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import gymnasium as gym
import h5py
from gymnasium.envs.registration import EnvSpec


PathLike = Union[str, bytes, os.PathLike]


class MinariStorage:
    def __init__(self, data_path: PathLike):
        """Initialize properties of the Minari storage.

        Args:
            data_path (str): full path to the `main_data.hdf5` file of the dataset.
        """
        self._data_path = data_path
        self._extra_data_id = 0
        with h5py.File(self._data_path, "r") as f:
            flatten_observations = f.attrs["flatten_observation"].item()
            assert isinstance(flatten_observations, bool)
            self._flatten_observations = flatten_observations

            flatten_actions = f.attrs["flatten_action"].item()
            assert isinstance(flatten_actions, bool)
            self._flatten_actions = flatten_actions

            self._env_spec = EnvSpec.from_json(f.attrs["env_spec"])

            total_episodes = f.attrs["total_episodes"].item()
            assert isinstance(total_episodes, int)
            self._total_episodes: int = total_episodes

            total_steps = f.attrs["total_steps"].item()
            assert isinstance(total_steps, int)
            self._total_steps: int = total_steps

            dataset_id = f.attrs["dataset_id"]
            assert isinstance(dataset_id, str)
            self._dataset_id = dataset_id

            self._combined_datasets = f.attrs.get("combined_datasets", default=[])

            env = gym.make(self._env_spec)

            self._observation_space = env.observation_space
            self._action_space = env.action_space

            env.close()

    def apply(
        self,
        function: Callable[[h5py.Group], Any],
        episode_indices: Optional[Iterable] = None,
    ) -> List[Any]:
        """Apply a function to a slice of the data.

        Args:
            function (Callable): function to apply to episodes
            episode_indices (Optional[Iterable]): epsiodes id to consider

        Returns:
            outs (list): list of outputs returned by the function applied to episodes
        """
        if episode_indices is None:
            episode_indices = range(self.total_episodes)
        out = []
        with h5py.File(self._data_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                assert isinstance(ep_group, h5py.Group)
                out.append(function(ep_group))

        return out

    def _filter_episode_data(self, episode: h5py.Group) -> Dict[str, Any]:
        episode_data = {
            "id": episode.attrs.get("id"),
            "total_timesteps": episode.attrs.get("total_steps"),
            "seed": episode.attrs.get("seed"),
            "observations": episode["observations"][()],
            "actions": episode["actions"][()],
            "rewards": episode["rewards"][()],
            "terminations": episode["terminations"][()],
            "truncations": episode["truncations"][()],
        }

        return episode_data

    def get_episodes(self, episode_indices: Iterable[int]) -> List[dict]:
        """Get a list of episodes.

        Args:
            episode_indices (Iterable[int]): episodes id to return

        Returns:
            episodes (List[dict]): list of episodes data
        """
        out = []
        with h5py.File(self._data_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                out.append(self._filter_episode_data(ep_group))

        return out

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
    def total_steps(self):
        """Total steps recorded in the Minari dataset along all episodes."""
        return self._total_steps

    @property
    def total_episodes(self):
        """Total episodes recorded in the Minari dataset."""
        return self._total_episodes

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

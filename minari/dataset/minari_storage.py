import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import gymnasium as gym
import h5py
import numpy as np
from gymnasium.envs.registration import EnvSpec

from minari.serialization import deserialize_space


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

            # ww will default to using the reconstructed observation and action spaces from the dataset
            # and fall back to the env spec env if the action and observation spaces are not both present
            # in the dataset.
            if "action_space" in f.attrs and "observation_space" in f.attrs:
                self._observation_space = deserialize_space(
                    f.attrs["observation_space"]
                )
                self._action_space = deserialize_space(f.attrs["action_space"])
            else:
                # checking if the base library of the environment is present in the environment
                entry_point = json.loads(f.attrs["env_spec"])["entry_point"]
                lib_full_path = entry_point.split(":")[0]
                base_lib = lib_full_path.split(".")[0]
                env_name = self._env_spec.id

                try:
                    env = gym.make(self._env_spec)
                    self._observation_space = env.observation_space
                    self._action_space = env.action_space
                    env.close()
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        f"Install {base_lib} for loading {env_name} data"
                    ) from e

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

    def _h5_group_to_dict_recursive(
        self, hdf_ref: Union[h5py.Group, h5py.Dataset], timestep
    ) -> Union[Dict, Tuple, np.ndarray]:

        if isinstance(hdf_ref, h5py.Dataset):
            return hdf_ref[timestep]
        elif isinstance(hdf_ref, h5py.Group):
            if "Tuple" in hdf_ref.attrs.keys():
                result = []
                for i in range(len(hdf_ref.keys())):
                    result.append(
                        self._h5_group_to_dict_recursive(
                            hdf_ref[f"_index_{i}"], timestep
                        )
                    )
                return tuple(result)
            else:
                result = {}
                for key in hdf_ref:
                    result[key] = self._h5_group_to_dict_recursive(
                        hdf_ref[key], timestep
                    )
                return result
        else:
            raise TypeError(
                f"hdf_ref of type {type(hdf_ref)} is not h5py.Group or h5py.Dict"
            )

    def _reconstruct_space_from_h5(
        self, hdf_ref: h5py.Group, timesteps: int
    ) -> List[Union[Tuple, Dict]]:
        result = []
        for i in range(timesteps):
            result.append(self._h5_group_to_dict_recursive(hdf_ref, i))
        return result

    def _filter_episode_data(self, episode: h5py.Group) -> Dict[str, Any]:

        episode_data = {
            "id": episode.attrs.get("id"),
            "total_timesteps": episode.attrs.get("total_steps"),
            "seed": episode.attrs.get("seed"),
            "rewards": episode["rewards"][()],
            "terminations": episode["terminations"][()],
            "truncations": episode["truncations"][()],
        }
        if isinstance(episode["observations"], h5py.Group):
            episode_data["observations"] = self._reconstruct_space_from_h5(
                episode["observations"], episode.attrs.get("total_steps") + 1
            )
        else:
            episode_data["observations"] = episode["observations"][()]

        if isinstance(episode["actions"], h5py.Group):
            episode_data["actions"] = self._reconstruct_space_from_h5(
                episode["actions"], episode.attrs.get("total_steps")
            )
        else:
            episode_data["actions"] = episode["actions"][()]

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
